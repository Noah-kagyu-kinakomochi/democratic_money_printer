# MoneyPrinter Architecture & Command Usage

## Overview

MoneyPrinter is an automated trading bot/system designed with a hybrid local-cloud architecture. It separates the daily operational trading logic from the heavy machine learning training processes. 

The core flow is:
1. **Cloud-Native Daily Execution**: The system fetches market data, runs inference using pre-trained models, and executes trades via Alpaca. **Crucially, this is recommended to run on a highly-available cloud instance (e.g., Cloud Run, EC2) rather than a residential connection to ensure 99.9% uptime for trading capital.**
2. **Cloud Training**: High-Compute ML training is offloaded to the cloud (e.g., GCP, Databricks). Local data is bundled, streamed directly to a Google Cloud Storage (GCS) bucket, trained on an ephemeral GPU instance, and the resulting artifacts (models, scalers) are saved back to GCS.

---

## 🏗️ Architecture: Core Components

### 1. Data Ingestion (`data/`)
Handles gathering real-time and historical data from brokers (Alpaca) and other sources.
- `ingestion.py`: Pulls OHLCV market data and stores it locally (Parquet).
- `processor.py` & `exporter.py`: Processes raw data, adds technical indicators (RSI, MACD, etc.), merges sentiment data, and exports it into ML-ready datasets (`.parquet`).

### 2. Trading Strategy & Inference (`strategy/`)
The brain of the daily operation. It runs various models to generate buy/sell signals.
- **`dl_model.py` (Deep Learning)**: Uses a pre-trained PyTorch LSTM model. **Crucially, this model does NOT train locally.** On startup, it dynamically authenticates to your GCS bucket, downloads `best_model.pth` and `scaler.pkl`, and loads them into memory for inference.
- **Other Models**: `moving_average.py`, `rsi_strategy.py`, `gb_strategy.py` (Gradient Boosting), etc., generate their own signals.
- **`democracy.py`**: An ensemble voter that aggregates signals from all the different strategy models to form a final consensus.
- **`regime.py`**: Detects the current market regime (e.g., Bull, Bear, Sideways) to weight the models accordingly.

### 3. Training Bundle (`training_bundle/`)
A self-contained package solely for cloud ML training.
- `train.py`: A standalone PyTorch training loop. It reads a consolidated dataset (`data.parquet`), trains an LSTM (`PricePredictor` from `model_def.py`), and spits out the `best_model.pth` and `scaler.pkl` artifacts.
- It is designed to be completely decoupled from the Alpaca API and daily trading logic.

### 4. Automation & Workflows (`.github/workflows/`)
- `gcp-gpu-training.yml`: An automated pipeline to spin up an ephemeral GCP GPU instance, transfer the `training_bundle` and project code, execute `train.py`, save the `.pt` models to a GCS bucket, and destroy the VM.

---

## 🔄 The Lifecycle (Where Things Happen)

### **Phase 1: Cloud Training (Occurs periodically)**
*Where: Ephemeral GCP GPU Instance or Databricks*
1. **Export & Upload**: Run `python main.py package` locally. This triggers `exporter.py` to create a massive `data.parquet` and directly uploads it to your GCS Bucket (`gs://your-bucket/datasets/data.parquet`).
2. **Provision & Run**: The GitHub action provisions a GCP VM. The VM uses `gsutil` to lightning-fast download the dataset right before running `train.py`.
3. **Artifacts**: The cloud instance produces `best_model.pth` and `scaler.pkl` and uploads them back to GCS (`gs://your-bucket/models/`).

### **Phase 2: Daily Operations (Occurs daily/minutely)**
*Where: Cloud Run, GCP e2-micro, AWS ECS (Highly-Available Environments Recommended)*
> **⚠️ WARNING:** Do not run your daily live trading execution loop on a residential Raspberry Pi or personal laptop. Internet dropouts or power failures while holding leveraged positions can result in catastrophic capital loss.

1. **Startup**: `dl_model.py` fetches the latest models from GCS.
2. **Ingest**: Run `python main.py ingest`.
3. **Execute**: Run `python main.py run`.
   - The engine loads the latest data.
   - The models analyze the data and vote.
   - Trades are sent to the Alpaca Broker.

---

## 💻 Key Command Usages (`main.py`)

The primary entry point is `main.py`. Here are the most critical commands:

### Daily Trading & Operations
- `python main.py run`: Runs one complete trading cycle (Ingests recent data -> Analyzes -> Votes -> Executes trades via Alpaca).
- `python main.py ingest`: Only downloads the latest market data and updates local storage (no trading).
- `python main.py analyze`: Runs the inference logic and prints the signals/regime detection to the console, but **does not execute trades**. Good for monitoring.

### ML Training Lifecycle (The GCP Flow)
- `python main.py export`: Uses `exporter.py` to calculate technical indicators and dump the data into ML-ready parquet files.
- `python main.py package`: Runs the export, and then strictly bundles all parquet files into a single `training_bundle/data.parquet` file. **Run this before pushing to trigger GCP training.**

### Utilities & Analysis
- `python main.py account`: Shows current Alpaca account balances.
- `python main.py positions`: Shows currently held positions.
- `python main.py backtest`: Runs historical backtests for the models to evaluate hypothetical performance.

---
## Summary of the Cloud-Local Split
- `training_bundle/train.py` = **Cloud (GCP/GPU)**
- `main.py run` (and everything in `strategy/`) = **Local/Serverless** 
