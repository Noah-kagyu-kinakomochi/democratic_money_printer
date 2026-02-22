# Strategy Models Documentation

This document details the strategy models available in the MoneyPrinter bot. 

> [!IMPORTANT]
> As of the migration to the serverless Cloud Run Architecture, **all legacy rule-based models (MA, RSI, MACD, Sentiment, Correlation) have been disabled**. 
> 
> The system now concentrates 100% of its memory, compute, and decision-making authority exclusively on the Deep Learning (`DL`) model. The DL model acts as an internal democracy, assigning Multi-Head Attention weights to various technical and macroeconomic inputs simultaneously.

---

## The Deep Learning Strategy (`DL`)

A PyTorch-based Time-Series Transformer network predicting 1-hour/15-minute price returns using a rich, dynamically weighted feature set.

### Architecture: Time-Series Transformer
The original Multi-Layer Perceptron (MLP) and LSTM architectures have been ripped out and upgraded to a highly capable **Encoder-Only Transformer**.
- **Positional Encoding**: Injects sine and cosine wave equations into the timeframes so the Transformer understands sequential order, even though it processes all data points simultaneously.
- **Multi-Head Self-Attention**: Allows the model to look at the entire sequence of pricing data and learn geometric relationships between specific features across time.
- **Optimizer**: `AdamW` (Adaptive Moment Estimation with decoupled weight decay) at `1e-4` LR.
- **Scheduler**: PyTorch `ReduceLROnPlateau` drops the learning rate dynamically if inference loss stagnates.
- **Decision Engine**:
  - `Predicted Return > 0.0` → **BUY**
  - `Predicted Return < 0.0` → **SELL**

### Inputs (8 Features per timeframe)
Instead of looking at 100+ raw normalized closing prices, the Transformer takes in sequences of exactly **8 engineered features** per bar:

1. **Asset Return**: Continuous $\ln$ return of the current asset (e.g., FXY).
2. **Asset Volume**: Raw trading volume of the asset.
3. **Sentiment Score**: The aggregated FinBERT NLP sentiment score for the asset over the last 24 hours from Alpha Vantage.
4. **VIX Level**: The raw CBOE Volatility Index representing broader market fear.
5. **SP500 Return**: The $\ln$ return of the broader stock market index.
6. **BTC Return**: The $\ln$ return of Bitcoin (utilized as a Crypto/Risk-On volatility indicator).
7. **XLK Return**: The $\ln$ return of the Technology Sector ETF.
8. **XLF Return**: The $\ln$ return of the Financial Sector ETF.

The Transformer analyzes sequences of 10 of these bars (so 80 discrete data points per batch) to predict the very next expected return period.

### Execution Flow
- **Training (Batch Process)**: Runs weekly via the `moneyprinter-train-job` leveraging an ephemeral Google Cloud NVIDIA Tesla L4 GPU.
- **Inference (Live Execution)**: The `moneyprinter-run-job` downloads the newly calculated `best_model.pth` artifact from GCS into RAM every 15 minutes, executes ultra-fast CPU inference over current market prices, casts the democratic vote, and submits Alpaca trades.
