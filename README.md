# MoneyPrinter 🖨️💰

Automated trading bot using democratic ensemble of strategy models with adaptive weight learning.

## Quick Start

```bash
# 1. Setup
cp .env.example .env          # Add your Alpaca API keys
pip install -r requirements.txt

# 2. Train weights (benchmark-relative scoring)
python main.py weights

# 3. Run simulation with trained weights
python main.py simulate
```

## Commands

| Command | Description |
|---------|-------------|
| `python main.py weights` | **Train** — learn adaptive weights via benchmark-relative scoring. Blends with previous weights (exponential smoothing) and saves to `data/weights.json` |
| `python main.py simulate` | **Evaluate** — full portfolio backtest (1 year, $1000). Loads saved weights. Reports alpha vs buy-and-hold, Sharpe, Sortino, Calmar |
| `python main.py backtest` | **Backtest** — individual model backtesting with per-model Sharpe ratios |
| `python main.py run` | **Live** — full trading cycle: weight learning → data ingestion → democratic voting → trade execution |
| `python main.py ingest` | **Ingest** — download market data only (no trading) |
| `python main.py analyze` | **Analyze** — run strategy analysis with **Regime Detection** (no trading) |
| `python main.py view macro` | **View Macro** — view historical macro data (SP500, VIX, BTC) |
| `python main.py account` | **Account** — show Alpaca account info |
| `python main.py positions` | **Positions** — show open positions |
| `python tools/harvest_daily.py` | **Harvest** — Cron script to fetch yesterday's OHLCV and append to storage |

## Key Features

### 🌊 Regime Awareness
The engine detects the current market regime (Trending Up/Down, Sideways Low/High Vol) using ADX and Volatility.
- **Trending Markets**: Boosts Trend-Following models (MA, MACD).
- **Sideways Markets**: Boosts Mean-Reversion models (RSI, Bollinger).
- **Dynamic Weighting**: Weights are adjusted in real-time based on the detected regime.

### 🌾 Hybrid Data Loading
Combines live Alapca data with historical macro data from Yahoo Finance (`yfinance`) to address data scarcity.
- **Macro Factors**: SP500, VIX, Treasury Yields, BTC, Sector ETFs.
- **Deep Learning**: The neural network strategy (`DL_1Min`) uses these 100+ features for price prediction.

## Architecture

```
main.py                     ← CLI entry point
├── core/engine.py          ← MoneyPrinterEngine (orchestrator)
├── strategy/
│   ├── democracy.py        ← Democratic voting engine (Regime Aware)
│   ├── regime.py           ← Market Regime Detector (ADX + Vol)
│   ├── base.py             ← StrategyModel base class
│   └── models/             ← 7 strategy models (MA, RSI, MACD, AutoReg, Corr, Sentiment, DL)
├── backtesting/
│   ├── engine.py           ← Single-model backtester
│   ├── scorer.py           ← Performance metrics (alpha, Sortino, Calmar, Info Ratio)
│   ├── weight_learner.py   ← Benchmark-relative weight learning + blending
│   ├── weight_store.py     ← JSON weight persistence
│   └── portfolio_backtest.py ← Full portfolio simulator
├── data/                   ← Data fetching & ingestion
│   ├── loader.py           ← Hybrid Data Loader (Macro + Live)
│   └── sources/            ← Alpaca, AlphaVantage, yfinance
├── tools/                  ← Utility scripts (harvest_daily.py)
├── storage/                ← SQLite + Parquet storage
├── trading/                ← Order execution
└── config/                 ← Settings & .env loading
```

### Ensemble

7 strategies × 2 timeframes (1Min, 1Day) = **14 models** voting democratically:

| Strategy | Technique |
|----------|-----------|
| Moving Average | Trend Following |
| RSI | Mean Reversion |
| MACD | Momentum |
| AutoRegression | Statistical Prediction |
| Correlation | Regime Detection |
| Sentiment | News Analysis |
| Deep Learning | Neural Network (Price + Macro) |

👉 **[See Detailed Model Documentation](docs/models.md)**

### Training Score (Anti Long-Bias)

Weight learning uses a composite score that penalizes always-buy strategies:

| Component | Weight | Purpose |
|-----------|--------|---------|
| Alpha-Sharpe | 30% | Edge over buy-and-hold |
| Sortino | 30% | Downside-risk-adjusted return |
| Information Ratio | 20% | Consistency of alpha |
| 1 − Max Drawdown | 20% | Capital preservation |

## Tests

```bash
python -m pytest tests/ -v    # Run unit tests
python tests/verify_data_integrity.py # Audit data pipeline (timezone/look-ahead)
```
