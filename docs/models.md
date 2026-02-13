# Strategy Models Documentation

This document details the 7 strategy models available in the MoneyPrinter bot. Each model operates independently within the Democratic Ensemble.

## 1. Deep Learning Strategy (`DL`)

A PyTorch-based neural network predicting 1-hour price returns using a rich feature set.

### Architecture
- **Type**: Multi-Layer Perceptron (MLP)
- **Structure**: Input Layer -> Dense(64, ReLU) -> Dropout(0.2) -> Dense(32, ReLU) -> Dense(1)
- **Output**: Predicted % Turn for the next 1 hour.

### Inputs (103 Features)
1. **Price History (96)**: Log-normalized returns of the last 24 hours (resampled to 15-min intervals).
   - *Note*: For Daily timeframe models (`DL_1Day`), this uses the last 96 days of daily closes.
2. **Volume (1)**: Recent volume normalized by average volume.
3. **Sentiment (1)**: News Sentiment Score (from Alpha Vantage).
4. **Macro Data (5)**:
   - **SP500**: S&P 500 Index Close
   - **VIX**: Volatility Index
   - **BTC**: Bitcoin Price
   - **XLK**: Tech Sector ETF
   - **XLF**: Financial Sector ETF

### Logic
- **Training**: Auto-trains on historical data when running `python main.py weights`.
- **Inference**:
  - `Predicted Return > 0.5%` → **BUY**
  - `Predicted Return < -0.5%` → **SELL**

---

## 2. Moving Average Crossover (`MA`)

Classic trend-following strategy.

- **Logic**: Compares Short-term MA vs Long-term MA.
  - `Short MA > Long MA` → **BUY**
  - `Short MA < Long MA` → **SELL**
- **Parameters**:
  - `short_window`: 10 (bars)
  - `long_window`: 30 (bars)

---

## 3. RSI Strategy (`RSI`)

Mean-reversion strategy based on the Relative Strength Index.

- **Logic**:
  - `RSI < 30` (Oversold) → **BUY**
  - `RSI > 70` (Overbought) → **SELL**
- **Parameters**:
  - `period`: 14
  - `oversold`: 30
  - `overbought`: 70

---

## 4. MACD Strategy (`MACD`)

Momentum strategy using Moving Average Convergence Divergence.

- **Logic**:
  - `MACD Line > Signal Line` → **BUY**
  - `MACD Line < Signal Line` → **SELL**
- **Parameters**:
  - `fast_period`: 12
  - `slow_period`: 26
  - `signal_period`: 9

---

## 5. AutoRegression (`AutoReg`)

Statistical model predicting future price based on past lags.

- **Logic**: Uses Ridge Regression to fit recent price changes.
  - `Predicted Price > Current Price` → **BUY** (if confidence high)
  - `Predicted Price < Current Price` → **SELL**
- **Parameters**:
  - `lags`: 30
  - `train_window`: 120

---

## 6. Correlation Regime (`CorrRegime`)

Detects market regimes using Price-Volume Correlation.

- **Logic**:
  - **High Correlation (> 0.5)**: Trend is strong (Volume confirming Price).
    - If Price Up + Vol Confirmed → **BUY**
    - If Price Down + Vol Confirmed → **SELL**
  - **Low Correlation**: Market is chopping/uncertain → **HOLD**
- **Parameters**:
  - `corr_window`: 30
  - `shift_lookback`: 5

---

## 7. Sentiment Strategy (`Sentiment`)

Fundamental analysis using news sentiment.

- **Logic**: Aggregates news sentiment scores (0 to 1) over the last 24 hours.
  - `Average Score > 0.15` (Positive) → **BUY**
  - `Average Score < -0.15` (Negative) → **SELL**
- **Data Source**: Alpha Vantage News API.
- **Parameters**:
  - `threshold`: 0.15
  - `lookback_hours`: 24
