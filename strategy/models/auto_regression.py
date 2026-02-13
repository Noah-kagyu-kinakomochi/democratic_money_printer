"""
Auto-Regression Strategy Model.

Inspired by LulutasoAI/FInancial_Analysis Auto_Correlation_Ten.py.

Uses lagged price features (t-1, t-2, ... t-N) with Linear Regression
to predict next-bar closing price direction.

Signals:
- BUY when predicted price > current price (bullish prediction)
- SELL when predicted price < current price (bearish prediction)
- HOLD when prediction is too close to current price

Confidence is based on the magnitude of predicted percentage change.
"""

import logging

import numpy as np
import pandas as pd

from models.market import Signal, SignalType
from strategy.base import ModelConfig, StrategyModel

logger = logging.getLogger(__name__)


class AutoRegressionStrategy(StrategyModel):
    """
    Lagged auto-regression strategy.

    Builds features from N lagged closing prices, trains a simple
    linear regression on a rolling window, and predicts next-bar
    direction. This is a time-series momentum model.
    """

    def __init__(self, config: ModelConfig = None):
        if config is None:
            config = ModelConfig(name="AutoReg", weight=1.0, params={
                "lags": 30,
                "train_window": 120,
            })
        super().__init__(config)
        self.lags = config.params.get("lags", 30)
        self.train_window = config.params.get("train_window", 120)

    @property
    def min_data_points(self) -> int:
        return self.lags + self.train_window + 5

    def _build_lagged_features(
        self, close: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build lagged feature matrix and target vector.

        For each bar t:
            X[t] = [close[t-1], close[t-2], ..., close[t-lags]]
            y[t] = close[t]

        Returns:
            (X, y) where X is (n_samples, lags) and y is (n_samples,)
        """
        values = close.values.astype(float)
        n = len(values)
        rows = n - self.lags

        X = np.zeros((rows, self.lags))
        y = np.zeros(rows)

        for i in range(rows):
            # Features: previous `lags` prices in reverse chronological order
            for lag in range(self.lags):
                X[i, lag] = values[self.lags + i - 1 - lag]
            # Target: current price
            y[i] = values[self.lags + i]

        return X, y

    def _fit_predict(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Fit linear regression on training data and predict next value.

        Uses normal equation (X^T X)^{-1} X^T y for speed —
        no sklearn dependency needed.
        """
        # Use only the most recent `train_window` samples for training
        if len(X) > self.train_window:
            X_train = X[-self.train_window - 1:-1]
            y_train = y[-self.train_window - 1:-1]
        else:
            X_train = X[:-1]
            y_train = y[:-1]

        # Normalize features for numerical stability
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std[X_std == 0] = 1.0  # avoid division by zero

        X_norm = (X_train - X_mean) / X_std

        # Add bias column
        ones = np.ones((X_norm.shape[0], 1))
        X_bias = np.hstack([ones, X_norm])

        # Normal equation with regularization (ridge)
        lambda_reg = 0.01
        XtX = X_bias.T @ X_bias + lambda_reg * np.eye(X_bias.shape[1])

        try:
            w = np.linalg.solve(XtX, X_bias.T @ y_train)
        except np.linalg.LinAlgError:
            return float(y[-1])  # fallback: predict current price

        # Predict using the latest features
        x_latest = X[-1:]
        x_latest_norm = (x_latest - X_mean) / X_std
        x_latest_bias = np.hstack([[1.0], x_latest_norm.flatten()])
        predicted = float(x_latest_bias @ w)

        return predicted

    def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        if len(data) < self.min_data_points:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                model_name=self.name,
                symbol=symbol,
                reason=f"Insufficient data ({len(data)} bars, need {self.min_data_points})",
            )

        close = data["close"]

        # Build features and predict
        X, y = self._build_lagged_features(close)
        if len(X) < 10:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                model_name=self.name,
                symbol=symbol,
                reason="Not enough samples after feature construction",
            )

        predicted_price = self._fit_predict(X, y)
        current_price = float(close.iloc[-1])

        if current_price <= 0:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                model_name=self.name,
                symbol=symbol,
                reason="Invalid current price",
            )

        # Calculate predicted change
        pct_change = (predicted_price - current_price) / current_price
        dead_zone = 0.001  # 0.1% — ignore noise

        if abs(pct_change) < dead_zone:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.1,
                model_name=self.name,
                symbol=symbol,
                reason=f"AutoReg prediction flat: {pct_change:+.4%}",
            )

        # Confidence: map |pct_change| to [0.3, 0.9]
        confidence = min(0.3 + abs(pct_change) * 30, 0.9)

        if pct_change > 0:
            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                model_name=self.name,
                symbol=symbol,
                reason=f"AutoReg predicts +{pct_change:.4%} (${current_price:.2f}→${predicted_price:.2f})",
            )
        else:
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                model_name=self.name,
                symbol=symbol,
                reason=f"AutoReg predicts {pct_change:.4%} (${current_price:.2f}→${predicted_price:.2f})",
            )
