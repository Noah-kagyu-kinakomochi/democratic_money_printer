"""
MACD (Moving Average Convergence Divergence) Strategy Model.

Signals:
- BUY when MACD line crosses above signal line (bullish momentum)
- SELL when MACD line crosses below signal line (bearish momentum)
- HOLD when no crossover detected

Confidence is based on histogram magnitude relative to price.
"""

import logging

import pandas as pd

from models.market import Signal, SignalType
from strategy.base import ModelConfig, StrategyModel

logger = logging.getLogger(__name__)


class MACDStrategy(StrategyModel):
    """MACD crossover strategy with histogram-based confidence."""

    def __init__(self, config: ModelConfig = None):
        if config is None:
            config = ModelConfig(name="MACD", weight=1.0, params={
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
            })
        super().__init__(config)
        self.fast_period = config.params.get("fast_period", 12)
        self.slow_period = config.params.get("slow_period", 26)
        self.signal_period = config.params.get("signal_period", 9)

    @property
    def min_data_points(self) -> int:
        return self.slow_period + self.signal_period + 5

    def _calculate_macd(self, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD line, signal line, and histogram."""
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        if len(data) < self.min_data_points:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                model_name=self.name,
                symbol=symbol,
                reason=f"Insufficient data ({len(data)} bars, need {self.min_data_points})",
            )

        macd_line, signal_line, histogram = self._calculate_macd(data["close"])

        curr_macd = macd_line.iloc[-1]
        curr_signal = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        curr_hist = histogram.iloc[-1]

        if pd.isna(curr_macd) or pd.isna(curr_signal):
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                model_name=self.name,
                symbol=symbol,
                reason="MACD calculation returned NaN",
            )

        # Histogram magnitude as % of price for confidence
        current_price = data["close"].iloc[-1]
        hist_pct = abs(curr_hist) / current_price if current_price > 0 else 0

        # Detect crossover
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            # Bullish crossover
            confidence = min(0.5 + hist_pct * 50, 1.0)
            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                model_name=self.name,
                symbol=symbol,
                reason=f"Bullish MACD crossover: MACD={curr_macd:.4f} > Signal={curr_signal:.4f}",
            )
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            # Bearish crossover
            confidence = min(0.5 + hist_pct * 50, 1.0)
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                model_name=self.name,
                symbol=symbol,
                reason=f"Bearish MACD crossover: MACD={curr_macd:.4f} < Signal={curr_signal:.4f}",
            )
        else:
            # No crossover — use histogram direction for lean
            if curr_hist > 0:
                confidence = min(hist_pct * 20, 0.35)
                return Signal(
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    model_name=self.name,
                    symbol=symbol,
                    reason=f"Bullish momentum: histogram={curr_hist:.4f} (no crossover)",
                )
            elif curr_hist < 0:
                confidence = min(hist_pct * 20, 0.35)
                return Signal(
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    model_name=self.name,
                    symbol=symbol,
                    reason=f"Bearish momentum: histogram={curr_hist:.4f} (no crossover)",
                )
            else:
                return Signal(
                    signal_type=SignalType.HOLD,
                    confidence=0.1,
                    model_name=self.name,
                    symbol=symbol,
                    reason="MACD flat — no clear momentum",
                )
