"""
RSI (Relative Strength Index) Strategy Model.

Signals:
- BUY when RSI drops below oversold threshold (default 30) — market is oversold
- SELL when RSI rises above overbought threshold (default 70) — market is overbought
- HOLD when RSI is in the neutral zone

Confidence scales with how extreme the RSI reading is.
"""

import logging

import pandas as pd

from models.market import Signal, SignalType
from strategy.base import ModelConfig, StrategyModel

logger = logging.getLogger(__name__)


class RSIStrategy(StrategyModel):
    """RSI-based mean reversion strategy."""

    def __init__(self, config: ModelConfig = None):
        if config is None:
            config = ModelConfig(name="RSI", weight=1.0, params={
                "period": 14,
                "oversold": 30,
                "overbought": 70,
            })
        super().__init__(config)
        self.period = config.params.get("period", 14)
        self.oversold = config.params.get("oversold", 30)
        self.overbought = config.params.get("overbought", 70)

    @property
    def min_data_points(self) -> int:
        return self.period + 5

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI using the standard formula."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, float('inf'))
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        if len(data) < self.min_data_points:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                model_name=self.name,
                symbol=symbol,
                reason=f"Insufficient data ({len(data)} bars, need {self.min_data_points})",
            )

        rsi = self._calculate_rsi(data["close"])
        current_rsi = rsi.iloc[-1]

        if pd.isna(current_rsi):
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                model_name=self.name,
                symbol=symbol,
                reason="RSI calculation returned NaN",
            )

        if current_rsi <= self.oversold:
            # Oversold — potential bounce / buy opportunity
            # More extreme = higher confidence
            extremity = (self.oversold - current_rsi) / self.oversold
            confidence = min(0.5 + extremity, 1.0)
            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                model_name=self.name,
                symbol=symbol,
                reason=f"Oversold: RSI={current_rsi:.1f} (threshold={self.oversold})",
            )
        elif current_rsi >= self.overbought:
            # Overbought — potential pullback / sell opportunity
            extremity = (current_rsi - self.overbought) / (100 - self.overbought)
            confidence = min(0.5 + extremity, 1.0)
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                model_name=self.name,
                symbol=symbol,
                reason=f"Overbought: RSI={current_rsi:.1f} (threshold={self.overbought})",
            )
        else:
            # Neutral zone
            # Slight lean based on which threshold is closer
            distance_to_oversold = current_rsi - self.oversold
            distance_to_overbought = self.overbought - current_rsi
            neutral_range = self.overbought - self.oversold

            confidence = 0.1 + 0.2 * (1 - min(distance_to_oversold, distance_to_overbought) / (neutral_range / 2))

            return Signal(
                signal_type=SignalType.HOLD,
                confidence=confidence,
                model_name=self.name,
                symbol=symbol,
                reason=f"Neutral: RSI={current_rsi:.1f} (range {self.oversold}-{self.overbought})",
            )
