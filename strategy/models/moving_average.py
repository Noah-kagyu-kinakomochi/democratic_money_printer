"""
Moving Average Crossover Strategy Model.

Signals:
- BUY when short MA crosses above long MA (golden cross)
- SELL when short MA crosses below long MA (death cross)
- HOLD otherwise

Confidence is based on the magnitude of the MA separation.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from models.market import Signal, SignalType
from strategy.base import ModelConfig, StrategyModel

logger = logging.getLogger(__name__)


class MovingAverageCrossover(StrategyModel):
    """Dual moving average crossover strategy."""

    def __init__(self, config: ModelConfig = None):
        if config is None:
            config = ModelConfig(name="MA_Crossover", weight=1.0, params={
                "short_window": 5,
                "long_window": 20,
            })
        super().__init__(config)
        self.short_window = config.params.get("short_window", 10)
        self.long_window = config.params.get("long_window", 30)

    @property
    def min_data_points(self) -> int:
        return self.long_window + 5  # need some extra for crossover detection

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

        # Calculate moving averages
        short_ma = close.rolling(window=self.short_window).mean()
        long_ma = close.rolling(window=self.long_window).mean()

        # Current and previous values
        curr_short = short_ma.iloc[-1]
        curr_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]

        # Calculate MA spread as % of price for confidence
        current_price = close.iloc[-1]
        spread_pct = abs(curr_short - curr_long) / current_price

        # Detect crossover
        if prev_short <= prev_long and curr_short > curr_long:
            # Golden cross — short MA just crossed above long MA
            confidence = min(0.5 + spread_pct * 10, 1.0)
            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                model_name=self.name,
                symbol=symbol,
                reason=f"Golden cross: SMA{self.short_window}={curr_short:.2f} > SMA{self.long_window}={curr_long:.2f}",
            )
        elif prev_short >= prev_long and curr_short < curr_long:
            # Death cross — short MA just crossed below long MA
            confidence = min(0.5 + spread_pct * 10, 1.0)
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                model_name=self.name,
                symbol=symbol,
                reason=f"Death cross: SMA{self.short_window}={curr_short:.2f} < SMA{self.long_window}={curr_long:.2f}",
            )
        else:
            # No crossover — trend continuation
            if curr_short > curr_long:
                # Bullish trend but not a new cross
                confidence = min(spread_pct * 5, 0.4)
                return Signal(
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    model_name=self.name,
                    symbol=symbol,
                    reason=f"Bullish trend: SMA{self.short_window} above SMA{self.long_window} (no new cross)",
                )
            elif curr_short < curr_long:
                confidence = min(spread_pct * 5, 0.4)
                return Signal(
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    model_name=self.name,
                    symbol=symbol,
                    reason=f"Bearish trend: SMA{self.short_window} below SMA{self.long_window} (no new cross)",
                )
            else:
                return Signal(
                    signal_type=SignalType.HOLD,
                    confidence=0.1,
                    model_name=self.name,
                    symbol=symbol,
                    reason="MAs converged — no clear signal",
                )
