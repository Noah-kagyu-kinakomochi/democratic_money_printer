"""
Sentiment Analysis Strategy.

Signals:
- BUY if average sentiment score > threshold
- SELL if average sentiment score < -threshold
- HOLD otherwise

Requires access to DataFetcher to retrieve sentiment data.
"""

from datetime import timedelta
import logging

import pandas as pd

from models.market import Signal, SignalType
from strategy.base import ModelConfig, StrategyModel

logger = logging.getLogger(__name__)

class SentimentStrategy(StrategyModel):
    """
    Trading strategy based on News Sentiment.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.fetcher = None
        self.threshold = config.params.get("threshold", 0.15)
        self.lookback_hours = config.params.get("lookback_hours", 24)

    def set_fetcher(self, fetcher):
        """Dependency injection for DataFetcher."""
        self.fetcher = fetcher

    @property
    def min_data_points(self) -> int:
        return 0  # We don't strictly need OHLCV history, but engine might enforce it

    def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Analyze sentiment score.
        Note: 'data' (OHLCV) is ignored, we fetch sentiment directly.
        """
        if not self.fetcher:
            return Signal(
                SignalType.HOLD, 
                0.0, 
                self.name, 
                symbol, 
                "Error: DataFetcher not configured"
            )

        # Calculate lookback window
        window = timedelta(hours=self.lookback_hours)
        
        # Determine "now" from the data (for backtesting accuracy)
        # If live, data.iloc[-1] is effectively now.
        if data.empty:
             return Signal(SignalType.HOLD, 0.0, self.name, symbol, "No data for timestamp")

        current_time = data.iloc[-1]["timestamp"]

        # Fetch score
        score = self.fetcher.get_sentiment_score(symbol, window, end_date=current_time)
        
        # Logic
        if score > self.threshold:
            confidence = min(abs(score) * 2, 1.0) # Map 0.15->0.3, 0.5->1.0
            return Signal(
                SignalType.BUY,
                confidence,
                self.name,
                symbol,
                f"Positive Sentiment: {score:.2f} > {self.threshold}"
            )
        elif score < -self.threshold:
            confidence = min(abs(score) * 2, 1.0)
            return Signal(
                SignalType.SELL,
                confidence,
                self.name,
                symbol,
                f"Negative Sentiment: {score:.2f} < -{self.threshold}"
            )
        else:
            return Signal(
                SignalType.HOLD,
                0.0,
                self.name,
                symbol,
                f"Neutral Sentiment: {score:.2f} (within ±{self.threshold})"
            )
