"""
Abstract Data Source Interface.
All market data providers must implement this interface.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


class DataSource(ABC):
    """
    Abstract interface for market data sources.
    
    Implementations:
        - AlpacaDataSource   (stocks + crypto)
        - AlphaVantageSource (supplementary historical data — future)
    """

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data for a symbol.
        
        Args:
            symbol: Ticker symbol (e.g. "AAPL")
            start: Start datetime
            end: End datetime (defaults to now)
            timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        ...

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Get the current market price for a symbol."""
        ...

    @abstractmethod
    def get_account_info(self) -> dict:
        """Get account information (balance, equity, etc.)."""
        ...
