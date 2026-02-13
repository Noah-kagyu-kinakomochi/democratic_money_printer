"""
Abstract Storage Adapter Interface.

All storage backends (SQLite, Parquet, Databricks) must implement this interface.
This ensures the rest of the system is completely decoupled from the storage engine.
To switch backends, just swap the implementation — zero business logic changes needed.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from models.market import Trade


class StorageAdapter(ABC):
    """
    Abstract interface for all data storage operations.
    
    Implementations:
        - SqliteStore   (local, Phase 1)
        - ParquetStore  (local, Phase 1)
        - DatabricksStore (cloud, Phase 3 — future)
    """

    # ─── OHLCV Market Data ────────────────────────────────────────────

    @abstractmethod
    def save_ohlcv(self, symbol: str, data: pd.DataFrame, timeframe: str = "1Day") -> int:
        """
        Save OHLCV candlestick data for a symbol and timeframe.
        DataFrame must have columns: timestamp, open, high, low, close, volume.
        Returns the number of rows saved.
        """
        ...

    @abstractmethod
    def load_ohlcv(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol and timeframe within an optional date range.
        Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        ...

    # ─── Trades ───────────────────────────────────────────────────────

    @abstractmethod
    def save_trade(self, trade: Trade) -> None:
        """Save a trade execution record."""
        ...

    @abstractmethod
    def load_trades(
        self,
        symbol: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[Trade]:
        """Load trade records, optionally filtered by symbol and date range."""
        ...

    # ─── Metadata / Utilities ─────────────────────────────────────────

    @abstractmethod
    def get_latest_timestamp(self, symbol: str, timeframe: str = "1Day") -> Optional[datetime]:
        """Get the most recent data timestamp for a symbol+timeframe. Used for incremental ingestion."""
        ...

    @abstractmethod
    def get_first_timestamp(self, symbol: str, timeframe: str = "1Day") -> Optional[datetime]:
        """Get the earliest data timestamp for a symbol+timeframe. Used to check if we need backfill."""
        ...

    @abstractmethod
    def list_symbols(self) -> list[str]:
        """List all symbols that have stored data."""
        ...

    # ─── Alternative Data (Sentiment) ─────────────────────────────────

    @abstractmethod
    def save_sentiment(self, symbol: str, data: pd.DataFrame) -> int:
        """
        Save sentiment data for a symbol.
        DataFrame columns: [timestamp, score, label, summary, url, source]
        Returns number of rows saved.
        """
        ...

    @abstractmethod
    def load_sentiment(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load sentiment data for a symbol.
        Returns DataFrame with sentiment columns.
        """
        ...

    # ─── Lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    def initialize(self) -> None:
        """Set up the storage backend (create tables, directories, etc.)."""
        ...

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass
