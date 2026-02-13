"""
DataFetcher — Read-Only Data Access Interface.

Provides a clean, read-only interface for strategy models and the backtester
to query historical market data. Wraps the StorageAdapter but only exposes
read operations — models can never accidentally write.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from storage.base import StorageAdapter

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Read-only data access layer for strategy models and backtesting.
    
    Usage:
        fetcher = DataFetcher(storage)
        data = fetcher.get_ohlcv("AAPL", start, end)
        latest = fetcher.get_latest_bars("AAPL", n=30)
    """

    def __init__(self, storage: StorageAdapter):
        self._storage = storage

    def get_ohlcv(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a symbol and timeframe within date range.
        Returns DataFrame with [timestamp, open, high, low, close, volume].
        """
        # Smart Resampling:
        # If requesting 5Min/15Min, load 1Min data and resample it.
        # This saves us from storing redundant data.
        target_tf = timeframe
        load_tf = timeframe
        if timeframe in ["5Min", "15Min", "1Hour"]:
            load_tf = "1Min"

        df = self._storage.load_ohlcv(symbol, start=start, end=end, timeframe=load_tf)
        
        if not df.empty and load_tf != target_tf:
            df = self._resample(df, target_tf)
            logger.info(f"🔄 Resampled {len(df)} rows ({load_tf} -> {target_tf}) for {symbol}")
        elif not df.empty:
            logger.info(f"💾 Storage: Loaded {len(df)} rows for {symbol} ({timeframe})")
            
        return df

    def get_latest_bars(self, symbol: str, n: int = 60, timeframe: str = "1Day") -> pd.DataFrame:
        """
        Get the last N bars for a symbol at a given timeframe.
        Useful for models that need a fixed lookback window.
        """
        # Smart Resampling logic
        target_tf = timeframe
        load_tf = timeframe
        if timeframe in ["5Min", "15Min", "1Hour"]:
            load_tf = "1Min"
            
        all_data = self._storage.load_ohlcv(symbol, timeframe=load_tf)
        
        if not all_data.empty and load_tf != target_tf:
            all_data = self._resample(all_data, target_tf)
            logger.info(f"🔄 Resampled latest bars ({load_tf} -> {target_tf}) for {symbol}")
        elif not all_data.empty:
            logger.info(f"💾 Storage: Loaded {len(all_data)} rows for {symbol} ({timeframe})")
        
        if all_data.empty:
            return all_data

        return all_data.tail(n).reset_index(drop=True)

    def _resample(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample 1Min DataFrame to target timeframe."""
        freq_map = {
            "5Min": "5min",
            "15Min": "15min",
            "1Hour": "1h",
        }
        
        if target_timeframe not in freq_map:
            logger.warning(f"⚠️ Unknown resampling target: {target_timeframe}")
            return df
            
        freq = freq_map[target_timeframe]
        
        # Ensure timestamp is index
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        
        df = df.sort_index()
        
        # Resample OHLCV
        # Open=first, High=max, Low=min, Close=last, Volume=sum
        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        
        return resampled.reset_index()

    def get_bars_for_range(
        self,
        symbol: str,
        end_date: datetime,
        lookback_days: int = 60,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Get bars for a specific lookback window ending at end_date.
        Perfect for backtesting: simulate what a model saw on a given date.
        """
        start = end_date - timedelta(days=lookback_days)
        return self.get_ohlcv(symbol, start=start, end=end_date, timeframe=timeframe)

    def get_available_symbols(self) -> list[str]:
        """List all symbols with stored data."""
        return self._storage.list_symbols()

    def get_date_range(self, symbol: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get the earliest and latest timestamps for a symbol."""
        data = self._storage.load_ohlcv(symbol)
        if data.empty:
            return None, None
        return data["timestamp"].min().to_pydatetime(), data["timestamp"].max().to_pydatetime()

    def get_sentiment_score(self, symbol: str, lookback_window: timedelta, end_date: datetime = None) -> float:
        """
        Calculate average sentiment score for the window ending at end_date (default now).
        Returns 0.0 if no data found.
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Load all sentiment (not filtered by date in storage usually, depends on impl)
        # SqliteStore.load_sentiment(symbol) returns all rows
        df = self._storage.load_sentiment(symbol)
        
        if df.empty:
            return 0.0
            
        # Filter by date [end_date - window, end_date]
        start_date = end_date - lookback_window
        
        # Ensure df['timestamp'] is comparable
        # Assume storage loads with timezone info if native, or we align them
        
        # Optimized filtering
        mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
        recent = df.loc[mask]
        
        if recent.empty:
            return 0.0
            
        return float(recent["score"].mean())
