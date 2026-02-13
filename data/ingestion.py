"""
Data Ingestion Coordinator.
Fetches market data from sources and stores it via the storage adapter.
Supports incremental ingestion (only fetches new data since last stored timestamp).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from data.sources.base import DataSource
from data.sources.alpha_vantage import AlphaVantageSource
from storage.base import StorageAdapter

logger = logging.getLogger(__name__)


class DataIngestionService:
    """
    Coordinates data fetching from sources and persistence to storage.
    Handles incremental updates to avoid re-fetching existing data.
    """

    def __init__(
        self,
        source: DataSource,
        storage: StorageAdapter,
        sentiment_source: Optional[AlphaVantageSource] = None
    ):
        self.source = source
        self.storage = storage
        self.sentiment_source = sentiment_source

    def ingest_symbol(
        self,
        symbol: str,
        lookback_days: int = 60,
        timeframe: str = "1Day",
    ) -> int:
        """
        Ingest OHLCV data for a single symbol.
        Performs incremental ingestion if data already exists.
        Skips entirely if stored data is fresh enough.
        
        Returns:
            Number of new rows stored (0 if skipped).
        """
        # Check what we already have
        latest = self.storage.get_latest_timestamp(symbol, timeframe)
        first = self.storage.get_first_timestamp(symbol, timeframe)
        
        now = datetime.now(timezone.utc)
        desired_start = now - timedelta(days=lookback_days)

        if latest and first:
            # Coverage check: do we have the requested history?
            # We allow a 5-day buffer for "start" to avoid tiny re-fetches
            has_history = first <= desired_start + timedelta(days=5)
            
            # Freshness check: is the latest data recent?
            age = now - latest
            max_age = timedelta(days=2) if timeframe == "1Day" else timedelta(days=1)
            is_fresh = age < max_age

            if has_history and is_fresh:
                logger.info(
                    f"💾 Storage: {symbol}/{timeframe} is fresh & covers history "
                    f"(range: {first.date()} to {latest.date()}, age: {age.total_seconds()/3600:.1f}h) — skipping API"
                )
                return 0

            if not has_history:
                # We need older data
                start = desired_start
                logger.info(f"Backfilling {symbol}: stored data starts {first.date()}, need {start.date()}")
            else:
                # We have history, just need to catch up to now
                start = latest - timedelta(days=1)
                logger.info(f"Incremental update for {symbol} from {start.date()}")
        else:
            # No data or partial metadata — full backfill
            start = desired_start
            logger.info(f"Full backfill for {symbol} from {start.date()} ({lookback_days} days)")

        # Fetch from source
        data = self.source.fetch_ohlcv(
            symbol=symbol,
            start=start,
            timeframe=timeframe,
        )

        if data.empty:
            logger.warning(f"No new data for {symbol}")
            return 0

        # Store
        rows_saved = self.storage.save_ohlcv(symbol, data, timeframe)
        logger.info(f"Stored {rows_saved} rows for {symbol} ({timeframe})")
        return rows_saved

    def ingest_symbols(
        self,
        symbols: list[str],
        lookback_days: int = 60,
        timeframe: str = "1Day",
    ) -> dict[str, int]:
        """
        Ingest OHLCV data for multiple symbols.
        
        Returns:
            Dict mapping symbol -> rows stored.
        """
        results = {}
        for symbol in symbols:
            try:
                rows = self.ingest_symbol(symbol, lookback_days, timeframe)
                results[symbol] = rows
            except Exception as e:
                logger.error(f"Failed to ingest {symbol}: {e}")
                results[symbol] = -1

        total = sum(v for v in results.values() if v > 0)
        logger.info(f"Ingestion complete: {total} total rows across {len(symbols)} symbols")
        return results

    def ingest_all_timeframes(
        self,
        symbols: list[str],
        timeframes: list[str],
        lookback_days: int = 60,
    ) -> dict[str, dict[str, int]]:
        """
        Ingest OHLCV data for multiple symbols and timeframes.

        Returns:
            Nested dict: {symbol: {timeframe: rows_stored}}
        """
        results = {}
        for symbol in symbols:
            results[symbol] = {}
            for tf in timeframes:
                try:
                    rows = self.ingest_symbol(symbol, lookback_days, tf)
                    results[symbol][tf] = rows
                except Exception as e:
                    logger.error(f"Failed to ingest {symbol}/{tf}: {e}")
                    results[symbol][tf] = -1

        total = sum(
            v for sym in results.values()
            for v in sym.values() if v > 0
        )
        logger.info(
            f"Multi-TF ingestion complete: {total} total rows across "
            f"{len(symbols)} symbols × {len(timeframes)} timeframes"
        )
        return results

    # ─── Sentiment Ingestion ──────────────────────────────────────────

    def ingest_sentiment(self, symbols: list[str], limit: int = 200) -> dict[str, int]:
        """
        Ingest News Sentiment for multiple symbols.
        
        Args:
            symbols: List of symbols to fetch sentiment for.
            limit: Max news items to fetch per symbol (default 200).
        """
        if not self.sentiment_source:
            logger.warning("⚠️ No sentiment source configured. Skipping sentiment ingestion.")
            return {}

        results = {}
        for symbol in symbols:
            try:
                logger.info(f"📰 Ingesting sentiment for {symbol}...")
                data = self.sentiment_source.fetch_news_sentiment(symbol, limit=limit)
                
                if data.empty:
                    logger.info(f"No sentiment data found for {symbol}")
                    results[symbol] = 0
                    continue
                
                saved = self.storage.save_sentiment(symbol, data)
                logger.info(f"Saved {saved} news items for {symbol}")
                results[symbol] = saved
                
            except Exception as e:
                logger.error(f"Failed to ingest sentiment for {symbol}: {e}")
                results[symbol] = -1
        
        return results
