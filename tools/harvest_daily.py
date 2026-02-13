#!/usr/bin/env python3
"""
Daily Harvest Script 🌾

Fetches the previous day's OHLCV data for all tracked symbols and appends it 
to the local Parquet storage.

Usage:
    python tools/harvest_daily.py

Scheduling:
    Run via cron every morning (e.g., 06:00 JST / 21:00 UTC).
    0 21 * * * cd /path/to/moneyprinter && /path/to/venv/bin/python tools/harvest_daily.py >> harvest.log 2>&1
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.getcwd())

from config.settings import load_settings
from core.engine import MoneyPrinterEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DailyHarvest")

def harvest():
    logger.info("🚜 Starting Daily Harvest...")
    
    try:
        settings = load_settings()
        
        # Initialize Engine (headless)
        engine = MoneyPrinterEngine(settings)
        
        symbols = settings.strategy.default_symbols
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        logger.info(f"📅 Target Date: {yesterday.strftime('%Y-%m-%d')}")
        logger.info(f"🌾 Symbols: {', '.join(symbols)}")
        
        # We use the ingestion service, but specifically target "recent" data.
        # ingest_all_timeframes usually handles backfill + recent.
        # By calling it, we ensure we have the latest.
        # It handles de-duplication internally.
        
        results = engine.ingestion.ingest_all_timeframes(
            symbols=symbols,
            timeframes=settings.strategy.timeframes,
            lookback_days=3, # Fetch last 3 days to be safe and cover weekends/holidays gaps
        )
        
        total_bars = 0
        for sym, tf_res in results.items():
            bars = sum(v for v in tf_res.values() if v >= 0)
            total_bars += bars
            logger.info(f"  ✅ {sym}: {bars} new bars")
            
        logger.info(f"🚜 Harvest Complete. Total bars: {total_bars}")
        
        # Also refresh macro data while we are at it
        from data.loader import HybridDataLoader
        try:
            loader = HybridDataLoader()
            loader.fetch_macro_history(lookback_years=5)
            logger.info("🌍 Macro data refreshed.")
        except Exception as e:
            logger.error(f"Failed to refresh macro data: {e}")

    except Exception as e:
        logger.critical(f"🔥 Harvest Failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'engine' in locals():
            engine.shutdown()

if __name__ == "__main__":
    harvest()
