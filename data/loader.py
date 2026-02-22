"""
Hybrid Data Loader.

Combines:
1. Historical Macro Data (from Yahoo Finance)
2. Live Market Data (from Alpaca/AlphaVantage)

Purpose: Provide a rich feature set for Regime Detection even if 
Alpaca history is limited.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    yf = None
    logger.warning("yfinance not found. Hybrid Data Loader will be limited.")


class HybridDataLoader:
    """
    Manages fetching and merging of Macro-Economic data.
    """
    
    MACRO_SYMBOLS = {
        "TNX": "^TNX",      # US 10-Year Treasury Yield
        "JGBS10": "^JGBS10",# JGB 10-Year Yield
        "DXY": "DX-Y.NYB",  # US Dollar Index
    }
    
    def __init__(self, storage_dir: str = "data/storage"):
        self.storage_dir = Path(storage_dir)
        self.macro_file = self.storage_dir / "macro_history.parquet"
        self.macro_data = pd.DataFrame()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def fetch_macro_history(self, lookback_years: int = 5):
        """
        Download historical macro data from Yahoo Finance.
        Saves to parquet for cache.
        """
        if not yf:
            logger.error("yfinance module missing. Cannot fetch macro history.")
            return

        logger.info(f"🌍 Fetching {lookback_years}y macro history via yfinance...")
        
        start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime("%Y-%m-%d")
        
        dfs = []
        for name, ticker in self.MACRO_SYMBOLS.items():
            try:
                # specific workarounds for yfinance randomness
                # We download daily data to ensure stability
                df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
                
                if df.empty:
                    logger.warning(f"  ❌ No data for {name} ({ticker})")
                    continue
                    
                # normalize index
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                
                # Keep only Close and maybe Volume
                # Rename columns: e.g. "SP500_Close"
                clean_df = df[["Close"]].copy()
                clean_df.columns = [f"{name}_Close"]
                
                # Resample to common timeline (1D) to align
                # This fills missing weekends for crypto but leaves gaps for stocks?
                # Actually index intersection is safer later.
                dfs.append(clean_df)
                logger.info(f"  ✅ {name} ({ticker}): {len(clean_df)} rows")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to fetch {name}: {e}")

        if not dfs:
            logger.error("Failed to fetch any macro data.")
            return

        # Merge all on index (outer join to keep crypto weekends if possible, 
        # but stocks are M-F. Forward fill is needed.)
        combined = pd.concat(dfs, axis=1)
        
        # Forward fill gaps (e.g. stock data on weekends for crypto alignment)
        combined = combined.ffill().fillna(0) # naive 0 fill for start
        
        # Save
        combined.to_parquet(self.macro_file)
        self.macro_data = combined
        logger.info(f"💾 Saved macro history to {self.macro_file} ({len(combined)} rows)")

    def load_macro_data(self):
        """Load cached macro data."""
        if self.macro_file.exists():
            try:
                self.macro_data = pd.read_parquet(self.macro_file)
                logger.info(f"📂 Loaded macro history ({len(self.macro_data)} rows)")
            except Exception as e:
                logger.error(f"Failed to load macro data: {e}")

    def get_latest_macro_context(self) -> dict:
        """
        Get the most recent known values for macro indicators.
        Useful for live trading context.
        """
        if self.macro_data.empty:
            self.load_macro_data()
            
        if self.macro_data.empty:
            return {}
            
        # Get last row
        last_row = self.macro_data.iloc[-1]
        timestamp = self.macro_data.index[-1]
        
        # Convert to simple dict
        context = last_row.to_dict()
        context['macro_timestamp'] = timestamp
        
        # Calculate derived metrics (e.g. Yield Diff)
        if "TNX_Close" in context and "JGBS10_Close" in context:
            context["Yield_Diff"] = context["TNX_Close"] - context["JGBS10_Close"]
            
        return context

    def merge_macro_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich a dataframe (OHLCV) with macro features.
        
        Args:
            df: DataFrame with DatetimeIndex (UTC)
            
        Returns:
            DataFrame with added macro columns (forward filled)
        """
        if self.macro_data.empty:
            self.load_macro_data()
            
        if self.macro_data.empty:
            return df
            
        # Handle case where timestamp is a column (not index)
        # DataFetcher.get_latest_bars returns RangeIndex with 'timestamp' column
        is_timestamp_col = False
        if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
            df = df.set_index("timestamp")
            is_timestamp_col = True
            
        # Ensure input df is timezone aware (UTC)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
            
        # FORCE NS RESOLUTION (Fix for pandas 2.0+ us/ns mismatch)
        df.index = df.index.astype("datetime64[ns, UTC]")
        if not self.macro_data.empty:
            self.macro_data.index = self.macro_data.index.astype("datetime64[ns, UTC]")
            
        # Merge logic
        # Macro data is usually 1Day resolution. Target might be 1Min.
        # We want to forward fill the macro data to the target timestamps.
        
        # Sort both to be safe
        df = df.sort_index()
        macro = self.macro_data.sort_index()
        
        # CRITICAL: Prevent Look-ahead Bias
        # Macro data comes with timestamps like "2023-10-27 00:00:00" which represents the Close of that day.
        # If we match this to "2023-10-27 10:00:00", we are giving the model the future Close price.
        # We must shift macro data by 1 day so that "2023-10-27" row contains data from "2023-10-26".
        # Actually, let's just shift the valuesdown by 1, keeping index? 
        # No, shift(1) moves data from t to t+1. 
        # If we shift(1), the row "2023-10-27" will contain "2023-10-26" data. 
        # Then merge_asof("2023-10-27 10:00:00") will pick up "2023-10-27" (which has old data). Correct.
        
        macro = macro.shift(1)
        
        # We can use merge_asof if inputs are sorted
        # But merge_asof requires columns to merge on, or index.
        # Let's use reindex/ffill approach which is robust for time series
        
        # Subset macro to relevant range (optimization)
        start = df.index[0] - timedelta(days=5) # buffer
        end = df.index[-1]
        macro_subset = macro[start:end]
        
        if macro_subset.empty:
            # Fallback for very recent data if macro isn't updated yet?
            # Or just use the last known value
            pass
            
        # Join: Left join on df.index
        # But we need ffill behavior.
        # pandas.merge_asof is best for "latest known value"
        
        merged = pd.merge_asof(
            df,
            macro,
            left_index=True,
            right_index=True,
            direction="backward" # fetch previous known value
        )
        
        # Restore original index/columns structure just in case
        if is_timestamp_col:
            merged = merged.reset_index()
            
        return merged
