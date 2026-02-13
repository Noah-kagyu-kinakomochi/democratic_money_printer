
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import ta
from storage.base import StorageAdapter
from data.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)

class DatasetExporter:
    """
    Exports rich, ML-ready datasets from stored data.
    Combines:
    - OHLCV (Price/Volume)
    - Technical Indicators (RSI, MACD, etc.)
    - News Sentiment (aggregated)
    """

    def __init__(self, storage: StorageAdapter):
        self.storage = storage
        self.fetcher = DataFetcher(storage)

    def export_dataset(self, symbols: list[str], output_dir: str = "dataset") -> None:
        """
        Export data for the given symbols to the output directory.
        Generates two datasets per symbol:
        1. Daily (1Day) with indicators and sentiment
        2. Intraday (15Min) with indicators
        """
        base_path = Path(output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Starting export to {output_dir} for {len(symbols)} symbols")
        
        for symbol in symbols:
            self._export_symbol_timeframe(symbol, "1Day", base_path)
            self._export_symbol_timeframe(symbol, "15Min", base_path)
            
        logger.info(f"✅ Export complete. Data saved to {base_path}")

    def _export_symbol_timeframe(self, symbol: str, timeframe: str, base_path: Path):
        """Process and export a single symbol/timeframe combination."""
        try:
            # 1. Load OHLCV
            df = self.fetcher.get_ohlcv(symbol, timeframe=timeframe)
            
            if df.empty:
                logger.warning(f"⚠️  No data for {symbol} ({timeframe}) — skipping")
                return

            # 2. Feature Engineering: Technical Indicators
            df = self._add_technical_indicators(df)
            
            # 3. Feature Engineering: Sentiment (only for Daily/1Day for now)
            if timeframe == "1Day":
                df = self._add_sentiment_features(df, symbol)
            
            # 4. Save to Parquet
            filename = f"{symbol}_{timeframe}.parquet"
            tf_dir = base_path / timeframe
            tf_dir.mkdir(exist_ok=True)
            
            output_path = tf_dir / filename
            df.to_parquet(output_path)
            logger.info(f"  Saved {symbol} ({timeframe}): {len(df)} rows -> {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export {symbol} ({timeframe}): {e}")

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI, MACD, BB, ATR, SMA to the DataFrame using `ta` library."""
        df = df.copy()
        
        # Ensure clean data
        df = df.ffill().bfill()
        
        # Technical Indicators require 'close', 'high', 'low' columns
        # ta expects Series
        
        # Trend
        if len(df) > 50:
            df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
            df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
        
        if len(df) > 200:
            df["sma_200"] = ta.trend.sma_indicator(df["close"], window=200)
        
        # Momentum
        if len(df) > 26:
            df["rsi"] = ta.momentum.rsi(df["close"], window=14)
            
            macd = ta.trend.MACD(df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_diff"] = macd.macd_diff()
            
            df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        
        # Volatility
        if len(df) > 20:
            bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
            df["bb_high"] = bb.bollinger_hband()
            df["bb_low"] = bb.bollinger_lband()
            df["bb_width"] = bb.bollinger_wband()
        
        return df

    def _add_sentiment_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Merge aggregated news sentiment into the OHLCV DataFrame."""
        sentiment_df = self.storage.load_sentiment(symbol)
        
        if sentiment_df.empty:
            df["sentiment_score"] = 0.0
            return df
            
        # Resample sentiment to Daily mean
        if "timestamp" in sentiment_df.columns:
            sentiment_df = sentiment_df.set_index("timestamp")
            
        # Resample and aggregate score
        daily_sentiment = sentiment_df.resample("D").agg({
            "score": "mean"
        }).dropna()
        
        # Merge on date component
        # We assume df has 'timestamp' column
        df["date"] = df["timestamp"].dt.date
        daily_sentiment["date"] = daily_sentiment.index.date
        
        # Use simple merge
        # Using suffix just in case
        merged = pd.merge(df, daily_sentiment[["date", "score"]], on="date", how="left")
        merged.rename(columns={"score": "sentiment_score"}, inplace=True)
        merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
        
        merged.drop(columns=["date"], inplace=True)
        
        return merged
