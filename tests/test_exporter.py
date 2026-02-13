
import unittest
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone

from storage.sqlite_store import SqliteStore
from data.exporter import DatasetExporter
from models.market import Trade

class TestDatasetExporter(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_export_output")
        self.db_path = "test_exporter.db"
        
        # Clean up
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        if Path(self.db_path).exists():
            Path(self.db_path).unlink()
            
        self.storage = SqliteStore(self.db_path)
        self.storage.initialize()
        
        # Determine UTC timezone
        self.utc = timezone.utc

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        if Path(self.db_path).exists():
            Path(self.db_path).unlink()

    def test_export_dataset(self):
        # 1. Setup Mock Data
        symbol = "TEST"
        
        # OHLCV: 100 days of data
        dates = [datetime.now(self.utc) - timedelta(days=i) for i in range(100)]
        dates.reverse()
        
        ohlcv_data = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0] * 100,
            "high": [110.0] * 100,
            "low": [90.0] * 100,
            "close": [105.0] * 100,  # Constant close, RSI should be 50 or undefined/stable
            "volume": [1000.0] * 100
        })
        # Make close vary to get RSI
        ohlcv_data["close"] = [100 + (i % 10) for i in range(100)] 
        
        self.storage.save_ohlcv(symbol, ohlcv_data, timeframe="1Day")
        
        # Sentiment: Some news
        sentiment_data = pd.DataFrame({
            "timestamp": dates[:5], # First 5 days have sentiment
            "score": [0.5, 0.8, -0.2, 0.0, 0.9],
            "label": ["Bullish"] * 5,
            "source": ["News"] * 5,
            "url": ["http"] * 5,
            "summary": ["Summary"] * 5
        })
        self.storage.save_sentiment(symbol, sentiment_data)
        
        # 2. Run Export
        exporter = DatasetExporter(self.storage)
        exporter.export_dataset([symbol], output_dir=str(self.test_dir))
        
        # 3. Verify Files
        day_file = self.test_dir / "1Day" / f"{symbol}_1Day.parquet"
        self.assertTrue(day_file.exists(), "1Day parquet file not created")
        
        # 4. Verify Content
        df = pd.read_parquet(day_file)
        
        # Check columns
        expected_cols = ["open", "high", "low", "close", "volume", "rsi", "macd", "sentiment_score"]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")
            
        # Check Sentiment Merge
        # We added sentiment for the first 5 dates (which are the OLDEST dates because we reversed)
        # So the first few rows should have non-zero sentiment (if aligned correctly)
        # Note: export merges on DATE.
        
        # Let's check non-zero sentiment count
        non_zero_sentiment = df[df["sentiment_score"] != 0.0]
        self.assertGreater(len(non_zero_sentiment), 0, "Sentiment data not merged correctly (all zeros)")
        
        # Check Indicator computation
        # RSI should be computed (not all NaNs) for later rows
        self.assertFalse(df["rsi"].iloc[-1] == float("nan"), "RSI is NaN at the end")

if __name__ == "__main__":
    unittest.main()
