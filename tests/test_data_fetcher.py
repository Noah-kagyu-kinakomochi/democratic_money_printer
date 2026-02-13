
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from data.data_fetcher import DataFetcher
from storage.base import StorageAdapter

@pytest.fixture
def mock_storage():
    return MagicMock(spec=StorageAdapter)

@pytest.fixture
def sample_1min_data():
    # Create 5 minutes of 1Min data
    # Minute 1: O=10 H=12 L=9 C=11 V=100
    # Minute 2: O=11 H=13 L=10 C=12 V=100
    # Minute 3: O=12 H=14 L=11 C=13 V=100
    # Minute 4: O=13 H=15 L=12 C=14 V=100
    # Minute 5: O=14 H=16 L=13 C=15 V=100
    
    # Expected 5Min bar:
    # Open=10 (min 1 open)
    # High=16 (max of highs)
    # Low=9 (min of lows)
    # Close=15 (min 5 close)
    # Volume=500 (sum)
    
    base = datetime(2023, 1, 1, 10, 0)
    data = []
    for i in range(5):
        row = {
            "timestamp": base + timedelta(minutes=i),
            "open": 10 + i,
            "high": 12 + i,
            "low": 9 + i,
            "close": 11 + i,
            "volume": 100
        }
        data.append(row)
    return pd.DataFrame(data)

def test_resample_1min_to_5min(mock_storage, sample_1min_data):
    # Setup
    mock_storage.load_ohlcv.return_value = sample_1min_data
    fetcher = DataFetcher(mock_storage)
    
    # Execute: Request 5Min data
    df_5min = fetcher.get_ohlcv("AAPL", timeframe="5Min")
    
    # Verify: Storage was asked for 1Min
    mock_storage.load_ohlcv.assert_called_with("AAPL", start=None, end=None, timeframe="1Min")
    
    # Verify Resampling Result
    assert len(df_5min) == 1
    row = df_5min.iloc[0]
    
    assert row["open"] == 10.0
    assert row["high"] == 16.0
    assert row["low"] == 9.0
    assert row["close"] == 15.0
    assert row["volume"] == 500
    assert row["timestamp"] == datetime(2023, 1, 1, 10, 0)

def test_no_resample_needed(mock_storage, sample_1min_data):
    # Setup
    mock_storage.load_ohlcv.return_value = sample_1min_data
    fetcher = DataFetcher(mock_storage)
    
    # Execute: Request 1Min data
    df_1min = fetcher.get_ohlcv("AAPL", timeframe="1Min")
    
    # Verify: Storage was asked for 1Min
    mock_storage.load_ohlcv.assert_called_with("AAPL", start=None, end=None, timeframe="1Min")
    
    # Verify: Data returned as is
    pd.testing.assert_frame_equal(df_1min, sample_1min_data)

def test_get_latest_bars_resampling(mock_storage):
    # Create 10 minutes of 1Min data (two 5Min bars)
    base = datetime(2023, 1, 1, 10, 0)
    data = []
    for i in range(10):
        row = {
            "timestamp": base + timedelta(minutes=i),
            "open": 100, "high": 105, "low": 95, "close": 100, "volume": 100
        }
        data.append(row)
    df_1min = pd.DataFrame(data)
    
    mock_storage.load_ohlcv.return_value = df_1min
    fetcher = DataFetcher(mock_storage)
    
    # Request latest 1 bar of 5Min
    df_latest = fetcher.get_latest_bars("AAPL", n=1, timeframe="5Min")
    
    # Verify we get exactly 1 bar
    assert len(df_latest) == 1
    # Check it's the LAST 5min bar (10:05)
    assert df_latest.iloc[0]["timestamp"] == datetime(2023, 1, 1, 10, 5)
    assert df_latest.iloc[0]["volume"] == 500
