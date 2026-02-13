"""
Unit tests for storage adapters (SQLite and Parquet).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import tempfile
import shutil
from datetime import datetime

from storage.sqlite_store import SqliteStore
from storage.parquet_store import ParquetStore
from tests.helpers import generate_ohlcv


class TestSqliteStore:
    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s = SqliteStore(db_path)
        s.initialize()
        yield s
        s.close()

    def test_save_and_load(self, store):
        df = generate_ohlcv(n=30)
        saved = store.save_ohlcv("AAPL", df)
        assert saved == 30

        loaded = store.load_ohlcv("AAPL")
        assert len(loaded) == 30

    def test_deduplication(self, store):
        df = generate_ohlcv(n=30)
        store.save_ohlcv("AAPL", df)
        store.save_ohlcv("AAPL", df)  # same data again

        loaded = store.load_ohlcv("AAPL")
        assert len(loaded) == 30  # should not duplicate

    def test_multiple_symbols(self, store):
        df1 = generate_ohlcv(n=20, seed=1)
        df2 = generate_ohlcv(n=25, seed=2)
        store.save_ohlcv("AAPL", df1)
        store.save_ohlcv("TSLA", df2)

        assert len(store.load_ohlcv("AAPL")) == 20
        assert len(store.load_ohlcv("TSLA")) == 25
        assert set(store.list_symbols()) == {"AAPL", "TSLA"}

    def test_latest_timestamp(self, store):
        df = generate_ohlcv(n=30)
        store.save_ohlcv("AAPL", df)
        ts = store.get_latest_timestamp("AAPL")
        assert ts is not None

    def test_empty_load(self, store):
        loaded = store.load_ohlcv("NOEXIST")
        assert loaded.empty

    def test_first_timestamp(self, store):
        df = generate_ohlcv(n=30)
        store.save_ohlcv("AAPL", df)
        first = store.get_first_timestamp("AAPL")
        latest = store.get_latest_timestamp("AAPL")
        
        # generate_ohlcv produces daily data starting from now-30days
        # so first should be earlier than latest
        assert first is not None
        assert latest is not None
        assert first < latest
        
        # Verify it matches the data
        assert first == df["timestamp"].min()


class TestParquetStore:
    @pytest.fixture
    def store(self, tmp_path):
        pdir = str(tmp_path / "parquet_data")
        s = ParquetStore(pdir)
        s.initialize()
        yield s

    def test_save_and_load(self, store):
        df = generate_ohlcv(n=30)
        saved = store.save_ohlcv("AAPL", df)
        assert saved == 30

        loaded = store.load_ohlcv("AAPL")
        assert len(loaded) == 30

    def test_deduplication(self, store):
        df = generate_ohlcv(n=30)
        store.save_ohlcv("AAPL", df)
        store.save_ohlcv("AAPL", df)

        loaded = store.load_ohlcv("AAPL")
        assert len(loaded) == 30

    def test_multiple_symbols(self, store):
        df1 = generate_ohlcv(n=20, seed=1)
        df2 = generate_ohlcv(n=25, seed=2)
        store.save_ohlcv("AAPL", df1)
        store.save_ohlcv("TSLA", df2)

        assert len(store.load_ohlcv("AAPL")) == 20
        assert len(store.load_ohlcv("TSLA")) == 25

    def test_empty_load(self, store):
        loaded = store.load_ohlcv("NOEXIST")
        assert loaded.empty

    def test_first_timestamp(self, store):
        df = generate_ohlcv(n=30)
        store.save_ohlcv("AAPL", df)
        first = store.get_first_timestamp("AAPL")
        
        assert first is not None
        assert first == df["timestamp"].min()


class TestStorageAdapterSwap:
    """Verify that SQLite and Parquet produce identical results (storage is truly modular)."""

    def test_data_parity(self, tmp_path):
        df = generate_ohlcv(n=50)

        sqlite = SqliteStore(str(tmp_path / "test.db"))
        sqlite.initialize()
        sqlite.save_ohlcv("AAPL", df)

        parquet = ParquetStore(str(tmp_path / "parquet_data"))
        parquet.initialize()
        parquet.save_ohlcv("AAPL", df)

        sq_data = sqlite.load_ohlcv("AAPL")
        pq_data = parquet.load_ohlcv("AAPL")

        assert len(sq_data) == len(pq_data)
        # Verify close prices are identical
        assert list(sq_data["close"]) == list(pq_data["close"])

        sqlite.close()
