"""
Parquet Storage Adapter.
Stores OHLCV data as Parquet files for fast columnar reads.
Ideal for time-series analysis and seamless Pandas integration.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from models.market import OrderSide, OrderStatus, Trade
from storage.base import StorageAdapter


class ParquetStore(StorageAdapter):
    """Parquet file-based storage adapter."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self._ohlcv_dir = self.base_dir / "ohlcv"
        self._trades_dir = self.base_dir / "trades"
        self._sentiment_dir = self.base_dir / "sentiment"

    def initialize(self) -> None:
        """Create directory structure."""
        self._ohlcv_dir.mkdir(parents=True, exist_ok=True)
        self._trades_dir.mkdir(parents=True, exist_ok=True)
        self._sentiment_dir.mkdir(parents=True, exist_ok=True)

    def _ohlcv_path(self, symbol: str, timeframe: str = "1Day") -> Path:
        """Path to a symbol+timeframe OHLCV parquet file."""
        return self._ohlcv_dir / f"{symbol.upper()}_{timeframe}.parquet"

    def _trades_path(self) -> Path:
        """Path to the trades parquet file."""
        return self._trades_dir / "trades.parquet"

    def _sentiment_path(self, symbol: str) -> Path:
        """Path to a symbol's sentiment parquet file."""
        return self._sentiment_dir / f"{symbol.upper()}.parquet"

    # ─── OHLCV ────────────────────────────────────────────────────────

    def save_ohlcv(self, symbol: str, data: pd.DataFrame, timeframe: str = "1Day") -> int:
        """Append or overwrite OHLCV data for a symbol+timeframe."""
        if data.empty:
            return 0

        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        path = self._ohlcv_path(symbol, timeframe)

        if path.exists():
            existing = pd.read_parquet(path)
            # Merge & deduplicate by timestamp
            combined = pd.concat([existing, data]).drop_duplicates(
                subset=["timestamp"], keep="last"
            )
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        else:
            combined = data.sort_values("timestamp").reset_index(drop=True)

        table = pa.Table.from_pandas(combined)
        pq.write_table(table, path, compression="snappy")
        return len(data)

    def load_ohlcv(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Read OHLCV data from parquet with optional date filtering."""
        path = self._ohlcv_path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if start:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end:
            df = df[df["timestamp"] <= pd.Timestamp(end)]

        return df.sort_values("timestamp").reset_index(drop=True)

    # ─── Trades ───────────────────────────────────────────────────────

    def save_trade(self, trade: Trade) -> None:
        """Append a trade record to the trades parquet file."""
        if trade.id is None:
            trade.id = str(uuid.uuid4())

        new_row = pd.DataFrame([trade.to_dict()])
        path = self._trades_path()

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row

        table = pa.Table.from_pandas(combined)
        pq.write_table(table, path, compression="snappy")

    def load_trades(
        self,
        symbol: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[Trade]:
        """Load trade records from parquet."""
        path = self._trades_path()
        if not path.exists():
            return []

        df = pd.read_parquet(path)

        if symbol:
            df = df[df["symbol"] == symbol]
        if start:
            df = df[df["timestamp"] >= start.isoformat()]
        if end:
            df = df[df["timestamp"] <= end.isoformat()]

        trades = []
        for _, row in df.iterrows():
            trades.append(Trade(
                id=row.get("id"),
                symbol=row["symbol"],
                side=OrderSide(row["side"]),
                qty=float(row["qty"]),
                price=float(row["price"]),
                status=OrderStatus(row["status"]),
                timestamp=datetime.fromisoformat(str(row["timestamp"])),
                broker_order_id=row.get("broker_order_id"),
            ))
        return trades

    # ─── Sentiment ────────────────────────────────────────────────────

    def save_sentiment(self, symbol: str, data: pd.DataFrame) -> int:
        """Save sentiment data to Parquet."""
        if data.empty:
            return 0
            
        data = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        
        path = self._sentiment_path(symbol)
        
        if path.exists():
            existing = pd.read_parquet(path)
            # Deduplicate by timestamp + source if possible, but simplest is timestamp
            combined = pd.concat([existing, data]).drop_duplicates(
                subset=["timestamp", "source", "url"], keep="last"
            )
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        else:
            combined = data.sort_values("timestamp").reset_index(drop=True)
            
        table = pa.Table.from_pandas(combined)
        pq.write_table(table, path, compression="snappy")
        return len(data)

    def load_sentiment(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load sentiment data from Parquet."""
        path = self._sentiment_path(symbol)
        if not path.exists():
            return pd.DataFrame()
            
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        if start:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
            
        return df.sort_values("timestamp").reset_index(drop=True)

    # ─── Metadata ─────────────────────────────────────────────────────

    def get_latest_timestamp(self, symbol: str, timeframe: str = "1Day") -> Optional[datetime]:
        """Get the most recent timestamp from a symbol+timeframe parquet file."""
        path = self._ohlcv_path(symbol, timeframe)
        if not path.exists():
            return None

        df = pd.read_parquet(path, columns=["timestamp"])
        if df.empty:
            return None

        max_ts = pd.to_datetime(df["timestamp"]).max()
        return max_ts.to_pydatetime()

    def get_first_timestamp(self, symbol: str, timeframe: str = "1Day") -> Optional[datetime]:
        """Get the earliest timestamp from a symbol+timeframe parquet file."""
        path = self._ohlcv_path(symbol, timeframe)
        if not path.exists():
            return None

        # Parquet optimization: reading a single column is fast
        df = pd.read_parquet(path, columns=["timestamp"])
        if df.empty:
            return None

        min_ts = pd.to_datetime(df["timestamp"]).min()
        return min_ts.to_pydatetime()

    def list_symbols(self) -> list[str]:
        """List all unique symbols with stored parquet files."""
        symbols = set()
        if self._ohlcv_dir.exists():
            for p in self._ohlcv_dir.glob("*.parquet"):
                # Filename format: SYMBOL_TIMEFRAME.parquet
                name = p.stem
                parts = name.rsplit("_", 1)
                symbols.add(parts[0].upper())
        return sorted(list(symbols))

    def close(self) -> None:
        pass
