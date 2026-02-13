"""
SQLite Storage Adapter.
Stores OHLCV data and trade records in a local SQLite database.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from models.market import OrderSide, OrderStatus, Trade
from storage.base import StorageAdapter


class SqliteStore(StorageAdapter):
    """SQLite-based storage adapter for local development."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection for a transaction block."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        """Create tables if they don't exist and handle migrations."""
        with self._connect() as conn:
            # OHLCV Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL DEFAULT '1Day',
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
            """)
            
            # Trades Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                pnl REAL DEFAULT 0.0
            )
            """)

            # ─── Migration: Check for pnl column in trades ───
            cursor = conn.execute("PRAGMA table_info(trades)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "pnl" not in columns:
                conn.execute("ALTER TABLE trades ADD COLUMN pnl REAL DEFAULT 0.0")

            # Sentiment Table (New)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment (
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                score REAL NOT NULL,
                label TEXT,
                summary TEXT,
                url TEXT,
                source TEXT,
                PRIMARY KEY (symbol, timestamp, source)
            )
            """)
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_tf ON ohlcv (symbol, timeframe)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_sym ON trades (symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_sym ON sentiment (symbol)")

    def save_ohlcv(self, symbol: str, data: pd.DataFrame, timeframe: str = "1Day") -> int:
        """Insert or replace OHLCV rows."""
        if data.empty:
            return 0

        rows = []
        for _, row in data.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, pd.Timestamp) or isinstance(ts, datetime):
                ts = ts.isoformat()
            
            rows.append((
                symbol, timeframe, ts,
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                float(row["volume"]),
            ))

        with self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO ohlcv
                   (symbol, timeframe, timestamp, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)

    def load_ohlcv(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Query OHLCV data with optional date filters."""
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE symbol = ? AND timeframe = ?"
        params: list = [symbol, timeframe]

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())

        query += " ORDER BY timestamp ASC"

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.tz_localize(timezone.utc)
            
        return df

    def save_trade(self, trade: Trade) -> None:
        """Insert a trade record."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO trades
                   (symbol, side, qty, price, timestamp, pnl)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    trade.symbol, trade.side.value,
                    trade.qty, trade.price,
                    trade.timestamp.isoformat(), 
                    trade.pnl if hasattr(trade, 'pnl') else 0.0
                ),
            )

    def load_trades(
        self,
        symbol: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[Trade]:
        """Query trade records."""
        query = "SELECT id, symbol, side, qty, price, timestamp, pnl FROM trades WHERE 1=1"
        params: list = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())

        query += " ORDER BY timestamp ASC"

        trades = []
        with self._connect() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                trades.append(Trade(
                    id=str(row["id"]),
                    symbol=row["symbol"],
                    side=OrderSide(row["side"]),
                    qty=float(row["qty"]),
                    price=float(row["price"]),
                    status=OrderStatus.FILLED,  # implicit
                    timestamp=datetime.fromisoformat(row["timestamp"]).replace(tzinfo=timezone.utc),
                    pnl=float(row["pnl"]) if row["pnl"] is not None else 0.0
                ))
        return trades

    def get_latest_timestamp(self, symbol: str, timeframe: str = "1Day") -> Optional[datetime]:
        """Get the most recent OHLCV timestamp for incremental fetching."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT MAX(timestamp) as max_ts FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
            row = cursor.fetchone()
            if row and row["max_ts"]:
                return datetime.fromisoformat(row["max_ts"]).replace(tzinfo=timezone.utc)
        return None

    def get_first_timestamp(self, symbol: str, timeframe: str = "1Day") -> Optional[datetime]:
        """Get the earliest OHLCV timestamp."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT MIN(timestamp) as min_ts FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
            row = cursor.fetchone()
            if row and row["min_ts"]:
                return datetime.fromisoformat(row["min_ts"]).replace(tzinfo=timezone.utc)
        return None

    def list_symbols(self) -> list[str]:
        """List all symbols with stored data."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol")
            return [row["symbol"] for row in cursor.fetchall()]

    # ─── Sentiment ────────────────────────────────────────────────────

    def save_sentiment(self, symbol: str, data: pd.DataFrame) -> int:
        """Save sentiment data for a symbol."""
        if data.empty:
            return 0
            
        records = []
        for _, row in data.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, pd.Timestamp) or isinstance(ts, datetime):
                ts = ts.isoformat()
                
            records.append((
                symbol,
                ts,
                float(row["score"]),
                str(row.get("label", "")),
                str(row.get("summary", "")),
                str(row.get("url", "")),
                str(row.get("source", ""))
            ))
            
        query = """
        INSERT OR REPLACE INTO sentiment (symbol, timestamp, score, label, summary, url, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        with self._connect() as conn:
            conn.executemany(query, records)
        return len(records)

    def load_sentiment(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load sentiment data for a symbol."""
        query = "SELECT timestamp, score, label, summary, url, source FROM sentiment WHERE symbol = ?"
        params = [symbol]
        
        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())
            
        query += " ORDER BY timestamp ASC"
        
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.tz_localize(timezone.utc)
            
        return df

    def close(self) -> None:
        """Legacy close method, not strictly needed with _connect context managers but good for interface."""
        pass
