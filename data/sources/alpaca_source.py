"""
Alpaca Market Data Source.
Fetches OHLCV data and account info from Alpaca's API.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from data.sources.base import DataSource

logger = logging.getLogger(__name__)


class AlpacaDataSource(DataSource):
    """Alpaca-based market data source using the official SDK."""

    # Mapping from our timeframe strings to Alpaca's TimeFrame
    _TIMEFRAME_MAP = {
        "1Min": None,   # Will be set in __init__ after import
        "5Min": None,
        "15Min": None,
        "1Hour": None,
        "1Day": None,
    }

    def __init__(self, api_key: str, secret_key: str, base_url: str = ""):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url

        # Lazy import alpaca SDK
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.trading.client import TradingClient

        self._data_client = StockHistoricalDataClient(api_key, secret_key)
        self._trading_client = TradingClient(api_key, secret_key)

        # Set up timeframe mapping
        self._TIMEFRAME_MAP = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Fetch historical bars from Alpaca."""
        from alpaca.data.requests import StockBarsRequest

        if end is None:
            end = datetime.now(timezone.utc) - timedelta(minutes=15)

        tf = self._TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
        )

        try:
            bars = self._data_client.get_stock_bars(request)
            bar_list = bars[symbol]
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        if not bar_list:
            logger.warning(f"No bars returned for {symbol} from {start} to {end}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        rows = []
        for bar in bar_list:
            rows.append({
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            })

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"🌍 API: Fetched {len(df)} bars for {symbol} ({timeframe})")
        return df

    def get_latest_price(self, symbol: str) -> float:
        """Get latest trade price from Alpaca."""
        from alpaca.data.requests import StockLatestTradeRequest

        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        try:
            trades = self._data_client.get_stock_latest_trade(request)
            return float(trades[symbol].price)
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            return 0.0

    def get_account_info(self) -> dict:
        """Get Alpaca account details."""
        try:
            account = self._trading_client.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "status": account.status.value if hasattr(account.status, 'value') else str(account.status),
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}
