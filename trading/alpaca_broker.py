"""
Alpaca Broker Implementation.
Executes trades via Alpaca's Trading API (paper or live).
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from models.market import OrderSide, OrderStatus, Trade
from trading.base import Broker

logger = logging.getLogger(__name__)


class AlpacaBroker(Broker):
    """Alpaca-based trade execution."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        self.paper = paper
        mode = "PAPER" if paper else "LIVE"
        logger.info(f"Alpaca Broker initialized ({mode} mode)")

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: str = "market",
    ) -> Trade:
        """Submit an order to Alpaca."""
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        alpaca_side = AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=alpaca_side,
            time_in_force=TimeInForce.DAY,
        )

        try:
            order = self._client.submit_order(request)
            logger.info(
                f"Order submitted: {side.value} {qty} {symbol} "
                f"(ID: {order.id}, status: {order.status})"
            )

            return Trade(
                id=str(order.id),
                symbol=symbol,
                side=side,
                qty=float(qty),
                price=float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                status=self._map_status(str(order.status)),
                timestamp=datetime.now(timezone.utc),
                broker_order_id=str(order.id),
            )
        except Exception as e:
            logger.error(f"Order failed: {side.value} {qty} {symbol} — {e}")
            return Trade(
                symbol=symbol,
                side=side,
                qty=float(qty),
                status=OrderStatus.REJECTED,
                timestamp=datetime.now(timezone.utc),
            )

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get current position for a symbol."""
        try:
            position = self._client.get_open_position(symbol)
            return {
                "symbol": position.symbol,
                "qty": float(position.qty),
                "avg_entry_price": float(position.avg_entry_price),
                "market_value": float(position.market_value),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc),
                "side": position.side,
            }
        except Exception:
            return None

    def get_all_positions(self) -> list[dict]:
        """Get all open positions."""
        try:
            positions = self._client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_account(self) -> dict:
        """Get account information."""
        try:
            account = self._client.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "status": str(account.status),
            }
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return {"error": str(e)}

    def close_position(self, symbol: str) -> Optional[Trade]:
        """Close all shares of a position."""
        try:
            order = self._client.close_position(symbol)
            logger.info(f"Closed position: {symbol} (order ID: {order.id})")
            return Trade(
                id=str(order.id),
                symbol=symbol,
                side=OrderSide.SELL,
                qty=float(order.qty) if order.qty else 0,
                status=self._map_status(str(order.status)),
                timestamp=datetime.now(timezone.utc),
                broker_order_id=str(order.id),
            )
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return None

    @staticmethod
    def _map_status(alpaca_status: str) -> OrderStatus:
        """Map Alpaca order status to our OrderStatus enum."""
        mapping = {
            "new": OrderStatus.PENDING,
            "accepted": OrderStatus.PENDING,
            "pending_new": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(alpaca_status.lower(), OrderStatus.PENDING)
