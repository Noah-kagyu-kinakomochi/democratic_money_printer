"""
Abstract Broker Interface.
All trade execution backends must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional

from models.market import OrderSide, Trade


class Broker(ABC):
    """
    Abstract interface for trade execution.
    
    Implementations:
        - AlpacaBroker (stocks + crypto via Alpaca)
        - CcxtBroker   (crypto via CCXT — future)
    """

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: str = "market",
    ) -> Trade:
        """
        Submit a market order.
        
        Args:
            symbol: Ticker symbol
            side: BUY or SELL
            qty: Number of shares/units
            order_type: "market" or "limit"
        
        Returns:
            Trade record with broker order ID and status.
        """
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[dict]:
        """Get current position for a symbol. Returns None if no position."""
        ...

    @abstractmethod
    def get_all_positions(self) -> list[dict]:
        """Get all current positions."""
        ...

    @abstractmethod
    def get_account(self) -> dict:
        """Get account info (equity, cash, buying power)."""
        ...

    @abstractmethod
    def close_position(self, symbol: str) -> Optional[Trade]:
        """Close an entire position for a symbol."""
        ...
