"""
Core data models for MoneyPrinter.
These are the shared types used across all modules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SignalType(Enum):
    """Trading signal produced by a strategy model."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class OHLCV:
    """A single candlestick bar."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class Signal:
    """A trading signal produced by a single strategy model."""
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    model_name: str
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""

    def __repr__(self) -> str:
        return (
            f"Signal({self.model_name}: {self.signal_type.value} "
            f"{self.symbol} @ {self.confidence:.1%} — {self.reason})"
        )


@dataclass
class Vote:
    """A weighted vote from a strategy model in the democratic engine."""
    signal: Signal
    weight: float  # model's weight in the ensemble

    @property
    def weighted_score(self) -> float:
        """Positive for BUY, negative for SELL, zero for HOLD."""
        if self.signal.signal_type == SignalType.BUY:
            return self.weight * self.signal.confidence
        elif self.signal.signal_type == SignalType.SELL:
            return -self.weight * self.signal.confidence
        return 0.0


@dataclass
class ConsensusSignal:
    """The final democratic decision from the ensemble of models."""
    signal_type: SignalType
    confidence: float  # aggregated confidence
    symbol: str
    votes: list[Vote]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def vote_summary(self) -> dict:
        """Returns count of BUY/SELL/HOLD votes."""
        summary = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for v in self.votes:
            summary[v.signal.signal_type.value] += 1
        return summary

    def __repr__(self) -> str:
        counts = self.vote_summary
        return (
            f"Consensus({self.signal_type.value} {self.symbol} @ {self.confidence:.1%} "
            f"| BUY:{counts['BUY']} SELL:{counts['SELL']} HOLD:{counts['HOLD']})"
        )


@dataclass
class Trade:
    """A record of an executed trade."""
    id: Optional[str] = None
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    qty: float = 0.0
    price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    consensus: Optional[ConsensusSignal] = None  # the decision that triggered this trade
    broker_order_id: Optional[str] = None

    @property
    def value(self) -> float:
        return self.qty * self.price

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": self.qty,
            "price": self.price,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "broker_order_id": self.broker_order_id,
        }
