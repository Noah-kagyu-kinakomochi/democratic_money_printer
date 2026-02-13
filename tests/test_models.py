"""
Unit tests for data models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timezone
from models.market import (
    OHLCV, Signal, SignalType, Vote, ConsensusSignal,
    Trade, OrderSide, OrderStatus,
)


class TestOHLCV:
    def test_creation(self):
        bar = OHLCV(
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            open=150.0, high=155.0, low=148.0, close=152.0, volume=1000000,
        )
        assert bar.close == 152.0
        assert bar.volume == 1000000

    def test_fields_present(self):
        bar = OHLCV(
            timestamp=datetime.now(timezone.utc),
            symbol="TSLA",
            open=100.0, high=110.0, low=90.0, close=105.0, volume=500000,
        )
        assert bar.open == 100.0
        assert bar.high == 110.0
        assert bar.low == 90.0


class TestSignal:
    def test_buy_signal(self):
        sig = Signal(
            signal_type=SignalType.BUY,
            confidence=0.85,
            model_name="test",
            symbol="AAPL",
            reason="test reason",
        )
        assert sig.signal_type == SignalType.BUY
        assert sig.confidence == 0.85
        assert sig.symbol == "AAPL"

    def test_hold_signal(self):
        sig = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.1,
            model_name="test",
            symbol="TSLA",
            reason="no signal",
        )
        assert sig.signal_type == SignalType.HOLD

    def test_repr(self):
        sig = Signal(
            signal_type=SignalType.SELL,
            confidence=0.7,
            model_name="RSI",
            symbol="AAPL",
            reason="overbought",
        )
        r = repr(sig)
        assert "RSI" in r
        assert "SELL" in r


class TestVote:
    def test_weighted_score_buy(self):
        sig = Signal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            model_name="test",
            symbol="AAPL",
            reason="test",
        )
        vote = Vote(signal=sig, weight=1.5)
        # BUY: +confidence * weight = 0.8 * 1.5 = 1.2
        assert vote.weighted_score == pytest.approx(1.2)

    def test_weighted_score_sell(self):
        sig = Signal(
            signal_type=SignalType.SELL,
            confidence=0.6,
            model_name="test",
            symbol="AAPL",
            reason="test",
        )
        vote = Vote(signal=sig, weight=1.0)
        # SELL: -confidence * weight = -0.6
        assert vote.weighted_score == pytest.approx(-0.6)

    def test_weighted_score_hold(self):
        sig = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            model_name="test",
            symbol="AAPL",
            reason="test",
        )
        vote = Vote(signal=sig, weight=1.0)
        # HOLD: 0
        assert vote.weighted_score == 0.0


class TestConsensusSignal:
    def test_vote_summary(self):
        votes = [
            Vote(signal=Signal(SignalType.BUY, 0.8, "m1", "AAPL", ""), weight=1.0),
            Vote(signal=Signal(SignalType.BUY, 0.6, "m2", "AAPL", ""), weight=1.0),
            Vote(signal=Signal(SignalType.SELL, 0.5, "m3", "AAPL", ""), weight=1.0),
        ]
        consensus = ConsensusSignal(
            signal_type=SignalType.BUY,
            confidence=0.5,
            symbol="AAPL",
            votes=votes,
        )
        summary = consensus.vote_summary
        assert summary["BUY"] == 2
        assert summary["SELL"] == 1
        assert summary["HOLD"] == 0


class TestTrade:
    def test_creation(self):
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=10,
            price=150.0,
            status=OrderStatus.FILLED,
        )
        assert trade.symbol == "AAPL"
        assert trade.side == OrderSide.BUY
        assert trade.qty == 10
        assert trade.status == OrderStatus.FILLED
