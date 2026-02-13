"""
Unit tests for the Democratic Engine — voting methods and weight updates.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models.market import Signal, SignalType, ConsensusSignal
from strategy.base import ModelConfig
from strategy.models.moving_average import MovingAverageCrossover
from strategy.models.rsi_strategy import RSIStrategy
from strategy.models.macd_strategy import MACDStrategy
from strategy.democracy import DemocraticEngine
from tests.helpers import generate_ohlcv


def _make_models():
    return [
        MovingAverageCrossover(ModelConfig(name="MA", weight=1.0, params={"short_window": 10, "long_window": 30})),
        RSIStrategy(ModelConfig(name="RSI", weight=1.0, params={"period": 14})),
        MACDStrategy(ModelConfig(name="MACD", weight=1.0, params={"fast_period": 12, "slow_period": 26, "signal_period": 9})),
    ]


class TestDemocraticEngineVoting:
    def test_weighted_vote_produces_consensus(self):
        engine = DemocraticEngine(models=_make_models(), voting_method="weighted", min_confidence=0.3)
        data = generate_ohlcv(n=60)
        consensus = engine.vote("AAPL", data)
        assert isinstance(consensus, ConsensusSignal)
        assert consensus.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)
        assert 0.0 <= consensus.confidence <= 1.0

    def test_majority_vote_produces_consensus(self):
        engine = DemocraticEngine(models=_make_models(), voting_method="majority", min_confidence=0.3)
        data = generate_ohlcv(n=60)
        consensus = engine.vote("AAPL", data)
        assert isinstance(consensus, ConsensusSignal)

    def test_unanimous_vote_produces_consensus(self):
        engine = DemocraticEngine(models=_make_models(), voting_method="unanimous", min_confidence=0.0)
        data = generate_ohlcv(n=60)
        consensus = engine.vote("AAPL", data)
        assert isinstance(consensus, ConsensusSignal)

    def test_vote_summary_counts(self):
        engine = DemocraticEngine(models=_make_models(), voting_method="weighted", min_confidence=0.0)
        data = generate_ohlcv(n=60)
        consensus = engine.vote("AAPL", data)
        summary = consensus.vote_summary
        assert summary["BUY"] + summary["SELL"] + summary["HOLD"] == 3

    def test_minimum_confidence_threshold(self):
        # With very high threshold, consensus should default to HOLD
        engine = DemocraticEngine(models=_make_models(), voting_method="weighted", min_confidence=0.99)
        data = generate_ohlcv(n=60)
        consensus = engine.vote("AAPL", data)
        # With 99% threshold, almost always HOLD
        assert consensus.signal_type == SignalType.HOLD


class TestDemocraticEngineWeightUpdate:
    def test_update_weights(self):
        models = _make_models()
        engine = DemocraticEngine(models=models, voting_method="weighted")

        new_weights = {"MA": 0.5, "RSI": 0.3, "MACD": 0.2}
        engine.update_weights(new_weights)

        for m in engine.models:
            assert m.weight == pytest.approx(new_weights[m.name])

    def test_update_weights_partial(self):
        models = _make_models()
        engine = DemocraticEngine(models=models, voting_method="weighted")

        # Only update one model
        engine.update_weights({"RSI": 2.0})

        for m in engine.models:
            if m.name == "RSI":
                assert m.weight == pytest.approx(2.0)
            else:
                assert m.weight == pytest.approx(1.0)  # unchanged

    def test_no_models_raises(self):
        with pytest.raises(ValueError):
            DemocraticEngine(models=[], voting_method="weighted")

    def test_model_status(self):
        engine = DemocraticEngine(models=_make_models(), voting_method="weighted")
        status = engine.get_model_status()
        assert len(status) == 3
        assert all("name" in s and "weight" in s for s in status)
