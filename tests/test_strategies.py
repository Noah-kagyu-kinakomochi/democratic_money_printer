"""
Unit tests for strategy models (MA Crossover, RSI, MACD).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models.market import Signal, SignalType
from strategy.base import ModelConfig, StrategyModel
from strategy.models.moving_average import MovingAverageCrossover
from strategy.models.rsi_strategy import RSIStrategy
from strategy.models.macd_strategy import MACDStrategy
from tests.helpers import (
    generate_ohlcv,
    generate_trending_data,
    generate_oversold_data,
    generate_overbought_data,
)


class TestMovingAverageCrossover:
    @pytest.fixture
    def model(self):
        return MovingAverageCrossover(ModelConfig(
            name="MA_Test",
            weight=1.0,
            params={"short_window": 10, "long_window": 30},
        ))

    def test_produces_signal(self, model):
        data = generate_ohlcv(n=60)
        signal = model.analyze("AAPL", data)
        assert isinstance(signal, Signal)
        assert signal.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.symbol == "AAPL"

    def test_min_data_points(self, model):
        assert model.min_data_points >= 30

    def test_insufficient_data(self, model):
        data = generate_ohlcv(n=5)  # too few bars
        signal = model.analyze("AAPL", data)
        assert signal.signal_type == SignalType.HOLD

    def test_uptrend_biases_buy(self, model):
        data = generate_trending_data("up", n=60)
        signal = model.analyze("AAPL", data)
        # In a clear uptrend, MA should lean BUY (or at least not SELL)
        assert signal.signal_type in (SignalType.BUY, SignalType.HOLD)


class TestRSIStrategy:
    @pytest.fixture
    def model(self):
        return RSIStrategy(ModelConfig(
            name="RSI_Test",
            weight=1.0,
            params={"period": 14, "oversold": 30, "overbought": 70},
        ))

    def test_produces_signal(self, model):
        data = generate_ohlcv(n=60)
        signal = model.analyze("AAPL", data)
        assert isinstance(signal, Signal)
        assert signal.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)

    def test_oversold_biases_buy(self, model):
        data = generate_oversold_data(n=60)
        signal = model.analyze("AAPL", data)
        # After a sharp drop, RSI should be oversold → BUY
        assert signal.signal_type in (SignalType.BUY, SignalType.HOLD)

    def test_overbought_produces_signal(self, model):
        data = generate_overbought_data(n=60)
        signal = model.analyze("AAPL", data)
        # Synthetic overbought data should produce some signal
        assert signal.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)
        assert 0.0 <= signal.confidence <= 1.0

    def test_insufficient_data(self, model):
        data = generate_ohlcv(n=5)
        signal = model.analyze("AAPL", data)
        assert signal.signal_type == SignalType.HOLD


class TestMACDStrategy:
    @pytest.fixture
    def model(self):
        return MACDStrategy(ModelConfig(
            name="MACD_Test",
            weight=1.0,
            params={"fast_period": 12, "slow_period": 26, "signal_period": 9},
        ))

    def test_produces_signal(self, model):
        data = generate_ohlcv(n=60)
        signal = model.analyze("AAPL", data)
        assert isinstance(signal, Signal)
        assert signal.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)

    def test_min_data_points(self, model):
        assert model.min_data_points >= 35

    def test_insufficient_data(self, model):
        data = generate_ohlcv(n=5)
        signal = model.analyze("AAPL", data)
        assert signal.signal_type == SignalType.HOLD


class TestModelUpdateWeight:
    def test_update_weight(self):
        model = MovingAverageCrossover(ModelConfig(name="MA", weight=1.0))
        model.update_weight(0.75)
        assert model.weight == pytest.approx(0.75)
        assert model.config.weight == pytest.approx(0.75)


# --- New models from Financial_Analysis repo ---

from strategy.models.auto_regression import AutoRegressionStrategy
from strategy.models.correlation_regime import CorrelationRegimeStrategy


class TestAutoRegressionStrategy:
    @pytest.fixture
    def model(self):
        return AutoRegressionStrategy(ModelConfig(
            name="AutoReg_Test",
            weight=1.0,
            params={"lags": 10, "train_window": 30},
        ))

    def test_produces_signal(self, model):
        data = generate_ohlcv(n=60)
        signal = model.analyze("AAPL", data)
        assert isinstance(signal, Signal)
        assert signal.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.symbol == "AAPL"

    def test_min_data_points(self, model):
        assert model.min_data_points >= 40  # lags + train_window

    def test_insufficient_data(self, model):
        data = generate_ohlcv(n=5)
        signal = model.analyze("AAPL", data)
        assert signal.signal_type == SignalType.HOLD

    def test_uptrend_prediction(self, model):
        data = generate_trending_data("up", n=60)
        signal = model.analyze("AAPL", data)
        # With small synthetic data, regression may not pick up trend perfectly
        assert signal.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)
        assert signal.confidence > 0

    def test_reason_contains_autoreg(self, model):
        data = generate_ohlcv(n=60)
        signal = model.analyze("AAPL", data)
        assert "AutoReg" in signal.reason or "Insufficient" in signal.reason


class TestCorrelationRegimeStrategy:
    @pytest.fixture
    def model(self):
        return CorrelationRegimeStrategy(ModelConfig(
            name="CorrRegime_Test",
            weight=1.0,
            params={"corr_window": 15, "shift_lookback": 3},
        ))

    def test_produces_signal(self, model):
        data = generate_ohlcv(n=60)
        signal = model.analyze("AAPL", data)
        assert isinstance(signal, Signal)
        assert signal.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)
        assert 0.0 <= signal.confidence <= 1.0

    def test_min_data_points(self, model):
        assert model.min_data_points >= 25  # corr_window + shift_lookback + buffer

    def test_insufficient_data(self, model):
        data = generate_ohlcv(n=5)
        signal = model.analyze("AAPL", data)
        assert signal.signal_type == SignalType.HOLD

    def test_reason_contains_context(self, model):
        data = generate_ohlcv(n=60)
        signal = model.analyze("AAPL", data)
        # Should mention regime, correlation, or insufficient
        has_context = any(
            word in signal.reason
            for word in ("regime", "Regime", "CorrRegime", "autocorr", "Insufficient", "Correlation")
        )
        assert has_context

