"""
Unit tests for backtesting engine, scorer, and weight learner.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult
from backtesting.scorer import ModelScorer, ModelScore
from backtesting.weight_learner import WeightLearner, WeightLearnerConfig
from strategy.base import ModelConfig
from strategy.models.moving_average import MovingAverageCrossover
from strategy.models.rsi_strategy import RSIStrategy
from strategy.models.macd_strategy import MACDStrategy
from tests.helpers import generate_ohlcv, generate_trending_data


class TestBacktestEngine:
    @pytest.fixture
    def engine(self):
        return BacktestEngine(BacktestConfig(initial_capital=100_000, slippage_pct=0.0))

    @pytest.fixture
    def model(self):
        return MovingAverageCrossover(ModelConfig(
            name="MA_Test", weight=1.0,
            params={"short_window": 10, "long_window": 30},
        ))

    def test_basic_backtest(self, engine, model):
        data = generate_ohlcv(n=60)
        result = engine.run(model, "AAPL", data)
        assert isinstance(result, BacktestResult)
        assert result.model_name == "MA_Test"
        assert result.symbol == "AAPL"
        assert result.initial_capital == 100_000

    def test_empty_data(self, engine, model):
        import pandas as pd
        result = engine.run(model, "AAPL", pd.DataFrame())
        assert result.total_trades == 0
        assert result.final_value == 100_000

    def test_equity_curve_produced(self, engine, model):
        data = generate_ohlcv(n=60)
        result = engine.run(model, "AAPL", data)
        assert len(result.equity_curve) > 0
        assert len(result.daily_returns) > 0

    def test_total_return_calculation(self, engine, model):
        data = generate_ohlcv(n=60)
        result = engine.run(model, "AAPL", data)
        expected = (result.final_value - result.initial_capital) / result.initial_capital
        assert result.total_return == pytest.approx(expected)

    def test_win_rate_bounds(self, engine, model):
        data = generate_ohlcv(n=60)
        result = engine.run(model, "AAPL", data)
        assert 0.0 <= result.win_rate <= 1.0


class TestModelScorer:
    @pytest.fixture
    def scorer(self):
        return ModelScorer()

    def test_score_basic(self, scorer):
        engine = BacktestEngine(BacktestConfig(slippage_pct=0.0))
        model = MovingAverageCrossover(ModelConfig(name="MA", weight=1.0, params={"short_window": 10, "long_window": 30}))
        data = generate_ohlcv(n=60)
        result = engine.run(model, "AAPL", data)

        score = scorer.score(result)
        assert isinstance(score, ModelScore)
        assert score.model_name == "MA"
        assert isinstance(score.sharpe_ratio, float)
        assert isinstance(score.max_drawdown, float)
        assert 0.0 <= score.max_drawdown <= 1.0

    def test_max_drawdown_flat_curve(self, scorer):
        mdd = scorer._max_drawdown([100, 100, 100, 100])
        assert mdd == pytest.approx(0.0)

    def test_max_drawdown_simple(self, scorer):
        # Peak at 100, drops to 80 = 20% drawdown
        mdd = scorer._max_drawdown([100, 90, 80, 85, 90])
        assert mdd == pytest.approx(0.2)

    def test_max_drawdown_empty(self, scorer):
        mdd = scorer._max_drawdown([])
        assert mdd == 0.0


class TestWeightLearner:
    def _make_models(self):
        return [
            MovingAverageCrossover(ModelConfig(name="MA", weight=1.0, params={"short_window": 10, "long_window": 30})),
            RSIStrategy(ModelConfig(name="RSI", weight=1.0, params={"period": 14})),
            MACDStrategy(ModelConfig(name="MACD", weight=1.0, params={"fast_period": 12, "slow_period": 26, "signal_period": 9})),
        ]

    def test_learn_weights_normalizes(self):
        learner = WeightLearner(WeightLearnerConfig(lookback_days=60, floor_weight=0.05))
        models = self._make_models()
        data = generate_ohlcv(n=60)

        update = learner.learn_weights(models, {"AAPL": data})

        # Weights should sum to ~1.0
        total = sum(update.weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_all_models_get_floor(self):
        learner = WeightLearner(WeightLearnerConfig(floor_weight=0.1))
        models = self._make_models()
        data = generate_ohlcv(n=60)

        update = learner.learn_weights(models, {"AAPL": data})

        # Every model should have at least floor weight
        for name, weight in update.weights.items():
            assert weight >= 0.1 - 1e-6

    def test_scores_produced(self):
        learner = WeightLearner()
        models = self._make_models()
        data = generate_ohlcv(n=60)

        update = learner.learn_weights(models, {"AAPL": data})

        assert len(update.scores) == 3
        assert all(isinstance(s, ModelScore) for s in update.scores)

    def test_empty_data(self):
        learner = WeightLearner()
        models = self._make_models()

        update = learner.learn_weights(models, {})

        # With no data, all models get equal weight (all scored 0)
        assert len(update.scores) == 3
        for score in update.scores:
            assert score.sharpe_ratio == 0.0

    def test_softmax_normalization(self):
        """Verify the _normalize_weights method produces valid softmax output."""
        learner = WeightLearner(WeightLearnerConfig(floor_weight=0.0, temperature=1.0))
        scores = [
            ModelScore(model_name="A", training_score=2.0),
            ModelScore(model_name="B", training_score=1.0),
            ModelScore(model_name="C", training_score=0.5),
        ]
        weights = learner._normalize_weights(scores)

        # A should have the highest weight, C the lowest
        assert weights["A"] > weights["B"] > weights["C"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_negative_score_clamped(self):
        """Models with negative training score should be clamped to 0."""
        learner = WeightLearner(WeightLearnerConfig(floor_weight=0.0, temperature=1.0))
        scores = [
            ModelScore(model_name="Good", training_score=2.0),
            ModelScore(model_name="Bad", training_score=-1.5),
        ]
        weights = learner._normalize_weights(scores)

        # Good should dominate, Bad gets minimal weight
        assert weights["Good"] > weights["Bad"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


class TestBenchmarkRelativeScoring:
    """Tests for benchmark-relative scoring (alpha, Sortino, Calmar, Info Ratio)."""

    @pytest.fixture
    def scorer(self):
        return ModelScorer()

    def test_alpha_with_benchmark(self, scorer):
        """Alpha should be model return minus benchmark return."""
        result = BacktestResult(
            model_name="Test", symbol="AAPL",
            initial_capital=100_000, final_value=110_000,
            daily_returns=[0.001] * 60,
            equity_curve=[100_000 + i * 166.67 for i in range(60)],
        )
        # Benchmark that also went up (like the market)
        bench_returns = [0.001] * 60
        score = scorer.score(result, benchmark_returns=bench_returns)
        # Same returns → alpha ≈ 0
        assert abs(score.alpha) < 0.01

    def test_alpha_outperformance(self, scorer):
        """Model beating benchmark should have positive alpha."""
        result = BacktestResult(
            model_name="Test", symbol="AAPL",
            initial_capital=100_000, final_value=110_000,
            daily_returns=[0.002] * 60,
            equity_curve=[100_000 + i * 166.67 for i in range(60)],
        )
        # Flat benchmark
        bench_returns = [0.0] * 60
        score = scorer.score(result, benchmark_returns=bench_returns)
        assert score.alpha > 0

    def test_sortino_ignores_upside(self, scorer):
        """Sortino should not penalize positive volatility."""
        # All positive returns = no downside deviation
        result = BacktestResult(
            model_name="Test", symbol="AAPL",
            initial_capital=100_000, final_value=105_000,
            daily_returns=[0.001] * 60,
            equity_curve=[100_000 + i * 83.33 for i in range(60)],
        )
        score = scorer.score(result)
        # Should be positive (no downside to penalize)
        assert score.sortino_ratio >= 0

    def test_composite_score_zero_without_benchmark(self, scorer):
        """Without benchmark, alpha-based components are 0, only drawdown contributes."""
        result = BacktestResult(
            model_name="Test", symbol="AAPL",
            initial_capital=100_000, final_value=100_000,
            daily_returns=[0.0] * 10,
            equity_curve=[100_000] * 10,
        )
        score = scorer.score(result)
        # training_score should only reflect drawdown component (1-0) * 0.2 = 0.2
        assert score.training_score == pytest.approx(0.2)

    def test_calmar_ratio_positive(self, scorer):
        """Calmar should be positive for profitable strategy with drawdown."""
        result = BacktestResult(
            model_name="Test", symbol="AAPL",
            initial_capital=100_000, final_value=110_000,
            daily_returns=[0.001, -0.0005] * 30,
            equity_curve=[100_000 + i * 166.67 for i in range(60)],
        )
        score = scorer.score(result)
        assert score.calmar_ratio >= 0


class TestWeightStore:
    """Tests for JSON weight persistence."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saved weights should load back identically."""
        from backtesting.weight_store import save_weights, load_weights
        path = tmp_path / "weights.json"

        weights = {"MA_1Day": 0.3, "RSI_1Day": 0.5, "MACD_1Day": 0.2}
        save_weights(weights, path=path)

        stored = load_weights(path=path)
        assert stored is not None
        assert stored.weights == weights
        assert stored.model_count == 3

    def test_load_missing_file(self, tmp_path):
        """Loading from nonexistent path returns None."""
        from backtesting.weight_store import load_weights
        stored = load_weights(path=tmp_path / "nope.json")
        assert stored is None

    def test_age_hours(self, tmp_path):
        """Freshly saved weights should be very young."""
        from backtesting.weight_store import save_weights, load_weights
        path = tmp_path / "weights.json"

        save_weights({"A": 1.0}, path=path)
        stored = load_weights(path=path)
        # Should be less than 1 minute old
        assert stored.age_hours < 0.02


class TestWeightBlending:
    """Tests for exponential smoothing (blend_with_previous)."""

    def test_blend_math(self):
        """Verify blending produces correct ratio."""
        learner = WeightLearner(WeightLearnerConfig(blend=0.4))
        fresh = {"A": 0.6, "B": 0.4}
        previous = {"A": 0.3, "B": 0.7}

        blended = learner.blend_with_previous(fresh, previous)

        # Before normalization: A = 0.4*0.6 + 0.6*0.3 = 0.42
        #                       B = 0.4*0.4 + 0.6*0.7 = 0.58
        # After normalization: sum = 1.0
        assert sum(blended.values()) == pytest.approx(1.0, abs=1e-6)
        assert blended["A"] < blended["B"]  # B had higher previous

    def test_new_model_gets_average(self):
        """New model (not in previous) gets average of previous weights."""
        learner = WeightLearner(WeightLearnerConfig(blend=0.5))
        fresh = {"A": 0.4, "B": 0.3, "C": 0.3}  # C is new
        previous = {"A": 0.6, "B": 0.4}  # No C

        blended = learner.blend_with_previous(fresh, previous)

        assert "C" in blended
        assert sum(blended.values()) == pytest.approx(1.0, abs=1e-6)

    def test_no_previous_passthrough(self):
        """With no previous weights, fresh weights pass through."""
        learner = WeightLearner(WeightLearnerConfig(blend=0.3))
        fresh = {"A": 0.5, "B": 0.5}

        blended = learner.blend_with_previous(fresh, {})

        assert blended == fresh
