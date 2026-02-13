"""
Weight Learner.

Runs backtests for all models over a rolling window,
scores their performance using benchmark-relative metrics,
and normalizes scores into weights using softmax.

Training mode uses a composite score that penalizes long-bias:
  composite = 0.3 * alpha_sharpe + 0.3 * sortino + 0.2 * info_ratio + 0.2 * (1-MDD)

An always-buy model gets alpha ≈ 0, so it won't dominate the ensemble.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.scorer import ModelScore, ModelScorer
from strategy.base import StrategyModel

logger = logging.getLogger(__name__)


@dataclass
class WeightLearnerConfig:
    """Configuration for weight learning."""
    lookback_days: int = 30  # rolling backtest window
    temperature: float = 1.0  # softmax temperature (higher = more uniform)
    floor_weight: float = 0.00  # minimum weight per model (allow disabling bad models)
    blend: float = 0.3  # exponential smoothing: 0.3 = 30% new, 70% historical
    use_composite_score: bool = True  # True = training mode, False = raw Sharpe
    backtest_config: BacktestConfig = None

    def __post_init__(self):
        if self.backtest_config is None:
            self.backtest_config = BacktestConfig()


@dataclass
class WeightUpdate:
    """Result of a weight learning cycle."""
    weights: dict[str, float]  # model_name -> normalized weight
    scores: list[ModelScore]
    timestamp: str = ""

    def __repr__(self) -> str:
        w_str = ", ".join(f"{k}={v:.3f}" for k, v in self.weights.items())
        return f"WeightUpdate({w_str})"



def _run_backtest_job(model, symbol, data, engine_config) -> Optional[ModelScore]:
    """
    Helper function for parallel execution.
    Re-instantiates BacktestEngine to ensure thread/process safety.
    """
    # Create a fresh engine for this process
    local_engine = BacktestEngine(engine_config)
    
    # Run backtest
    result = local_engine.run(model, symbol, data)
    
    # Compute benchmark
    benchmark_returns = BacktestEngine.compute_benchmark_returns(
        data, min_lookback=model.min_data_points
    )
    
    # Score
    scorer = ModelScorer()
    score = scorer.score(result, benchmark_returns=benchmark_returns)
    return score


class WeightLearner:
    """
    Learns optimal model weights by backtesting each model independently,
    scoring their performance against a buy-and-hold benchmark,
    and normalizing into weights.

    Training weight formula:
    1. Backtest each model on [d-lookback, d-1]
    2. Compute benchmark (buy-and-hold) returns for same window
    3. Score = composite(alpha_sharpe, sortino, info_ratio, 1-MDD)
    4. Weights = softmax(scores / temperature)
    5. Apply floor weight (ensure no model goes to zero)
    """

    def __init__(self, config: WeightLearnerConfig = None):
        self.config = config or WeightLearnerConfig()
        # engine is not used directly in parallel mode, but kept for init check
        self.engine = BacktestEngine(self.config.backtest_config)
        self.scorer = ModelScorer()

    def learn_weights(
        self,
        models: list[StrategyModel],
        data_by_symbol: dict,
    ) -> WeightUpdate:
        """
        Run backtests for all models and compute normalized weights.

        Args:
            models: List of strategy models to evaluate
            data_by_symbol: Dict of symbol -> {timeframe -> DataFrame} (nested)
                            OR symbol -> DataFrame (flat, legacy)

        Returns:
            WeightUpdate with normalized weights and individual scores
        """
        from joblib import Parallel, delayed
        
        logger.info(f"🎓 Weight learning: {len(models)} models × {len(data_by_symbol)} symbols (Parallel)")

        tasks = []

        # Prepare tasks
        for model in models:
            tf = getattr(model.config, 'timeframe', '1Day')

            for symbol, sym_data in data_by_symbol.items():
                if isinstance(sym_data, dict):
                    data = sym_data.get(tf, pd.DataFrame())
                else:
                    data = sym_data

                if data.empty or len(data) < model.min_data_points + 5:
                    continue

                tasks.append((model, symbol, data))

        # Run in parallel
        # specific n_jobs=-1 uses all cores. 
        # We pass self.config.backtest_config to re-init engine in workers
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_run_backtest_job)(model, symbol, data, self.config.backtest_config)
            for model, symbol, data in tasks
        )

        # Aggregate results by model
        scores_by_model: dict[str, list[ModelScore]] = {}
        
        for score in results:
            if score:
                scores_by_model.setdefault(score.model_name, []).append(score)

        all_scores: list[ModelScore] = []

        # Average metrics across symbols for each model
        for model in models:
            model_scores = scores_by_model.get(model.name, [])
            
            if model_scores:
                avg_training = float(np.mean([s.training_score for s in model_scores]))
                avg_sharpe = float(np.mean([s.sharpe_ratio for s in model_scores]))
                avg_alpha = float(np.mean([s.alpha for s in model_scores]))
                avg_sortino = float(np.mean([s.sortino_ratio for s in model_scores]))
                avg_info = float(np.mean([s.information_ratio for s in model_scores]))
            else:
                avg_training = avg_sharpe = avg_alpha = avg_sortino = avg_info = 0.0

            all_scores.append(ModelScore(
                model_name=model.name,
                sharpe_ratio=avg_sharpe,
                alpha=avg_alpha,
                sortino_ratio=avg_sortino,
                information_ratio=avg_info,
                training_score=avg_training,
                total_return=0.0,
                total_trades=0,
            ))

            logger.info(
                f"  {model.name}: training={avg_training:.3f}, "
                f"alpha={avg_alpha:+.4f}, sharpe={avg_sharpe:.3f}"
            )

        # Normalize scores into weights
        weights = self._normalize_weights(all_scores)

        update = WeightUpdate(weights=weights, scores=all_scores)
        logger.info(f"🎓 Weight update: {update}")
        return update

    def _normalize_weights(self, scores: list[ModelScore]) -> dict[str, float]:
        """
        Convert scores into normalized weights using softmax.

        Uses composite training_score (benchmark-relative) when
        use_composite_score is True, otherwise falls back to raw Sharpe.

        Steps:
        1. Select score metric (composite or Sharpe)
        2. Clamp negatives to 0
        3. Apply softmax with temperature
        4. Apply floor weight and re-normalize
        """
        if not scores:
            return {}

        names = [s.model_name for s in scores]

        if self.config.use_composite_score:
            raw_scores = np.array([max(s.training_score, 0.0) for s in scores])
        else:
            raw_scores = np.array([max(s.sharpe_ratio, 0.0) for s in scores])

        # Handle all-zero case
        if np.sum(raw_scores) == 0:
            # Equal weights if all models performed equally poorly
            n = len(scores)
            return {name: 1.0 / n for name in names}

        # Softmax normalization
        temp = self.config.temperature
        exp_scores = np.exp(raw_scores / temp)
        softmax_weights = exp_scores / np.sum(exp_scores)

        # Apply floor weight
        floor = self.config.floor_weight
        n = len(scores)
        total_floor = floor * n

        if total_floor >= 1.0:
            # Floor alone exceeds 1.0, just use equal weights
            return {name: 1.0 / n for name in names}

        # Scale softmax weights into remaining space after floors
        remaining = 1.0 - total_floor
        final_weights = {}
        for i, name in enumerate(names):
            final_weights[name] = floor + remaining * float(softmax_weights[i])

        # Verify normalization
        total = sum(final_weights.values())
        if abs(total - 1.0) > 1e-6:
            # Re-normalize to sum exactly 1.0
            final_weights = {k: v / total for k, v in final_weights.items()}

        return final_weights

    def blend_with_previous(
        self,
        fresh_weights: dict[str, float],
        previous_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Exponential smoothing: blend fresh scores with historical weights.

        final = blend * fresh + (1 - blend) * previous

        New models (not in previous) get the average previous weight as prior.
        Removed models (not in fresh) are dropped.
        """
        blend = self.config.blend

        if not previous_weights:
            return fresh_weights

        # Average previous weight — used as prior for new models
        avg_previous = sum(previous_weights.values()) / len(previous_weights) if previous_weights else 0.0

        blended = {}
        for name, fresh_w in fresh_weights.items():
            prev_w = previous_weights.get(name, avg_previous)
            blended[name] = blend * fresh_w + (1 - blend) * prev_w

        # Re-normalize to sum to 1.0
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        return blended
