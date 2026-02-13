"""
Model Scorer.

Calculates performance metrics from backtest results.
Supports both raw metrics and benchmark-relative (alpha) metrics.

Training mode: uses composite score (alpha + risk-adjusted)
Evaluation mode: reports alpha and all metrics for review
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backtesting.engine import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class ModelScore:
    """Performance score for a single model."""
    model_name: str
    # Raw metrics
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    annualized_return: float = 0.0
    volatility: float = 0.0
    # Benchmark-relative metrics
    alpha: float = 0.0  # excess return over buy-and-hold
    sortino_ratio: float = 0.0  # return / downside volatility
    calmar_ratio: float = 0.0  # annualized return / max drawdown
    information_ratio: float = 0.0  # alpha / tracking error
    # Composite training score
    training_score: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Score({self.model_name}: sharpe={self.sharpe_ratio:.3f}, "
            f"alpha={self.alpha:+.2%}, sortino={self.sortino_ratio:.3f}, "
            f"calmar={self.calmar_ratio:.3f}, info_ratio={self.information_ratio:.3f}, "
            f"train={self.training_score:.3f})"
        )


class ModelScorer:
    """
    Scores backtest results into performance metrics.

    Two modes:
    - Training: composite score penalizes long-bias (always-buy gets ~0)
    - Evaluation: reports alpha and all metrics for review
    """

    TRADING_DAYS_PER_YEAR = 252

    # Training composite weights
    WEIGHT_ALPHA_SHARPE = 0.30
    WEIGHT_SORTINO = 0.30
    WEIGHT_INFO_RATIO = 0.20
    WEIGHT_DRAWDOWN = 0.20

    def score(
        self,
        result: BacktestResult,
        benchmark_returns: Optional[list[float]] = None,
    ) -> ModelScore:
        """
        Calculate performance metrics from a backtest result.

        Args:
            result: BacktestResult from the backtesting engine
            benchmark_returns: Daily returns of buy-and-hold benchmark.
                If provided, computes alpha-based metrics.

        Returns:
            ModelScore with all metrics
        """
        returns = np.array(result.daily_returns) if result.daily_returns else np.array([0.0])

        # --- Raw metrics ---

        # Sharpe ratio (annualized, risk-free rate = 0)
        sharpe = self._sharpe(returns)

        # Annualized return
        n_days = len(returns) if len(returns) > 0 else 1
        annualized = (1 + result.total_return) ** (self.TRADING_DAYS_PER_YEAR / max(n_days, 1)) - 1

        # Volatility (annualized)
        vol = float(np.std(returns) * np.sqrt(self.TRADING_DAYS_PER_YEAR)) if len(returns) > 1 else 0.0

        # Max drawdown
        mdd = self._max_drawdown(result.equity_curve)

        # Sortino ratio
        sortino = self._sortino(returns)

        # Calmar ratio
        calmar = annualized / mdd if mdd > 0 else 0.0

        # --- Benchmark-relative metrics ---
        alpha = 0.0
        info_ratio = 0.0
        alpha_sharpe = 0.0

        if benchmark_returns is not None:
            bench = np.array(benchmark_returns)
            # Align lengths
            min_len = min(len(returns), len(bench))
            model_ret = returns[:min_len]
            bench_ret = bench[:min_len]

            # Alpha = cumulative model return - cumulative benchmark return
            model_cum = float(np.prod(1 + model_ret) - 1)
            bench_cum = float(np.prod(1 + bench_ret) - 1)
            alpha = model_cum - bench_cum

            # Excess daily returns (over benchmark)
            excess = model_ret - bench_ret

            # Information ratio = mean(excess) / std(excess) * sqrt(252)
            if len(excess) > 1 and np.std(excess) > 0:
                info_ratio = float(
                    (np.mean(excess) / np.std(excess)) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                )

            # Alpha-Sharpe: Sharpe of excess returns
            alpha_sharpe = self._sharpe(excess)

        # --- Composite training score ---
        # Penalizes always-buy: alpha ≈ 0 means no edge
        training_score = self._composite_score(
            alpha_sharpe=alpha_sharpe,
            sortino=sortino,
            info_ratio=info_ratio,
            max_drawdown=mdd,
        )

        score = ModelScore(
            model_name=result.model_name,
            sharpe_ratio=float(sharpe),
            total_return=result.total_return,
            win_rate=result.win_rate,
            max_drawdown=mdd,
            total_trades=result.total_trades,
            annualized_return=annualized,
            volatility=vol,
            alpha=alpha,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            training_score=training_score,
        )

        logger.info(f"  📈 {score}")
        return score

    def _composite_score(
        self,
        alpha_sharpe: float,
        sortino: float,
        info_ratio: float,
        max_drawdown: float,
    ) -> float:
        """
        Compute composite training score.

        Components (all clamped to [0, ∞)):
        - Alpha Sharpe (30%): edge over buy-and-hold
        - Sortino (30%): downside-risk-adjusted return
        - Information Ratio (20%): consistency of alpha
        - 1 - Max Drawdown (20%): capital preservation

        An always-buy model gets alpha_sharpe ≈ 0, so it scores poorly.
        """
        c_alpha = max(alpha_sharpe, 0.0)
        c_sortino = max(sortino, 0.0)
        c_info = max(info_ratio, 0.0)
        c_dd = max(1.0 - max_drawdown, 0.0)

        return (
            self.WEIGHT_ALPHA_SHARPE * c_alpha
            + self.WEIGHT_SORTINO * c_sortino
            + self.WEIGHT_INFO_RATIO * c_info
            + self.WEIGHT_DRAWDOWN * c_dd
        )

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return float(
            (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        )

    @staticmethod
    def _sortino(returns: np.ndarray) -> float:
        """
        Annualized Sortino ratio.
        Only penalizes downside deviation (returns < 0).
        """
        if len(returns) < 2:
            return 0.0
        downside = returns[returns < 0]
        if len(downside) == 0:
            # No downside — perfect (but cap it)
            return float(np.mean(returns) * np.sqrt(252)) if np.mean(returns) > 0 else 0.0
        downside_std = float(np.std(downside))
        if downside_std == 0:
            return 0.0
        return float(
            (np.mean(returns) / downside_std) * np.sqrt(252)
        )

    @staticmethod
    def _max_drawdown(equity_curve: list[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0

        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdowns = np.where(peak > 0, (peak - eq) / peak, 0.0)

        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
