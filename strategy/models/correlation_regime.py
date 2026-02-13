"""
Correlation Regime Strategy Model.

Inspired by LulutasoAI/FInancial_Analysis check_correlation.py
and Correlation_Analysis.py.

Detects regime changes by monitoring rolling correlation between
price returns and volume. When the price-volume relationship
breaks down or shifts, it signals potential trend changes.

Additionally uses a self-correlation (autocorrelation) measure
to detect momentum persistence or mean-reversion regimes.

Signals:
- BUY when positive autocorrelation + rising price-volume correlation
  (momentum regime, trend likely to continue up)
- SELL when negative autocorrelation or correlation breakdown
  (regime shift, trend likely reversing)
- HOLD when correlation is ambiguous

Confidence is based on the strength and consistency of the correlation signal.
"""

import logging

import numpy as np
import pandas as pd

from models.market import Signal, SignalType
from strategy.base import ModelConfig, StrategyModel

logger = logging.getLogger(__name__)


class CorrelationRegimeStrategy(StrategyModel):
    """
    Correlation regime detection strategy.

    Monitors two correlation signals:
    1. Price-Volume correlation: high correlation = trend strength,
       breakdown = potential reversal
    2. Return autocorrelation: positive = momentum regime,
       negative = mean-reversion regime

    Regime shifts (sudden correlation changes) generate the
    strongest signals.
    """

    def __init__(self, config: ModelConfig = None):
        if config is None:
            config = ModelConfig(name="CorrRegime", weight=1.0, params={
                "corr_window": 30,
                "shift_lookback": 5,
                "autocorr_lag": 1,
            })
        super().__init__(config)
        self.corr_window = config.params.get("corr_window", 30)
        self.shift_lookback = config.params.get("shift_lookback", 5)
        self.autocorr_lag = config.params.get("autocorr_lag", 1)

    @property
    def min_data_points(self) -> int:
        return self.corr_window + self.shift_lookback + 10

    def _rolling_correlation(
        self, series_a: pd.Series, series_b: pd.Series, window: int
    ) -> pd.Series:
        """Compute rolling Pearson correlation between two series."""
        return series_a.rolling(window).corr(series_b)

    def _log_returns(self, close: pd.Series) -> pd.Series:
        """Compute log returns."""
        return np.log(close / close.shift(1))

    def _autocorrelation(self, returns: pd.Series, lag: int, window: int) -> pd.Series:
        """Rolling autocorrelation of returns at a given lag."""
        shifted = returns.shift(lag)
        return returns.rolling(window).corr(shifted)

    def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        if len(data) < self.min_data_points:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                model_name=self.name,
                symbol=symbol,
                reason=f"Insufficient data ({len(data)} bars, need {self.min_data_points})",
            )

        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        # 1. Log returns
        log_ret = self._log_returns(close)

        # 2. Price-Volume rolling correlation
        #    high positive = volume confirms trend
        #    breakdown (near 0 or negative) = divergence
        pv_corr = self._rolling_correlation(
            log_ret, volume.pct_change(), self.corr_window
        )

        # 3. Return autocorrelation (momentum measure)
        #    positive = momentum regime (trend persists)
        #    negative = mean-reversion regime
        auto_corr = self._autocorrelation(
            log_ret, self.autocorr_lag, self.corr_window
        )

        # Get current and recent values
        curr_pv = pv_corr.iloc[-1]
        curr_auto = auto_corr.iloc[-1]

        if pd.isna(curr_pv) or pd.isna(curr_auto):
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                model_name=self.name,
                symbol=symbol,
                reason="Correlation calculation returned NaN",
            )

        # 4. Detect correlation regime SHIFT
        #    Compare recent correlation to slightly older correlation
        recent_pv = pv_corr.iloc[-self.shift_lookback:].mean()
        older_pv = pv_corr.iloc[-2 * self.shift_lookback:-self.shift_lookback].mean()
        pv_shift = recent_pv - older_pv  # positive = strengthening, negative = weakening

        recent_auto = auto_corr.iloc[-self.shift_lookback:].mean()

        # 5. Recent price direction
        recent_return = float(log_ret.iloc[-self.shift_lookback:].sum())

        # Combine signals
        if pd.isna(pv_shift) or pd.isna(recent_auto):
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.1,
                model_name=self.name,
                symbol=symbol,
                reason="Insufficient correlation history",
            )

        # Score: combine autocorrelation and regime shift
        # Positive autocorrelation + recent up = momentum buy
        # Negative autocorrelation + recent up = mean-reversion sell
        momentum_score = float(recent_auto) * np.sign(recent_return)

        # Regime shift adds conviction
        shift_bonus = float(pv_shift) * np.sign(recent_return)

        combined = momentum_score + shift_bonus * 0.5

        # Map to signal
        dead_zone = 0.05

        if abs(combined) < dead_zone:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.1,
                model_name=self.name,
                symbol=symbol,
                reason=f"CorrRegime neutral: score={combined:.3f} (autocorr={recent_auto:.3f})",
            )

        confidence = min(0.3 + abs(combined) * 1.5, 0.85)

        if combined > 0:
            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                model_name=self.name,
                symbol=symbol,
                reason=(
                    f"Momentum regime: autocorr={recent_auto:.3f}, "
                    f"PV-shift={pv_shift:+.3f}, return={recent_return:+.4f}"
                ),
            )
        else:
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                model_name=self.name,
                symbol=symbol,
                reason=(
                    f"Reversal regime: autocorr={recent_auto:.3f}, "
                    f"PV-shift={pv_shift:+.3f}, return={recent_return:+.4f}"
                ),
            )
