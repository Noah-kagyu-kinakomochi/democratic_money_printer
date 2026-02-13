import logging
import pandas as pd
from config.settings import RegimeConfig
from strategy.regime import MarketRegime
from strategy.models.moving_average import MovingAverageCrossover
from strategy.models.macd_strategy import MACDStrategy
from strategy.models.rsi_strategy import RSIStrategy
from strategy.base import StrategyModel
from models.market import ConsensusSignal, Signal, SignalType, Vote

logger = logging.getLogger(__name__)


class DemocraticEngine:
    """
    Ensemble decision engine that aggregates signals from multiple strategy models.
    """

    def __init__(
        self,
        models: list[StrategyModel],
        voting_method: str = "weighted",
        min_confidence: float = 0.3,
        min_weight: float = 0.01,
        regime_config: RegimeConfig = None,
    ):
        self.models = [m for m in models if m.config.enabled]
        self.voting_method = voting_method
        self.min_confidence = min_confidence
        self.min_weight = min_weight
        self.regime_config = regime_config or RegimeConfig()

        if not self.models:
            raise ValueError("At least one enabled strategy model is required")

        logger.info(
            f"Democratic Engine initialized with {len(self.models)} models: "
            f"{[m.name for m in self.models]} | voting={voting_method} | min_weight={min_weight}"
        )

    def vote(self, symbol: str, data: pd.DataFrame, regime: MarketRegime = MarketRegime.UNKNOWN) -> ConsensusSignal:
        """
        Run all models and aggregate their votes into a consensus signal.
        """
        votes: list[Vote] = []

        for model in self.models:
            # Optimization: Skip models with low weight (pre-adjustment)
            if model.weight < self.min_weight:
                continue

            try:
                signal = model.analyze(symbol, data)
                
                # Dynamic Weighting based on Regime
                weight = self._apply_regime_adjustment(model, regime)
                
                vote = Vote(signal=signal, weight=weight)
                votes.append(vote)
                logger.info(f"  🗳️  {model.name}: {signal} (w={weight:.3f})")
            except Exception as e:
                logger.error(f"  ❌  {model.name} failed: {e}")
                abstain = Signal(SignalType.HOLD, 0.0, model.name, symbol, f"Error: {e}")
                votes.append(Vote(signal=abstain, weight=0.0))

        if not votes:
            return ConsensusSignal(SignalType.HOLD, 0.0, symbol, [])

        return self._tally_votes(symbol, votes)

    def vote_multi_tf(self, symbol: str, data_by_timeframe: dict[str, pd.DataFrame], regime: MarketRegime = MarketRegime.UNKNOWN) -> ConsensusSignal:
        """
        Run all models, each on its own timeframe data, and aggregate votes with regime awareness.
        """
        votes: list[Vote] = []

        for model in self.models:
            if model.weight < self.min_weight:
                continue

            tf = model.config.timeframe
            data = data_by_timeframe.get(tf)

            if data is None or data.empty:
                abstain = Signal(SignalType.HOLD, 0.0, model.name, symbol, f"No data for timeframe {tf}")
                votes.append(Vote(signal=abstain, weight=0.0))
                continue

            try:
                signal = model.analyze(symbol, data)
                
                # Dynamic Weighting based on Regime
                weight = self._apply_regime_adjustment(model, regime)
                
                vote = Vote(signal=signal, weight=weight)
                votes.append(vote)
                logger.info(f"  🗳️  {model.name}: {signal} (w={weight:.3f})")
            except Exception as e:
                logger.error(f"  ❌  {model.name} failed: {e}")
                abstain = Signal(SignalType.HOLD, 0.0, model.name, symbol, f"Error: {e}")
                votes.append(Vote(signal=abstain, weight=0.0))

        if not votes:
            return ConsensusSignal(SignalType.HOLD, 0.0, symbol, [])

        return self._tally_votes(symbol, votes)

    def _tally_votes(self, symbol: str, votes: list[Vote]) -> ConsensusSignal:
        """Shared voting tally logic."""
        if self.voting_method == "weighted":
            consensus = self._weighted_vote(symbol, votes)
        elif self.voting_method == "majority":
            consensus = self._majority_vote(symbol, votes)
        elif self.voting_method == "unanimous":
            consensus = self._unanimous_vote(symbol, votes)
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")

        # Threshold check
        if consensus.confidence < self.min_confidence:
            logger.info(
                f"  📊 Consensus: {consensus.signal_type.value} @ {consensus.confidence:.1%} "
                f"— BELOW threshold ({self.min_confidence:.0%}), switching to HOLD"
            )
            return ConsensusSignal(SignalType.HOLD, consensus.confidence, symbol, votes)
        else:
            logger.info(
                f"  📊 Consensus: {consensus.signal_type.value} @ {consensus.confidence:.1%} "
                f"— ABOVE threshold ({self.min_confidence:.0%})"
            )
            return consensus

    def _apply_regime_adjustment(self, model: StrategyModel, regime: MarketRegime) -> float:
        """
        Adjust model weight based on market regime.
        Trend following: Boost in Trend, Penalize in Chop.
        Mean Reversion: Boost in Chop, Penalize in Trend.
        """
        weight = model.weight
        if regime == MarketRegime.UNKNOWN:
            return weight

        is_trend_model = isinstance(model, (MovingAverageCrossover, MACDStrategy))
        is_mean_rev = isinstance(model, RSIStrategy)
        
        # Strategies not explicitly categorized (like Sentiment, DL) are neutral (no adj)
        if not (is_trend_model or is_mean_rev):
            return weight

        cfg = self.regime_config
        
        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            if is_trend_model:
                weight *= cfg.trend_boost
            elif is_mean_rev:
                weight *= cfg.trend_penalty
        
        elif regime in (MarketRegime.SIDEWAYS_LOW_VOL, MarketRegime.SIDEWAYS_HIGH_VOL):
            if is_trend_model:
                weight *= cfg.chop_penalty
            elif is_mean_rev:
                weight *= cfg.chop_boost
                
        return weight

    def _weighted_vote(self, symbol: str, votes: list[Vote]) -> ConsensusSignal:
        """
        Weighted voting: sum up weighted scores.
        Positive total = BUY, Negative total = SELL, Near-zero = HOLD.
        """
        total_weight = sum(v.weight for v in votes if v.weight > 0)
        if total_weight == 0:
            return ConsensusSignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                symbol=symbol,
                votes=votes,
            )

        weighted_sum = sum(v.weighted_score for v in votes)
        normalized_score = weighted_sum / total_weight  # range: -1.0 to 1.0

        if normalized_score > 0.05:
            signal_type = SignalType.BUY
        elif normalized_score < -0.05:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        confidence = min(abs(normalized_score), 1.0)

        return ConsensusSignal(
            signal_type=signal_type,
            confidence=confidence,
            symbol=symbol,
            votes=votes,
        )

    def _majority_vote(self, symbol: str, votes: list[Vote]) -> ConsensusSignal:
        """Simple majority: most common signal wins."""
        counts = {SignalType.BUY: 0, SignalType.SELL: 0, SignalType.HOLD: 0}
        for v in votes:
            counts[v.signal.signal_type] += 1

        winner = max(counts, key=counts.get)
        total = len(votes)
        majority_pct = counts[winner] / total

        # Average confidence of winning votes
        winning_votes = [v for v in votes if v.signal.signal_type == winner]
        avg_confidence = sum(v.signal.confidence for v in winning_votes) / len(winning_votes) if winning_votes else 0

        return ConsensusSignal(
            signal_type=winner,
            confidence=avg_confidence * majority_pct,
            symbol=symbol,
            votes=votes,
        )

    def _unanimous_vote(self, symbol: str, votes: list[Vote]) -> ConsensusSignal:
        """Unanimous: all models must agree, otherwise HOLD."""
        active_votes = [v for v in votes if v.signal.signal_type != SignalType.HOLD]

        if not active_votes:
            return ConsensusSignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                symbol=symbol,
                votes=votes,
            )

        first_signal = active_votes[0].signal.signal_type
        all_agree = all(v.signal.signal_type == first_signal for v in active_votes)

        if all_agree:
            avg_confidence = sum(v.signal.confidence for v in active_votes) / len(active_votes)
            return ConsensusSignal(
                signal_type=first_signal,
                confidence=avg_confidence,
                symbol=symbol,
                votes=votes,
            )
        else:
            return ConsensusSignal(
                signal_type=SignalType.HOLD,
                confidence=0.1,
                symbol=symbol,
                votes=votes,
            )

    def update_weights(self, weights: dict[str, float]) -> None:
        """
        Update model weights from the WeightLearner.
        
        Args:
            weights: Dict mapping model_name -> new normalized weight.
                     Models not in the dict keep their current weight.
        """
        for model in self.models:
            if model.name in weights:
                old_w = model.weight
                model.update_weight(weights[model.name])
                logger.info(f"  ⚖️  {model.name}: weight {old_w:.3f} → {weights[model.name]:.3f}")

    def get_model_status(self) -> list[dict]:
        """Get current models and their configuration."""
        return [
            {
                "name": m.name,
                "weight": m.weight,
                "enabled": m.config.enabled,
                "type": m.__class__.__name__,
            }
            for m in self.models
        ]
