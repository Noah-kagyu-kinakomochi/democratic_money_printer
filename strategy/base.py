"""
Abstract Strategy Model Interface.

Each strategy model is a 'voter' in the democratic engine.
It analyzes market data and casts a vote (BUY / SELL / HOLD) with a confidence score.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from models.market import Signal


@dataclass
class ModelConfig:
    """Configuration for a single strategy model (voter)."""
    name: str
    weight: float = 1.0  # influence in the democratic vote
    params: dict = field(default_factory=dict)
    timeframe: str = "1Day"  # timeframe this model instance operates on
    enabled: bool = True


class StrategyModel(ABC):
    """
    Abstract base class for all strategy models.
    
    Each model:
    1. Receives historical OHLCV data
    2. Analyzes it using its specific methodology
    3. Returns a Signal (BUY/SELL/HOLD + confidence + reasoning)
    
    Models are pluggable — add new models by subclassing and registering.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
        self.weight = config.weight

    @abstractmethod
    def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Analyze OHLCV data and produce a trading signal.
        
        Args:
            symbol: The ticker symbol being analyzed
            data: DataFrame with columns [timestamp, open, high, low, close, volume]
                  sorted by timestamp ascending. Must have at least enough history
                  for the model's lookback period.
        
        Returns:
            Signal with signal_type, confidence, and reason.
        """
        ...

    def update_weight(self, new_weight: float) -> None:
        """Update this model's weight (called by WeightLearner)."""
        old_weight = self.weight
        self.weight = new_weight
        self.config.weight = new_weight

    @property
    def min_data_points(self) -> int:
        """Minimum number of OHLCV bars needed for analysis."""
        return 30  # override in subclass if needed

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, weight={self.weight:.3f})"
