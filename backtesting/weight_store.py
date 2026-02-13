"""
Weight Store.

Persists model weights to JSON for reuse between runs.
Stores weights, scores, and timestamp so the weight learner
can blend fresh scores with previous weights (exponential smoothing).
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default location: data/weights.json (next to SQLite/Parquet files)
DEFAULT_WEIGHTS_PATH = Path("data/weights.json")


@dataclass
class StoredWeights:
    """Weights loaded from disk."""
    weights: dict[str, float]
    scores: dict[str, dict]  # model_name -> {training_score, alpha, ...}
    timestamp: str
    model_count: int = 0

    @property
    def age_hours(self) -> float:
        """How old these weights are, in hours."""
        try:
            saved_dt = datetime.fromisoformat(self.timestamp)
            now = datetime.now(timezone.utc)
            if saved_dt.tzinfo is None:
                saved_dt = saved_dt.replace(tzinfo=timezone.utc)
            return (now - saved_dt).total_seconds() / 3600
        except (ValueError, TypeError):
            return float("inf")


def save_weights(
    weights: dict[str, float],
    scores: Optional[dict[str, dict]] = None,
    path: Path = DEFAULT_WEIGHTS_PATH,
) -> None:
    """
    Save weights to JSON.

    Args:
        weights: model_name -> weight mapping
        scores: optional model_name -> score dict
        path: file path (created if doesn't exist)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_count": len(weights),
        "weights": weights,
        "scores": scores or {},
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"💾 Saved {len(weights)} model weights to {path}")


def load_weights(path: Path = DEFAULT_WEIGHTS_PATH) -> Optional[StoredWeights]:
    """
    Load weights from JSON.

    Returns:
        StoredWeights if file exists and is valid, None otherwise.
    """
    if not path.exists():
        logger.info(f"No saved weights found at {path}")
        return None

    try:
        with open(path) as f:
            data = json.load(f)

        stored = StoredWeights(
            weights=data.get("weights", {}),
            scores=data.get("scores", {}),
            timestamp=data.get("timestamp", ""),
            model_count=data.get("model_count", 0),
        )

        logger.info(
            f"📂 Loaded {len(stored.weights)} weights from {path} "
            f"(age: {stored.age_hours:.1f}h)"
        )
        return stored

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to load weights from {path}: {e}")
        return None
