"""
Test helpers — Synthetic data generators for reproducible unit tests.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_ohlcv(
    n: int = 60,
    start_price: float = 150.0,
    start_date: str = "2025-01-01",
    seed: int = 42,
    trend: str = "neutral",
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.
    
    Args:
        n: Number of bars
        start_price: Starting close price
        start_date: Starting date string
        seed: Random seed for reproducibility
        trend: 'up', 'down', or 'neutral'
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start=start_date, periods=n, freq="D")
    
    drift = {"up": 0.5, "down": -0.5, "neutral": 0.0}[trend]
    price = start_price
    rows = []

    for d in dates:
        change = rng.randn() * 2 + drift
        o = price
        h = price + abs(rng.randn()) * 1.5
        l = price - abs(rng.randn()) * 1.5
        c = price + change
        v = rng.randint(1_000_000, 5_000_000)
        rows.append({
            "timestamp": d,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": int(v),
        })
        price = c

    return pd.DataFrame(rows)


def generate_trending_data(direction: str = "up", n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate clearly trending data for testing signal direction."""
    return generate_ohlcv(n=n, trend=direction, seed=seed)


def generate_oversold_data(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate data with a recent sharp drop (triggers RSI oversold)."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
    price = 150.0
    rows = []

    for i, d in enumerate(dates):
        if i > n - 15:
            # Sharp decline in last 15 bars
            change = -abs(rng.randn()) * 2
        else:
            change = rng.randn() * 1
        
        o = price
        h = price + abs(rng.randn()) * 0.5
        l = price - abs(rng.randn()) * 0.5
        c = max(price + change, 1.0)
        v = rng.randint(1_000_000, 5_000_000)
        rows.append({
            "timestamp": d,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": int(v),
        })
        price = c

    return pd.DataFrame(rows)


def generate_overbought_data(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate data with a recent sharp rally (triggers RSI overbought)."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
    price = 100.0
    rows = []

    for i, d in enumerate(dates):
        if i > n - 15:
            # Sharp rally in last 15 bars
            change = abs(rng.randn()) * 2.5
        else:
            change = rng.randn() * 1
        
        o = price
        h = price + abs(rng.randn()) * 0.5
        l = price - abs(rng.randn()) * 0.5
        c = price + change
        v = rng.randint(1_000_000, 5_000_000)
        rows.append({
            "timestamp": d,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": int(v),
        })
        price = c

    return pd.DataFrame(rows)
