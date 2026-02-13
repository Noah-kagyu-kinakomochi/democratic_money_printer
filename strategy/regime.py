"""
Market Regime Detector.

Uses PyTorch to classify market state based on Trend (ADX) and Volatility.
"""

from enum import Enum
import logging
import torch
import pandas as pd
import numpy as np
from config.settings import RegimeConfig

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SIDEWAYS_LOW_VOL = "SIDEWAYS_LOW_VOL"
    SIDEWAYS_HIGH_VOL = "SIDEWAYS_HIGH_VOL"
    UNKNOWN = "UNKNOWN"


class RegimeDetector:
    """
    PyTorch-based Market Regime Detector.
    
    Logic:
    1. Calculate ADX (Average Directional Index) for Trend Strength.
    2. Calculate Volatility (Std Dev of returns).
    3. Apply smoothing (rolling mean) to prevent flickering.
    4. Classify based on thresholds.
    """

    def __init__(self, config: RegimeConfig):
        self.config = config

    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime from OHLCV data.
        """
        if df.empty or len(df) < self.config.adx_period + 10:
            return MarketRegime.UNKNOWN

        # specific check for torch availability
        if not torch:
            logger.warning("PyTorch not found, skipping regime detection")
            return MarketRegime.UNKNOWN

        try:
            # Convert to PyTorch Tensors
            high = torch.tensor(df["high"].values, dtype=torch.float32)
            low = torch.tensor(df["low"].values, dtype=torch.float32)
            close = torch.tensor(df["close"].values, dtype=torch.float32)
            
            # 1. Calculate ADX (Vectorized)
            adx, plus_di, minus_di = self._calculate_adx_torch(high, low, close)
            
            # 2. Calculate Volatility (Vectorized)
            # Log returns
            returns = torch.log(close[1:] / close[:-1])
            # Rolling std dev (simple approximation using unfolding)
            # We want recent volatility
            vol_window = 20
            if len(returns) > vol_window:
                recent_vol = torch.std(returns[-vol_window:])
            else:
                recent_vol = torch.std(returns)

            # 3. Smoothing / Hysteresis
            # We take the mean of the last N ADX values
            smoothing = self.config.smoothing_window
            current_adx = torch.mean(adx[-smoothing:]).item()
            
            # Direction
            # If +DI > -DI, it's Uptrend dominant
            p_di = torch.mean(plus_di[-smoothing:]).item()
            m_di = torch.mean(minus_di[-smoothing:]).item()
            
            current_vol = recent_vol.item()

            # 4. Classification
            is_trending = current_adx > self.config.adx_threshold
            is_high_vol = current_vol > self.config.volatility_threshold

            if is_trending:
                if p_di > m_di:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            else:
                if is_high_vol:
                    return MarketRegime.SIDEWAYS_HIGH_VOL
                else:
                    return MarketRegime.SIDEWAYS_LOW_VOL

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return MarketRegime.UNKNOWN

    def _calculate_adx_torch(self, high, low, close):
        """
        Calculate ADX using PyTorch operations.
        """
        n = self.config.adx_period
        epsilon = 1e-8

        # 1. True Range
        tr1 = high[1:] - low[1:]
        tr2 = torch.abs(high[1:] - close[:-1])
        tr3 = torch.abs(low[1:] - close[:-1])
        tr = torch.max(tr1, torch.max(tr2, tr3))

        # 2. Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = torch.where((up_move > down_move) & (up_move > 0), up_move, torch.zeros_like(up_move))
        minus_dm = torch.where((down_move > up_move) & (down_move > 0), down_move, torch.zeros_like(down_move))

        # 3. Smoothed Moving Averages (Wilder's Smoothing)
        # Using simple exponential moving average as proxy for efficiency in torch
        # alpha = 1 / n
        
        def ema_torch(series, alpha):
            # This is hard to fully vectorize without a loop or cumulative product trick
            # For simplicity and speed in this context, we use a loop for the recursive part
            # or a convolution if we want pure vectorization.
            # Given the constraint "No loops over time dimension" is strict, 
            # we can use conv1d for SMA, but EMA is recursive.
            # However, for ADX, Wilder's is often approximated by EMA.
            
            # Let's use a cumulative sum approach for SMA first as it's purely vectorized
            # SMA is often "good enough" for regime detection if we don't need exact TA-lib match.
            # But let's try to be better.
            
            # PyTorch doesn't have a native ewma.
            # We will use simple Moving Average (Unfold) for purely vectorized "Trend" detection
            # It reacts slightly differently but is robust.
            
            # Unfold creates a sliding window view
            # [T] -> [T-n+1, n]
            if len(series) < n:
                return series
                
            windows = series.unfold(0, n, 1)
            return torch.mean(windows, dim=1)

        # Using SMA for vectorization speed (Constraint 2)
        # We need to slice arrays to align
        
        tr_smooth = ema_torch(tr, 1/n)
        plus_smooth = ema_torch(plus_dm, 1/n)
        minus_smooth = ema_torch(minus_dm, 1/n)

        # 4. DI
        plus_di = 100 * (plus_smooth / (tr_smooth + epsilon))
        minus_di = 100 * (minus_smooth / (tr_smooth + epsilon))

        # 5. DX
        dx = 100 * torch.abs(plus_di - minus_di) / (plus_di + minus_di + epsilon)

        # 6. ADX (Smoothed DX)
        adx = ema_torch(dx, 1/n)
        
        # Pad/Align outputs to match original length (roughly) for the detector to grab tail
        # We just return the valid parts
        return adx, plus_di, minus_di
