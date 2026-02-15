""" 
Deep Learning Strategy Model.

Predicts 1-hour price change using a PyTorch LSTM based on sequence data.
Inputs:
- Sequence Length: 10 steps (e.g., 10x 15-min bars)
- Features (103):
  - Price History (Log Returns)
  - Volume (Normalized)
  - Sentiment Score
  - Macro Data (SP500, VIX, JPY, EUR, etc.)

Output:
- Predicted % return for next 1 hour.
"""

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from data.processor import DataProcessor

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None

from models.market import Signal, SignalType
from strategy.base import ModelConfig, StrategyModel

logger = logging.getLogger(__name__)


try:
    from training_bundle.model_def import PricePredictor
except ImportError:
    # Fallback/Local dev
    class PricePredictor(nn.Module):
        """LSTM for sequence-based price prediction (Fallback)."""
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True,
                dropout=0.2
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)


class DeepLearningStrategy(StrategyModel):
    """
    Deep Learning Strategy using LSTM + Macro Data + Robust Scaling.
    Training is offloaded to Cloud; this class handles Inference only.
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.fetcher = None
        
        # Shared Cloud Artifacts
        self.model_path = "data/best_model.pth"
        self.processor = DataProcessor("data/scaler.pkl")
        
        # Hyperparams
        self.seq_len = 10
        self.resample_tf = "15Min" 
        
        # Features (example count):
        # Price(1) + Volume(1) + Sentiment(1) + Macro(5) = 8 base features
        self.input_dim = 8 
        
        self.model = None
        self.is_trained = False
        
        if torch:
            self.model = PricePredictor(self.input_dim)
            self._load_model()

    def set_fetcher(self, fetcher):
        self.fetcher = fetcher

    def _load_model(self):
        """
        Load model:
        1. Try MLflow Registry (Production).
        2. Fallback to local disk (data/best_model.pth).
        """
        # 1. Load from Local Disk
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
                self.model.eval()
                self.is_trained = True
                logger.info(f"🧠 Loaded Local model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load local DL model: {e}")
        else:
            logger.warning(f"⚠️  No model found at {self.model_path}. Train locally or place model file.")

    def save_model(self):
        """Save model to disk (Disabled for cloud workflow)."""
        pass

    @property
    def min_data_points(self) -> int:
        # Need enough for sequence + horizon
        return self.seq_len + 10

    def _extract_features(self, df: pd.DataFrame, sentiment_score: float) -> np.ndarray:
        """
        Extract features for a specific DataFrame window.
        Returns array of shape (N, features).
        """
        # Calculate returns
        df = df.copy()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # Macro cols
        macro_cols = ["SP500_Close", "VIX_Close", "BTC_Close", "XLK_Close", "XLF_Close"]
        
        # Feature list construction
        features = []
        
        # Log Return
        features.append(df['log_ret'].values.reshape(-1, 1))
        
        # Volume (Log or Norm? RobustScaler handles it)
        features.append(df['volume'].values.reshape(-1, 1))
        
        # Sentiment (Static for this window or time-variant? Here static approximation)
        sent_arr = np.full((len(df), 1), sentiment_score)
        features.append(sent_arr)

        # Macro
        for col in macro_cols:
            if col in df.columns:
                if col == "VIX_Close":
                    # VIX is level
                    features.append(df[col].values.reshape(-1, 1))
                else:
                    # Others are returns
                    feat = np.log(df[col] / df[col].shift(1)).fillna(0).values.reshape(-1, 1)
                    features.append(feat)
            else:
                features.append(np.zeros((len(df), 1)))
                
        # Stack: (N, 8)
        return np.hstack(features)

    def prepare_input(self, symbol: str, data: pd.DataFrame, sentiment_score: float) -> torch.Tensor:
        """
        Prepare SINGLE sequence input for inference.
        Returns tensor (1, seq_len, features) or None.
        """
        if data.empty:
            return None
            
        # Resample
        is_daily = "Day" in self.config.timeframe
        tf = "1Day" if is_daily else self.resample_tf
        
        agg = {"close": "last", "volume": "sum"}
        macro_cols = ["SP500_Close", "VIX_Close", "BTC_Close", "XLK_Close", "XLF_Close"]
        for c in macro_cols:
            if c in data.columns: agg[c] = "last"
            
        try:
            df = data.set_index("timestamp").resample(tf).agg(agg).dropna()
        except:
            if isinstance(data.index, pd.DatetimeIndex):
                df = data.resample(tf).agg(agg).dropna()
            else:
                return None
        
        if len(df) < self.seq_len + 1:
            return None
            
        # Extract RAW features (unscaled)
        raw_feats = self._extract_features(df, sentiment_score)
        
        # Take last seq_len
        raw_seq = raw_feats[-self.seq_len:]
        
        # Scale
        # Important: Processor must be loaded with cloud scaler
        if not self.processor.is_fitted:
             # Try loading again?
             pass
             
        scaled_seq = self.processor.transform(pd.DataFrame(raw_seq), list(range(self.input_dim)))
        
        # To Tensor
        return torch.FloatTensor(scaled_seq).unsqueeze(0) # (1, seq, feat)

    def train(self, data_map: dict):
        """
        Local training disabled.
        Expects artifacts in data/best_model.pth and data/scaler.pkl
        """
        logger.info("☁️  Deep Learning training offloaded to cloud. Check data/ directory for artifacts.")
        self._load_model()


    def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        if not torch:
            return Signal(SignalType.HOLD, 0.0, self.name, symbol, "PyTorch missing")
        
        if not self.is_trained:
            return Signal(SignalType.HOLD, 0.0, self.name, symbol, "Model not trained")
            
        sent = 0.0
        if self.fetcher:
             # Use timestamp of data end
            end_dt = data.iloc[-1]["timestamp"]
            sent = self.fetcher.get_sentiment_score(symbol, timedelta(hours=24), end_date=end_dt)
            
        inp_tensor = self.prepare_input(symbol, data, sent)
        if inp_tensor is None:
            return Signal(SignalType.HOLD, 0.0, self.name, symbol, "Insufficient data")
            
        with torch.no_grad():
            pred_return = self.model(inp_tensor).item()
            
        # Dynamic Threshold based on Volatility (ATR-like)
        # Calculate Rolling StdDev of returns (last 24h)
        # Use existing data DataFrame
        
        # Crude Volatility Estimate:
        # Calculate log returns of the input data
        # We need to access the 'close' column
        closes = data["close"]
        rets = np.log(closes / closes.shift(1)).dropna()
        vol = rets.std() 
        # If data is 1Min, this is 1Min vol. 
        # Model predicts 15Min/1Day return?
        # Model predicts NEXT STEP return. If resample_tf is 15Min, it predicts 15Min return.
        # Volatility should be scaled to the horizon.
        # If vol is 1Min, we need sqrt(15) scaling to get 15Min vol?
        
        if "Day" in self.config.timeframe: 
            # Daily data -> Daily vol
            period_vol = vol 
        else:
            # Intraday (1Min) -> Scaled to 15Min
            period_vol = vol * np.sqrt(15)
            
        if pd.isna(period_vol) or period_vol == 0:
            period_vol = 0.001 # fallback 0.1% volatility
            
        # Threshold: 0.5 * Volatility
        threshold = 0.5 * period_vol
        
        confidence = min(abs(pred_return) / (threshold + 1e-6), 1.0)
        
        if pred_return > threshold:
            return Signal(SignalType.BUY, confidence, self.name, symbol, f"Pred {pred_return:.2%} > {threshold:.2%} (Vol)")
        elif pred_return < -threshold:
            return Signal(SignalType.SELL, confidence, self.name, symbol, f"Pred {pred_return:.2%} < -{threshold:.2%} (Vol)")
            
        return Signal(SignalType.HOLD, 0.0, self.name, symbol, f"Pred {pred_return:.2%} within {threshold:.2%} band")
