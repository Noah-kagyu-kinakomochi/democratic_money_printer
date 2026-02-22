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
from data.gcs_utils import GCSManager
from config.settings import GCSConfig

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
    import math

    class PositionalEncoding(nn.Module):
        """Injects positional information into the sequence for the Transformer."""
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:, :x.size(1), :]
            return x

    class PricePredictor(nn.Module):
        """Transformer for sequence-based price prediction"""
        def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.2):
            super().__init__()
            self.input_projection = nn.Linear(input_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
                dropout=dropout, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc_out = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            nn.Linear(32, 3) # 3 Classes: DOWN (0), FLAT (1), UP (2)
        )

        def forward(self, x):
            x = self.input_projection(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            out = x[:, -1, :]
            return self.fc_out(out)

def frac_diff(series: pd.Series, d: float = 0.4, thres: float = 0.01) -> pd.Series:
    """Apply fractional differentiation to a pandas Series to achieve fractionally integrated stationarity."""
    w = [1.]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1]).reshape(-1, 1)
    
    res = []
    # Pad with NaN until we have enough history for the window
    for iloc in range(len(series)):
        if iloc < len(w) - 1:
            res.append(np.nan)
        else:
            window = series.iloc[iloc - len(w) + 1 : iloc + 1].values
            res.append(np.dot(w.T, window)[0])
            
    return pd.Series(res, index=series.index)


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
        self.seq_len = 96
        self.resample_tf = "15Min" 
        
        # Features: Frac-Diff(1) + Vol(1) + Sent(1) + Yield Diff(1) + DXY(1) + ATR(1) + MACD(1) = 7
        self.input_dim = 7
        
        self.model = None
        self.is_trained = False
        
        if torch:
            self.model = PricePredictor(self.input_dim)
            self._load_model()

    def set_fetcher(self, fetcher):
        self.fetcher = fetcher

    def _load_model(self):
        """
        Load model artifacts:
        1. Attempt to pull the latest `best_model.pth` and `scaler.pkl` from GCS.
        2. Fall back to local disk if GCS is unconfigured or files are absent.
        """
        gcs = GCSManager()
        gcs_paths = GCSConfig()  # Provides canonical path defaults

        if gcs.bucket_name:
            logger.info("☁️  Pulling latest model artifacts from GCS...")
            gcs.download_file(gcs_paths.model_path, self.model_path, ignore_missing=True)
            gcs.download_file(gcs_paths.scaler_path, "data/scaler.pkl", ignore_missing=True)
        else:
            logger.debug("GCS_BUCKET_NAME not set — using local model artifacts only.")

        # Load from disk (may have just been refreshed from GCS)
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
                self.model.eval()
                self.is_trained = True
                logger.info(f"🧠 Successfully loaded DL model from {self.model_path}")
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
        # Feature 1: Fractionally Differentiated Price (d=0.4)
        df['frac_diff'] = frac_diff(df['close'], d=0.4, thres=0.01)
        features.append(df['frac_diff'].fillna(0).values.reshape(-1, 1))
        
        # Feature 2: Volume 
        features.append(df['volume'].values.reshape(-1, 1))
        
        # Feature 3: Sentiment
        sent_arr = np.full((len(df), 1), sentiment_score)
        features.append(sent_arr)

        # Feature 4: US/JPY 10-Year Yield Differential
        tnx = df.get("TNX_Close", pd.Series(np.zeros(len(df)), index=df.index)).ffill().fillna(0)
        jgbs10 = df.get("JGBS10_Close", pd.Series(np.zeros(len(df)), index=df.index)).ffill().fillna(0)
        yield_diff = tnx - jgbs10
        features.append(yield_diff.values.reshape(-1, 1))

        # Feature 5: DXY (US Dollar Index) Return
        dxy = df.get("DXY_Close", pd.Series(np.zeros(len(df)), index=df.index)).ffill()
        dxy_ret = np.log(dxy / dxy.shift(1)).fillna(0)
        features.append(dxy_ret.values.reshape(-1, 1))
        
        # Feature 6: ATR (Average True Range) - Rolling 14-period Volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().fillna(0)
        features.append(atr.values.reshape(-1, 1))
        
        # Feature 7: MACD Divergence
        # (EMA 12 - EMA 26) - Signal(EMA 9)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_div = (macd - signal).fillna(0)
        features.append(macd_div.values.reshape(-1, 1))
            
        # Stack: (N, 7)
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
        
        try:
            agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            macro_cols = ["TNX_Close", "JGBS10_Close", "DXY_Close"]
            for c in macro_cols:
                if c in data.columns: agg[c] = "last"
            
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
            logits = self.model(inp_tensor) # (1, 3)
            # Apply Softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()
            
        # Classes: DOWN (0), FLAT (1), UP (2)
        prob_down = float(probs[0])
        prob_flat = float(probs[1])
        prob_up   = float(probs[2])
        
        predicted_class = np.argmax(probs)
        confidence = float(np.max(probs))
        
        # Threshold for action
        # The model is trained to identify > 1.5 sigma moves.
        # So any "UP" or "DOWN" class prediction is already a strong signal.
        if predicted_class == 2 and confidence > 0.4:
            return Signal(
                SignalType.BUY, 
                confidence, 
                self.name, 
                symbol, 
                f"UP Breakout prob: {prob_up:.1%} (FLAT: {prob_flat:.1%}, DOWN: {prob_down:.1%})"
            )
        elif predicted_class == 0 and confidence > 0.4:
            return Signal(
                SignalType.SELL, 
                confidence, 
                self.name, 
                symbol, 
                f"DOWN Breakout prob: {prob_down:.1%} (FLAT: {prob_flat:.1%}, UP: {prob_up:.1%})"
            )
            
        return Signal(
            SignalType.HOLD, 
            prob_flat, 
            self.name, 
            symbol, 
            f"FLAT predicted. Probabilities -> UP: {prob_up:.1%}, FLAT: {prob_flat:.1%}, DOWN: {prob_down:.1%}"
        )
