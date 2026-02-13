import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data scaling and sequence generation for Deep Learning models.
    """
    def __init__(self, scaler_path: str = "data/scaler.pkl"):
        self.scaler = RobustScaler()
        self.scaler_path = Path(scaler_path)
        self.is_fitted = False
        self._load_scaler()

    def fit(self, data: pd.DataFrame, feature_cols: list):
        """
        Fit scaler on training data.
        """
        if data.empty:
            return

        # Fit only on specified feature columns
        self.scaler.fit(data[feature_cols])
        self.is_fitted = True
        self._save_scaler()
        logger.info(f"⚖️  DataProcessor fitted on {len(data)} rows.")

    def transform(self, data: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """
        Scale data using saved parameters.
        Returns numpy array of scaled features.
        """
        if not self.is_fitted:
            logger.warning("⚠️ Scaler not fitted. Returning raw data (RISKY).")
            return data[feature_cols].values

        try:
            scaled = self.scaler.transform(data[feature_cols])
            return scaled
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            return data[feature_cols].values

    def create_sequences(self, data: np.ndarray, seq_len: int, target: np.ndarray = None):
        """
        Create sliding window sequences for LSTM.
        
        Args:
            data: (N, features) array
            seq_len: sequence length (time steps)
            target: (N,) array of targets (optional)
            
        Returns:
            X: (batch, seq_len, features)
            y: (batch,) or None
        """
        xs = []
        ys = []
        
        if len(data) < seq_len:
            return np.array([]), np.array([])

        for i in range(len(data) - seq_len):
            x = data[i:(i + seq_len)]
            xs.append(x)
            if target is not None:
                ys.append(target[i + seq_len]) # Predict next step after sequence
        
        return np.array(xs), np.array(ys) if target is not None else None

    def _save_scaler(self):
        try:
            self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, self.scaler_path)
        except Exception as e:
            logger.error(f"Failed to save scaler: {e}")

    def _load_scaler(self):
        if self.scaler_path.exists():
            try:
                self.scaler = joblib.load(self.scaler_path)
                self.is_fitted = True
            except Exception as e:
                logger.warning(f"Failed to load scaler: {e}. Will refit.")
