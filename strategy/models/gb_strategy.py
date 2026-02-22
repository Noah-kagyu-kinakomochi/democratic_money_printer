"""
Gradient Boosting Strategy using HistGradientBoostingRegressor.
Predicts next period return based on Technical Indicators and Lagged Returns.
"""

import logging
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Optional

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from models.market import Signal, SignalType
from strategy.base import StrategyModel, ModelConfig

logger = logging.getLogger(__name__)

class GradientBoostingStrategy(StrategyModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_path = "data/gb_model.pkl"
        self.model = GradientBoostingRegressor(
            loss='squared_error',
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        self.is_trained = False
        self._load_model()
        
        # Hyperparams
        self.forecast_horizon = 60 # 60 period ahead (Alpha)
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                logger.info(f"🧠 Loaded Gradient Boosting model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load GB model: {e}")
                
    def save_model(self):
        try:
            joblib.dump(self.model, self.model_path)
            logger.info(f"💾 Saved Gradient Boosting model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save GB model: {e}")

    def _compute_rsi(self, series: pd.Series, window=14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-6)
        return 100 - (100 / (1 + rs))

    def _compute_macd(self, series: pd.Series, slow=26, fast=12, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract technical features and targets.
        Returns DataFrame with features.
        """
        df = df.copy()
        
        # 1. Lags
        df['ret_1'] = np.log(df['close'] / df['close'].shift(1))
        df['ret_2'] = df['ret_1'].shift(1)
        df['ret_3'] = df['ret_1'].shift(2)
        df['ret_5'] = df['ret_1'].shift(4)
        
        # 2. Volatility
        df['vol_5'] = df['ret_1'].rolling(5).std()
        df['vol_20'] = df['ret_1'].rolling(20).std()
        
        # 3. Momentum
        df['rsi'] = self._compute_rsi(df['close'], 14)
        macd, sig = self._compute_macd(df['close'])
        df['macd'] = macd
        df['macd_sig'] = sig
        df['macd_hist'] = macd - sig
        
        # 4. Trend
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['dist_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # 5. Volume
        df['vol_chg'] = df['volume'].pct_change()
        
        # 6. Macro Features (if available)
        # SP500_Close, VIX_Close, BTC_Close (from HybridDataLoader)
        if 'SP500_Close' in df.columns:
            df['sp500_ret'] = df['SP500_Close'].pct_change().fillna(0)
        else:
            df['sp500_ret'] = 0.0
            
        if 'VIX_Close' in df.columns:
            df['vix'] = df['VIX_Close'].fillna(20.0) # default to 20
        else:
            df['vix'] = 20.0
            
        if 'BTC_Close' in df.columns:
            df['btc_ret'] = df['BTC_Close'].pct_change().fillna(0)
        else:
            df['btc_ret'] = 0.0

        # Calculate 60-period return for TARGET (shift back to align current row with future return)
        # ret_60 = (Price(t+60) - Price(t)) / Price(t)
        # Log return sum
        df['ret_60'] = df['ret_1'].rolling(window=60).sum().shift(-60)

        return df

    def prepare_training_data(self, data_map: Dict[str, pd.DataFrame]):
        """
        Consolidate data from all symbols into a single X, y dataset.
        """
        X_list = []
        y_list = []
        
        for symbol, df in data_map.items():
            if len(df) < 100:
                continue
                
            df_feat = self._extract_features(df)
            
            # Target: Next 60m Return
            # Already shifted in _extract_features
            df_feat['target'] = df_feat['ret_60']
            
            # Drop NaNs created by lags/indicators
            df_feat = df_feat.dropna()
            
            if df_feat.empty:
                continue
                
            # Features to use (Added Macro)
            feature_cols = [
                'ret_1', 'ret_2', 'ret_3', 'ret_5',
                'vol_5', 'vol_20',
                'rsi', 'macd', 'macd_hist',
                'dist_sma20', 'vol_chg',
                'sp500_ret', 'vix', 'btc_ret'
            ]
            
            X_list.append(df_feat[feature_cols].values)
            y_list.append(df_feat['target'].values)
            
        if not X_list:
            return None, None
            
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        return X, y

    def train(self, data_map: Dict[str, pd.DataFrame]):
        logger.info(f"🚂 Training Gradient Boosting Strategy on {len(data_map)} symbols...")
        
        X, y = self.prepare_training_data(data_map)
        
        if X is None or len(X) == 0:
            logger.warning("Insufficient data for training GB model.")
            return
            
        logger.info(f"Training on {len(X)} samples. Features: {X.shape[1]}")
        
        # Fit model
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"✅ Training Complete. Score (R2): {self.model.score(X, y):.4f}")
        
        self.save_model()

    def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        if not self.is_trained:
            return Signal(SignalType.HOLD, 0.0, self.name, symbol, "Model not trained")
            
        if len(data) < 60: # Need enough for indicators
            return Signal(SignalType.HOLD, 0.0, self.name, symbol, "Insufficient liquidity/history")
            
        # Extract features for latest bar
        df_feat = self._extract_features(data)
        
        # Get last row
        last_row = df_feat.iloc[-1:]
        
        feature_cols = [
            'ret_1', 'ret_2', 'ret_3', 'ret_5',
            'vol_5', 'vol_20',
            'rsi', 'macd', 'macd_hist',
            'dist_sma20', 'vol_chg',
            'sp500_ret', 'vix', 'btc_ret'
        ]
        
        # Check for NaNs (if not enough data populated)
        if last_row[feature_cols].isnull().values.any():
             return Signal(SignalType.HOLD, 0.0, self.name, symbol, "Features not ready (NaN)")
             
        X_pred = last_row[feature_cols].values
        
        # Predict
        pred_ret = self.model.predict(X_pred)[0]
        
        # Dynamic Scaled Threshold
        # Convert 1m vol to 15m vol
        vol_1m = last_row['vol_20'].item() if 'vol_20' in last_row else 0.001
        if pd.isna(vol_1m) or vol_1m == 0: vol_1m = 0.001
        
        vol_15m = vol_1m * np.sqrt(15)
        
        # Sniper Mode: Aggressive Threshold (0.02 * Vol_15m)
        # We want to trade on almost any positive expectation
        threshold = 0.02 * vol_15m
        confidence = min(abs(pred_ret) / (threshold + 1e-6), 1.0)
        
        # Trend Filter: DISABLED (Aggressive Mode)
        # We trust the model's prediction completely.
        
        if pred_ret > threshold:
            return Signal(SignalType.BUY, confidence, self.name, symbol, f"Pred {pred_ret:.2%} > {threshold:.2%} (Aggressive)")
                
        elif pred_ret < -threshold:
             return Signal(SignalType.SELL, confidence, self.name, symbol, f"Pred {pred_ret:.2%} < -{threshold:.2%} (Aggressive)")
             
        return Signal(SignalType.HOLD, 0.0, self.name, symbol, f"Pred {pred_ret:.2%} within {threshold:.2%} band")
