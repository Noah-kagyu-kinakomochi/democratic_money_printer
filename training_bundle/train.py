import argparse
import logging
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from model_def import PricePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

def extract_features(df: pd.DataFrame, sentiment_score: float = 0.0) -> np.ndarray:
    """
    Extract 7 strictly orthogonal features for the Transformer sequence.
    Returns array of shape (N, 7).
    """
    df = df.copy()
    features = []
    
    # Feature 1: Fractionally Differentiated Price (d=0.4)
    # Retains memory of the price curve while achieving stationarity
    df['frac_diff'] = frac_diff(df['close'], d=0.4, thres=0.01)
    features.append(df['frac_diff'].fillna(0).values.reshape(-1, 1))
    
    # Feature 2: Volume 
    features.append(df['volume'].values.reshape(-1, 1))
    
    # Feature 3: Sentiment
    sent_arr = np.full((len(df), 1), sentiment_score)
    features.append(sent_arr)

    # Feature 4: US/JPY 10-Year Yield Differential
    # Fallbacks to 0 if data is missing
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

def create_sequences(data: np.ndarray, seq_len: int, targets: np.ndarray):
    """
    Create sliding window sequences.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = targets[i + seq_len - 1] # Target is the NEXT step return relative to sequence end?
        # In dl_model.py: targets = feats[1:, 0] (shifted by 1).
        # feats[i] aligns with target[i] which is return at i+1.
        # So at step i, we want to predict target[i].
        # Sequence x[i:i+seq] -> Predict target[i+seq-1]? 
        # Let's align with dl_model.py logic:
        # Loop over processed_dfs where targets are already shifted.
        
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train(data_path: str, output_dir: str, epochs: int = 20, batch_size: int = 32):
    """
    Main training loop.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
    except Exception:
        # Fallback to recursively reading directory if path is a dir
        if os.path.isdir(data_path):
             df = pd.read_parquet(data_path)
        else:
             raise

    logger.info(f"Loaded {len(df)} rows.")
    
    # Filter by symbol? specific logic? 
    # For now assume input contains 'symbol' col or is a single symbol file.
    # If multiple symbols, group by symbol.
    
    if 'symbol' not in df.columns:
        logger.warning("No 'symbol' column found. Treating as single sequence.")
        df['symbol'] = 'UNKNOWN'
        
    symbols = df['symbol'].unique()
    logger.info(f"Found symbols: {symbols}")
    
    all_features = []
    processed_dfs = []
    
    seq_len = 96 # 24 Hours of 15Min bars
    input_dim = 7 # 7 Orthogonal Features
    
    # 1. Feature Extraction
    for sym in symbols:
        sub = df[df['symbol'] == sym].sort_values("timestamp") # Ensure sorted
        if len(sub) < 50: continue
        
        # Resample to 15Min/1Day?
        # Assuming input is already appropriate resolution or 1Min.
        # Let's resample to 15Min for training consistency.
        try:
            agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            macro_cols = ["TNX_Close", "JGBS10_Close", "DXY_Close"]
            for c in macro_cols:
                if c in sub.columns: agg[c] = "last"
            
            resampled = sub.set_index("timestamp").resample("15Min").agg(agg).dropna()
        except:
             # Fallback
             resampled = sub.set_index("timestamp")
             
        if len(resampled) < seq_len + 5: continue
        
        feats = extract_features(resampled)
        
        # 96-Bar Standard Deviation Volatility Threshold for Classification
        # We classify next return relative to historical rolling std-dev.
        targets = np.zeros(len(resampled))
        returns = np.log(resampled['close'] / resampled['close'].shift(1)).fillna(0)
        rolling_std = returns.rolling(96).std().fillna(0.001)
        
        for i in range(len(resampled) - 1):
            next_ret = returns.iloc[i + 1]
            sigma = rolling_std.iloc[i]
            if next_ret > 1.5 * sigma:
                targets[i] = 2  # UP
            elif next_ret < -1.5 * sigma:
                targets[i] = 0  # DOWN
            else:
                targets[i] = 1  # FLAT
                
        # feats and targets align at index `i` mapping to `return at i+1`
        targets = targets[:-1] 
        feats = feats[:-1]
        
        # Drop rows where frac_diff is NaN (the burn-in period)
        # Assuming frac_diff is the first feature (idx 0)
        valid_idx = ~np.isnan(feats[:, 0])
        feats = feats[valid_idx]
        targets = targets[valid_idx]
        
        if len(feats) < seq_len + 5: continue
        
        processed_dfs.append((feats, targets))
        all_features.append(feats)
        
    if not all_features:
        logger.error("No valid data for training.")
        return

    # 2. Fit Scaler
    logger.info("Fitting Scaler...")
    concat_feats = np.vstack(all_features)
    scaler = RobustScaler()
    scaler.fit(concat_feats)
    
    # Save Scaler!
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # 3. Create Sequences
    X_train_list, y_train_list = [], []
    for feats, targets in processed_dfs:
        scaled = scaler.transform(feats)
        xs, ys = create_sequences(scaled, seq_len, targets)
        if len(xs) > 0:
            X_train_list.append(xs)
            y_train_list.append(ys)
            
    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    X_tensor = torch.FloatTensor(X_train)
    # Target is integer class for Cross Entropy
    y_tensor = torch.LongTensor(y_train)
    
    logger.info(f"Training data shape: {X_tensor.shape}")
    
    # 4. Train
    model = PricePredictor(input_size=input_dim)
    
    # Transformers need lower starting LRs without Warmup schedulers
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Categorical Classification Objective
    criterion = nn.CrossEntropyLoss()
    
    # Early Stopping params
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = 25
    
    model.train()
    for epoch in range(epochs):
        # Batching?
        permutation = torch.randperm(X_tensor.size()[0])
        
        epoch_loss = 0
        for i in range(0, X_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_tensor[indices], y_tensor[indices]
            
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / (len(X_tensor) / batch_size)
        
        # Step the scheduler
        scheduler.step(avg_loss)
        
        # Log learning rate and loss
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
        
        # Early Stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            # 5. Save Model (only when it improves)
            model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"  🌟 New best model saved to {model_path} (Loss: {best_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.warning(f"🛑 Early stopping triggered after {epoch+1} epochs! No improvement for {patience} epochs.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to data.parquet in the same directory as the script
    default_data_path = os.path.join(os.path.dirname(__file__), "data.parquet")
    parser.add_argument("--data", type=str, default=default_data_path, help="Path to input .parquet file or directory")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--epochs", type=int, default=150) # Bumped up significantly since we have early stopping
    args = parser.parse_args()
    
    train(args.data, args.output, args.epochs)
