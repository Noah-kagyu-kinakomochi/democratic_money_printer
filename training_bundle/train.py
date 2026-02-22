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

def extract_features(df: pd.DataFrame, sentiment_score: float = 0.0) -> np.ndarray:
    """
    Extract features for a specific DataFrame window.
    Must match logic in `dl_model.py`.
    Returns array of shape (N, features).
    """
    df = df.copy()
    # 1. Log Return
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    features = []
    
    # Feature 1: Log Return
    features.append(df['log_ret'].values.reshape(-1, 1))
    
    # Feature 2: Volume 
    features.append(df['volume'].values.reshape(-1, 1))
    
    # Feature 3: Sentiment (Static approximation)
    sent_arr = np.full((len(df), 1), sentiment_score)
    features.append(sent_arr)

    # Macro cols
    macro_cols = ["SP500_Close", "VIX_Close", "BTC_Close", "XLK_Close", "XLF_Close"]
    for col in macro_cols:
        if col in df.columns:
            if col == "VIX_Close":
                # Feature 4-8: VIX Level / Others Returns
                features.append(df[col].values.reshape(-1, 1))
            else:
                feat = np.log(df[col] / df[col].shift(1)).fillna(0).values.reshape(-1, 1)
                features.append(feat)
        else:
            features.append(np.zeros((len(df), 1)))
            
    # Stack: (N, 8)
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
    
    seq_len = 10
    input_dim = 8
    
    # 1. Feature Extraction
    for sym in symbols:
        sub = df[df['symbol'] == sym].sort_values("timestamp") # Ensure sorted
        if len(sub) < 50: continue
        
        # Resample to 15Min/1Day?
        # Assuming input is already appropriate resolution or 1Min.
        # Let's resample to 15Min for training consistency.
        try:
            agg = {"close": "last", "volume": "sum"}
            macro_cols = ["SP500_Close", "VIX_Close", "BTC_Close", "XLK_Close", "XLF_Close"]
            for c in macro_cols:
                if c in sub.columns: agg[c] = "last"
            
            resampled = sub.set_index("timestamp").resample("15Min").agg(agg).dropna()
        except:
             # Fallback
             resampled = sub.set_index("timestamp")
             
        if len(resampled) < seq_len + 5: continue
        
        feats = extract_features(resampled)
        
        # Shift targets (Next close return)
        # feats column 0 is log_ret.
        # Target for step i is return at i+1.
        targets = feats[1:, 0] 
        feats = feats[:-1]
        
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
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
    logger.info(f"Training data shape: {X_tensor.shape}")
    
    # 4. Train
    model = PricePredictor(input_size=input_dim)
    
    # Transformers need lower starting LRs without Warmup schedulers
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    criterion = nn.MSELoss()
    
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
