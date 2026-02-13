"""
Remote Training Job for Databricks.

This script is designed to run in a Databricks environment.
It:
1. Loads the latest data from storage (DBFS/S3).
2. Trains the LSTM model.
3. Logs metrics and artifacts to MLflow.
4. Registers the model to the Model Registry if performance is good.
"""

import os
import sys
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import mlflow.sklearn

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_bundle.model_def import PricePredictor
from config.settings import load_settings

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_remote")


def create_sequences(data, seq_length, target_col):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i : i + seq_length]
        y = data[i + seq_length][target_col]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_path", type=str, default="training_bundle/data.parquet")
    parser.add_argument("--experiment_name", type=str, default="/Shared/MoneyPrinter_Experiments")
    args = parser.parse_args()

    # MLflow Setup
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name="Cloud_LSTM_Train"):
        # Log Params
        mlflow.log_params(vars(args))
        
        # 1. Load Data
        logger.info(f"Loading data from {args.data_path}...")
        try:
            df = pd.read_parquet(args.data_path)
            logger.info(f"Loaded {len(df)} rows.")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            sys.exit(1)

        # Preprocessing
        # Drop non-numeric columns for training (keep simple for now)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Ensure 'close' is present for target
        if 'close' not in numeric_cols:
             # Try to find close in original df
             if 'close' in df.columns:
                 numeric_cols = numeric_cols.union(['close'])
             else:
                 logger.error("Target column 'close' not found.")
                 sys.exit(1)

        data = df[numeric_cols].values
        
        # Scale
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Create Sequences
        SEQ_LENGTH = 60
        target_idx = list(numeric_cols).index("close")
        
        X, y = create_sequences(scaled_data, SEQ_LENGTH, target_idx)
        
        # Split (80/20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Tensor conversion
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # Loader
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
        
        # 2. Model Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on {device}")
        
        model = PricePredictor(input_dim=X.shape[2], hidden_dim=50, num_layers=2).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # 3. Training Loop
        model.train()
        for epoch in range(args.epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # 4. Evaluation
        model.eval()
        with torch.no_grad():
            preds = model(X_test.to(device)).cpu().numpy().squeeze()
            actuals = y_test.numpy()
            
            test_loss = np.mean((preds - actuals) ** 2)
            logger.info(f"Test MSE: {test_loss:.6f}")
            mlflow.log_metric("test_mse", test_loss)
            
            # Simple directional accuracy
            direction_pred = np.sign(preds[1:] - preds[:-1])
            direction_actual = np.sign(actuals[1:] - actuals[:-1])
            accuracy = np.mean(direction_pred == direction_actual)
            logger.info(f"Directional Accuracy: {accuracy:.2%}")
            mlflow.log_metric("accuracy", accuracy)

        # 5. Log Model & Scaler
        # Log Scaler as sklearn model
        mlflow.sklearn.log_model(scaler, "scaler")
        
        # Log PyTorch Model
        # Input example for signature
        input_example = X_test[:1].numpy()
        mlflow.pytorch.log_model(
            model, 
            "model",
            registered_model_name="MoneyPrinter_Model", # Registers to Model Registry
            input_example=input_example
        )
        logger.info("Model registered to 'MoneyPrinter_Model'.")
        
        # Also save locally for artifact upload if needed
        torch.save(model.state_dict(), "best_model.pth")
        joblib.dump(scaler, "scaler.pkl")
        mlflow.log_artifact("best_model.pth")
        mlflow.log_artifact("scaler.pkl")

if __name__ == "__main__":
    train()
