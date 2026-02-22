import torch.nn as nn

class PricePredictor(nn.Module):
    """LSTM with Batch Normalization for sequence-based price prediction."""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.3
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take last time step output
        out = out[:, -1, :]
        # Apply Batch Normalization + Activation
        out = self.bn(out)
        out = self.relu(out)
        return self.fc(out)
