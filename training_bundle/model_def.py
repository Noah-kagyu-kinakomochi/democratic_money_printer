import torch
import torch.nn as nn
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
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class PricePredictor(nn.Module):
    """Transformer for sequence-based price prediction."""
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.2):
        super().__init__()
        
        # Project input features to d_model space
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression head
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 3) # 3 Classes: DOWN (0), FLAT (1), UP (2)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.input_projection(x)          # (batch, seq, d_model)
        x = self.pos_encoder(x)               # Add positional encoding
        
        # Pass through Transformer
        x = self.transformer_encoder(x)       # (batch, seq, d_model)
        
        # Take the output of the LAST sequence step
        out = x[:, -1, :]                     # (batch, d_model)
        
        return self.fc_out(out)
