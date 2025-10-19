# transformer_model.py
import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_dim, seq_len, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        # project feature_dim -> d_model
        self.input_proj = nn.Linear(feature_dim, d_model)
        # positional enc
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)  # predict scalar (scaled Close)
        )

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = self.input_proj(x)  # -> (batch, seq_len, d_model)
        x = x + self.pos_enc[:, :x.size(1), :]
        h = self.transformer(x)  # (batch, seq_len, d_model)
        out = self.output(h[:, -1, :])  # use last timestep representation
        return out
