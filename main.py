import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn



#Downloads data
data = yf.download(tickers='NVDA')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Calculate moving averages
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['MA_200'] = data['Close'].rolling(window=200).mean()
data = data.dropna()


# Preprocesses data
scaler = MinMaxScaler()
scaled_Data = scaler.fit_transform(data)

def create_sequence(data, length=30):
    x, y = [], []
    for i in range(len(data) - length):
        x.append(data[i:i+length])
        y.append(data[i+length, 3])
    return np.array(x), np.array(y)

sequence_length = 30
x, y = create_sequence(scaled_Data, sequence_length)

#Creates Tensors

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsequeeze(1)


#Transformer Model
class Transformer(nn.Module):
    def __init__(self, input_size, length, nhead=1, num_layers=2, hidden_dim=128):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(1, length, input_size))
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(input_size, 1)
    
    def forward(self, x):
        x = x + self.pos_encoder
        x = self.transformer(x)
        return self.decoder(x[:,-1, :])
    
#Train
model = Transformer(input_size=x.shape[2], seq_len=sequence_length)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    optimizer.zero_grad()
    output = model(x_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}")

#Prediction
preds = model(x_tensor).detach().numpy()
# Convert back to original price scale
preds_real = scaler.inverse_transform(
    np.concatenate([np.zeros((len(preds), x.shape[2]-1)), preds], axis=1)
)[:, -1]


#Simple trend signal
signals = ['BUY' if preds_real[i+1] > preds_real[i] else 'SELL' for i in range(len(preds_real)-1)]
print(signals)












# Calculate RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain, loss = delta.where(delta > 0, 0.0), delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = abs(loss.rolling(window=period).mean())

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    data['RSI'] = rsi
    return data
data = calculate_rsi(data)
print(data.head())






