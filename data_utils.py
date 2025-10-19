import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

def download_and_prepare(ticker='AAPL', start='2018-01-01', end=None, seq_len=30):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    # technical features (simple)
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    values = df.values.astype(np.float32)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len, 3])  # scaled Close
    X = np.array(X)  # shape (N, seq_len, features)
    y = np.array(y).reshape(-1, 1)  # (N,1)

    # split train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    return {
        'X_train': X_train_t, 'y_train': y_train_t,
        'X_test': X_test_t, 'y_test': y_test_t,
        'scaler': scaler,
        'raw_df': df
    }
