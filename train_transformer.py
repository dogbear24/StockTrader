# train_transformer.py
import torch
from transformer_model import TimeSeriesTransformer
from data_utils import download_and_prepare
import os

def train_and_save(ticker='AAPL', seq_len=30, epochs=30, save_path='ts_transformer.pth', device='cpu'):
    data = download_and_prepare(ticker=ticker, seq_len=seq_len)
    X_train = data['X_train'].to(device)
    y_train = data['y_train'].to(device)
    X_test = data['X_test'].to(device)
    y_test = data['y_test'].to(device)

    # feature_dim = X_train.shape[2]
    feature_dim = 7
    model = TimeSeriesTransformer(feature_dim, seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_test)
                val_loss = loss_fn(val_pred, y_test).item()
            print(f"Epoch {epoch}/{epochs} train_loss={loss.item():.6f} val_loss={val_loss:.6f}")

    torch.save({
        'model_state': model.state_dict(),
        'scaler': data['scaler'],
        'raw_df': data['raw_df']
    }, save_path)
    print("Saved transformer to", save_path)
    return save_path

if __name__ == '__main__':
    train_and_save()
