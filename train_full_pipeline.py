# train_full_pipeline.py (top of file)
import torch
from transformer_model import TimeSeriesTransformer
from data_utils import download_and_prepare
from trading_env import TradingEnv
from rl_agent import PolicyNet, train_reinforce
from sklearn.preprocessing._data import MinMaxScaler
import numpy as np

# # Allow unpickling of needed numpy + sklearn objects
# torch.serialization.add_safe_globals([
#     MinMaxScaler,
#     np._core.multiarray._reconstruct,
#     np.ndarray,
#     np.dtype,
#     np.dtypes.Float32DType  # <-- NEW addition
# ])

def load_transformer(path='ts_transformer.pth', device='cpu'):
    
    
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # assume feature_dim used originally was 5 (Open High Low Close Volume)
    # but we saved only state dict; need to know seq_len and feature dims we trained with
    # For simplicity, recreate a model compatible with saved state keys:
    # Option: load state into a model with same architecture you used in training script.
    # Here we try to infer d_model from state dict keys
    # -> simpler: create a model with known params used previously
    seq_len = 30
    feature_dim = 7  # match training
    model = TimeSeriesTransformer(feature_dim=feature_dim, seq_len=seq_len)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    scaler = ckpt['scaler']
    raw_df = ckpt['raw_df']
    return model, scaler, raw_df

if __name__ == '__main__':
    device = 'cpu'
    # load transformer trained earlier
    model, scaler, raw_df = load_transformer('ts_transformer.pth', device=device)
    env = TradingEnv(raw_df=raw_df, scaler=scaler, transformer=model, seq_len=30, initial_cash=10000, device=device)
    state_dim = len(env._get_state())
    policy = PolicyNet(state_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    train_reinforce(env, policy, optimizer, episodes=200, gamma=0.99, device=device)
    # after training, you can save the policy
    torch.save(policy.state_dict(), 'policy_net.pth')
    print("Finished RL training; policy saved.")
