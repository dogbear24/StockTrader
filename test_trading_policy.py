import torch
import numpy as np
from transformer_model import TimeSeriesTransformer
from data_utils import download_and_prepare
from trading_env import TradingEnv
from rl_agent import PolicyNet
from sklearn.preprocessing._data import MinMaxScaler

# ----------------------------
# Configuration
# ----------------------------
TICKER = "AAPL"
SEQ_LEN = 30
INITIAL_CASH = 10000
DEVICE = 'cpu'
POLICY_PATH = 'policy_net.pth'
TRANSFORMER_PATH = 'ts_transformer.pth'

# ----------------------------
# Safe globals for loading transformer checkpoint
# ----------------------------
torch.serialization.add_safe_globals([
    MinMaxScaler,
    np.core.multiarray._reconstruct,
    np.ndarray
])

# ----------------------------
# Load Transformer and Scaler
# ----------------------------
def load_transformer(path=TRANSFORMER_PATH, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    feature_dim = 7  # must match transformer training
    model = TimeSeriesTransformer(feature_dim=feature_dim, seq_len=SEQ_LEN).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    scaler = ckpt['scaler']
    raw_df = ckpt['raw_df']
    return model, scaler, raw_df

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Load transformer
    transformer, scaler, raw_df = load_transformer()

    # Initialize environment
    env = TradingEnv(raw_df=raw_df, scaler=scaler, transformer=transformer,
                     seq_len=SEQ_LEN, initial_cash=INITIAL_CASH, device=DEVICE)
    
    # Load policy network
    state_dim = len(env._get_state())
    policy = PolicyNet(state_dim).to(DEVICE)
    policy.load_state_dict(torch.load(POLICY_PATH, map_location=DEVICE))
    policy.eval()

    # Run a single trading episode
    state = env.reset()
    done = False
    total_reward = 0.0

    print("Starting trading simulation...\n")
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_probs = policy(state_tensor)
        action = torch.argmax(action_probs, dim=1).item()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state

    print("Simulation finished.")
    print(f"Final net worth: ${info['net_worth']:.2f}")
    print(f"Total reward: {total_reward:.2f}")
