# trading_env.py
import numpy as np
import torch

class TradingEnv:
    """
    Simple trading env:
    - Discrete actions: 0 = hold, 1 = buy (buy 1 share), 2 = sell (sell 1 share)
    - Uses price series from raw_df['Close'] (unscaled)
    - State: last seq_len raw features (scaled) + transformer_pred (scaled) + holdings + cash_norm
    """
    def __init__(self, raw_df, scaler, transformer, seq_len=30, initial_cash=10000, device='cpu'):
        self.raw_df = raw_df.reset_index(drop=True)
        self.prices = raw_df['Close'].values.astype(np.float32)
        self.seq_len = seq_len
        self.scaler = scaler
        self.transformer = transformer.to(device)
        self.device = device
        self.initial_cash = initial_cash
        self.reset()

    def reset(self, start_idx=None):
        if start_idx is None:
            # pick a random start so episodes vary
            self.start = np.random.randint(0, len(self.raw_df) - self.seq_len - 1)
        else:
            self.start = start_idx
        self.current = self.start + self.seq_len  # index of the step to predict/trade
        self.cash = float(self.initial_cash)
        self.holdings = 0.0
        self.done = False
        return self._get_state()

    def _get_window(self, idx):
        """
        Returns seq_len x 7 features:
        - first 5 columns scaled (Open, High, Low, Close, Volume)
        - last 2 columns normalized (MA10, MA50)
        """
        window = self.raw_df.iloc[idx - self.seq_len: idx][['Open','High','Low','Close','Volume','MA10','MA50']].values.astype(np.float32)
        # scale first 5 features
        scaled = self.scaler.transform(window)
        # normalize last 2 features manually
        # ma_norm = window[:, 5:]
        # ma_norm = (ma_norm - ma_norm.min(axis=0)) / (np.ptp(ma_norm, axis=0) + 1e-9)  # column-wise normalization
        # full_window = np.concatenate([scaled, ma_norm], axis=1)
        return scaled  # shape seq_len x 7

    def _predict_next(self, window_scaled):
        """
        Feed full 7-feature window to transformer
        """
        x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, seq_len, 7)
        with torch.no_grad():
            out = self.transformer(x).cpu().numpy().squeeze()
        return float(out)

    def _get_state(self):
        w = self._get_window(self.current)
        # feed full window to transformer (all 7 features)
        transformer_pred = self._predict_next(w)
        close_seq = w[:, 3]  # scaled close prices
        state = np.concatenate([
            close_seq,
            [transformer_pred],
            [self.cash / (self.initial_cash*10)],
            [self.holdings / 100.0]
        ])
        return state.astype(np.float32)

    def step(self, action):
        # actions: 0 hold, 1 buy 1 share, 2 sell 1 share
        price = float(self.prices[self.current])  # raw price
        if action == 1:  # buy
            if self.cash >= price:
                self.cash -= price
                self.holdings += 1.0
        elif action == 2:  # sell
            if self.holdings >= 1.0:
                self.cash += price
                self.holdings -= 1.0

        self.current += 1
        if self.current >= len(self.raw_df):
            self.done = True

        next_state = self._get_state() if not self.done else None
        net_worth = self.cash + self.holdings * price
        # reward = change in net worth (simple)
        reward = net_worth - self.initial_cash
        # for learning, we can give incremental reward as net_worth - prev_net_worth; here we return net_worth - initial for simplicity
        info = {'net_worth': net_worth}
        return next_state, float(reward), self.done, info
