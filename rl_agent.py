# rl_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden=128, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

def train_reinforce(env, policy, optimizer, episodes=200, gamma=0.99, device='cpu'):
    policy.to(device)
    for ep in range(1, episodes+1):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        while not done:
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            probs = policy(s)
            m = torch.distributions.Categorical(probs)
            action = m.sample().item()
            log_prob = m.log_prob(torch.tensor(action).to(device))
            next_state, reward, done, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state if next_state is not None else state

        # compute discounted returns (simple total reward baseline)
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_p, R in zip(log_probs, returns):
            policy_loss.append(-log_p * R)
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

        if ep % 10 == 0:
            print(f"Episode {ep}/{episodes}  total_reward={sum(rewards):.2f} loss={loss.item():.4f}")
