import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.common import device

try:
    from tqdm.auto import tqdm
except Exception:
    def _tqdm(x, **k):
        return x


    tqdm = _tqdm


class ActorCritic(nn.Module):
    """
    A2C 用的共享网络：
    forward(obs) -> (logits, value)
    """

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi_head = nn.Linear(hidden, act_dim)
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        logits = self.pi_head(h)
        v = self.v_head(h).squeeze(-1)
        return logits, v


def train_a2c(env, episodes=200, gamma=0.99, lr=1e-3, progress_interval=100, out_dir="training_figs"):
    """
    Advantage Actor-Critic (A2C) 训练
    返回：ActorCritic 网络，其 forward(obs) -> (logits, value)
    在 run_experiment 里会用：
        logits, v = model(o)
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    ep_rewards = []
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    for ep in tqdm(range(episodes), desc="[A2C] Training"):
        obs, _ = env.reset()
        done = False

        log_probs = []
        values = []
        rewards = []

        ep_ret = 0.0
        steps = 0

        while not done:
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, v = net(o)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            act = dist.sample()

            logp = dist.log_prob(act)
            obs2, rew, done, _, _ = env.step(int(act.item()))

            log_probs.append(logp)
            values.append(v.squeeze(0))
            rewards.append(torch.tensor(rew, dtype=torch.float32, device=device))

            obs = obs2
            ep_ret += rew
            steps += 1

        # 计算 returns 和 advantage
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.append(R)
        returns.reverse()

        returns = torch.stack(returns)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)

        advantages = returns - values

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = (advantages ** 2).mean()
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_ret = ep_ret / max(steps, 1)
        ep_rewards.append(ep_ret)
        if (ep + 1) % progress_interval == 0 or (ep + 1) == episodes:
            tqdm.write(f"[A2C] Episode {ep + 1}/{episodes}, avg_return={avg_ret:.3f}")

    # 保存训练曲线
    try:
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, f"a2c_training_{start_time}.png")
        plt.figure(figsize=(8, 4))
        plt.plot(ep_rewards, label='episode_return')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('A2C Training Returns')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print(f"[A2C] Training curve saved to {fig_path}")
    except Exception as e:
        print(f"[A2C] Failed to save training plot: {e}")

    return net
