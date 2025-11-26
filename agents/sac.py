import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.common import device

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **k: x


class ReplayBuffer:
    def __init__(self, size=50000):
        self.size = size
        self.buf = []

    def push(self, *tr):
        if len(self.buf) >= self.size:
            self.buf.pop(0)
        self.buf.append(tr)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return list(zip(*batch))

    def __len__(self):
        return len(self.buf)


class SoftQNetwork(nn.Module):
    """
    Q(s,a) 网络：输入 obs 和 one-hot action 拼接
    """

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act_onehot):
        x = torch.cat([obs, act_onehot], dim=-1)
        return self.net(x).squeeze(-1)


class PolicyNetwork(nn.Module):
    """
    策略网络：输出各离散动作的概率分布
    forward(obs) -> probs
    """

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs):
        logits = self.net(obs)
        probs = torch.softmax(logits, dim=-1)
        return probs


def train_sac(env, episodes=200, gamma=0.99, alpha=0.1, lr=3e-4, batch_size=64, progress_interval=100,
              out_dir="training_figs"):
    """
    简化版离散 SAC：
      - Q 网络：SoftQNetwork
      - 策略：PolicyNetwork（Categorical）
      - 返回 policy_model，其 forward(obs)->probs 可直接用于抽样动作
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    q_net = SoftQNetwork(obs_dim, act_dim).to(device)
    q_target = SoftQNetwork(obs_dim, act_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())

    pi_net = PolicyNetwork(obs_dim, act_dim).to(device)

    q_opt = optim.Adam(q_net.parameters(), lr=lr)
    pi_opt = optim.Adam(pi_net.parameters(), lr=lr)

    buf = ReplayBuffer()

    ep_rewards = []
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    for ep in tqdm(range(episodes), desc="[SAC] Training"):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0

        while not done:
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                probs = pi_net(o)
                dist = torch.distributions.Categorical(probs)
                act = dist.sample().item()

            obs2, rew, done, _, _ = env.step(act)
            buf.push(obs, act, rew, obs2, done)
            obs = obs2

            ep_ret += rew
            steps += 1

            # 更新阶段
            if len(buf) > 1000:
                s, a, r, s2, d = buf.sample(batch_size)
                s = torch.tensor(np.array(s), dtype=torch.float32, device=device)
                a = torch.tensor(a, dtype=torch.long, device=device)
                r = torch.tensor(r, dtype=torch.float32, device=device)
                s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32, device=device)

                # -------- 更新 Q 网络 --------
                with torch.no_grad():
                    probs2 = pi_net(s2)
                    dist2 = torch.distributions.Categorical(probs2)
                    a2 = dist2.sample()
                    logp2 = dist2.log_prob(a2)
                    a2_onehot = torch.nn.functional.one_hot(a2, act_dim).float()
                    q2 = q_target(s2, a2_onehot)
                    target = r + gamma * (1 - d) * (q2 - alpha * logp2)

                a_onehot = torch.nn.functional.one_hot(a, act_dim).float()
                q = q_net(s, a_onehot)
                q_loss = ((q - target) ** 2).mean()

                q_opt.zero_grad()
                q_loss.backward()
                q_opt.step()

                # -------- 更新策略网络 --------
                probs = pi_net(s)
                dist = torch.distributions.Categorical(probs)
                a_samp = dist.sample()
                logp = dist.log_prob(a_samp)
                a_samp_onehot = torch.nn.functional.one_hot(a_samp, act_dim).float()
                q_pi = q_net(s, a_samp_onehot)

                pi_loss = (alpha * logp - q_pi).mean()

                pi_opt.zero_grad()
                pi_loss.backward()
                pi_opt.step()

                # 软更新目标 Q
                with torch.no_grad():
                    tau = 0.01
                    for p, pt in zip(q_net.parameters(), q_target.parameters()):
                        pt.data.mul_(1 - tau).add_(tau * p.data)

        avg_ret = ep_ret / max(steps, 1)
        ep_rewards.append(ep_ret)
        if (ep + 1) % progress_interval == 0 or (ep + 1) == episodes:
            tqdm.write(f"[SAC] Episode {ep + 1}/{episodes}, avg_return={avg_ret:.3f}")

    # 保存训练曲线
    try:
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, f"sac_training_{start_time}.png")
        plt.figure(figsize=(8, 4))
        plt.plot(ep_rewards, label='episode_return')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('SAC Training Returns')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print(f"[SAC] Training curve saved to {fig_path}")
    except Exception as e:
        print(f"[SAC] Failed to save training plot: {e}")

    # 返回策略网络即可，评估时只用 pi_net
    return pi_net
