import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.common import device
from agents.a2c import ActorCritic

try:
    from tqdm.auto import tqdm
except Exception:
    def _tqdm(x, **k):
        return x


    tqdm = _tqdm


class PPOAgent:
    """
    PPO Agent:
      - self.net: ActorCritic
      - act(obs) 返回: action, logp, value
      - update(trajectories) 做若干轮 PPO 更新
    其中 trajectories 是列表: (obs, act, logp_old, return, advantage)
    """

    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2):
        self.net = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip = clip

    def act(self, obs):
        o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, v = self.net(o)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        act = dist.sample()
        logp = dist.log_prob(act)
        return int(act.item()), logp.detach(), v.detach()

    def update(self, trajectories, epochs=5, batch_size=64):
        obs = torch.tensor(
            np.array([t[0] for t in trajectories]),
            dtype=torch.float32, device=device
        )
        acts = torch.tensor(
            [t[1] for t in trajectories],
            dtype=torch.long, device=device
        )
        logp_old = torch.stack([t[2] for t in trajectories]).to(device)
        returns = torch.stack([t[3] for t in trajectories]).to(device)
        adv = torch.stack([t[4] for t in trajectories]).to(device)

        # 归一化adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(trajectories)
        for _ in range(epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)

            for start in range(0, n, batch_size):
                end = start + batch_size
                batch = idx[start:end]

                b_obs = obs[batch]
                b_acts = acts[batch]
                b_logp_old = logp_old[batch]
                b_returns = returns[batch]
                b_adv = adv[batch]

                logits, v = self.net(b_obs)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logp = dist.log_prob(b_acts)

                ratio = torch.exp(logp - b_logp_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_adv

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((b_returns - v) ** 2).mean()
                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


def train_ppo(env, episodes=200, gamma=0.99, lam=0.95, progress_interval=100, out_dir="training_figs"):
    """
    PPO 训练函数：
      - 返回 PPOAgent 对象
      - 在 run_experiment 中会用 ppo_agent.net(...) 进行决策
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = PPOAgent(obs_dim, act_dim, gamma=gamma, lam=lam)

    ep_rewards = []
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    for ep in tqdm(range(episodes), desc="[PPO] Training"):
        obs, _ = env.reset()
        done = False
        traj = []

        ep_ret = 0.0
        steps = 0

        while not done:
            act, logp, v = agent.act(obs)
            obs2, rew, done, _, _ = env.step(act)

            traj.append((obs, act, logp, rew, v.item()))
            obs = obs2
            ep_ret += rew
            steps += 1

        # 计算 GAE-style returns / advantages
        returns = []
        advs = []
        R = 0.0
        A = 0.0
        for (_, _, _, rew, v) in reversed(traj):
            R = rew + gamma * R
            delta = rew + gamma * R - v
            A = delta + gamma * lam * A
            returns.append(torch.tensor(R, dtype=torch.float32))
            advs.append(torch.tensor(A, dtype=torch.float32))
        returns.reverse()
        advs.reverse()

        trajectories = []
        for (o, a, logp, rew, v), R, A in zip(traj, returns, advs):
            trajectories.append((o, a, logp, R, A))

        agent.update(trajectories)

        avg_ret = ep_ret / max(steps, 1)
        ep_rewards.append(ep_ret)
        if (ep + 1) % progress_interval == 0 or (ep + 1) == episodes:
            tqdm.write(f"[PPO] Episode {ep + 1}/{episodes}, avg_return={avg_ret:.3f}")

    # 保存训练曲线
    try:
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, f"ppo_training_{start_time}.png")
        plt.figure(figsize=(8, 4))
        plt.plot(ep_rewards, label='episode_return')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('PPO Training Returns')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print(f"[PPO] Training curve saved to {fig_path}")
    except Exception as e:
        print(f"[PPO] Failed to save training plot: {e}")

    return agent
