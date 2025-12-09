import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# tqdm may not be available in all environments; fall back to identity if import fails
try:
    from tqdm.auto import tqdm
except Exception:
    def _tqdm(x, **k):
        return x


    tqdm = _tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.common import device
from utils.plotting import plot_dqn_training_returns


class MLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Simple residual block with two Linear+LayerNorm+ReLU layers.

    Input/Output shape: [batch, hidden]
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = out + residual
        out = self.relu(out)
        return out


class DuelingMLP(nn.Module):
    """Deeper dueling network with LayerNorm and residual blocks for stability.

    Input:  [batch, obs_dim]
    Output: [batch, act_dim] Q-values
    """

    def __init__(self, obs_dim, act_dim, hidden: int = 128, n_res_blocks: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        # input projection
        self.fc_in = nn.Linear(obs_dim, hidden)
        self.ln_in = nn.LayerNorm(hidden)
        self.relu = nn.ReLU()

        # residual stack on shared features
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_res_blocks)])

        # optional final shared layer before heads
        self.fc_shared_out = nn.Linear(hidden, hidden)
        self.ln_shared_out = nn.LayerNorm(hidden)

        # value stream
        self.value_fc = nn.Linear(hidden, hidden)
        self.value_out = nn.Linear(hidden, 1)

        # advantage stream
        self.adv_fc = nn.Linear(hidden, hidden)
        self.adv_out = nn.Linear(hidden, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = self.fc_in(x)
        h = self.ln_in(h)
        h = self.relu(h)

        for block in self.res_blocks:
            h = block(h)

        h = self.fc_shared_out(h)
        h = self.ln_shared_out(h)
        h = self.relu(h)

        v = self.relu(self.value_fc(h))
        v = self.value_out(v)  # [B, 1]

        a = self.relu(self.adv_fc(h))
        a = self.adv_out(a)  # [B, A]

        a_mean = a.mean(dim=1, keepdim=True)
        q = v + a - a_mean
        return q


class ReplayBuffer:
    def __init__(self, size=50000):
        self.size = size
        self.buf = []

    def push(self, *tr):
        if len(self.buf) >= self.size:
            self.buf.pop(0)
        self.buf.append(tr)

    def sample(self, batch):
        d = random.sample(self.buf, batch)
        return list(zip(*d))

    def __len__(self):
        return len(self.buf)


# ---------------- Prioritized Replay (Proportional) -----------------
class PrioritizedReplayBuffer:
    """A lightweight proportional prioritized replay buffer emphasizing burst phases.

    Each transition stored as (s,a,r,s2,d,phase,priority).
    Priority initialization:
      base_priority = max(existing priorities) or 1
      if phase in burst_phases -> base_priority * burst_scale
      else -> base_priority
    Sampling probability p_i = priority_i ** alpha / sum_j priority_j ** alpha.
    Importance sampling weight w_i = (N * p_i)^{-beta} normalized by max.
    After a training step updating TD errors, new priority = |td_error| + eps.
    To keep burst emphasis, we multiply by burst_scale again if phase is burst.
    """

    def __init__(self, capacity=50000, alpha=0.6, beta_start=0.4, beta_frames=100000,
                 burst_phases=None, burst_scale=5.0, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.eps = eps
        self.burst_phases = set(burst_phases or [])
        self.burst_scale = burst_scale
        self.storage = []  # list of tuples
        self.priorities = []  # parallel list of priorities

    def __len__(self):
        return len(self.storage)

    def _current_beta(self):
        # linear anneal beta to 1.0 over beta_frames
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))

    def push(self, s, a, r, s2, d, phase):
        if len(self.storage) == self.capacity:
            # remove oldest
            self.storage.pop(0)
            self.priorities.pop(0)
        max_pri = max(self.priorities) if self.priorities else 1.0
        base = max_pri
        if phase in self.burst_phases:
            base *= self.burst_scale
        self.storage.append((s, a, r, s2, d, phase))
        self.priorities.append(base)

    def sample(self, batch_size):
        if len(self.storage) == 0:
            raise ValueError("PrioritizedReplayBuffer is empty")
        # Clean priorities: replace non-positive or NaN with small epsilon
        pri_raw = np.array(self.priorities, dtype=np.float64)
        pri = np.where(np.isnan(pri_raw) | (pri_raw <= 0), 1e-6, pri_raw)
        probs = pri ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0 or np.isnan(probs_sum):
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum
        actual_batch = min(batch_size, len(self.storage))
        # sample without replacement if enough, else with replacement
        replace_flag = actual_batch > len(self.storage) or actual_batch < batch_size
        indices = np.random.choice(len(self.storage), actual_batch, p=probs, replace=replace_flag)
        samples = [self.storage[i] for i in indices]
        beta = self._current_beta()
        self.frame += 1
        N = len(self.storage)
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max() + 1e-12
        s, a, r, s2, d, phase = list(zip(*samples))
        return (list(s), list(a), list(r), list(s2), list(d), list(phase), indices, weights.astype(np.float32))

    def update_priorities(self, indices, td_errors, phases):
        for idx, err, ph in zip(indices, td_errors, phases):
            new_p = float(abs(err) + self.eps)
            if ph in self.burst_phases:
                new_p *= self.burst_scale
            self.priorities[idx] = new_p


def train_dqn(env, episodes=500, progress_interval=100, out_dir="training_figs",
              prioritized=True, alpha=0.6, beta_start=0.4, burst_quantile=0.75,
              burst_scale=5.0, progress_tag: str | None = None,
              net_type: str = "dueling"):
    """Train DQN agent.

    prioritized: if True, use prioritized replay emphasizing burst phases (high arrival intensity phases).
    burst phases determined by row-sum of env.D1 >= quantile(burst_quantile).

    progress_tag: optional string shown in tqdm progress bar to indicate
    the current experiment configuration, e.g. overall task index and
    (map_mode, corr, load, algo) parameters.

    net_type: "mlp" for original shallow MLP, "dueling" for deeper dueling network.
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    if net_type == "mlp":
        q = MLP(obs_dim, act_dim).to(device)
        q_t = MLP(obs_dim, act_dim).to(device)
    else:
        # default to dueling network
        q = DuelingMLP(obs_dim, act_dim).to(device)
        q_t = DuelingMLP(obs_dim, act_dim).to(device)

    q_t.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=1e-3)

    # Determine burst phases from MAP arrival matrix (row sums of D1)
    burst_phases = []
    if prioritized and hasattr(env, 'D1') and env.D1 is not None:
        try:
            arrivals_per_phase = np.array(env.D1).sum(axis=1)
            thresh = np.quantile(arrivals_per_phase, burst_quantile)
            burst_phases = list(np.where(arrivals_per_phase >= thresh)[0])
            print(f"[DQN][PR] burst_phases={burst_phases}, thresh={thresh:.4f}, arrivals={arrivals_per_phase}")
        except Exception as e:
            print(f"[DQN][PR] Failed to compute burst_phases: {e}; fallback to empty list.")
            burst_phases = []

    if prioritized:
        buf = PrioritizedReplayBuffer(capacity=200000, alpha=alpha, beta_start=beta_start,
                                      burst_phases=burst_phases, burst_scale=burst_scale)
    else:
        buf = ReplayBuffer(size=200000)

    gamma = 0.99
    tau = 0.01

    ep_rewards = []
    loss_history = []  # record per-episode mean loss
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    train_interval = 1  # 新增：每 4 步训练一次
    global_step = 0

    # === new: build richer tqdm description with tag and episodes ===
    if progress_tag:
        desc = f"[DQN] {progress_tag} (episodes={episodes})"
    else:
        desc = f"[DQN] Training (episodes={episodes})"

    for ep in tqdm(range(episodes), desc=desc):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        episode_losses = []

        steps = 0
        while not done:
            steps += 1
            global_step += 1
            eps = max(0.05, 0.5 - ep / episodes)
            if random.random() < eps:
                act = env.action_space.sample()
            else:
                with torch.no_grad():
                    o = torch.tensor(obs, dtype=torch.float32, device=device)
                    act = q(o).argmax().item()

            phase = int(obs[0])  # current MAP phase for burst emphasis
            obs2, rew, done, _, info = env.step(act)
            if prioritized:
                buf.push(obs, act, rew, obs2, done, phase)
            else:
                buf.push(obs, act, rew, obs2, done)
            obs = obs2
            ep_ret += rew

            # Training step
            if len(buf) >= 256 and global_step % train_interval == 0:  # wait until minimal batch size reached
                if prioritized:
                    s, a, r, s2, d, ph, idxs, w = buf.sample(64)
                else:
                    s, a, r, s2, d = buf.sample(64)

                # Convert lists of arrays to numpy arrays first for efficiency
                s = np.asarray(s, dtype=np.float32)
                s2 = np.asarray(s2, dtype=np.float32)
                a = np.asarray(a, dtype=np.int64)
                r = np.asarray(r, dtype=np.float32)
                d = np.asarray(d, dtype=np.float32)

                # Then create torch tensors from numpy arrays and move to device
                s_t = torch.from_numpy(s).to(device)
                s2_t = torch.from_numpy(s2).to(device)
                a_t = torch.from_numpy(a).to(device).long()
                r_t = torch.from_numpy(r).to(device)
                d_t = torch.from_numpy(d).to(device)

                q_pred = q(s_t).gather(1, a_t.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    q_tar = q_t(s2_t).max(1)[0]
                    y = r_t + gamma * (1 - d_t) * q_tar
                td_errors = (q_pred - y).detach().cpu().numpy()

                if prioritized:
                    w_t = torch.from_numpy(w).to(device)
                    loss = (w_t * (q_pred - y) ** 2).mean()
                else:
                    loss = ((q_pred - y) ** 2).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                if prioritized:
                    buf.update_priorities(idxs, td_errors, ph)

                episode_losses.append(loss.item())

                # Soft update target network
                with torch.no_grad():
                    for p, p2 in zip(q.parameters(), q_t.parameters()):
                        p2.data.mul_(1 - tau).add_(tau * p.data)

        ep_rewards.append(ep_ret)
        if len(episode_losses) > 0:
            loss_history.append(float(sum(episode_losses) / len(episode_losses)))
        else:
            loss_history.append(0.0)

    info = {
        "ep_rewards": ep_rewards,
        "loss_history": loss_history,
        "start_time": start_time,
    }

    # Plot training returns (optional helper)
    try:
        plot_dqn_training_returns(ep_rewards, out_dir=out_dir, timestamp=start_time)
    except Exception as e:
        print(f"[DQN][WARN] Failed to plot training returns: {e}")

    return q, info
