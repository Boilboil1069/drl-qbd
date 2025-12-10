import os
import sys

import gymnasium as gym
import numpy as np
import simpy
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from env.map_process import MAPSource


class ParallelQueueEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, D0, D1, mus, horizon_time=1000):
        super().__init__()
        self.D0, self.D1 = D0, D1
        self.map = MAPSource(D0, D1)
        self.mus = mus
        self.n = len(mus)
        self.horizon = horizon_time
        self.imbalance_weight = 2.0

        # 观测: [phase, normalized workloads...] ，长度 = 1 + n
        # workload_i ≈ min(q_i / mu_i, W_MAX) / W_MAX ∈ [0,1]
        self.workload_cap = 20.0

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1 + self.n,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n)

        self.env = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.map.reset()
        self.current_time = 0

        self.env = simpy.Environment()
        self.queue_lengths = np.zeros(self.n, dtype=int)
        self.stores = [simpy.Store(self.env) for _ in range(self.n)]

        for i in range(self.n):
            self.env.process(self._server(i))

        self.total_q = np.zeros(self.n)
        self.arrivals = np.zeros(self.n)

        obs = self._get_obs()
        return obs, {}

    def _server(self, r):
        mu = self.mus[r]
        while True:
            job = yield self.stores[r].get()
            self.queue_lengths[r] -= 1
            yield self.env.timeout(np.random.exponential(1 / mu))

    def step(self, action):
        assert self.action_space.contains(action)
        tau, is_arrival, new_phase = self.map.sample_event()
        self.current_time += tau

        # 先推进服务过程再处理本次 MAP 事件
        self.env.run(until=self.env.now + tau)

        # 仅当 MAP 事件为“到达”时才真正入队
        if is_arrival:
            self.stores[action].put(1)
            self.queue_lengths[action] += 1
            self.arrivals[action] += 1

        # 记录当前队长样本，用于后续估计平均队长
        self.total_q += self.queue_lengths

        q_float = self.queue_lengths.astype(float)
        mean_q = np.mean(q_float)
        imbalance = np.mean(np.abs(q_float - mean_q))
        reward = -(np.sum(q_float) + self.imbalance_weight * imbalance)
        done = (self.current_time >= self.horizon)

        obs = self._get_obs()
        info = {}
        return obs, reward, done, False, info

    def _get_obs(self):
        # 将队长转换为近似“工作量”并做截断 + 归一化，便于网络学习
        q = self.queue_lengths.astype(float)
        mus = np.asarray(self.mus, dtype=float)
        mus_safe = np.where(mus > 0, mus, 1e-8)
        workloads = q / mus_safe  # 单位服务时间下的排队工作量
        workloads = np.minimum(workloads, self.workload_cap) / self.workload_cap
        return np.concatenate(([float(self.map.phase)], workloads))

    def get_stats(self):
        return (self.total_q / (np.sum(self.arrivals) + 1e-6), self.arrivals)
