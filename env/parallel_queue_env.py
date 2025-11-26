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

        self.observation_space = spaces.Box(
            low=0, high=200, shape=(1 + self.n,), dtype=np.int32
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

        self.stores[action].put(1)
        self.queue_lengths[action] += 1
        self.arrivals[action] += 1

        tau, is_arrival, new_phase = self.map.sample_event()
        self.current_time += tau

        self.env.run(until=self.env.now + tau)
        self.total_q += self.queue_lengths

        reward = -np.sum(self.queue_lengths)
        done = (self.current_time >= self.horizon)

        obs = self._get_obs()
        info = {}
        return obs, reward, done, False, info

    def _get_obs(self):
        return np.concatenate(([self.map.phase], self.queue_lengths))

    def get_stats(self):
        return (self.total_q / (np.sum(self.arrivals) + 1e-6), self.arrivals)
