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

    def __init__(self, D0, D1, mus, horizon_time=1000, burn_in_time: float = 0.0):
        super().__init__()
        self.D0, self.D1 = D0, D1
        self.map = MAPSource(D0, D1)
        self.mus = mus
        self.n = len(mus)
        self.horizon = horizon_time
        self.imbalance_weight = 2.0

        # burn-in: statistics are only accumulated after current_time >= burn_in_time
        # this helps align simulation with steady-state QBD theory by discarding
        # the initial transient from an empty system.
        self.burn_in_time = float(burn_in_time)

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
        self.current_time = 0.0

        self.env = simpy.Environment()
        self.queue_lengths = np.zeros(self.n, dtype=int)
        self.stores = [simpy.Store(self.env) for _ in range(self.n)]

        for i in range(self.n):
            self.env.process(self._server(i))

        # legacy event-count statistics (kept for backward compatibility)
        self.total_q = np.zeros(self.n)
        self.arrivals = np.zeros(self.n)

        # time-averaged statistics: ∫ Q_r(t) dt over time
        self.area_q = np.zeros(self.n, dtype=float)

        # we only start accumulating statistics after burn-in
        self._stats_started = False

        obs = self._get_obs()
        return obs, {}

    def _server(self, r):
        mu = self.mus[r]
        while True:
            job = yield self.stores[r].get()
            self.queue_lengths[r] -= 1
            yield self.env.timeout(np.random.exponential(1 / mu))

    def _maybe_start_stats(self):
        """Mark that we should start accumulating statistics after burn-in.

        This is called from step() after current_time has been advanced. Once the
        burn-in time is passed, subsequent calls are no-ops.
        """
        if (not self._stats_started) and (self.current_time >= self.burn_in_time):
            self._stats_started = True

    def step(self, action):
        assert self.action_space.contains(action)
        tau, is_arrival, new_phase = self.map.sample_event()

        # advance continuous time and service processes
        self.current_time += tau
        self.env.run(until=self.env.now + tau)

        # check whether we should start collecting statistics (after burn-in)
        self._maybe_start_stats()

        # only when MAP event is an arrival do we enqueue a new job
        if is_arrival:
            self.stores[action].put(1)
            self.queue_lengths[action] += 1
            if self._stats_started:
                self.arrivals[action] += 1

        # accumulate queue-length statistics
        if self._stats_started:
            # time-weighted statistics: area under Q(t)
            self.area_q += self.queue_lengths * tau
            # legacy event-count statistics for backward compatibility
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
        # 将队长转化为近似“工作量”并做截断 + 归一化，便于网络学习
        q = self.queue_lengths.astype(float)
        mus = np.asarray(self.mus, dtype=float)
        mus_safe = np.where(mus > 0, mus, 1e-8)
        workloads = q / mus_safe  # 单位服务时间下的排队工作量
        workloads = np.minimum(workloads, self.workload_cap) / self.workload_cap
        return np.concatenate(([float(self.map.phase)], workloads))

    def get_stats(self):
        """Return legacy event-averaged queue length (for backward compatibility).

        This corresponds to the previous implementation where we simply counted
        queue lengths at each MAP event and normalised by the total number of
        arrivals. It is kept to avoid breaking old analysis code, but new
        experiments comparing with QBD theory should prefer get_time_avg_stats().
        """
        return (self.total_q / (np.sum(self.arrivals) + 1e-6), self.arrivals)

    def get_time_avg_stats(self):
        """Return time-averaged queue length vector over the effective horizon.

        L_r ≈ (1 / T_eff) ∫_0^{T_eff} Q_r(t) dt, where T_eff is the time elapsed
        after burn-in. This is the natural quantity to compare against the
        steady-state QBD theoretical mean queue length.
        """
        if not self._stats_started:
            # no time after burn-in: return zeros to avoid NaN
            return np.zeros(self.n, dtype=float), self.arrivals

        effective_time = max(self.current_time - self.burn_in_time, 1e-6)
        L_vec = self.area_q / effective_time
        return L_vec, self.arrivals

