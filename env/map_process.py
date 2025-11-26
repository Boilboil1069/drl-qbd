import warnings

import numpy as np


class MAPSource:
    """
    MAP(D0,D1) sample_event() -> (tau, is_arrival, new_phase)
    """

    def __init__(self, D0, D1):
        self.D0 = D0
        self.D1 = D1
        self.Q = D0 + D1
        self.m = D0.shape[0]
        self.reset()

        self.rates = -np.diag(self.Q)

    def reset(self):
        self.phase = np.random.randint(0, self.m)

    def sample_event(self):
        i = self.phase
        # ensure we operate on a Python float (avoid numpy-scalar typing issues)
        rate_i = float(self.rates[i])

        # protect against non-positive rates (avoid division by zero)
        if rate_i <= 0:
            warnings.warn(
                f"MAPSource: non-positive exit rate detected for phase {i} (rate={rate_i}). "
                "Falling back to small positive rate and uniform transition probabilities.")
            rate_i = max(rate_i, 1e-12)

        tau = np.random.exponential(1.0 / rate_i)

        # compute transition probabilities and make robust to tiny negative/rounding errors
        probs = self.Q[i, :].astype(float) / rate_i
        # Clip small negative values (numerical noise) to zero
        probs = np.maximum(probs, 0.0)
        s = probs.sum()
        if s <= 0 or not np.isfinite(s):
            # if sum is zero (or NaN/Inf), fall back to uniform probabilities
            warnings.warn(
                f"MAPSource: invalid transition probabilities from phase {i} (sum={s}). "
                "Using uniform probabilities instead.")
            probs = np.ones(self.m, dtype=float) / float(self.m)
        else:
            probs = probs / s

        j = np.random.choice(np.arange(self.m), p=probs)
        is_arrival = self.D1[i, j] > 0
        self.phase = int(j)
        return tau, is_arrival, j
