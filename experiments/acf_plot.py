import numpy as np
import matplotlib.pyplot as plt
from env.map_process import MAPSource


def compute_lags_acf(x, max_lag=50):
    x = np.array(x)
    x = x - np.mean(x)
    acf = []
    denom = np.sum(x * x)

    for lag in range(max_lag):
        if lag == 0:
            num = denom
        else:
            num = np.sum(x[:-lag] * x[lag:])
        acf.append(num / denom)

    return np.array(acf)


def plot_map_acf(D0, D1, T=3000, max_lag=50):
    map_src = MAPSource(D0, D1)
    t = 0
    intervals = []

    while t < T:
        tau, is_arrival, phase = map_src.sample_event()
        t += tau
        intervals.append(tau)

    acf = compute_lags_acf(intervals, max_lag=max_lag)

    plt.figure(figsize=(7,4))
    plt.stem(range(max_lag), acf, use_line_collection=True)
    plt.title("Autocorrelation Function of MAP Interarrival Times")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
