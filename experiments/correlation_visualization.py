import numpy as np
import matplotlib.pyplot as plt
from env.map_process import MAPSource


def simulate_arrivals(D0, D1, T=2000):
    map_src = MAPSource(D0, D1)
    t = 0
    arrivals = []
    phases = []

    while t < T:
        tau, is_arrival, phase = map_src.sample_event()
        t += tau
        phases.append(phase)
        if is_arrival:
            arrivals.append(t)

    return np.array(arrivals), np.array(phases)


def plot_correlation_level(D0_base, D1_base, levels=(0.0, 0.3, 0.6, 1.0), T=800):
    fig, axes = plt.subplots(len(levels), 1, figsize=(10, 2.5*len(levels)), sharex=True)

    for ax, level in zip(axes, levels):
        # modify burst intensity
        D0 = D0_base
        D1 = D1_base.copy()
        D1[1, 0] = 5 * level

        arrivals, phases = simulate_arrivals(D0, D1, T=T)

        ax.plot(phases[:1000], linewidth=1)
        ax.set_title(f"Correlation Level = {level}")
        ax.set_ylabel("MAP Phase")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.xlabel("Event Index")
    plt.tight_layout()
    plt.show()
