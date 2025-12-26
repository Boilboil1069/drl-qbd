import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.parallel_queue_env import ParallelQueueEnv
from scenario_design import map_with_correlation, scale_load
from agents.dqn import train_dqn
from run_experiment import evaluate_policy, estimate_routing_probs
from qbd.qbd_solver import theoretical_L
from utils.common import set_seed, device


def build_env(corr=0.2, load=0.9, mus=(4.5, 4.0, 3.5, 3.0), horizon=1000.0, burn_in=0.0):
    D0_base, D1_base = map_with_correlation(corr)
    D0, D1 = scale_load(D0_base, D1_base, factor=load)
    env = ParallelQueueEnv(D0, D1, mus, horizon_time=horizon, burn_in_time=burn_in)
    return env, D0, D1


def run_single_episode_jsq(horizon=1000.0, burn_in=0.0, load=0.9, corr=0.2, mus=(4.5, 4.0, 3.5, 3.0)):
    env, _, _ = build_env(corr=corr, load=load, mus=mus, horizon=horizon, burn_in=burn_in)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        qlens = obs[1:]
        act = int(np.argmin(qlens))
        obs, rew, done, _, _ = env.step(act)
        steps += 1

    L_vec, arrivals = env.get_time_avg_stats()

    print("=== Env Sanity Check (JSQ policy) ===")
    print(f"horizon_time={horizon}, burn_in_time={burn_in}, corr={corr}, load={load}")
    print(f"current_time={env.current_time:.3f}, steps={steps}, total_arrivals={arrivals.sum()} arr_vec={arrivals}")
    print(f"area_q={env.area_q}")
    print(f"L_vec (time-avg)={L_vec}, sum={L_vec.sum():.6f}")


def compare_jsq_vs_random():
    """Compare L_sim (time-avg) between JSQ and random policies for one scenario."""
    corr = 0.2
    load = 0.9
    mus = (4.5, 4.0, 3.5, 3.0)
    horizon = 1000.0
    burn_in = 0.0
    episodes = 5

    print("\n=== Compare JSQ vs Random (L_sim only) ===")
    print(f"corr={corr}, load={load}, mus={mus}, horizon={horizon}, episodes={episodes}")

    # JSQ
    env_jsq, _, _ = build_env(corr=corr, load=load, mus=mus, horizon=horizon, burn_in=burn_in)
    L_jsq = evaluate_policy(env_jsq, "jsq", model=None, episodes=episodes, use_time_avg=True)
    print(f"[JSQ] L_sim_vec={L_jsq}, sum={L_jsq.sum():.6f}")

    # Random
    env_rand, _, _ = build_env(corr=corr, load=load, mus=mus, horizon=horizon, burn_in=burn_in)
    L_rand = evaluate_policy(env_rand, "random", model=None, episodes=episodes, use_time_avg=True)
    print(f"[Random] L_sim_vec={L_rand}, sum={L_rand.sum():.6f}")


def compare_jsq_vs_dqn_with_theory():
    """Single-scenario debug: JSQ baseline vs DQN, including L_sim and L_theory."""
    corr = 0.2
    load = 0.9
    mus = (4.5, 4.0, 3.5, 3.0)
    horizon_train = 500.0
    horizon_eval = 1000.0
    burn_in = 0.0
    episodes_eval = 5
    routing_samples = 5000
    seed = 2024

    print("\n=== Single-Scenario Debug: JSQ vs DQN (L_sim & L_theory) ===")
    print(f"corr={corr}, load={load}, mus={mus}")

    # Build base MAP
    env_tmp, D0, D1 = build_env(corr=corr, load=load, mus=mus, horizon=horizon_eval, burn_in=burn_in)

    # ---- JSQ baseline ----
    set_seed(seed)
    env_jsq, _, _ = build_env(corr=corr, load=load, mus=mus, horizon=horizon_eval, burn_in=burn_in)
    L_jsq_sim = evaluate_policy(env_jsq, "jsq", model=None, episodes=episodes_eval, use_time_avg=True)
    P_jsq = estimate_routing_probs(env_jsq, "jsq", model=None, num_samples=routing_samples)
    L_jsq_th = theoretical_L(D0, D1, mus, P_jsq)

    print("[JSQ] L_sim_vec=", L_jsq_sim, "sum=", float(L_jsq_sim.sum()))
    print("[JSQ] P_r(j)=\n", P_jsq)
    print("[JSQ] L_theory_vec=", L_jsq_th, "sum=", float(L_jsq_th.sum()))

    # ---- DQN: train then evaluate ----
    set_seed(seed)
    env_train, _, _ = build_env(corr=corr, load=load, mus=mus, horizon=horizon_train, burn_in=burn_in)
    print("\n[DQN] Training...")
    dqn_model, info = train_dqn(env_train, episodes=32, prioritized=True)

    # New env for evaluation
    env_eval, _, _ = build_env(corr=corr, load=load, mus=mus, horizon=horizon_eval, burn_in=burn_in)
    print("[DQN] Evaluating policy...")
    L_dqn_sim = evaluate_policy(env_eval, "dqn", dqn_model, episodes=episodes_eval, use_time_avg=True)
    P_dqn = estimate_routing_probs(env_eval, "dqn", dqn_model, num_samples=routing_samples)
    L_dqn_th = theoretical_L(D0, D1, mus, P_dqn)

    print("[DQN] L_sim_vec=", L_dqn_sim, "sum=", float(L_dqn_sim.sum()))
    print("[DQN] P_r(j)=\n", P_dqn)
    print("[DQN] L_theory_vec=", L_dqn_th, "sum=", float(L_dqn_th.sum()))


if __name__ == "__main__":
    # 1) 原始 JSQ 单集 sanity check
    run_single_episode_jsq()

    # 2) JSQ vs Random 平均队长对比
    compare_jsq_vs_random()

    # 3) 单场景 JSQ vs DQN，包含理论队长
    compare_jsq_vs_dqn_with_theory()
