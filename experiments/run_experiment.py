import os

import sys
from datetime import datetime
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.plotting import (
    configure_matplotlib_for_chinese,
    ensure_dir,
    plot_mean_queue_vs_load,
    plot_mean_queue_vs_load_fixed_corr,
    plot_error_vs_corr,
    plot_error_heatmap,
    plot_error_vs_load_1d,
    plot_dqn_q_loss,
)
from utils.export_latex import export_latex_table
from utils.persistence import save_experiment_data  # 新增: 数据持久化保存


# configure font early so all plots use it
configure_matplotlib_for_chinese()

from env.parallel_queue_env import ParallelQueueEnv
from agents.dqn import train_dqn
from agents.a2c import train_a2c
from agents.ppo import train_ppo
from agents.sac import train_sac
from qbd.qbd_solver import theoretical_L
from scenario_design import map_with_correlation, scale_load
from utils.common import set_seed, device
from advanced_maps import mmpp2_with_level, hawkes_like_with_level, super_burst_with_level

"""
==============================================================
JSQ 策略的理论最优性简述（可直接搬进论文理论/相关工作小节）
==============================================================

在对称的 M/M/n 系统中（n 个独立、同质的指数服务台，服务率相同），
经典结果表明：

  - 在许多“工作负载凸序”（workload convex order）意义下，
    JSQ（Join-the-Shortest-Queue）在所有非预emptive、非抢占式路由策略中，
    最小化队长向量的凸函数（例如总队长、最大队长等）的期望值。

典型文献包括 Winston (1977) 以及后续对多服务器排队网络的推广，
结论可大致概括为：

  - 若所有服务器同质、服务为指数分布、到达为泊松（或更一般的 MAP），
    且路由策略在给定队长向量时仅基于队长信息进行决策，
    则 JSQ 在队长向量的凸序意义下是“最优”的路由规则之一。

在 MAP/M/n 的设定下，可以把 MAP 看成对泊松到达的马尔可夫调制：
外部到达仍然满足 PASTA 型性质（在相位条件下）。
在给定某个背景相位 j 时，系统瞬时到达过程近似为参数 λ_j 的泊松流；
若所有服务器仍然对称，则对每个固定相位 j，
JSQ 对“条件队长分布”的优化性质与 M/M/n 模型类似。

因此，在本文的 “外部 MAP + 并联 M/1 + DRL 路由” 框架中：

  - JSQ 可以看作一个经典的强基线（strong baseline）；
  - DRL 策略应当在很多场景下至少达到 JSQ 的性能，
    或在存在异质服务器/复杂成本时超越 JSQ；
  - 使用 QBD 理论 + 仿真对比，可以定量评估 DRL 相对于 JSQ
    在复杂到达（MAP）和复杂代价函数下的优势或劣势。

==============================================================
"""


# ============================================================
# 策略动作选择（包括 DRL + 多个经典 baseline）
# ============================================================

def select_action(policy_type, model, env, obs):
    """
    根据策略类型 + 模型 + 当前观测 obs 选择动作。
    支持的策略：
      - random      : 随机路由
      - jsq         : Join-the-Shortest-Queue
      - pod2        : Power-of-d with d=2（两随机队列中选短者）
      - jiq         : Join-Idle-Queue（有空队列时选空队列，否则 JSQ）
      - dqn / a2c / ppo / sac : 深度强化学习策略
    """
    # 纯随机
    if policy_type == "random" or model is None:
        return env.action_space.sample()

    # JSQ：选当前队长最短的队列
    if policy_type == "jsq":
        qlens = obs[1:]  # 观测为 [phase, q1, q2, ...]
        return int(np.argmin(qlens))

    # Power-of-d (d=2)：随机选两个队列，选队长较短的
    if policy_type == "pod2":
        qlens = obs[1:]
        n = len(qlens)
        i, j = np.random.choice(n, size=2, replace=False)
        return int(i if qlens[i] <= qlens[j] else j)

    # Join-Idle-Queue (JIQ)：如有空队列优先选空，否则用 JSQ
    if policy_type == "jiq":
        qlens = obs[1:]
        idle_idx = np.where(qlens == 0)[0]
        if len(idle_idx) > 0:
            return int(np.random.choice(idle_idx))
        return int(np.argmin(qlens))

    # LC（在本框架下与 JSQ 等价）
    if policy_type == "lc":
        qlens = obs[1:]
        return int(np.argmin(qlens))

    # LW：Least-Work
    if policy_type == "lw":
        qlens = obs[1:]
        mus = np.array(env.mus, dtype=float)
        mus_safe = np.where(mus > 0, mus, 1e-8)
        workloads = qlens / mus_safe
        return int(np.argmin(workloads))

    # RR：Round-Robin
    if policy_type == "rr":
        n = len(obs[1:])  # 或 env.n
        if not hasattr(select_action, "_rr_index"):
            select_action._rr_index = 0
        idx = select_action._rr_index % n
        select_action._rr_index = (idx + 1) % n
        return int(idx)

    # 下面是 DRL 类策略
    o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    if policy_type == "dqn":
        with torch.no_grad():
            q = model(o)
            return int(q.argmax().item())

    if policy_type == "a2c":
        with torch.no_grad():
            logits, v = model(o)
            probs = torch.softmax(logits, dim=-1)
            return int(probs.argmax().item())

    if policy_type == "ppo":
        with torch.no_grad():
            logits, v = model.net(o)
            probs = torch.softmax(logits, dim=-1)
            return int(probs.argmax().item())

    if policy_type == "sac":
        with torch.no_grad():
            probs = model(o)
            return int(probs.argmax().item())

    # fallback
    return env.action_space.sample()


# ============================================================
# 策略训练统一入口
# ============================================================

def train_policy(env, algo_name, episodes=100):
    """
    统一训练接口：
      - random, jsq, pod2, jiq 都不训练（rule-based）
      - dqn / a2c / ppo / sac 会调用对应的 train_xxx
    """
    if algo_name in ["random", "jsq", "pod2", "jiq", "lw", "lc", "rr"]:
        print(f"[TRAIN] {algo_name}: rule-based，无需训练")
        return None

    if algo_name == "dqn":
        print("[TRAIN] DQN ...")
        model, info = train_dqn(env, episodes=episodes)
        # Q-loss 绘图在主实验脚本 main() 中统一完成
        return model, info

    if algo_name == "a2c":
        print("[TRAIN] A2C ...")
        return train_a2c(env, episodes=episodes)

    if algo_name == "ppo":
        print("[TRAIN] PPO ...")
        return train_ppo(env, episodes=episodes)

    if algo_name == "sac":
        print("[TRAIN] SAC ...")
        return train_sac(env, episodes=episodes)

    raise ValueError(f"Unknown algo_name = {algo_name}")


# ============================================================
# 仿真评估 + 路由概率估计
# ============================================================

def evaluate_policy(env, policy_type, model, episodes=5):
    """
    多次仿真，估计平均队长向量 L_sim (对每条队列)
    """
    all_L = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            act = select_action(policy_type, model, env, obs)
            obs, rew, done, _, info = env.step(act)

        L_vec, arrivals = env.get_stats()
        all_L.append(L_vec)

    return np.mean(np.array(all_L), axis=0)


def estimate_routing_probs(env, policy_type, model, num_samples=2000):
    """
    估计路由概率 p_r(j)
    返回矩阵 P: shape (n_servers, m_phases)
      P[r, j] ≈ P{路由到队列 r | MAP 相位=j}
    """
    m = env.D0.shape[0]
    n = env.n

    counts = np.zeros((n, m), dtype=float)
    totals = np.zeros(m, dtype=float)

    obs, _ = env.reset()
    done = False
    collected = 0

    while collected < num_samples:
        phase = int(obs[0])
        act = select_action(policy_type, model, env, obs)

        counts[act, phase] += 1
        totals[phase] += 1
        collected += 1

        obs, _, done, _, _ = env.step(act)
        if done:
            obs, _ = env.reset()
            done = False

    P = np.zeros_like(counts)
    for j in range(m):
        if totals[j] > 0:
            P[:, j] = counts[:, j] / totals[j]
        else:
            P[:, j] = 1.0 / n
    return P


# ============================================================
# 实验网格：相关性 × 负载
# ============================================================

def run_grid_experiment(
        algos=("random", "jsq", "pod2", "jiq", "dqn"),
        corr_levels=(0.0, 0.2, 0.6, 0.8, 1.0),
        load_factors=(0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4),
        mus=(4.5, 4.0, 3.5, 3.0),
        horizon_time=200.0,
        train_episodes=80,
        eval_episodes=5,
        routing_samples=2000,
        seed=2024,
        map_mode="base",      # 单模式（向后兼容）
        map_modes=None,        # 若提供序列，将对每个模式运行并汇总
):
    """
    在 (相关性 × 负载) 网格上跑所有策略，并记录：
      - L_sim: 仿真平均总队长
      - L_theory: QBD 理论平均总队长
      - err: 误差 |L_theory - L_sim|

    map_mode 控制使用的到达模型:
      base        -> scenario_design.map_with_correlation(level)
      mmpp2       -> advanced_maps.mmpp2_with_level(level)
      hawkes      -> advanced_maps.hawkes_like_with_level(level)
      super_burst -> advanced_maps.super_burst_with_level(level)
    两种用法：
    1) 单模式：传入 map_mode（默认 "base"），返回与旧版本兼容的结果结构。
    2) 多模式：传入 map_modes 序列（如 ("base","mmpp2","hawkes","super_burst")），忽略单个 map_mode，
       对每个模式分别跑网格，返回 {"multi": True, "per_mode": {mode: 单模式结果结构}, ...}。
    """
    set_seed(seed)
    corr_levels = list(corr_levels)
    load_factors = list(load_factors)
    mus = np.array(mus, dtype=float)

    # ---------------- internal: run one mode -----------------
    def _run_single_mode(one_mode:str):
        def build_map(level):
            if one_mode == "base":
                return map_with_correlation(level)
            if one_mode == "mmpp2":
                return mmpp2_with_level(level)
            if one_mode == "hawkes":
                return hawkes_like_with_level(level)
            if one_mode == "super_burst":
                return super_burst_with_level(level)
            raise ValueError(f"Unknown map_mode={one_mode}")

        mode_results = {}
        for algo in algos:
            mode_results[algo] = {
                "L_sim": np.zeros((len(corr_levels), len(load_factors))),
                "L_theory": np.zeros((len(corr_levels), len(load_factors))),
                "err": np.zeros((len(corr_levels), len(load_factors))),
                # 按负载聚合的一维结果：results_by_load[algo][load_factor] = {L_sim, L_theory}
                "by_load": {lf: {"L_sim": 0.0, "L_theory": 0.0} for lf in load_factors},
            }

        for ic, corr in enumerate(corr_levels):
            for il, lf in enumerate(load_factors):
                print("\n" + "=" * 70)
                print(f"[场景] map_mode={one_mode} corr={corr}, load_factor={lf}")
                print("=" * 70)
                D0_base, D1_base = build_map(corr)
                D0, D1 = scale_load(D0_base, D1_base, factor=lf)
                for algo in algos:
                    print(f"\n--- 策略: {algo} ---")
                    env = ParallelQueueEnv(D0, D1, mus, horizon_time=horizon_time)

                    # 训练策略：DQN 返回 (model, info)，其余只返回 model 或 None
                    train_ret = train_policy(env, algo, episodes=train_episodes)
                    if isinstance(train_ret, tuple):
                        model = train_ret[0]
                    else:
                        model = train_ret

                    L_sim_vec = evaluate_policy(env, algo, model, episodes=eval_episodes)
                    L_sim_total = float(np.sum(L_sim_vec))
                    print(f"  仿真平均队长 L_sim_vec={L_sim_vec}, sum={L_sim_total:.3f}")
                    routing_probs = estimate_routing_probs(env, algo, model, num_samples=routing_samples)
                    print(f"  路由概率矩阵 P_r(j):\n{routing_probs}")
                    L_th_vec = theoretical_L(D0, D1, mus, routing_probs)
                    L_th_total = float(np.sum(L_th_vec))
                    print(f"  理论平均队长 L_theory_vec={L_th_vec}, sum={L_th_total:.3f}")
                    err = abs(L_th_total - L_sim_total)
                    print(f"  误差 |L_theory - L_sim| = {err:.3f}")
                    mode_results[algo]["L_sim"][ic, il] = L_sim_total
                    mode_results[algo]["L_theory"][ic, il] = L_th_total
                    mode_results[algo]["err"][ic, il] = err

                    # 记录按负载聚合的一维数据：对同一 load_factor 取相关性平均
                    prev_sim = mode_results[algo]["by_load"][lf]["L_sim"]
                    prev_theory = mode_results[algo]["by_load"][lf]["L_theory"]
                    # 简单取平均：累加后除以场景数（这里用在线平均，避免额外存储）
                    # 当前第 ic+1 次遇到该负载
                    k = ic + 1
                    mode_results[algo]["by_load"][lf]["L_sim"] = prev_sim + (L_sim_total - prev_sim) / k
                    mode_results[algo]["by_load"][lf]["L_theory"] = prev_theory + (L_th_total - prev_theory) / k
        return {
            "results": mode_results,
            "corr_levels": corr_levels,
            "load_factors": load_factors,
            "mus": mus,
            "map_mode": one_mode,
        }

    # -------------- multi-mode branch -----------------
    if map_modes is not None:
        agg = {}
        mode_list = list(map_modes)
        for m in mode_list:
            print("\n" + "#" * 80)
            print(f"### 开始运行 map_mode = {m} ###")
            print("#" * 80)
            agg[m] = _run_single_mode(m)
        return {
            "multi": True,
            "per_mode": agg,
            "corr_levels": corr_levels,
            "load_factors": load_factors,
            "mus": mus,
            "map_modes": mode_list,
        }

    # -------------- single-mode branch (compat) ---------
    return _run_single_mode(map_mode)


# ============================================================
# main
# ============================================================

def main():
    start_time = time.time()
    algos = ("random", "jsq", "jiq", "pod2", "lw", "lc", "rr", "dqn")
    corr_levels = (0.25, 0.5, 0.75, 1.0)
    load_factors = (0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4)
    mus = (4.5, 4.0, 3.5, 3.0)

    exp_data = run_grid_experiment(
        algos=algos,
        corr_levels=corr_levels,
        load_factors=load_factors,
        mus=mus,
        horizon_time=500.0,
        train_episodes=100,
        eval_episodes=4,
        routing_samples=15000,
        seed=2024,
        map_modes=("base", "mmpp2", "hawkes", "super_burst"),
    )

    base_fig_dir = "figures"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = os.path.join(base_fig_dir, timestamp)
    ensure_dir(root_out)

    # 保存数据：统一转成 per_mode 结构以复用 save_experiment_data
    if exp_data.get("multi"):
        per_mode_results = exp_data["per_mode"]
    else:
        per_mode_results = {exp_data["map_mode"]: exp_data}
    metadata = {
        "algos": algos,
        "corr_levels": corr_levels,
        "load_factors": load_factors,
        "mus": mus,
        "horizon_time": 500.0,
        "train_episodes": 100,
        "eval_episodes": 4,
        "routing_samples": 15000,
        "seed": 2024,
        "map_modes": tuple(per_mode_results.keys()),
        "timestamp": timestamp,
    }
    json_saved_path = save_experiment_data(per_mode_results, root_out, metadata)
    print(f"[DATA] Experiment data saved to {json_saved_path}")

    fixed_corr = 0.5

    if exp_data.get("multi"):
        for mode, data_mode in exp_data["per_mode"].items():
            out_dir = os.path.join(root_out, mode)
            ensure_dir(out_dir)
            plot_mean_queue_vs_load(data_mode, algos=algos, out_dir=out_dir)
            plot_mean_queue_vs_load_fixed_corr(
                data_mode,
                corr=fixed_corr,
                algos=algos,
                out_dir=out_dir,
                filename_prefix=f"mean_queue_vs_load_fixed_corr_{mode}",
            )
            plot_error_vs_corr(data_mode, algos=algos, out_dir=out_dir)
            plot_error_heatmap(data_mode, algo="dqn", out_dir=out_dir)
            plot_error_heatmap(data_mode, algo="jsq", out_dir=out_dir)
            export_latex_table(data_mode, algos=algos, filename=f"results_{mode}.tex", out_dir=out_dir)
            plot_error_vs_load_1d(
                data_mode,
                algos=algos,
                out_dir=out_dir,
                filename_prefix=f"error_vs_load_{mode}",
                title=f"理论 vs 仿真误差随利用率变化（{mode}）",
            )
    else:
        out_dir = os.path.join(root_out, exp_data['map_mode'])
        ensure_dir(out_dir)
        plot_mean_queue_vs_load(exp_data, algos=algos, out_dir=out_dir)
        plot_mean_queue_vs_load_fixed_corr(
            exp_data,
            corr=fixed_corr,
            algos=algos,
            out_dir=out_dir,
            filename_prefix=f"mean_queue_vs_load_fixed_corr_{exp_data['map_mode']}",
        )
        plot_error_vs_corr(exp_data, algos=algos, out_dir=out_dir)
        plot_error_heatmap(exp_data, algo="dqn", out_dir=out_dir)
        plot_error_heatmap(exp_data, algo="jsq", out_dir=out_dir)
        export_latex_table(exp_data, algos=algos, filename=f"results_{exp_data['map_mode']}.tex", out_dir=out_dir)
        plot_error_vs_load_1d(
            exp_data,
            algos=algos,
            out_dir=out_dir,
            filename_prefix=f"error_vs_load_{exp_data['map_mode']}",
            title=f"理论 vs 仿真误差随利用率变化（{exp_data['map_mode']}）",
        )

    # 可选：单独跑一次 DQN 训练，画 Q-loss 曲线
    try:
        from env.parallel_queue_env import ParallelQueueEnv
        from scenario_design import map_with_correlation, scale_load
        D0_base, D1_base = map_with_correlation(0.5)
        D0, D1 = scale_load(D0_base, D1_base, factor=0.8)
        mus_arr = np.array(mus, dtype=float)
        env_debug = ParallelQueueEnv(D0, D1, mus_arr, horizon_time=500.0)
        model_debug, info_debug = train_dqn(env_debug, episodes=80)
        loss_hist = info_debug.get("loss_history") if isinstance(info_debug, dict) else None
        qloss_dir = os.path.join(root_out, "dqn_debug")
        ensure_dir(qloss_dir)
        ts_qloss = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dqn_q_loss(loss_hist, out_dir=qloss_dir, timestamp=ts_qloss)
    except Exception as e:
        print(f"[WARN] Failed to run/plot standalone DQN Q-loss: {e}")

    from plot_corr_vs_acf import big_corr_vs_acf
    big_corr_vs_acf(levels=corr_levels, models=("base", "mmpp2", "hawkes", "super_burst"), out_dir=root_out)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[TIME] Total experiment runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()
