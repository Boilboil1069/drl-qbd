import os
import sys
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import numpy as np  # 新增：用于将嵌套 list 转成 ndarray

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 不直接在顶层导入 run_grid_experiment，避免循环依赖和未使用告警
from utils.plotting import (
    configure_matplotlib_for_chinese,
    ensure_dir,
    plot_mean_queue_vs_load,
    plot_mean_queue_vs_load_fixed_corr,
    plot_error_vs_corr,
    plot_error_heatmap,
    plot_error_vs_load_1d,
)
from utils.export_latex import export_latex_table
from utils.persistence import save_experiment_data  # 新增: 数据持久化


# 全局配置一次 matplotlib 字体
configure_matplotlib_for_chinese()


def _run_single_scenario(mode: str,
                         algo: str,
                         corr: float,
                         lf: float,
                         mus,
                         horizon_train: float,
                         horizon_eval: float,
                         train_episodes: int,
                         eval_episodes: int,
                         routing_samples: int,
                         seed: int,
                         ic: int,
                         il: int,
                         algo_index: int,
                         total_algos: int,
                         total_corrs: int,
                         total_loads: int,
                         progress_tag: str | None = None):
    """在子进程中跑单个 (mode, corr, load, algo) 组合，返回原 run_grid_experiment
    中 _run_single_mode 对应位置的结果切片。

    progress_tag: 由主进程构造的全局进度前缀，如
      "TASK 37/1560 | mode=base, algo=dqn, corr=0.4, load=0.3"，
    将会在 DQN 训练的 tqdm 描述中展示全局进度。

    返回: (mode, algo, ic, il, L_sim_total, L_th_total, err, by_load_increment)
    其中 by_load_increment 是本 load_factor 下的 (L_sim_total, L_th_total)，
    供主进程做在线平均聚合。
    """
    from experiments.run_experiment import run_grid_experiment as _rg
    # 子进程只打印简单 worker 信息，不再使用 [TASK x/y] 语义，避免 1/1 误导
    print(
        f"[WORKER] mode={mode}, algo={algo} ({algo_index+1}/{total_algos}), "
        f"corr={corr} ({ic+1}/{total_corrs}), load={lf} ({il+1}/{total_loads})"
    )

    # 这里 progress_tag 目前主要用于 DQN tqdm 描述，由 run_grid_experiment 内部
    # 再拼接场景信息使用；为保持接口简单，先通过全局变量挂载或后续扩展
    # （当前实现中 run_grid_experiment 自身已根据 (mode, algo, corr, load) 构造 tag，
    # 这里的全局 TASK 信息会在后续需要时整合进去）。
    # For DQN: do a short-horizon training phase (faster), then a long-horizon
    # evaluation phase to compute stable metrics. For other algos, run only the
    # evaluation horizon (no extra fast training step here).
    if algo == "dqn":
        # training phase (no evaluation during this call)
        _rg(
            algos=(algo,),
            corr_levels=(corr,),
            load_factors=(lf,),
            mus=mus,
            horizon_time=horizon_train,
            train_episodes=train_episodes,
            eval_episodes=0,
            routing_samples=routing_samples,
            seed=seed,
            map_mode=mode,
            map_modes=None,
            verbose_task=False,
            global_progress_tag=progress_tag,
        )
        # evaluation phase (no further training)
        exp = _rg(
            algos=(algo,),
            corr_levels=(corr,),
            load_factors=(lf,),
            mus=mus,
            horizon_time=horizon_eval,
            train_episodes=0,
            eval_episodes=eval_episodes,
            routing_samples=routing_samples,
            seed=seed,
            map_mode=mode,
            map_modes=None,
            verbose_task=False,
            global_progress_tag=progress_tag,
        )
    else:
        exp = _rg(
            algos=(algo,),
            corr_levels=(corr,),
            load_factors=(lf,),
            mus=mus,
            horizon_time=horizon_eval,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            routing_samples=routing_samples,
            seed=seed,
            map_mode=mode,
            map_modes=None,
            verbose_task=False,
            global_progress_tag=progress_tag,
        )

    # exp 是单模式返回结构：{"results": {algo: {...}}, "corr_levels": [...], ...}
    res_algo = exp["results"][algo]
    L_sim = float(res_algo["L_sim"][0, 0])
    L_th = float(res_algo["L_theory"][0, 0])
    err = float(res_algo["err"][0, 0])

    by_load_increment = (L_sim, L_th)
    return mode, algo, ic, il, L_sim, L_th, err, lf, by_load_increment


def main():
    parser = argparse.ArgumentParser(description="Fine-grained parallel grid experiment")
    parser.add_argument(
        "--net",
        type=str,
        default="dueling",
        choices=["mlp", "dueling"],
        help="Q-network architecture for DQN: 'mlp' or 'dueling'",
    )
    args, _ = parser.parse_known_args()

    """更细粒度并行版本：每个 (mode, corr, load, algo) 组合独立子进程。

    注意：总组合数较大时要适当调小 workers，以避免 CPU 过载或显存争用。
    """
    start_time = time.time()
    algos = ("random", "jsq", "jiq", "pod2", "lw", "lc", "rr", "dqn")
    corr_levels = (0.2, 0.4, 0.6, 0.8, 1.0)
    load_factors = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
    # mus = (4.5, 4.0, 3.5, 3.0)
    mus = (4.0, 4.0, 4.0, 4.0)

    # Use separate horizons: shorter for training (to speed up DQN updates),
    # longer for evaluation (to get stable final metrics)
    horizon_train = 500.0
    horizon_eval = 1000.0
    train_episodes = 64
    eval_episodes = 5
    routing_samples = 10000
    seed = 2024
    workers = 8

    map_modes = ("base", "hawkes", "super_burst")

    n_modes = len(map_modes)
    n_corrs = len(corr_levels)
    n_loads = len(load_factors)
    n_algos = len(algos)
    total_tasks = n_modes * n_corrs * n_loads * n_algos

    base_fig_dir = "figures"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = os.path.join(base_fig_dir, timestamp)
    ensure_dir(root_out)

    print("#" * 80)
    print("[PARALLEL] Start fine-grained parallel grid experiment")
    print(
        f"[PARALLEL] 总实验组合数: modes={n_modes} × corrs={n_corrs} × loads={n_loads} × algos={n_algos} = {total_tasks}"
    )
    print(
        f"[PARALLEL] 关键训练参数: horizon_time={horizon_train}, train_episodes={train_episodes}, "
        f"eval_episodes={eval_episodes}, routing_samples={routing_samples}, seed={seed}"
    )
    print("#" * 80)

    # 预先为每个 mode / algo 创建结果容器，与 run_grid_experiment 单模式结构一致
    per_mode_results: dict[str, dict] = {}
    for mode in map_modes:
        mode_results = {}
        for algo in algos:
            mode_results[algo] = {
                "L_sim": [[0.0 for _ in load_factors] for _ in corr_levels],
                "L_theory": [[0.0 for _ in load_factors] for _ in corr_levels],
                "err": [[0.0 for _ in load_factors] for _ in corr_levels],
                "by_load": {lf: {"L_sim": 0.0, "L_theory": 0.0} for lf in load_factors},
            }
        per_mode_results[mode] = {
            "results": mode_results,
            "corr_levels": list(corr_levels),
            "load_factors": list(load_factors),
            "mus": mus,
            "map_mode": mode,
        }

    # 计数器，用于按 (corr, load) 在线平均 by_load（同一负载下不同 corr 的平均）
    by_load_counts: dict[str, dict[float, int]] = {
        mode: {lf: 0 for lf in load_factors} for mode in map_modes
    }

    submitted = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        global_task_idx = 0
        for mode_idx, mode in enumerate(map_modes):
            for ic, corr in enumerate(corr_levels):
                for il, lf in enumerate(load_factors):
                    for algo_index, algo in enumerate(algos):
                        global_task_idx += 1
                        # 构造全局 progress_tag，传入子进程用于 DQN tqdm 描述
                        progress_tag = (
                            f"TASK {global_task_idx}/{total_tasks} | "
                            f"mode={mode}, algo={algo}, corr={corr}, load={lf}"
                        )
                        fut = executor.submit(
                            _run_single_scenario,
                            mode,
                            algo,
                            float(corr),
                            float(lf),
                            mus,
                            horizon_train,
                            horizon_eval,
                            train_episodes,
                            eval_episodes,
                            routing_samples,
                            seed,
                            ic,
                            il,
                            algo_index,
                            n_algos,
                            n_corrs,
                            n_loads,
                            progress_tag,
                        )
                        futures.append(fut)
                        submitted += 1

        print(f"[PARALLEL] Submitted {submitted} futures")

        for i, fut in enumerate(as_completed(futures), start=1):
            mode, algo, ic, il, L_sim, L_th, err, lf, by_load_inc = fut.result()
            print(
                f"[TASK {i}/{submitted}] 当前策略={algo}, map_mode={mode}, "
                f"corr={corr_levels[ic]}, load_factor={load_factors[il]} | "
                f"L_sim={L_sim:.3f}, L_th={L_th:.3f}, err={err:.3f}"
            )

            res = per_mode_results[mode]["results"][algo]
            res["L_sim"][ic][il] = L_sim
            res["L_theory"][ic][il] = L_th
            res["err"][ic][il] = err

            # 在线平均更新 by_load（对同一 load_factor 跨 corr 平均）
            count = by_load_counts[mode][lf] + 1
            by_load_counts[mode][lf] = count
            prev_sim = res["by_load"][lf]["L_sim"]
            prev_th = res["by_load"][lf]["L_theory"]
            new_sim, new_th = by_load_inc
            res["by_load"][lf]["L_sim"] = prev_sim + (new_sim - prev_sim) / count
            res["by_load"][lf]["L_theory"] = prev_th + (new_th - prev_th) / count

    # 保存实验数据 (JSON + 每模式 arrays.npz)
    metadata = {
        "algos": algos,
        "corr_levels": corr_levels,
        "load_factors": load_factors,
        "mus": mus,
        "horizon_train": horizon_train,
        "horizon_eval": horizon_eval,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "routing_samples": routing_samples,
        "seed": seed,
        "map_modes": map_modes,
        "timestamp": timestamp,
    }

    # 在保存和绘图前，先把每个 algo 的 L_sim/L_theory/err 从嵌套 list 转成 numpy 数组，
    # 以保持与串行 run_experiment 返回结构一致，便于 plotting 代码使用切片 L_sim[ic, :]
    for mode in map_modes:
        mode_res = per_mode_results[mode]["results"]
        for algo in algos:
            if algo not in mode_res:
                continue
            arrs = mode_res[algo]
            for key in ("L_sim", "L_theory", "err"):
                if isinstance(arrs.get(key), list):
                    arrs[key] = np.array(arrs[key], dtype=float)

    json_saved_path = save_experiment_data(per_mode_results, root_out, metadata)
    print(f"[DATA] Experiment data saved to {json_saved_path}")

    # 统一画图 + 导出表格
    fixed_corr = 0.5

    for mode in map_modes:
        data_mode = per_mode_results[mode]
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

    from plot_corr_vs_acf import big_corr_vs_acf
    big_corr_vs_acf(levels=corr_levels, models=map_modes, out_dir=root_out)

    end_time = time.time()
    elapsed = end_time - start_time
    print(
        f"[TIME][PARALLEL] Total experiment runtime: {elapsed:.2f} seconds "
        f"({elapsed/60:.2f} minutes aka {elapsed/3600:.2f} hours)"
    )


if __name__ == "__main__":
    main()
