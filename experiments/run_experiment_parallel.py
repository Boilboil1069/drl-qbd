import os
import sys
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.run_experiment import (
    run_grid_experiment,
)
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


def _run_single_mode_parallel(mode: str,
                              algos,
                              corr_levels,
                              load_factors,
                              mus,
                              horizon_time: float,
                              train_episodes: int,
                              eval_episodes: int,
                              routing_samples: int,
                              seed: int,
                              total_tasks: int,
                              mode_index: int,
                              total_modes: int):
    """在子进程中跑单个 map_mode 的 run_grid_experiment。

    返回 (mode, exp_data_mode) 方便主进程汇总。
    """
    print(f"[PARALLEL][MODE {mode_index+1}/{total_modes}] Start map_mode='{mode}'")
    print(
        f"[PARALLEL][MODE {mode}] 总实验组合数: {total_tasks} = "
        f"len(corr_levels)={len(corr_levels)} × len(load_factors)={len(load_factors)} × len(algos)={len(algos)}"
    )
    print(
        f"[PARALLEL][MODE {mode}] 关键训练参数: horizon_time={horizon_time}, "
        f"train_episodes={train_episodes}, eval_episodes={eval_episodes}, routing_samples={routing_samples}, seed={seed}"
    )

    exp_data_mode = run_grid_experiment(
        algos=algos,
        corr_levels=corr_levels,
        load_factors=load_factors,
        mus=mus,
        horizon_time=horizon_time,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        routing_samples=routing_samples,
        seed=seed,
        map_mode=mode,
        map_modes=None,
    )
    return mode, exp_data_mode


def main():
    """并行版本的大实验入口：不同 map_mode 在多个进程中并行执行。"""
    start_time = time.time()
    algos = ("random", "jsq", "jiq", "pod2", "lw", "lc", "rr", "dqn")
    corr_levels = (0.2, 0.4, 0.6, 0.8, 1.0)
    load_factors = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
    mus = (4.5, 4.0, 3.5, 3.0)

    horizon_time = 500.0
    train_episodes = 30
    eval_episodes = 5
    routing_samples = 10000
    seed = 2024
    workers = 8

    map_modes = ("base", "hawkes", "super_burst")

    # 计算总实验组合数量，用于打印
    total_tasks = len(corr_levels) * len(load_factors) * len(algos)
    total_modes = len(map_modes)

    base_fig_dir = "figures"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = os.path.join(base_fig_dir, timestamp)
    ensure_dir(root_out)

    print("#" * 80)
    print("[PARALLEL] Start parallel grid experiment for modes:", map_modes)
    print(
        f"[PARALLEL] 全局总实验组合数: {total_modes} × {total_tasks} = {total_modes * total_tasks}"
    )
    print(
        f"[PARALLEL] 全局关键训练参数: horizon_time={horizon_time}, train_episodes={train_episodes}, "
        f"eval_episodes={eval_episodes}, routing_samples={routing_samples}, seed={seed}"
    )
    print("#" * 80)

    per_mode_results = {}

    # 并行跑每个 map_mode
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for idx, mode in enumerate(map_modes):
            fut = executor.submit(
                _run_single_mode_parallel,
                mode,
                algos,
                corr_levels,
                load_factors,
                mus,
                horizon_time,
                train_episodes,
                eval_episodes,
                routing_samples,
                seed,
                total_tasks,
                idx,
                total_modes,
            )
            futures.append(fut)

        for fut in as_completed(futures):
            mode, exp_data_mode = fut.result()
            per_mode_results[mode] = exp_data_mode
            print(f"[PARALLEL] Finished mode = {mode}")

    # 保存实验数据 (JSON + 每模式 arrays.npz)
    metadata = {
        "algos": algos,
        "corr_levels": corr_levels,
        "load_factors": load_factors,
        "mus": mus,
        "horizon_time": horizon_time,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "routing_samples": routing_samples,
        "seed": seed,
        "map_modes": map_modes,
        "timestamp": timestamp,
    }
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

    # ACF 综合图一次性画即可（不依赖 per_mode exp_data 结构）
    from plot_corr_vs_acf import big_corr_vs_acf
    big_corr_vs_acf(levels=corr_levels, models=map_modes, out_dir=root_out)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[TIME][PARALLEL] Total experiment runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes aka {elapsed/3600:.2f} hours)")


if __name__ == "__main__":
    main()
