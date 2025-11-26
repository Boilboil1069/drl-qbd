import os
import sys
import numpy as np
# 添加项目根路径以支持直接运行脚本时的包导入
root_path = os.path.dirname(os.path.dirname(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
print(f"[DEBUG] root_path={root_path}")
print(f"[DEBUG] sys.path[:4]={sys.path[:4]}")

import matplotlib.pyplot as plt
from experiments.acf_plot import compute_lags_acf
from env.map_process import MAPSource
from scenario_design import map_with_correlation
from advanced_maps import (
    mmpp2_with_level,
    hawkes_like_with_level,
    super_burst_with_level,
)
from datetime import datetime
import os


def _simulate_intervals(D0, D1, T):
    map_src = MAPSource(D0, D1)
    t = 0.0
    intervals = []
    while t < T:
        tau, is_arr, ph = map_src.sample_event()
        t += tau
        intervals.append(tau)
    return intervals


def plot_corr_vs_acf(levels=(0.0,0.3,0.6,1.0), T=3000, max_lag=60, out_dir="figures"):
    """旧版：单一基础 map_with_correlation 的相位轨迹+ACF 拼图（仍保留）。"""
    fig, axes = plt.subplots(2, len(levels),
                             figsize=(4*len(levels), 6),
                             sharex='col')

    for idx, level in enumerate(levels):
        D0, D1 = map_with_correlation(level)
        intervals = _simulate_intervals(D0, D1, T)

        # phase sequence 再单独模拟（为减少重复算到达区分，这里复用 MAPSource）
        map_src = MAPSource(D0, D1)
        phases = []
        t = 0.0
        while t < 800:  # 只截取前 800 时间单位的相位轨迹
            tau, is_arr, ph = map_src.sample_event()
            t += tau
            phases.append(ph)

        axes[0, idx].plot(phases, linewidth=1)
        axes[0, idx].set_title(f"corr={level}")
        axes[0, idx].set_ylabel("Phase")
        axes[0, idx].grid(True, alpha=0.4)

        acf = compute_lags_acf(intervals, max_lag=max_lag)
        axes[1, idx].stem(range(max_lag), acf)
        axes[1, idx].set_xlabel("Lag")
        axes[1, idx].set_ylabel("ACF")
        axes[1, idx].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


def big_corr_vs_acf(levels=(0.0, 0.25, 0.5, 0.75, 1.0), T=4000, max_lag=80,
                    models=("base", "mmpp2", "hawkes", "super_burst"),
                    out_dir="figures"):
    """生成“相关性 vs ACF 的大图”:
    行：不同 MAP 复杂模型
    列：不同相关性 level
    每个子图：给出该 level 下该模型的 interarrival ACF。
    base: 原 scenario_design.map_with_correlation
    mmpp2: 使用 mmpp2_with_level 调节 alpha
    hawkes: 使用 hawkes_like_with_level 调节自激发强度
    super_burst: 使用 super_burst_with_level 调节 burst 连续程度
    """
    levels = list(levels)
    models = list(models)

    n_rows = len(models)
    n_cols = len(levels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0*n_cols, 2.6*n_rows), sharex=True, sharey=True)
    axes_arr = np.atleast_2d(axes)  # 保证二维

    def get_map(model_name, level):
        if model_name == "base":
            return map_with_correlation(level)
        if model_name == "mmpp2":
            return mmpp2_with_level(level)
        if model_name == "hawkes":
            return hawkes_like_with_level(level)
        if model_name == "super_burst":
            return super_burst_with_level(level)
        raise ValueError(f"Unknown model_name={model_name}")

    # 预计算所有 ACF
    acf_cache = {}
    for m in models:
        for lv in levels:
            D0, D1 = get_map(m, lv)
            intervals = _simulate_intervals(D0, D1, T)
            acf_cache[(m, lv)] = compute_lags_acf(intervals, max_lag=max_lag)

    for i, m in enumerate(models):
        for j, lv in enumerate(levels):
            ax = axes_arr[i, j]
            acf = acf_cache[(m, lv)]
            ax.stem(range(max_lag), acf, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
            if i == 0:
                ax.set_title(f"level={lv}")
            if j == 0:
                ax.set_ylabel(f"{m}\nACF")
            ax.grid(True, alpha=0.35, linestyle='--')

    # 底行统一设置 x 轴标签
    for ax in axes_arr[-1, :]:
        ax.set_xlabel("Lag")

    fig.suptitle("复杂 MAP: 相关性 level vs Interarrival ACF", fontsize=14)
    plt.tight_layout(rect=(0,0,1,0.96))

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, timestamp)
    os.makedirs(out_path, exist_ok=True)
    pdf = os.path.join(out_path, "corr_vs_acf_big.pdf")
    png = os.path.join(out_path, "corr_vs_acf_big.png")
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, bbox_inches='tight')
    print(f"[FIG] saved big corr-vs-acf figure: {pdf} / {png}")
    plt.close(fig)


if __name__ == "__main__":
    # 示例调用：生成大图
    big_corr_vs_acf()
