import os
import platform

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Matplotlib 全局字体配置（支持中文）
# ============================================================


def configure_matplotlib_for_chinese():
    """Configure matplotlib font to properly display Chinese depending on OS.

    Picks a prioritized list of candidate fonts (matched by substring) and sets
    matplotlib's sans-serif list to those candidates. Falls back to DejaVu Sans
    or the first available fonts when no Chinese-capable font is detected.
    Also ensures the minus sign renders properly and PDF/SVG font embedding
    options are set so saved figures include the proper fonts.
    """
    try:
        system = platform.system()
        # candidate substrings (order = preference)
        if system == "Darwin":
            preferred = [
                "PingFang",
                "PingFang SC",
                "PingFang HK",
                "AppleGothic",
                "Heiti",
                "STHeiti",
                "Songti",
                "Noto Sans CJK",
            ]
        elif system == "Windows":
            preferred = [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "MS Gothic",
                "Noto Sans CJK",
            ]
        else:
            preferred = ["SimHei", "Microsoft YaHei", "Noto Sans CJK", "DejaVu Sans"]

        fm_list = fm.fontManager.ttflist
        available_names = [f.name for f in fm_list]

        # Build a prioritized list of actual available font names by substring matching.
        chosen_list = []
        for pref in preferred:
            for f in fm_list:
                if pref.lower() in f.name.lower():
                    if f.name not in chosen_list:
                        chosen_list.append(f.name)
        # Ensure DejaVu Sans (matplotlib default) is present as a safe fallback
        if "DejaVu Sans" in available_names and "DejaVu Sans" not in chosen_list:
            chosen_list.append("DejaVu Sans")

        # Final fallback: first few available fonts (guarantee non-empty)
        if not chosen_list:
            chosen_list = available_names[:6] if available_names else []

        # Apply to rcParams: prefer the chosen list for sans-serif and set family
        if chosen_list:
            mpl.rcParams["font.sans-serif"] = chosen_list
            mpl.rcParams["font.family"] = "sans-serif"

        # Improve PDF/SVG output so fonts are embedded/retained
        mpl.rcParams["pdf.fonttype"] = 42   # use TrueType fonts in PDFs
        mpl.rcParams["ps.fonttype"] = 42    # use TrueType for PS
        mpl.rcParams["svg.fonttype"] = "none"
        mpl.rcParams["axes.unicode_minus"] = True

        # Debug: show which actual font files matplotlib will select for the
        # first few chosen families (helps diagnose missing glyphs).
        resolved = []
        from matplotlib.font_manager import FontProperties
        for fam in (chosen_list[:6] if chosen_list else []):
            try:
                fp = FontProperties(family=fam)
                path = fm.findfont(fp, fallback_to_default=True)
                resolved.append((fam, path))
            except Exception:
                resolved.append((fam, None))

        print(f"[MATPLOTLIB] system={system}, font candidates={chosen_list[:8]}")
        print("[MATPLOTLIB] resolved font files:")
        for fam, path in resolved:
            print(f"  {fam} -> {path}")
    except Exception as e:
        print(f"[MATPLOTLIB] failed to configure font for Chinese: {e}")


# ============================================================
# 通用图像保存工具
# ============================================================


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def save_figure(fig, base_name: str, out_dir: str = "figures", dpi: int | None = None) -> None:
    """Save a Matplotlib figure as both SVG & PNG in given directory."""
    ensure_dir(out_dir)
    svg_path = os.path.join(out_dir, base_name + ".svg")
    png_path = os.path.join(out_dir, base_name + ".png")
    fig.savefig(svg_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    print(f"[FIG] saved: {svg_path} / {png_path}")


# ============================================================
# 通用理论 vs 仿真对比绘图
# ============================================================


def plot_theory_vs_sim(L_sim, L_th):
    """Bar plot comparing per-queue simulated vs theoretical mean length."""
    n = len(L_sim)
    x = np.arange(n)

    plt.figure(figsize=(6, 4))
    plt.bar(x - 0.15, L_sim, width=0.3, label="Simulation")
    plt.bar(x + 0.15, L_th, width=0.3, label="Theory (QBD)")
    plt.xticks(x, [f"Q{r}" for r in range(n)])
    plt.ylabel("Mean Queue Length")
    plt.title("Theory vs Simulation")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ============================================================
# DQN 训练相关绘图
# ============================================================


def plot_dqn_training_returns(ep_rewards, out_dir: str, timestamp: str):
    """Plot & save DQN per-episode returns curve."""
    ensure_dir(out_dir)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(ep_rewards, label="episode_return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("DQN Training Returns")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    png_name = f"dqn_training_{timestamp}"
    save_figure(fig, png_name, out_dir=out_dir)
    plt.close(fig)


def plot_dqn_q_loss(loss_hist, out_dir: str, timestamp: str):
    """Plot & save mean Q-loss curve per episode for DQN."""
    if loss_hist is None or len(loss_hist) == 0:
        print("[DQN] No loss history to plot.")
        return

    ensure_dir(out_dir)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(loss_hist, label="mean Q-loss per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("DQN Q-loss Training Curve")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    base_name = f"dqn_q_loss_{timestamp}"
    save_figure(fig, base_name, out_dir=out_dir, dpi=300)
    plt.close(fig)


# ============================================================
# run_grid_experiment 相关绘图
# ============================================================


def plot_mean_queue_vs_load(exp_data, algos=None, out_dir="figures"):
    results = exp_data["results"]
    corr_levels = exp_data["corr_levels"]
    load_factors = exp_data["load_factors"]

    if algos is None:
        algos = list(results.keys())

    for algo in algos:
        if algo not in results:
            print(f"[WARN] algo '{algo}' missing in results; skip mean_queue_vs_load plot.")
            continue
        if "L_sim" not in results[algo]:
            print(f"[WARN] algo '{algo}' lacks 'L_sim'; skip.")
            continue
        L_sim = results[algo]["L_sim"]

        fig, ax = plt.subplots(figsize=(6, 4))
        for ic, corr in enumerate(corr_levels):
            ax.plot(
                load_factors,
                L_sim[ic, :],
                marker="o",
                label=f"corr={corr:.2f}",
            )

        ax.set_xlabel("负载缩放因子 (load factor)")
        ax.set_ylabel("平均总队长 (仿真)")
        ax.set_title(f"策略 {algo}: 平均总队长 vs 负载")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        plt.tight_layout()

        save_figure(fig, f"mean_queue_vs_load_{algo}", out_dir)
        plt.close(fig)


def plot_mean_queue_vs_load_fixed_corr(exp_data, corr, algos=None, out_dir="figures",
                                       filename_prefix="mean_queue_vs_load_fixed_corr"):
    """
    固定相关性 corr，下画不同负载因子下平均总队长随负载变化。

    仅绘制 load_factor < 1.0 的部分，以聚焦稳定区间的行为。
    """
    results = exp_data["results"]
    corr_levels = exp_data["corr_levels"]
    load_factors = np.array(exp_data["load_factors"], dtype=float)

    if algos is None:
        algos = list(results.keys())

    # 找到最接近 corr 的索引
    corr_levels_arr = np.array(corr_levels, dtype=float)
    idx = int(np.argmin(np.abs(corr_levels_arr - float(corr))))
    corr_val = corr_levels_arr[idx]

    # 仅保留 load_factor < 1.0 的索引; 若没有则使用全部
    mask = load_factors < 1.0
    if not np.any(mask):
        mask = np.ones_like(load_factors, dtype=bool)
    x_vals = load_factors[mask]

    def _dedup_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        new_h, new_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                new_h.append(h)
                new_l.append(l)
        ax.legend(new_h, new_l, fontsize=12, ncol=2)

    fig, ax = plt.subplots(figsize=(8, 6))
    for algo in algos:
        if algo not in results or "L_sim" not in results[algo]:
            print(f"[WARN] algo '{algo}' missing L_sim; skip fixed_corr plot.")
            continue
        L_sim = results[algo]["L_sim"]  # shape (n_corr, n_load)
        if L_sim.shape[0] <= idx:
            print(f"[WARN] algo '{algo}' L_sim shape {L_sim.shape} insufficient for corr index {idx}; skip.")
            continue
        L_line_full = L_sim[idx, :]
        L_line = L_line_full[mask]
        ax.plot(x_vals, L_line, marker="o", linewidth=2, label=algo)

    ax.set_xlabel("负载缩放因子 (load factor)", fontsize=14)
    ax.set_ylabel("平均总队长 (仿真)", fontsize=14)
    ax.set_title(f"固定相关性 corr={corr_val:.2f} 时，不同策略平均总队长 vs 负载", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    _dedup_legend(ax)
    plt.tight_layout()

    ensure_dir(out_dir)
    base = f"{filename_prefix}_corr_{corr_val:.2f}".replace(".", "p")
    save_figure(fig, base, out_dir=out_dir)
    plt.close(fig)


def plot_error_vs_corr(exp_data, algos=None, out_dir="figures"):
    results = exp_data["results"]
    corr_levels = exp_data["corr_levels"]
    load_factors = exp_data["load_factors"]

    if algos is None:
        algos = list(results.keys())

    # 只保留 load factor < 1 的索引
    lf_arr = np.array(load_factors, dtype=float)
    valid_mask = lf_arr < 1.0
    valid_indices = np.where(valid_mask)[0]

    for algo in algos:
        if algo not in results:
            print(f"[WARN] algo '{algo}' missing; skip error_vs_corr.")
            continue
        rec = results[algo]
        if "err" not in rec:
            print(f"[WARN] algo '{algo}' has no 'err' key; available keys={list(rec.keys())}; skip.")
            continue
        err_full = rec["err"]  # shape (n_corr, n_load)
        fig, ax = plt.subplots(figsize=(6, 4))
        for il in valid_indices:
            if il >= err_full.shape[1]:
                continue
            lf = lf_arr[il]
            ax.plot(
                corr_levels,
                err_full[:, il],
                marker="s",
                label=f"load={lf:.2f}",
            )

        ax.set_xlabel("相关性水平 (corr)")
        ax.set_ylabel(r"$|L_{\mathrm{theory}} - L_{\mathrm{sim}}|$")
        ax.set_title(f"策略 {algo}: 理论-仿真误差 vs 相关性")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        plt.tight_layout()

        save_figure(fig, f"error_vs_corr_{algo}", out_dir)
        plt.close(fig)


def plot_error_heatmap(exp_data, algo, out_dir="figures"):
    results = exp_data.get("results", {})
    if algo not in results:
        print(f"[WARN] algo '{algo}' not found in results keys={list(results.keys())}; skip heatmap.")
        return
    rec = results[algo]
    if "err" not in rec:
        print(f"[WARN] algo '{algo}' has no 'err' key; keys={list(rec.keys())}; skip heatmap.")
        return
    err = rec["err"]
    if err is None or (isinstance(err, np.ndarray) and err.size == 0):
        print(f"[WARN] algo '{algo}' err array empty; skip heatmap.")
        return

    corr_levels = exp_data.get("corr_levels", [])
    load_factors = exp_data.get("load_factors", [])
    if len(corr_levels) == 0 or len(load_factors) == 0:
        print(f"[WARN] Missing corr_levels or load_factors; skip heatmap for '{algo}'.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        err,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=(load_factors[0], load_factors[-1], corr_levels[0], corr_levels[-1]),
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$|L_{\mathrm{theory}} - L_{\mathrm{sim}}|$")

    ax.set_xlabel("负载缩放 (load factor)")
    ax.set_ylabel("相关性 (corr)")
    ax.set_title(f"策略 {algo}: 理论-仿真误差热力图")
    plt.tight_layout()

    save_figure(fig, f"error_heatmap_{algo}", out_dir)
    plt.close(fig)


def plot_error_vs_load_1d(exp_data, algos=None,
                          title="理论 vs 仿真误差随利用率变化",
                          out_dir="figures",
                          filename_prefix="error_vs_load"):
    """
    基于 run_grid_experiment 生成的按负载聚合结果，绘制误差-负载折线图。
    """
    results = exp_data["results"]
    load_factors = list(exp_data["load_factors"])

    if algos is None:
        algos = list(results.keys())

    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(8, 6))

    for algo in algos:
        if algo not in results:
            print(f"[WARN] algo '{algo}' missing; skip error_vs_load_1d.")
            continue
        if "by_load" not in results[algo]:
            print(f"[WARN] algo '{algo}' missing 'by_load'; keys={list(results[algo].keys())}; skip.")
            continue
        errors = []
        for lf in load_factors:
            rec = results[algo]["by_load"].get(lf)
            if rec is None:
                errors.append(np.nan)
                continue
            L_sim = rec.get("L_sim", np.nan)
            L_theory = rec.get("L_theory", np.nan)
            errors.append(abs(L_sim - L_theory))
        ax.plot(load_factors, errors, marker="o", linewidth=2, label=algo)

    ax.set_xlabel("利用率 (load factor)", fontsize=14)
    ax.set_ylabel("误差 |L_sim − L_theory|", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=12)
    plt.tight_layout()

    base = filename_prefix
    save_figure(fig, base, out_dir=out_dir, dpi=300)
    plt.close(fig)
