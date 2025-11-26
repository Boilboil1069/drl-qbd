import os
import json
from typing import Dict, Any
import numpy as np


# ------------------------------------------------------------
# Serialization helpers for experiment data
# ------------------------------------------------------------

def _serialize_single_mode(data_mode: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single-mode experiment result into a JSON-serializable dict.
    Expected keys in data_mode: results, corr_levels, load_factors, mus, map_mode.
    """
    out: Dict[str, Any] = {
        "map_mode": data_mode.get("map_mode"),
        "corr_levels": list(map(float, data_mode.get("corr_levels", []))),
        "load_factors": list(map(float, data_mode.get("load_factors", []))),
        "mus": list(map(float, np.array(data_mode.get("mus", []), dtype=float))),
        "results": {},
    }
    results = data_mode.get("results", {})
    for algo, res in results.items():
        out_res: Dict[str, Any] = {
            "L_sim": np.array(res.get("L_sim", [])).tolist(),
            "L_theory": np.array(res.get("L_theory", [])).tolist(),
            "err": np.array(res.get("err", [])).tolist(),
            "by_load": {},
        }
        by_load = res.get("by_load", {})
        for lf, vals in by_load.items():
            out_res["by_load"][str(lf)] = {
                "L_sim": float(vals.get("L_sim", 0.0)),
                "L_theory": float(vals.get("L_theory", 0.0)),
            }
        out["results"][algo] = out_res
    return out


def save_experiment_data(per_mode_results: Dict[str, Dict[str, Any]], out_dir: str, metadata: Dict[str, Any]) -> str:
    """Save aggregated per-mode results produced by parallel or single experiment.

    per_mode_results: {mode: single_mode_result_dict}
    metadata: includes fields like algos, corr_levels, load_factors, mus, horizon_time, etc.

    Writes a JSON file (experiment_data.json) under out_dir and returns its path.
    Also saves per-mode numpy arrays to compressed .npz for precise reload.
    """
    os.makedirs(out_dir, exist_ok=True)
    data_json: Dict[str, Any] = {
        "version": 1,
        "created_at": metadata.get("timestamp"),
        "algos": list(metadata.get("algos", [])),
        "corr_levels": list(map(float, metadata.get("corr_levels", []))),
        "load_factors": list(map(float, metadata.get("load_factors", []))),
        "mus": list(map(float, np.array(metadata.get("mus", []), dtype=float))),
        "horizon_time": float(metadata.get("horizon_time", 0.0)),
        "train_episodes": int(metadata.get("train_episodes", 0)),
        "eval_episodes": int(metadata.get("eval_episodes", 0)),
        "routing_samples": int(metadata.get("routing_samples", 0)),
        "seed": int(metadata.get("seed", 0)),
        "map_modes": list(metadata.get("map_modes", [])),
        "per_mode": {},
    }
    per_mode_dict: Dict[str, Any] = data_json["per_mode"]
    for mode, single in per_mode_results.items():
        per_mode_dict[mode] = _serialize_single_mode(single)

        # precise arrays dump per mode
        mode_dir = os.path.join(out_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        arrays_path = os.path.join(mode_dir, "arrays.npz")
        arrays_to_save: Dict[str, Any] = {}
        for algo, res in single.get("results", {}).items():
            arrays_to_save[f"{algo}_L_sim"] = np.array(res.get("L_sim"), dtype=float)
            arrays_to_save[f"{algo}_L_theory"] = np.array(res.get("L_theory"), dtype=float)
            arrays_to_save[f"{algo}_err"] = np.array(res.get("err"), dtype=float)
        if arrays_to_save:
            np.savez_compressed(arrays_path, **arrays_to_save)

    json_path = os.path.join(out_dir, "experiment_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=2)
    return json_path


def load_experiment_data(json_path: str) -> Dict[str, Any]:
    """Load previously saved experiment JSON and reconstruct structure similar to
    run_grid_experiment multi-mode return.
    Returns dict with keys: multi=True, per_mode={mode: single_mode_dict}, corr_levels, load_factors, mus, map_modes.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    corr_levels = raw.get("corr_levels", [])
    load_factors = raw.get("load_factors", [])
    mus = np.array(raw.get("mus", []), dtype=float)
    per_mode_raw = raw.get("per_mode", {})
    per_mode: Dict[str, Any] = {}
    for mode, sm in per_mode_raw.items():
        single: Dict[str, Any] = {
            "map_mode": sm.get("map_mode", mode),
            "corr_levels": corr_levels,
            "load_factors": load_factors,
            "mus": mus,
            "results": {},
        }
        for algo, res in sm.get("results", {}).items():
            single["results"][algo] = {
                "L_sim": np.array(res.get("L_sim", []), dtype=float),
                "L_theory": np.array(res.get("L_theory", []), dtype=float),
                "err": np.array(res.get("err", []), dtype=float),
                "by_load": {float(k): {"L_sim": float(v.get("L_sim", 0.0)), "L_theory": float(v.get("L_theory", 0.0))}
                            for k, v in res.get("by_load", {}).items()},
            }
        per_mode[mode] = single
    return {
        "multi": True,
        "per_mode": per_mode,
        "corr_levels": corr_levels,
        "load_factors": load_factors,
        "mus": mus,
        "map_modes": raw.get("map_modes", list(per_mode.keys())),
        "algos": raw.get("algos", []),
        "metadata": {
            "horizon_time": raw.get("horizon_time"),
            "train_episodes": raw.get("train_episodes"),
            "eval_episodes": raw.get("eval_episodes"),
            "routing_samples": raw.get("routing_samples"),
            "seed": raw.get("seed"),
            "created_at": raw.get("created_at"),
            "version": raw.get("version"),
        },
    }


def replot_saved(json_path: str, out_dir: str | None = None, fixed_corr: float = 0.5) -> str:
    """Convenience function: load saved experiment_data.json and regenerate all plots & LaTeX.
    Returns the output directory used.
    """
    from utils.plotting import (
        ensure_dir,
        plot_mean_queue_vs_load,
        plot_mean_queue_vs_load_fixed_corr,
        plot_error_vs_corr,
        plot_error_heatmap,
        plot_error_vs_load_1d,
    )
    from utils.export_latex import export_latex_table

    data = load_experiment_data(json_path)
    if out_dir is None:
        # default to same directory containing JSON
        out_dir = os.path.dirname(json_path)
    ensure_dir(out_dir)

    algos = data.get("algos") or []
    map_modes = data.get("map_modes") or []

    for mode in map_modes:
        data_mode = data["per_mode"][mode]
        mode_dir = os.path.join(out_dir, mode)
        ensure_dir(mode_dir)
        plot_mean_queue_vs_load(data_mode, algos=algos, out_dir=mode_dir)
        plot_mean_queue_vs_load_fixed_corr(
            data_mode,
            corr=fixed_corr,
            algos=algos,
            out_dir=mode_dir,
            filename_prefix=f"mean_queue_vs_load_fixed_corr_{mode}",
        )
        plot_error_vs_corr(data_mode, algos=algos, out_dir=mode_dir)
        # heatmaps for representative algos if present
        for rep_algo in ["dqn", "jsq"]:
            if rep_algo in data_mode["results"]:
                plot_error_heatmap(data_mode, algo=rep_algo, out_dir=mode_dir)
        export_latex_table(data_mode, algos=algos, filename=f"results_{mode}.tex", out_dir=mode_dir)
        plot_error_vs_load_1d(
            data_mode,
            algos=algos,
            out_dir=mode_dir,
            filename_prefix=f"error_vs_load_{mode}",
            title=f"理论 vs 仿真误差随利用率变化（{mode}）",
        )

    # ACF 综合图
    try:
        # utils 目录下相对导入需要带上包名 experiments
        from experiments.plot_corr_vs_acf import big_corr_vs_acf  # 修正导入路径
        levels = data.get("corr_levels", [])
        big_corr_vs_acf(levels=levels, models=map_modes, out_dir=out_dir)
    except Exception as e:
        print(f"[WARN] replot_saved: failed to plot corr_vs_acf: {e}")

    return out_dir


__all__ = ["save_experiment_data", "load_experiment_data", "replot_saved"]
