import os
from typing import Any, Dict, Iterable

from .plotting import ensure_dir


def export_latex_table(exp_data: Dict[str, Any], algos: Iterable[str] | None = None,
                       filename: str = "results.tex", out_dir: str = "figures") -> None:
    """Export experiment results to a LaTeX table file.

    Parameters
    ----------
    exp_data : dict
        Output from `run_grid_experiment`, expected keys:
          - "results": mapping algo_name -> {"L_sim", "L_theory", "err"}
          - "corr_levels": list of correlation levels
          - "load_factors": list of load factors
    algos : iterable of str, optional
        Which algorithms to include as columns; defaults to all keys in
        `exp_data["results"]`.
    filename : str, optional
        LaTeX file name to write (under out_dir).
    out_dir : str, optional
        Directory where the LaTeX file will be written.
    """
    results = exp_data["results"]
    corr_levels = exp_data["corr_levels"]
    load_factors = exp_data["load_factors"]

    if algos is None:
        algos = list(results.keys())

    ensure_dir(out_dir)
    file_path = os.path.join(out_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{c|c|" + "c" * len(algos) + "}\\hline\n")

        header = " & ".join([a.upper() for a in algos])
        f.write("Corr & Load & " + header + " \\\\ \\hline\n")

        for ic, corr in enumerate(corr_levels):
            for il, lf in enumerate(load_factors):
                row = [f"{corr:.2f}", f"{lf:.2f}"]
                for algo in algos:
                    sim = results[algo]["L_sim"][ic, il]
                    th = results[algo]["L_theory"][ic, il]
                    err = results[algo]["err"][ic, il]
                    row.append(f"{sim:.2f}/{th:.2f}/{err:.2f}")
                f.write(" & ".join(row) + " \\\\ \n")
            f.write("\\hline\n")

        f.write("\\end{tabular}\n")
        f.write("\\caption{不同策略在相关性与负载组合下的仿真与理论结果（格式：Sim/Theory/Err）。}\n")
        f.write("\\label{tab:drl_qbd_results}\n")
        f.write("\\end{table}\n")

    print(f"[LaTeX] 表格代码已写入 {file_path}")
