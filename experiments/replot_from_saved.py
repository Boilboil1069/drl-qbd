import os
import sys
import argparse

# Ensure project root is on sys.path when running as a script
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.persistence import replot_saved
from utils.plotting import configure_matplotlib_for_chinese


def main():
    configure_matplotlib_for_chinese()
    parser = argparse.ArgumentParser(description="Replot figures from a saved experiment_data.json without rerunning simulations.")
    parser.add_argument("json_path", help="Path to experiment_data.json")
    parser.add_argument("--out_dir", default=None, help="Optional output directory (default: directory containing JSON)")
    parser.add_argument("--fixed_corr", type=float, default=0.5, help="Correlation level to use for fixed-corr plots")

    args = parser.parse_args()
    json_path = args.json_path

    if not os.path.isfile(json_path):
        print(f"[ERROR] JSON file not found: {json_path}")
        sys.exit(1)

    out_dir = replot_saved(json_path=json_path, out_dir=args.out_dir, fixed_corr=args.fixed_corr)
    print(f"[REPlot] Completed. Figures regenerated under: {out_dir}")


if __name__ == "__main__":
    main()

