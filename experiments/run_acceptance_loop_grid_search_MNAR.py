"""
Grid-based MNAR experiments for CV-based acceptance simulation.

Runs a Cartesian product of bias_percentage and hidden_corr configurations,
each with fixed init-sample=1500 and sample-size=300, saving results in
timestamped subdirectories under berebasl/data/simulations.

The hidden variable is controlled via --var-to-hide (default -1, i.e. the
last feature).  Override it on the command line; all grid cells share the
same var_to_hide value.

Usage
-----
  python run_acceptance_loop_grid_search_MNAR.py
  python run_acceptance_loop_grid_search_MNAR.py --var-to-hide 0 --num-gens 200
"""

import os
import sys
from datetime import datetime
from itertools import product

import torch

# Add base path for imports
base_path = os.path.abspath("..")
if base_path not in sys.path:
    sys.path.append(base_path)

from .acceptance_loop_cv_based_MNAR import (
    build_parser_for_cv_loop,
    process_args_cv_loop,
    run_cv_simulation,
)


def main():
    # Parse base arguments (includes --var-to-hide, --bias-percentage,
    # --hidden-corr and all the shared loop args).
    parser = build_parser_for_cv_loop()
    args = parser.parse_args([])

    # Process base params (bias_percentage / hidden_corr will be overridden
    # per grid cell below; var_to_hide is shared across all cells).
    params = process_args_cv_loop(args)

    # Override fixed values common to all grid cells
    params["init_sample"] = 1500
    params["sample_size"] = 300

    # ── Grid definition ───────────────────────────────────────────────────────
    # bias_percentage: quantile on the hidden variable that triggers forced
    #   acceptance.  0.0 means no MNAR distortion.
    bias_percentages: list = [0.0, 0.05, 0.10, 0.20, 0.35]

    # hidden_corr: Pearson correlation between the hidden variable and every
    #   visible variable in the DGP covariance.  0.0 = fully independent.
    hidden_corrs: list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # ─────────────────────────────────────────────────────────────────────────

    # Create timestamped base directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_sim_dir = os.path.abspath(
        os.path.join("berebasl", "data", "simulations", f"mnar_grid_{timestamp}")
    )
    os.makedirs(base_sim_dir, exist_ok=True)

    print(f"Running MNAR grid experiments in {base_sim_dir}")
    print(
        f"  var_to_hide      = {params['var_to_hide']}\n"
        f"  bias_percentages = {bias_percentages}\n"
        f"  hidden_corrs     = {hidden_corrs}\n"
        f"  Total cells      = {len(bias_percentages) * len(hidden_corrs)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Run grid ──────────────────────────────────────────────────────────────
    for bias_pct, h_corr in product(bias_percentages, hidden_corrs):
        # Build a human-readable subdirectory name
        bias_str = str(bias_pct).replace(".", "_")
        corr_str = str(h_corr).replace(".", "_")
        sub_dir_name = f"bias_{bias_str}_corr_{corr_str}"
        sim_dir_path = os.path.join(base_sim_dir, sub_dir_name)
        os.makedirs(sim_dir_path, exist_ok=True)

        # Build per-cell params
        run_params = params.copy()
        run_params["sim_dir_path"]    = sim_dir_path
        run_params["bias_percentage"] = bias_pct
        run_params["hidden_corr"]     = h_corr

        print(
            f"Starting experiment: bias_percentage={bias_pct}, "
            f"hidden_corr={h_corr}, var_to_hide={run_params['var_to_hide']} "
            f"→ {sim_dir_path}"
        )

        try:
            run_cv_simulation(run_params, device=device, dtype=torch.float64)
            print(
                f"Completed experiment: bias_percentage={bias_pct}, "
                f"hidden_corr={h_corr}"
            )
        except Exception as e:
            print(
                f"Error in experiment bias_percentage={bias_pct}, "
                f"hidden_corr={h_corr}: {e}"
            )
            continue


if __name__ == "__main__":
    main()
