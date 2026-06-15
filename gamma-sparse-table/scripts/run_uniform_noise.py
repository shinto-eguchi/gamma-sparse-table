#!/usr/bin/env python
"""Run correct-specification and uniform-noise comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from gamma_sparse_table.core import FitOptions, create_problem, default_gamma_grid
from gamma_sparse_table.experiments import run_uniform_noise
from gamma_sparse_table.plotting import plot_method_bar, plot_wj_mse_curve, summarize_wj_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/uniform_noise")
    parser.add_argument("--q", type=int, default=10)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--eps-uniform", type=float, default=0.07)
    parser.add_argument("--b-resample", type=int, default=10)
    parser.add_argument("--subsample-frac", type=float, default=0.70)
    parser.add_argument("--max-gamma", type=float, default=2.0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    problem = create_problem(q=args.q, n=args.n)
    gamma_grid = default_gamma_grid(max_gamma=args.max_gamma)
    options = FitOptions(ridge=args.ridge)
    _results, curves, summary = run_uniform_noise(
        problem=problem,
        out_dir=out_dir,
        n_reps=args.n_reps,
        n_jobs=args.n_jobs,
        seed=args.seed,
        gamma_grid=gamma_grid,
        eps_values=(("correct", 0.0), ("uniform_noise", args.eps_uniform)),
        b_resample=args.b_resample,
        subsample_frac=args.subsample_frac,
        options=options,
    )
    curve_summary = summarize_wj_curve(curves)
    curve_summary.to_csv(out_dir / "uniform_noise_wj_curve_summary.csv", index=False)
    for scenario in summary["scenario"].unique():
        sub = summary[summary["scenario"] == scenario]
        plot_method_bar(sub, "rmse_active_mean", "rmse_active_se", "Active-parameter RMSE", out_dir / f"active_rmse_{scenario}.png")
        plot_method_bar(sub, "prob_l2_mean", "prob_l2_se", "Clean-distribution L2", out_dir / f"prob_l2_{scenario}.png")
        plot_wj_mse_curve(curve_summary[curve_summary["scenario"] == scenario], out_dir / f"wj_mse_curve_{scenario}.png")


if __name__ == "__main__":
    main()
