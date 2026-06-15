#!/usr/bin/env python
"""Run the high-leverage sparse-noise stress experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from gamma_sparse_table.core import FitOptions, create_problem, default_gamma_grid
from gamma_sparse_table.experiments import run_high_leverage_stress
from gamma_sparse_table.plotting import plot_method_bar, plot_wj_mse_curve, summarize_wj_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/high_leverage_stress")
    parser.add_argument("--q", type=int, default=10)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--m-stress", type=int, default=100)
    parser.add_argument("--lambda-stress", type=float, default=3.0)
    parser.add_argument("--low-eta-quantile", type=float, default=0.15)
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
    _results, curves, summary = run_high_leverage_stress(
        problem=problem,
        out_dir=out_dir,
        n_reps=args.n_reps,
        n_jobs=args.n_jobs,
        seed=args.seed,
        gamma_grid=gamma_grid,
        m_stress=args.m_stress,
        lambda_stress=args.lambda_stress,
        low_eta_quantile=args.low_eta_quantile,
        b_resample=args.b_resample,
        subsample_frac=args.subsample_frac,
        options=options,
    )
    curve_summary = summarize_wj_curve(curves)
    curve_summary.to_csv(out_dir / "high_leverage_stress_wj_curve_summary.csv", index=False)
    plot_method_bar(summary, "rmse_active_mean", "rmse_active_se", "Active-parameter RMSE", out_dir / "high_leverage_stress_active_rmse.png")
    plot_method_bar(summary, "rmse_all_mean", "rmse_all_se", "All-parameter RMSE", out_dir / "high_leverage_stress_all_rmse.png")
    plot_method_bar(summary, "prob_l2_mean", "prob_l2_se", "Clean-distribution L2", out_dir / "high_leverage_stress_prob_l2.png")
    plot_wj_mse_curve(curve_summary, out_dir / "high_leverage_stress_wj_mse_curve.png")


if __name__ == "__main__":
    main()
