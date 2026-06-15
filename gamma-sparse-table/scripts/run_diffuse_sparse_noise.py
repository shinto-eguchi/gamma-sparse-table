#!/usr/bin/env python
"""Run the diffuse sparse-noise gamma-path experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from gamma_sparse_table.core import FitOptions, create_problem, default_gamma_grid
from gamma_sparse_table.experiments import run_diffuse_sparse_noise
from gamma_sparse_table.plotting import plot_gamma_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/diffuse_sparse_noise")
    parser.add_argument("--q", type=int, default=10)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--m-sparse", type=int, default=140)
    parser.add_argument("--lambda-sparse", type=float, default=1.5)
    parser.add_argument("--max-gamma", type=float, default=3.0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    problem = create_problem(q=args.q, n=args.n)
    gamma_grid = default_gamma_grid(max_gamma=args.max_gamma, extended=args.max_gamma > 2.0)
    options = FitOptions(ridge=args.ridge)
    _results, summary = run_diffuse_sparse_noise(
        problem=problem,
        out_dir=out_dir,
        n_reps=args.n_reps,
        n_jobs=args.n_jobs,
        seed=args.seed,
        gamma_grid=gamma_grid,
        m_sparse=args.m_sparse,
        lambda_sparse=args.lambda_sparse,
        options=options,
    )
    plot_gamma_curve(summary, "rmse_all_mean", "rmse_all_se", "All-parameter RMSE", out_dir / "gamma_rmse_all.png", zoom_width=0.6)
    plot_gamma_curve(summary, "rmse_active_mean", "rmse_active_se", "Active-parameter RMSE", out_dir / "gamma_rmse_active.png", zoom_width=0.6)
    plot_gamma_curve(summary, "prob_l2_mean", "prob_l2_se", "Probability L2 error to clean model", out_dir / "gamma_prob_l2.png", zoom_width=0.6)


if __name__ == "__main__":
    main()
