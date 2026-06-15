"""Monte Carlo experiment drivers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .core import (
    FitOptions,
    SparseTableProblem,
    default_gamma_grid,
    evaluate_theta,
    fit_gamma_path_from_counts,
    make_diffuse_sparse_noise_distribution,
    make_high_leverage_sparse_noise_distribution,
    make_uniform_noisy_distribution,
    summarize_by_gamma,
    summarize_methods,
    wj_select_gamma_from_counts,
)


def _nearest_index(grid: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(grid - target)))


def run_diffuse_sparse_noise(
    problem: SparseTableProblem,
    out_dir: str | Path,
    n_reps: int = 40,
    n_jobs: int = 1,
    seed: int = 20260607,
    gamma_grid: np.ndarray | None = None,
    m_sparse: int = 140,
    lambda_sparse: float = 1.5,
    options: FitOptions | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the diffuse sparse-noise gamma-path experiment."""

    if gamma_grid is None:
        gamma_grid = default_gamma_grid(max_gamma=3.0, extended=True)
    gamma_grid = np.asarray(gamma_grid, dtype=float)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_global = np.random.default_rng(seed)

    def one_rep(rep_id: int, rep_seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(rep_seed)
        p_data, noise_cells, eps_total = make_diffuse_sparse_noise_distribution(
            problem.p_clean, problem.n, m_sparse, lambda_sparse, rng
        )
        counts = rng.multinomial(problem.n, p_data)
        theta_path, _p_path, success, nit = fit_gamma_path_from_counts(
            counts, problem, gamma_grid, options=options
        )
        rows = []
        for i, gamma in enumerate(gamma_grid):
            metrics = evaluate_theta(theta_path[i], problem)
            rows.append(
                {
                    "scenario": "diffuse_sparse_noise",
                    "rep": rep_id,
                    "gamma": float(gamma),
                    "success": bool(success[i]),
                    "nit": int(nit[i]),
                    "n_nonzero_cells": int(np.sum(counts > 0)),
                    "n_noise_cells_observed": int(np.sum(counts[noise_cells] > 0)),
                    "eps_total": eps_total,
                    **metrics,
                }
            )
        return pd.DataFrame(rows)

    start = time.time()
    seeds = rng_global.integers(1, 10**9, size=n_reps)
    parts = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(one_rep)(r, int(seeds[r])) for r in range(n_reps)
    )
    result_df = pd.concat(parts, ignore_index=True)
    summary_df = summarize_by_gamma(result_df)
    result_df.to_csv(out_dir / "gamma_sparse_rmse_results.csv", index=False)
    summary_df.to_csv(out_dir / "gamma_sparse_rmse_summary.csv", index=False)
    print(f"Diffuse sparse-noise experiment elapsed: {(time.time() - start) / 60:.2f} min")
    return result_df, summary_df


def run_uniform_noise(
    problem: SparseTableProblem,
    out_dir: str | Path,
    n_reps: int = 40,
    n_jobs: int = 1,
    seed: int = 20260611,
    gamma_grid: np.ndarray | None = None,
    eps_values: tuple[tuple[str, float], ...] = (("correct", 0.00), ("uniform_noise", 0.07)),
    fixed_gamma_methods: dict[str, float] | None = None,
    pilot_gamma: float = 1.0,
    b_resample: int = 10,
    subsample_frac: float = 0.70,
    options: FitOptions | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run correct/uniform-noise comparisons with WJ selection."""

    if gamma_grid is None:
        gamma_grid = default_gamma_grid(max_gamma=2.0)
    gamma_grid = np.asarray(gamma_grid, dtype=float)
    if fixed_gamma_methods is None:
        fixed_gamma_methods = {
            "MLE": 0.0,
            "critical_0.5": 0.5,
            "gamma_0.8": 0.8,
            "pilot_1.0": 1.0,
        }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_global = np.random.default_rng(seed)

    def one_rep(rep_id: int, rep_seed: int, scenario: str, eps_uniform: float):
        rng = np.random.default_rng(rep_seed)
        p_data = make_uniform_noisy_distribution(problem.p_clean, eps_uniform)
        counts = rng.multinomial(problem.n, p_data)
        gamma_wj, curve, theta_path, _p_path, success = wj_select_gamma_from_counts(
            counts,
            problem,
            gamma_grid,
            rng,
            pilot_gamma=pilot_gamma,
            b_resample=b_resample,
            subsample_frac=subsample_frac,
            options=options,
        )
        rows = []
        for method, gamma_target in fixed_gamma_methods.items():
            idx = _nearest_index(gamma_grid, gamma_target)
            metrics = evaluate_theta(theta_path[idx], problem)
            rows.append(
                {
                    "scenario": scenario,
                    "rep": rep_id,
                    "method": method,
                    "gamma": float(gamma_grid[idx]),
                    "success": bool(success[idx]),
                    "eps_uniform": eps_uniform,
                    "uniform_expected_count_per_cell": problem.n * eps_uniform / problem.K,
                    **metrics,
                }
            )
        idx_wj = _nearest_index(gamma_grid, gamma_wj)
        metrics = evaluate_theta(theta_path[idx_wj], problem)
        rows.append(
            {
                "scenario": scenario,
                "rep": rep_id,
                "method": "WJ_selected",
                "gamma": gamma_wj,
                "success": bool(success[idx_wj]),
                "eps_uniform": eps_uniform,
                "uniform_expected_count_per_cell": problem.n * eps_uniform / problem.K,
                **metrics,
            }
        )
        curve = curve.copy()
        curve["scenario"] = scenario
        curve["rep"] = rep_id
        curve["eps_uniform"] = eps_uniform
        return pd.DataFrame(rows), curve

    all_results = []
    all_curves = []
    start = time.time()
    for scenario, eps_uniform in eps_values:
        seeds = rng_global.integers(1, 10**9, size=n_reps)
        parts = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(one_rep)(r, int(seeds[r]), scenario, eps_uniform) for r in range(n_reps)
        )
        all_results.append(pd.concat([p[0] for p in parts], ignore_index=True))
        all_curves.append(pd.concat([p[1] for p in parts], ignore_index=True))

    result_df = pd.concat(all_results, ignore_index=True)
    curve_df = pd.concat(all_curves, ignore_index=True)
    summary_df = summarize_methods(result_df)
    result_df.to_csv(out_dir / "uniform_noise_method_results.csv", index=False)
    curve_df.to_csv(out_dir / "uniform_noise_wj_curves.csv", index=False)
    summary_df.to_csv(out_dir / "uniform_noise_method_summary.csv", index=False)
    print(f"Uniform-noise experiment elapsed: {(time.time() - start) / 60:.2f} min")
    return result_df, curve_df, summary_df


def run_high_leverage_stress(
    problem: SparseTableProblem,
    out_dir: str | Path,
    n_reps: int = 40,
    n_jobs: int = 1,
    seed: int = 20260612,
    gamma_grid: np.ndarray | None = None,
    m_stress: int = 100,
    lambda_stress: float = 3.0,
    low_eta_quantile: float = 0.15,
    fixed_gamma_methods: dict[str, float] | None = None,
    pilot_gamma: float = 1.0,
    b_resample: int = 10,
    subsample_frac: float = 0.70,
    options: FitOptions | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the high-leverage sparse-noise stress experiment."""

    if gamma_grid is None:
        gamma_grid = default_gamma_grid(max_gamma=2.0)
    gamma_grid = np.asarray(gamma_grid, dtype=float)
    if fixed_gamma_methods is None:
        fixed_gamma_methods = {
            "MLE": 0.0,
            "critical_0.5": 0.5,
            "gamma_0.8": 0.8,
            "pilot_1.0": 1.0,
            "gamma_1.3": 1.3,
        }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_global = np.random.default_rng(seed)

    def one_rep(rep_id: int, rep_seed: int):
        rng = np.random.default_rng(rep_seed)
        p_data, stress_cells, eps_total = make_high_leverage_sparse_noise_distribution(
            problem,
            m_stress=m_stress,
            lambda_stress=lambda_stress,
            rng=rng,
            low_eta_quantile=low_eta_quantile,
        )
        counts = rng.multinomial(problem.n, p_data)
        gamma_wj, curve, theta_path, _p_path, success = wj_select_gamma_from_counts(
            counts,
            problem,
            gamma_grid,
            rng,
            pilot_gamma=pilot_gamma,
            b_resample=b_resample,
            subsample_frac=subsample_frac,
            options=options,
        )
        rows = []
        for method, gamma_target in fixed_gamma_methods.items():
            idx = _nearest_index(gamma_grid, gamma_target)
            metrics = evaluate_theta(theta_path[idx], problem)
            rows.append(
                {
                    "scenario": "high_leverage_sparse_noise",
                    "rep": rep_id,
                    "method": method,
                    "gamma": float(gamma_grid[idx]),
                    "success": bool(success[idx]),
                    "eps_stress": eps_total,
                    "m_stress": m_stress,
                    "lambda_stress": lambda_stress,
                    "n_stress_cells_observed": int(np.sum(counts[stress_cells] > 0)),
                    "stress_count_total": int(np.sum(counts[stress_cells])),
                    "mean_clean_prob_stress_cells": float(np.mean(problem.p_clean[stress_cells])),
                    "mean_eta_stress_cells": float(np.mean(problem.eta_clean[stress_cells])),
                    **metrics,
                }
            )
        idx_wj = _nearest_index(gamma_grid, gamma_wj)
        metrics = evaluate_theta(theta_path[idx_wj], problem)
        rows.append(
            {
                "scenario": "high_leverage_sparse_noise",
                "rep": rep_id,
                "method": "WJ_selected",
                "gamma": gamma_wj,
                "success": bool(success[idx_wj]),
                "eps_stress": eps_total,
                "m_stress": m_stress,
                "lambda_stress": lambda_stress,
                "n_stress_cells_observed": int(np.sum(counts[stress_cells] > 0)),
                "stress_count_total": int(np.sum(counts[stress_cells])),
                "mean_clean_prob_stress_cells": float(np.mean(problem.p_clean[stress_cells])),
                "mean_eta_stress_cells": float(np.mean(problem.eta_clean[stress_cells])),
                **metrics,
            }
        )
        curve = curve.copy()
        curve["scenario"] = "high_leverage_sparse_noise"
        curve["rep"] = rep_id
        curve["eps_stress"] = eps_total
        return pd.DataFrame(rows), curve

    start = time.time()
    seeds = rng_global.integers(1, 10**9, size=n_reps)
    parts = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(one_rep)(r, int(seeds[r])) for r in range(n_reps)
    )
    result_df = pd.concat([p[0] for p in parts], ignore_index=True)
    curve_df = pd.concat([p[1] for p in parts], ignore_index=True)
    summary_df = summarize_methods(result_df)

    method_order = {
        "MLE": 0,
        "critical_0.5": 1,
        "gamma_0.8": 2,
        "pilot_1.0": 3,
        "gamma_1.3": 4,
        "WJ_selected": 5,
    }
    summary_df["method_order"] = summary_df["method"].map(method_order)
    summary_df = summary_df.sort_values("method_order").drop(columns="method_order")

    result_df.to_csv(out_dir / "high_leverage_stress_method_results.csv", index=False)
    curve_df.to_csv(out_dir / "high_leverage_stress_wj_curves.csv", index=False)
    summary_df.to_csv(out_dir / "high_leverage_stress_method_summary.csv", index=False)
    print(f"High-leverage stress experiment elapsed: {(time.time() - start) / 60:.2f} min")
    return result_df, curve_df, summary_df
