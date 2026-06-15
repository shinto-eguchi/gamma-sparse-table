"""Plotting helpers for simulation summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_gamma_curve(
    summary: pd.DataFrame,
    mean_col: str,
    se_col: str,
    ylabel: str,
    output_path: str | Path,
    zoom_width: float | None = None,
) -> None:
    """Plot a gamma curve with 95% Monte Carlo error bars."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = summary["gamma"].to_numpy()
    y = summary[mean_col].to_numpy()
    se = summary[se_col].to_numpy()
    best_idx = int(np.argmin(y))
    best_gamma = float(x[best_idx])
    best_y = float(y[best_idx])

    plt.figure(figsize=(7.5, 5.2))
    plt.errorbar(x, y, yerr=1.96 * se, marker="o", capsize=3)
    plt.axvline(0.5, linestyle="--", linewidth=1, label=r"$\gamma=1/2$")
    plt.axvline(1.0, linestyle=":", linewidth=1, label=r"$\gamma=1$")
    plt.scatter([best_gamma], [best_y], s=110, zorder=5)
    plt.text(best_gamma, best_y, f"  min={best_gamma:.2f}", va="bottom")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(ylabel)
    plt.title(ylabel + r" vs $\gamma$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    if zoom_width is not None:
        left = max(float(x.min()), best_gamma - zoom_width)
        right = min(float(x.max()), best_gamma + zoom_width)
        mask = (x >= left) & (x <= right)
        if mask.sum() >= 2:
            zoom_path = output_path.with_name(output_path.stem + "_zoom" + output_path.suffix)
            plt.figure(figsize=(7.5, 5.2))
            plt.errorbar(x[mask], y[mask], yerr=1.96 * se[mask], marker="o", capsize=3)
            plt.axvline(0.5, linestyle="--", linewidth=1, label=r"$\gamma=1/2$")
            plt.axvline(1.0, linestyle=":", linewidth=1, label=r"$\gamma=1$")
            plt.scatter([best_gamma], [best_y], s=110, zorder=5)
            plt.text(best_gamma, best_y, f"  min={best_gamma:.2f}", va="bottom")
            plt.xlabel(r"$\gamma$")
            plt.ylabel(ylabel)
            plt.title(ylabel + r" vs $\gamma$ (zoom)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(zoom_path, dpi=220)
            plt.close()


def plot_method_bar(
    summary: pd.DataFrame,
    metric_mean: str,
    metric_se: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """Plot a bar chart for method comparisons."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(summary["method"], summary[metric_mean], yerr=1.96 * summary[metric_se], capsize=4)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_wj_mse_curve(curve_summary: pd.DataFrame, output_path: str | Path) -> None:
    """Plot WJ MSE, bias-squared, and variance curves."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gamma_min = float(curve_summary.loc[curve_summary["mse_p_mean"].idxmin(), "gamma"])
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        curve_summary["gamma"],
        curve_summary["mse_p_mean"],
        yerr=1.96 * curve_summary["mse_p_se"],
        marker="o",
        capsize=3,
        label="Estimated MSE",
    )
    plt.plot(curve_summary["gamma"], curve_summary["bias2_p_mean"], marker="s", linestyle="--", label="Bias squared")
    plt.plot(curve_summary["gamma"], curve_summary["var_p_mean"], marker="^", linestyle="--", label="Variance")
    plt.axvline(0.5, linestyle="--", linewidth=1, label=r"$\gamma=1/2$")
    plt.axvline(1.0, linestyle=":", linewidth=2, label=r"pilot $\gamma=1$")
    plt.axvline(gamma_min, linestyle="-", linewidth=2, label=fr"mean MSE min $\gamma={gamma_min:.2f}$")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Estimated MSE on fitted-distribution scale")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def summarize_wj_curve(curve_df: pd.DataFrame) -> pd.DataFrame:
    """Average WJ MSE curves over Monte Carlo replications."""

    group_cols = ["scenario", "gamma"] if "scenario" in curve_df.columns else ["gamma"]
    return (
        curve_df.groupby(group_cols)
        .agg(
            bias2_p_mean=("bias2_p", "mean"),
            var_p_mean=("var_p", "mean"),
            mse_p_mean=("mse_p", "mean"),
            mse_p_se=("mse_p", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        )
        .reset_index()
        .sort_values(group_cols)
    )
