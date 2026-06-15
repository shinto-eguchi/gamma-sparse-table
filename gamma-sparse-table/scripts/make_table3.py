#!/usr/bin/env python
"""Format Table 3 from high_leverage_stress_method_summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", nargs="?", default="results/high_leverage_stress/high_leverage_stress_method_summary.csv")
    parser.add_argument("--out", default="results/high_leverage_stress/table3_markdown.md")
    return parser.parse_args()


def fmt_mean_se(row: pd.Series, mean_col: str, se_col: str) -> str:
    return f"{row[mean_col]:.4f} ({row[se_col]:.4f})"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    rows = []
    for _, row in df.iterrows():
        gamma = f"{row['gamma_mean']:.3f} ({row['gamma_sd']:.3f})" if row["method"] == "WJ_selected" else f"{row['gamma_mean']:.1f}"
        rows.append(
            {
                "Method": row["method"],
                "gamma": gamma,
                "All RMSE": fmt_mean_se(row, "rmse_all_mean", "rmse_all_se"),
                "Active RMSE": fmt_mean_se(row, "rmse_active_mean", "rmse_active_se"),
                "Inactive RMSE": fmt_mean_se(row, "rmse_inactive_mean", "rmse_inactive_se"),
            }
        )
    table = pd.DataFrame(rows).to_markdown(index=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
