"""Aggregate results across seeds.

Reads outputs/results.jsonl and produces per-cell mean and standard deviation
for every scalar metric, grouped by (phase, scenario, model, train_lang,
test_lang, split). Rows without a seed field are treated as a single-seed run
and included as-is.

Outputs:
    outputs/tables/aggregated_mean.csv  — mean of each metric per group
    outputs/tables/aggregated_std.csv   — std of each metric per group
    outputs/tables/aggregated_summary.csv — mean ± std as formatted strings

Run:  python -m src.eval.aggregate_seeds
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import OUTPUTS_DIR, TABLES_DIR  # noqa: E402


RESULTS_PATH = OUTPUTS_DIR / "results.jsonl"

GROUP_COLS = ["phase", "scenario", "model", "train_lang", "test_lang", "split"]
METRIC_COLS_PREFIX = [
    "accuracy",
    "precision_",
    "recall_",
    "f1_",
    "pr_auc_",
    "n_train",
    "shots",
    "train_time_sec",
]


def _is_metric_col(col: str) -> bool:
    return any(col.startswith(p) or col == p.rstrip("_") for p in METRIC_COLS_PREFIX)


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"No results file at {RESULTS_PATH}")
        return

    rows = [json.loads(line) for line in RESULTS_PATH.open() if line.strip()]
    df = pd.DataFrame(rows)

    if "phase" not in df.columns:
        df["phase"] = "unknown"
    if "seed" not in df.columns:
        df["seed"] = 42
    for col in GROUP_COLS:
        if col not in df.columns:
            df[col] = "n/a"
        df[col] = df[col].fillna("n/a")

    numeric_metric_cols = [
        c for c in df.columns
        if _is_metric_col(c) and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"Loaded {len(df)} result rows.")
    print(f"Grouping by: {GROUP_COLS}")
    print(f"Numeric metric columns to aggregate: {numeric_metric_cols}")

    grouped = df.groupby(GROUP_COLS, dropna=False)
    n_seeds = grouped["seed"].nunique().rename("n_seeds")

    mean_df = grouped[numeric_metric_cols].mean().round(4)
    std_df = grouped[numeric_metric_cols].std(ddof=0).round(4).fillna(0.0)
    mean_df = mean_df.join(n_seeds)
    std_df = std_df.join(n_seeds)

    def _fmt(m, s):
        if pd.isna(m):
            return ""
        return f"{m:.4f} ± {s:.4f}"

    summary = mean_df.copy()
    for col in numeric_metric_cols:
        summary[col] = [
            _fmt(m, s) for m, s in zip(mean_df[col].values, std_df[col].values)
        ]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    mean_path = TABLES_DIR / "aggregated_mean.csv"
    std_path = TABLES_DIR / "aggregated_std.csv"
    summary_path = TABLES_DIR / "aggregated_summary.csv"
    mean_df.reset_index().to_csv(mean_path, index=False)
    std_df.reset_index().to_csv(std_path, index=False)
    summary.reset_index().to_csv(summary_path, index=False)

    print(f"\nWrote:\n  {mean_path}\n  {std_path}\n  {summary_path}")
    print(f"\nGroups: {len(mean_df)}. Multi-seed groups: {(mean_df['n_seeds'] > 1).sum()}.")


if __name__ == "__main__":
    main()
