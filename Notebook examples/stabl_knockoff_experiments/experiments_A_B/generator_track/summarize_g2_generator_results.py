#!/usr/bin/env python3
"""Replicate-level summaries and paired contrasts for G2 and G3."""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from research_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    mean_pairwise_jaccard,
    parse_selected_indices,
)

PRIMARY_METRICS = (
    "fdp",
    "power",
    "support_jaccard",
    "average_precision",
    "ranking_auroc",
    "mean_null_score",
    "n_selected",
    "estimated_fdp",
    "calibration_gap",
)


def bootstrap_mean_ci(values: np.ndarray, *, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def safe_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.allclose(values, 0.0):
        return 1.0
    try:
        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selection-rule", default="stabl_min")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=20260722)
    args = parser.parse_args()

    directory = Path(args.out_dir).expanduser().resolve()
    selection = pd.read_csv(directory / "selection_results.csv")
    valid = selection[
        selection.get("error", pd.Series("", index=selection.index)).fillna("").eq("")
    ].copy()
    valid = valid[valid["selection_rule"] == args.selection_rule].copy()
    if valid.empty:
        raise RuntimeError("No successful rows for the requested selection rule")

    stability_rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "p_setting", "actual_p", "replicate", "generator_label"]
    for key, block in valid.groupby(group_columns, dropna=False):
        sets = [parse_selected_indices(value) for value in block["selected_indices"]]
        stability_rows.append(
            {
                **dict(zip(group_columns, key)),
                "n_draws": len(sets),
                "draw_jaccard": mean_pairwise_jaccard(sets),
            }
        )
    stability = pd.DataFrame(stability_rows)

    metric_columns = [metric for metric in PRIMARY_METRICS if metric in valid.columns]
    metric_columns += [column for column in valid.columns if column.startswith("precision_at_")]
    replicate_columns = [
        "dataset",
        "p_setting",
        "actual_p",
        "replicate",
        "generator_label",
        "generator",
    ]
    replicates = (
        valid.groupby(replicate_columns, dropna=False)[metric_columns]
        .mean()
        .reset_index()
        .merge(stability, on=group_columns, how="left")
    )

    overall_rows: list[dict[str, Any]] = []
    overall_groups = ["dataset", "p_setting", "actual_p", "generator_label", "generator"]
    metric_names = [
        column for column in replicates.columns if column not in set(replicate_columns)
    ]
    for key, block in replicates.groupby(overall_groups, dropna=False):
        row: dict[str, Any] = {
            **dict(zip(overall_groups, key)),
            "n_replicates": int(block["replicate"].nunique()),
        }
        for metric in metric_names:
            values = pd.to_numeric(block[metric], errors="coerce").dropna()
            if len(values):
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)

    contrast_rows: list[dict[str, Any]] = []
    setting_groups = ["dataset", "p_setting", "actual_p"]
    for setting_key, setting_block in replicates.groupby(setting_groups, dropna=False):
        labels = sorted(setting_block["generator_label"].unique())
        for left_label, right_label in combinations(labels, 2):
            left = setting_block[setting_block["generator_label"] == left_label].set_index("replicate")
            right = setting_block[setting_block["generator_label"] == right_label].set_index("replicate")
            common = left.index.intersection(right.index)
            for metric_index, metric in enumerate(metric_names):
                if metric not in left or metric not in right or len(common) == 0:
                    continue
                difference = (
                    pd.to_numeric(right.loc[common, metric], errors="coerce")
                    - pd.to_numeric(left.loc[common, metric], errors="coerce")
                ).to_numpy(dtype=float)
                difference = difference[np.isfinite(difference)]
                if difference.size == 0:
                    continue
                low, high = bootstrap_mean_ci(
                    difference,
                    samples=args.bootstrap_samples,
                    seed=args.random_state + len(contrast_rows) * 97 + metric_index,
                )
                contrast_rows.append(
                    {
                        **dict(zip(setting_groups, setting_key)),
                        "contrast": f"{right_label}_minus_{left_label}",
                        "right_label": right_label,
                        "left_label": left_label,
                        "metric": metric,
                        "n_pairs": int(difference.size),
                        "mean_difference": float(np.mean(difference)),
                        "median_difference": float(np.median(difference)),
                        "ci_low": low,
                        "ci_high": high,
                        "wilcoxon_p": safe_wilcoxon(difference),
                    }
                )

    atomic_write_csv(stability, directory / "generator_draw_stability.csv")
    atomic_write_csv(replicates, directory / "generator_replicate_aggregates.csv")
    atomic_write_csv(overall, directory / "generator_summary_replicate_level.csv")
    atomic_write_csv(pd.DataFrame(contrast_rows), directory / "generator_paired_contrasts.csv")
    atomic_write_json(
        {
            "selection_rule": args.selection_rule,
            "successful_rows": int(len(valid)),
            "replicate_rows": int(len(replicates)),
            "contrast_rows": int(len(contrast_rows)),
        },
        directory / "generator_summary_status.json",
    )
    print(directory)


if __name__ == "__main__":
    main()
