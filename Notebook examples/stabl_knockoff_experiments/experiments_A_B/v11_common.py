"""Shared utilities for V11 missingness stress testing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from v91_common import paired_wilcoxon_p, percentile_bootstrap_mean_ci


def fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one missingness rate")
    if any(not 0.0 < item < 1.0 for item in values):
        raise ValueError("Every missingness rate must lie in (0, 1)")
    if len(set(values)) != len(values):
        raise ValueError("Missingness rates must be unique")
    return tuple(sorted(values))


def parse_items(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("Expected at least one comma-separated item")
    return items


def rate_label(rate: float) -> str:
    return f"rate_{rate:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def favorable_direction(metric: str) -> str:
    lower = {
        "fdp",
        "false_positives",
        "estimated_fdp",
        "median_signal_rank",
        "mean_null_score",
        "masked_rmse",
        "masked_mae",
        "sample_cov_fro_relative",
        "sample_corr_fro_relative",
        "pair_corr_abs_error_to_oracle",
        "s_relative_abs_error_to_oracle",
        "cov_kk_fro_relative",
        "cov_xk_offdiag_rmse",
    }
    return "lower" if metric in lower else "higher"


def favorable_difference(metric: str, difference: float) -> bool:
    return difference < 0 if favorable_direction(metric) == "lower" else difference > 0


def paired_contrast(
    frame: pd.DataFrame,
    *,
    index_columns: Iterable[str],
    metric: str,
    seed: int,
    n_bootstrap: int,
    reference: str = "median",
    comparator: str = "bayesianridge_mean",
) -> dict[str, Any]:
    required = set(index_columns) | {"completion", metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing paired contrast columns: {sorted(missing)}")
    wide = frame.pivot_table(
        index=list(index_columns),
        columns="completion",
        values=metric,
        aggfunc="mean",
    )
    if reference not in wide.columns or comparator not in wide.columns:
        raise ValueError(f"Missing {reference} or {comparator} for {metric}")
    wide = wide[[reference, comparator]].dropna()
    differences = wide[comparator].to_numpy(dtype=float) - wide[reference].to_numpy(dtype=float)
    ci_low, ci_high = percentile_bootstrap_mean_ci(
        differences,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    direction = favorable_direction(metric)
    favorable = differences < 0 if direction == "lower" else differences > 0
    return {
        "metric": metric,
        "reference": reference,
        "comparator": comparator,
        "favorable_direction": direction,
        "n_pairs": int(len(differences)),
        "reference_mean": float(wide[reference].mean()),
        "comparator_mean": float(wide[comparator].mean()),
        "mean_difference": float(np.mean(differences)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "paired_wilcoxon_p": paired_wilcoxon_p(differences),
        "fraction_pairs_favorable": float(np.mean(favorable)),
    }


def valid_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "error" in frame.columns:
        frame = frame[frame["error"].fillna("").eq("")].copy()
    return frame
