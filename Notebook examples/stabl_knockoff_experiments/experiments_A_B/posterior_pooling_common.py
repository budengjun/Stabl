#!/usr/bin/env python3
"""Shared helpers for Experiment A V6 posterior-pooling benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

PoolingAggregator = Literal["mean", "median"]


def parse_pool_sizes(value: str) -> tuple[int, ...]:
    """Parse a strictly increasing, unique list of positive pool sizes."""
    try:
        values = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError("pool sizes must be comma-separated positive integers") from exc
    if not values or values[0] < 1:
        raise ValueError("at least one positive pool size is required")
    return tuple(values)


def parse_aggregators(value: str) -> tuple[PoolingAggregator, ...]:
    allowed = {"mean", "median"}
    values = tuple(item.strip().lower() for item in str(value).split(",") if item.strip())
    if not values:
        raise ValueError("at least one pooling aggregator is required")
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"unsupported pooling aggregators: {invalid}")
    return values  # type: ignore[return-value]


def pool_score_paths(
    score_paths: Iterable[ArrayLike],
    *,
    pool_size: int,
    aggregator: PoolingAggregator,
) -> NDArray[np.float64]:
    """Pool the first ``pool_size`` aligned STABL score paths elementwise.

    Each component path is expected to have shape ``(p, n_lambda)``.  The V6
    primary estimator is the arithmetic mean, corresponding to

        mean_m Pi_j^(m)(lambda).

    Median pooling is retained as an optional robustness sensitivity analysis.
    """
    arrays = [np.asarray(item, dtype=float) for item in score_paths]
    if pool_size < 1 or pool_size > len(arrays):
        raise ValueError("pool_size is incompatible with the available components")
    chosen = arrays[:pool_size]
    reference_shape = chosen[0].shape
    if len(reference_shape) != 2:
        raise ValueError("component STABL scores must have shape (p, n_lambda)")
    if any(item.shape != reference_shape for item in chosen):
        raise ValueError("all pooled STABL score paths must share the same shape")
    stacked = np.stack(chosen, axis=0)
    if not np.isfinite(stacked).all():
        raise ValueError("component STABL scores must be finite")
    if aggregator == "mean":
        pooled = np.mean(stacked, axis=0)
    elif aggregator == "median":
        pooled = np.median(stacked, axis=0)
    else:
        raise ValueError(f"unsupported pooling aggregator: {aggregator}")
    return np.asarray(pooled, dtype=float)


def max_score_dispersion(
    score_paths: Iterable[ArrayLike], *, pool_size: int
) -> dict[str, float]:
    """Summarize component-to-component variation in maximum STABL scores."""
    arrays = [np.asarray(item, dtype=float) for item in score_paths][:pool_size]
    if not arrays:
        raise ValueError("at least one score path is required")
    maximum = np.stack([np.max(item, axis=1) for item in arrays], axis=0)
    if maximum.shape[0] == 1:
        feature_sd = np.zeros(maximum.shape[1], dtype=float)
    else:
        feature_sd = np.std(maximum, axis=0, ddof=1)
    return {
        "component_max_score_sd_mean": float(np.mean(feature_sd)),
        "component_max_score_sd_median": float(np.median(feature_sd)),
        "component_max_score_sd_max": float(np.max(feature_sd)),
    }


def mean_pairwise_jaccard(masks: Iterable[ArrayLike]) -> float:
    arrays = [np.asarray(item, dtype=bool).reshape(-1) for item in masks]
    if len(arrays) < 2:
        return 1.0
    values: list[float] = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            union = int(np.logical_or(arrays[i], arrays[j]).sum())
            intersection = int(np.logical_and(arrays[i], arrays[j]).sum())
            values.append(1.0 if union == 0 else intersection / union)
    return float(np.mean(values))


def encode_indices(mask: ArrayLike) -> str:
    values = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
    return ";".join(str(int(item)) for item in values)


def decode_indices(value: object, p: int) -> NDArray[np.bool_]:
    mask = np.zeros(int(p), dtype=bool)
    if value is None:
        return mask
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return mask
    indices = [int(item) for item in text.split(";") if item != ""]
    if indices and (min(indices) < 0 or max(indices) >= p):
        raise ValueError("encoded selected indices are outside the feature range")
    mask[np.asarray(indices, dtype=int)] = True
    return mask


def detect_score_directory(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.name == "scores" and candidate.is_dir():
        return candidate
    score_dir = candidate / "scores"
    if score_dir.is_dir():
        return score_dir
    raise FileNotFoundError(f"No scores directory found under {candidate}")


def validate_v5_reuse_config(
    source_out_dir: str | Path,
    target_config: dict[str, object],
) -> Path:
    """Validate that V5 component-0 scores are scientifically compatible.

    The V6 nested pool uses component draw 0 as the exact V5 single-draw
    baseline.  We therefore require all data-generation and STABL-path settings
    that determine those score arrays to match.  The V5 source may contain more
    replicates and more generator/completion conditions than the V6 target.
    """
    source = Path(source_out_dir).expanduser().resolve()
    config_path = source / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing V5 config: {config_path}")
    old = json.loads(config_path.read_text(encoding="utf-8"))
    required_equal = (
        "n",
        "p",
        "n_signal",
        "block_size",
        "rho",
        "task",
        "signal_strength",
        "missing_rate",
        "mar_driver_fraction",
        "n_bootstraps",
        "stabl_grid_size",
        "stabl_c_min",
        "stabl_c_max",
        "stabl_alpha_min",
        "stabl_alpha_max",
        "threshold_min",
        "threshold_max",
        "threshold_step",
        "iterative_max_iter",
        "iterative_nearest_features",
        "random_state",
    )
    mismatches = []
    for key in required_equal:
        if key in target_config and old.get(key) != target_config.get(key):
            mismatches.append((key, old.get(key), target_config.get(key)))
    old_reps = int(old.get("n_replicates", 0))
    target_reps = int(target_config.get("n_replicates", 0))
    if old_reps < target_reps:
        mismatches.append(("n_replicates", old_reps, f">={target_reps}"))
    if mismatches:
        detail = "; ".join(f"{k}: source={a!r}, target={b!r}" for k, a, b in mismatches)
        raise RuntimeError(f"V5 score source is incompatible with V6: {detail}")
    return detect_score_directory(source)
