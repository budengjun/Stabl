#!/usr/bin/env python3
"""Validity-aware aggregation rules for multiple-completion STABL.

V6 showed that averaging artificial-feature score paths by artificial-feature
identity can dilute the negative-control burden when each component regenerates
a new knockoff realization.  This module keeps the pooled real score path but
aggregates artificial evidence through component-wise exceedance counts.

The proposed rule is an empirical calibration device, not a theorem-backed
extension of STABL.  It must be evaluated using known-support simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from research_common import SelectionResult

PoolingRule = Literal[
    "mean_path_naive",
    "mean_real_component_artificial_count",
    "component_selection_vote",
]


@dataclass(frozen=True)
class ValidityAwareSelection:
    """Selection plus the full threshold-count diagnostics used to obtain it."""

    result: SelectionResult
    fdp_curve: NDArray[np.float64]
    real_count_curve: NDArray[np.float64]
    artificial_count_curve: NDArray[np.float64]
    artificial_count_sd_curve: NDArray[np.float64]
    naive_artificial_count_curve: NDArray[np.float64]


def parse_pooling_rules(value: str) -> tuple[PoolingRule, ...]:
    allowed = {
        "mean_path_naive",
        "mean_real_component_artificial_count",
        "component_selection_vote",
    }
    aliases = {
        "naive": "mean_path_naive",
        "validity_aware": "mean_real_component_artificial_count",
        "component_vote": "component_selection_vote",
    }
    items: list[str] = []
    for raw in str(value).split(","):
        item = raw.strip().lower()
        if not item:
            continue
        items.append(aliases.get(item, item))
    if not items:
        raise ValueError("at least one pooling rule is required")
    invalid = sorted(set(items) - allowed)
    if invalid:
        raise ValueError(f"unsupported pooling rules: {invalid}")
    # Preserve command-line order while removing duplicates.
    return tuple(dict.fromkeys(items))  # type: ignore[return-value]


def parse_vote_fractions(value: str) -> tuple[float, ...]:
    try:
        values = sorted({float(item.strip()) for item in str(value).split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError("vote fractions must be comma-separated numeric values") from exc
    if not values or any(not 0.0 < item <= 1.0 for item in values):
        raise ValueError("vote fractions must lie in (0, 1]")
    return tuple(values)


def _as_score_paths(paths: Iterable[ArrayLike], pool_size: int) -> list[NDArray[np.float64]]:
    arrays = [np.asarray(item, dtype=float) for item in paths]
    if pool_size < 1 or pool_size > len(arrays):
        raise ValueError("pool_size is incompatible with the available components")
    chosen = arrays[:pool_size]
    shape = chosen[0].shape
    if len(shape) != 2 or any(item.shape != shape for item in chosen):
        raise ValueError("all score paths must share shape (p, n_lambda)")
    if not all(np.isfinite(item).all() for item in chosen):
        raise ValueError("score paths must be finite")
    return chosen


def max_scores(scores: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(scores, dtype=float)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return np.max(array, axis=1)
    raise ValueError("scores must be one or two dimensional")


def component_exceedance_counts(
    score_paths: Iterable[ArrayLike],
    *,
    pool_size: int,
    threshold_grid: ArrayLike,
) -> NDArray[np.float64]:
    """Return a matrix with one row per component and one column per threshold."""
    arrays = _as_score_paths(score_paths, pool_size)
    grid = np.asarray(threshold_grid, dtype=float).reshape(-1)
    if grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("threshold grid must be finite and nonempty")
    maxima = np.stack([max_scores(item) for item in arrays], axis=0)
    return np.stack(
        [np.sum(maxima >= threshold, axis=1).astype(float) for threshold in grid],
        axis=1,
    )


def select_mean_real_component_artificial_count(
    pooled_real_scores: ArrayLike,
    artificial_component_paths: Iterable[ArrayLike],
    *,
    pool_size: int,
    threshold_grid: ArrayLike,
    naive_pooled_artificial_scores: ArrayLike | None = None,
    rule_name: str = "mean_real_component_artificial_count",
) -> ValidityAwareSelection:
    """Apply stabl_min using component-wise artificial exceedance counts.

    For threshold t, the numerator is

        1 + mean_m #{j: max_lambda artificial_score[m,j,lambda] >= t},

    while the denominator uses the pooled real path.  This prevents regenerated
    artificial identities from vanishing merely because each draw highlights a
    different knockoff feature.
    """
    real = max_scores(pooled_real_scores)
    grid = np.asarray(threshold_grid, dtype=float).reshape(-1)
    real_counts = np.asarray([np.sum(real >= threshold) for threshold in grid], dtype=float)
    component_counts = component_exceedance_counts(
        artificial_component_paths,
        pool_size=pool_size,
        threshold_grid=grid,
    )
    artificial_mean = component_counts.mean(axis=0)
    artificial_sd = (
        component_counts.std(axis=0, ddof=1)
        if pool_size > 1
        else np.zeros(grid.size, dtype=float)
    )
    if naive_pooled_artificial_scores is None:
        naive_counts = np.full(grid.size, np.nan, dtype=float)
    else:
        naive = max_scores(naive_pooled_artificial_scores)
        naive_counts = np.asarray([np.sum(naive >= threshold) for threshold in grid], dtype=float)
    curve = (1.0 + artificial_mean) / np.maximum(1.0, real_counts)
    index = int(np.argmin(curve))
    estimated = float(curve[index])
    if estimated > 1.0:
        selected = np.zeros(real.shape[0], dtype=bool)
        threshold = 1.0
    else:
        threshold = float(grid[index])
        selected = real >= threshold
    result = SelectionResult(selected, threshold, estimated, rule_name)
    return ValidityAwareSelection(
        result=result,
        fdp_curve=np.asarray(curve, dtype=float),
        real_count_curve=real_counts,
        artificial_count_curve=np.asarray(artificial_mean, dtype=float),
        artificial_count_sd_curve=np.asarray(artificial_sd, dtype=float),
        naive_artificial_count_curve=np.asarray(naive_counts, dtype=float),
    )


def select_component_vote(
    component_masks: Iterable[ArrayLike],
    *,
    pool_size: int,
    vote_fraction: float,
    rule_name: str | None = None,
) -> SelectionResult:
    """Select features appearing in at least ``vote_fraction`` of component sets.

    This is a sensitivity rule only.  Its ``estimated_fdp`` is deliberately NaN
    because a vote threshold does not inherit the original STABL FDP+ estimate.
    """
    masks = [np.asarray(item, dtype=bool).reshape(-1) for item in component_masks]
    if pool_size < 1 or pool_size > len(masks):
        raise ValueError("pool_size is incompatible with the component masks")
    if not 0.0 < vote_fraction <= 1.0:
        raise ValueError("vote_fraction must lie in (0, 1]")
    chosen = masks[:pool_size]
    p = chosen[0].shape[0]
    if any(item.shape != (p,) for item in chosen):
        raise ValueError("component masks must share the same length")
    vote_rate = np.mean(np.stack(chosen, axis=0), axis=0)
    selected = vote_rate >= vote_fraction
    name = rule_name or f"component_selection_vote_{vote_fraction:g}"
    return SelectionResult(selected, float(vote_fraction), float("nan"), name)


def curve_value_at_selected_threshold(
    selection: ValidityAwareSelection,
) -> dict[str, float]:
    """Return threshold-curve diagnostics at the selected threshold/minimum."""
    index = int(np.argmin(selection.fdp_curve))
    return {
        "real_count_at_threshold": float(selection.real_count_curve[index]),
        "mean_component_artificial_count_at_threshold": float(
            selection.artificial_count_curve[index]
        ),
        "sd_component_artificial_count_at_threshold": float(
            selection.artificial_count_sd_curve[index]
        ),
        "naive_pooled_artificial_count_at_threshold": float(
            selection.naive_artificial_count_curve[index]
        ),
        "artificial_dilution_at_threshold": float(
            selection.artificial_count_curve[index]
            - selection.naive_artificial_count_curve[index]
        ),
    }
