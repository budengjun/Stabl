#!/usr/bin/env python3
"""Shared utilities for Generator Track G2.2.

G2.2 has two goals:

1. Calibrate exchangeability diagnostics against known-valid and deliberately
   invalid controls.
2. Screen PLSKO configurations using a validity-first, STABL-aware decision
   rule rather than the official knockoff-filter tuning objective.

The module is intentionally Python-only. Official PLSKO generation is reached
through the existing Python ``OfficialPLSKOSampler`` bridge already present in
``experiments_A_B``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class C2STResult:
    auc_mean: float
    auc_sd: float
    auc_min: float
    auc_max: float
    n_repeats: int
    null_auc_mean: float = float("nan")
    null_auc_q95: float = float("nan")
    p_empirical: float = float("nan")


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: tuple[str, ...]


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def atomic_write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=json_default)
    os.replace(temporary, destination)


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    selected = np.zeros(values.size, dtype=bool)
    k = max(0, min(int(k), values.size))
    if k == 0:
        return selected
    order = np.lexsort((np.arange(values.size), -values))
    selected[order[:k]] = True
    return selected


def max_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return np.max(values, axis=1)
    raise ValueError(f"Expected one or two dimensional score array, got {values.shape}")


def selected_set_metrics(selected: np.ndarray, support: np.ndarray) -> dict[str, float]:
    chosen = np.asarray(selected, dtype=bool)
    truth = np.zeros(chosen.size, dtype=bool)
    truth[np.asarray(support, dtype=int)] = True
    tp = int(np.sum(chosen & truth))
    fp = int(np.sum(chosen & ~truth))
    count = int(np.sum(chosen))
    union = int(np.sum(chosen | truth))
    return {
        "n_selected": float(count),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "fdp": float(fp / max(1, count)),
        "power": float(tp / max(1, int(np.sum(truth)))),
        "support_jaccard": float(tp / max(1, union)),
    }


def pairwise_jaccard(selected_sets: Sequence[np.ndarray]) -> float:
    arrays = [np.asarray(item, dtype=bool) for item in selected_sets]
    if len(arrays) < 2:
        return float("nan")
    values: list[float] = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            union = int(np.sum(arrays[i] | arrays[j]))
            intersection = int(np.sum(arrays[i] & arrays[j]))
            values.append(float(intersection / max(1, union)))
    return float(np.mean(values))


def _paired_orientation_data(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("C2ST inputs must be finite two dimensional matrices of equal shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("C2ST inputs must be finite")
    n = a.shape[0]
    features = np.vstack([a, b])
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    groups = np.concatenate([np.arange(n), np.arange(n)])
    return features, labels, groups


def _classifier(kind: str, seed: int, rf_estimators: int):
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=10000,
                random_state=seed,
            ),
        )
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(rf_estimators),
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError("classifier kind must be logistic or extra_trees")


def _grouped_oof_auc(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    classifier_kind: str,
    seed: int,
    n_splits: int,
    rf_estimators: int,
    permute_orientation_labels: bool,
) -> float:
    unique_groups = np.unique(groups)
    if unique_groups.size < 4:
        return float("nan")
    splits = min(int(n_splits), int(unique_groups.size))
    if splits < 2:
        return float("nan")

    labels_for_fit = labels.copy()
    if permute_orientation_labels:
        rng = np.random.default_rng(seed + 918273)
        for group in unique_groups:
            rows = np.flatnonzero(groups == group)
            if rows.size != 2:
                raise ValueError("Paired C2ST expects exactly two rows per group")
            if rng.random() < 0.5:
                labels_for_fit[rows] = labels_for_fit[rows[::-1]]

    predictions = np.full(labels.size, np.nan, dtype=float)
    splitter = KFold(n_splits=splits, shuffle=True, random_state=seed)
    for train_group_idx, test_group_idx in splitter.split(unique_groups):
        train_groups = unique_groups[train_group_idx]
        test_groups = unique_groups[test_group_idx]
        train_rows = np.flatnonzero(np.isin(groups, train_groups))
        test_rows = np.flatnonzero(np.isin(groups, test_groups))
        model = _classifier(classifier_kind, seed, rf_estimators)
        model.fit(features[train_rows], labels_for_fit[train_rows])
        predictions[test_rows] = model.predict_proba(features[test_rows])[:, 1]

    if not np.isfinite(predictions).all():
        raise RuntimeError("C2ST failed to obtain complete out-of-fold predictions")
    auc = float(roc_auc_score(labels_for_fit, predictions))
    return max(auc, 1.0 - auc)


def repeated_paired_c2st(
    left: np.ndarray,
    right: np.ndarray,
    *,
    classifier_kind: str = "logistic",
    repeats: int = 5,
    n_splits: int = 5,
    seed: int = 0,
    null_permutations: int = 0,
    rf_estimators: int = 200,
) -> C2STResult:
    features, labels, groups = _paired_orientation_data(left, right)
    aucs = np.asarray(
        [
            _grouped_oof_auc(
                features,
                labels,
                groups,
                classifier_kind=classifier_kind,
                seed=seed + 1009 * repeat,
                n_splits=n_splits,
                rf_estimators=rf_estimators,
                permute_orientation_labels=False,
            )
            for repeat in range(int(repeats))
        ],
        dtype=float,
    )
    null_values: list[float] = []
    for permutation in range(int(null_permutations)):
        null_values.append(
            _grouped_oof_auc(
                features,
                labels,
                groups,
                classifier_kind=classifier_kind,
                seed=seed + 500003 + 7919 * permutation,
                n_splits=n_splits,
                rf_estimators=rf_estimators,
                permute_orientation_labels=True,
            )
        )
    null_array = np.asarray(null_values, dtype=float)
    observed = float(np.mean(aucs))
    empirical = float("nan")
    if null_array.size:
        empirical = float((1 + np.sum(null_array >= observed)) / (1 + null_array.size))
    return C2STResult(
        auc_mean=observed,
        auc_sd=float(np.std(aucs, ddof=1)) if aucs.size > 1 else 0.0,
        auc_min=float(np.min(aucs)),
        auc_max=float(np.max(aucs)),
        n_repeats=int(aucs.size),
        null_auc_mean=float(np.mean(null_array)) if null_array.size else float("nan"),
        null_auc_q95=float(np.quantile(null_array, 0.95)) if null_array.size else float("nan"),
        p_empirical=empirical,
    )


def swap_pair_matrices(
    X: np.ndarray,
    X_tilde: np.ndarray,
    feature_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    real = np.asarray(X, dtype=float)
    knockoff = np.asarray(X_tilde, dtype=float)
    if real.shape != knockoff.shape:
        raise ValueError("X and X_tilde must have equal shapes")
    p = real.shape[1]
    indices = np.asarray(feature_indices, dtype=int)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= p):
        raise ValueError("Invalid swap feature indices")
    original = np.column_stack([real, knockoff])
    swapped_real = real.copy()
    swapped_knockoff = knockoff.copy()
    swapped_real[:, indices] = knockoff[:, indices]
    swapped_knockoff[:, indices] = real[:, indices]
    swapped = np.column_stack([swapped_real, swapped_knockoff])
    return original, swapped


def exchangeability_diagnostics(
    X: np.ndarray,
    X_tilde: np.ndarray,
    *,
    seed: int,
    repeats: int,
    n_splits: int,
    null_permutations: int,
    classifier_kinds: Sequence[str],
    n_single_features: int,
    block_size: int,
    rf_estimators: int,
) -> list[dict[str, Any]]:
    real = np.asarray(X, dtype=float)
    knockoff = np.asarray(X_tilde, dtype=float)
    n, p = real.shape
    rng = np.random.default_rng(seed + 331)
    single_features = np.sort(
        rng.choice(p, size=min(int(n_single_features), p), replace=False)
    )
    order = rng.permutation(p)
    blocks = [order[start : start + int(block_size)] for start in range(0, p, int(block_size))]

    comparisons: list[tuple[str, str, np.ndarray, np.ndarray, str]] = []
    comparisons.append(("marginal", "all", real, knockoff, "marginal"))
    global_left, global_right = swap_pair_matrices(real, knockoff, np.arange(p))
    comparisons.append(("global_swap", "all", global_left, global_right, "swap"))
    for feature in single_features:
        left, right = swap_pair_matrices(real, knockoff, [int(feature)])
        comparisons.append(("single_feature_swap", str(int(feature)), left, right, "swap"))
    for block_index, block in enumerate(blocks):
        left, right = swap_pair_matrices(real, knockoff, block)
        comparisons.append(
            (
                "block_swap",
                f"block_{block_index:03d}",
                left,
                right,
                "swap",
            )
        )

    rows: list[dict[str, Any]] = []
    for classifier_index, classifier_kind in enumerate(classifier_kinds):
        for comparison_index, (test, unit, left, right, family) in enumerate(comparisons):
            result = repeated_paired_c2st(
                left,
                right,
                classifier_kind=classifier_kind,
                repeats=repeats,
                n_splits=n_splits,
                seed=seed + 100000 * classifier_index + 101 * comparison_index,
                null_permutations=null_permutations,
                rf_estimators=rf_estimators,
            )
            rows.append(
                {
                    "test": test,
                    "test_family": family,
                    "unit": unit,
                    "classifier": classifier_kind,
                    **asdict(result),
                    "n": n,
                    "p": p,
                }
            )
    return rows


def invalid_knockoff(
    X: np.ndarray,
    *,
    kind: str,
    seed: int,
) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    n, p = values.shape
    if kind == "columnwise_permutation":
        result = np.empty_like(values)
        for j in range(p):
            result[:, j] = values[rng.permutation(n), j]
        return result
    if kind == "mean_shift":
        scale = np.std(values, axis=0, ddof=1)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return values + 0.35 * scale + rng.normal(scale=0.05 * scale, size=values.shape)
    if kind == "copy_jitter":
        scale = np.std(values, axis=0, ddof=1)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return values + rng.normal(scale=0.03 * scale, size=values.shape)
    raise ValueError(f"Unknown invalid control kind: {kind}")


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = 10000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(int(samples), array.size), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def paired_summary(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    metric: str,
    key_columns: Sequence[str],
    generator_column: str = "generator_label",
    seed: int = 0,
    bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    subset = frame[frame[generator_column].isin([candidate, baseline])].copy()
    pivot = subset.pivot_table(
        index=list(key_columns),
        columns=generator_column,
        values=metric,
        aggfunc="mean",
    ).dropna()
    if candidate not in pivot or baseline not in pivot:
        return {
            "candidate": candidate,
            "baseline": baseline,
            "metric": metric,
            "n_pairs": 0,
            "delta_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "wilcoxon_p": float("nan"),
        }
    delta = pivot[candidate].to_numpy(dtype=float) - pivot[baseline].to_numpy(dtype=float)
    low, high = bootstrap_mean_ci(delta, seed=seed, samples=bootstrap_samples)
    p_value = float("nan")
    if delta.size and not np.allclose(delta, 0.0):
        try:
            p_value = float(wilcoxon(delta, zero_method="pratt", alternative="two-sided").pvalue)
        except ValueError:
            p_value = float("nan")
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "n_pairs": int(delta.size),
        "candidate_mean": float(pivot[candidate].mean()),
        "baseline_mean": float(pivot[baseline].mean()),
        "delta_mean": float(np.mean(delta)) if delta.size else float("nan"),
        "ci_low": low,
        "ci_high": high,
        "wilcoxon_p": p_value,
    }


def derive_c2st_gate(
    summary: pd.DataFrame,
    *,
    synthetic_oracle_label: str,
    ssi_equicorr_label: str,
    ssi_mvr_label: str,
    absolute_cap: float = 0.80,
    margin: float = 0.10,
) -> dict[str, float]:
    """Derive conservative, pre-specified gates from calibration controls.

    The global-swap gate is the smaller of ``absolute_cap`` and the largest
    calibrated valid-control upper benchmark plus ``margin``. The same rule is
    applied to marginal C2ST. The result is not estimated from any PLSKO row.
    """
    valid_labels = {synthetic_oracle_label, ssi_equicorr_label, ssi_mvr_label}
    valid = summary[summary["generator_label"].isin(valid_labels)].copy()
    if valid.empty:
        raise ValueError("Calibration summary does not contain required valid controls")

    def gate_for(test: str) -> float:
        block = valid[valid["test"].eq(test)]
        if block.empty:
            raise ValueError(f"Calibration summary has no {test} rows")
        reference = float(block["auc_mean_mean"].max())
        q90 = float(block.get("auc_mean_q90", block["auc_mean_mean"]).max())
        return float(min(absolute_cap, max(reference, q90) + margin))

    return {
        "marginal_auc_max": gate_for("marginal"),
        "global_swap_auc_max": gate_for("global_swap"),
        "absolute_cap": float(absolute_cap),
        "calibration_margin": float(margin),
    }


def validity_gate_decision(
    row: pd.Series | dict[str, Any],
    gate: dict[str, float],
) -> GateDecision:
    values = dict(row)
    reasons: list[str] = []
    if float(values.get("marginal_auc_mean", math.inf)) > float(gate["marginal_auc_max"]):
        reasons.append("marginal_c2st")
    if float(values.get("global_swap_auc_mean", math.inf)) > float(gate["global_swap_auc_max"]):
        reasons.append("global_swap_c2st")
    max_slack = float(gate.get("draw_max_slack", 0.05))
    if "marginal_auc_max" in values and float(values["marginal_auc_max"]) > float(gate["marginal_auc_max"]) + max_slack:
        reasons.append("marginal_c2st_draw_max")
    if "global_swap_auc_max" in values and float(values["global_swap_auc_max"]) > float(gate["global_swap_auc_max"]) + max_slack:
        reasons.append("global_swap_c2st_draw_max")
    if int(values.get("near_constant_knockoff_columns_max", 0)) > 0:
        reasons.append("near_constant_knockoff_columns")
    return GateDecision(passed=not reasons, reasons=tuple(reasons))


def downstream_gate_decision(
    metrics: dict[str, float],
    *,
    max_fdp_delta: float,
    min_average_precision_delta: float,
    min_matched_power_delta: float,
    min_draw_jaccard_delta: float,
) -> GateDecision:
    reasons: list[str] = []
    if metrics.get("fdp_delta", math.inf) > max_fdp_delta:
        reasons.append("fdp_delta")
    if metrics.get("average_precision_delta", -math.inf) < min_average_precision_delta:
        reasons.append("average_precision_delta")
    if metrics.get("matched_power_delta", -math.inf) < min_matched_power_delta:
        reasons.append("matched_power_delta")
    if metrics.get("draw_jaccard_delta", -math.inf) < min_draw_jaccard_delta:
        reasons.append("draw_jaccard_delta")
    return GateDecision(passed=not reasons, reasons=tuple(reasons))
