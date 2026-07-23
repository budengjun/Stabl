#!/usr/bin/env python3
"""G2.1 saved-score threshold, selection-size, and exchangeability diagnosis.

This experiment consumes a completed Generator Track G2 directory. It does not
rerun STABL or regenerate knockoffs. The primary analysis reconstructs every
STABL threshold path from saved real and artificial selection-frequency arrays.
It then separates raw generator differences into two complementary components:

1. Exact matched-size ranking component, using the candidate top-k real scores
   where k is the baseline stabl_min selected count.
2. Threshold-choice component, evaluating the candidate at the baseline
   stabl_min threshold before allowing the candidate to use its own argmin.

An optional expanded C2ST audit reuses the saved knockoff pair cache and
reconstructs the standardized real X from config.json. It therefore adds no
new knockoff or STABL fitting.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wilcoxon
from sklearn.metrics import roc_auc_score

from research_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    cap_features_by_variance,
    c2st_diagnostics,
    fdp_plus_curve,
    load_real_dataset,
    select_stabl_min,
    standardize_completed_matrix,
)

SCORE_PATTERN = re.compile(
    r"p_(?P<p>\d+)__rep_(?P<rep>\d+)__(?P<generator>.+)__draw_(?P<draw>\d+)\.npz$"
)

METRIC_NAMES = (
    "n_selected",
    "true_positives",
    "false_positives",
    "realized_fdp",
    "power",
    "support_jaccard",
)


@dataclass(frozen=True)
class ScoreRun:
    p: int
    replicate: int
    generator: str
    draw: int
    real_scores: np.ndarray
    artificial_scores: np.ndarray
    real_max: np.ndarray
    artificial_max: np.ndarray
    support: np.ndarray
    feature_names: np.ndarray
    stabl_threshold: float
    stabl_estimated_fdp: float
    stabl_selected: np.ndarray
    stabl_metrics: dict[str, float]
    source_path: Path

    @property
    def key(self) -> tuple[int, int, int]:
        return self.p, self.replicate, self.draw


@dataclass(frozen=True)
class ContrastSpec:
    candidate: str
    baseline: str

    @property
    def label(self) -> str:
        return f"{self.candidate}_minus_{self.baseline}"


def max_scores(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return np.max(array, axis=1)
    raise ValueError(f"Unsupported score shape {array.shape}")


def parse_comparisons(text: str | None, generators: Sequence[str]) -> list[ContrastSpec]:
    if text:
        specs: list[ContrastSpec] = []
        for raw in text.split(","):
            item = raw.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    "Every comparison must use candidate:baseline syntax, for example "
                    "plsko_tuned_ssi:equicorr"
                )
            candidate, baseline = [piece.strip() for piece in item.split(":", 1)]
            if candidate not in generators or baseline not in generators:
                raise ValueError(
                    f"Unknown comparison {item}. Available generators are {sorted(generators)}"
                )
            if candidate == baseline:
                raise ValueError(f"Candidate and baseline are identical in {item}")
            specs.append(ContrastSpec(candidate=candidate, baseline=baseline))
        if not specs:
            raise ValueError("No comparisons were parsed")
        return specs

    preferred = [
        ContrastSpec("plsko_tuned_ssi", "equicorr"),
        ContrastSpec("plsko_tuned_ssi", "mvr"),
        ContrastSpec("plsko_default_diagnostic", "equicorr"),
        ContrastSpec("plsko_tuned_ssi", "plsko_default_diagnostic"),
    ]
    available = [
        item
        for item in preferred
        if item.candidate in generators and item.baseline in generators
    ]
    if available:
        return available
    return [ContrastSpec(right, left) for left, right in combinations(sorted(generators), 2)]


def selection_metrics(selected: np.ndarray, support: np.ndarray) -> dict[str, float]:
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
        "realized_fdp": float(fp / max(1, count)),
        "power": float(tp / max(1, int(np.sum(truth)))),
        "support_jaccard": float(tp / max(1, union)),
    }


def stable_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    chosen = np.zeros(values.size, dtype=bool)
    k = max(0, min(int(k), values.size))
    if k == 0:
        return chosen
    order = np.lexsort((np.arange(values.size), -values))
    chosen[order[:k]] = True
    return chosen


def select_at_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=float) >= float(threshold)


def finite_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    valid = np.isfinite(values)
    if np.unique(labels[valid]).size < 2:
        return float("nan")
    auc = float(roc_auc_score(labels[valid], values[valid]))
    return max(auc, 1.0 - auc)


def score_summary(run: ScoreRun) -> dict[str, Any]:
    truth = np.zeros(run.p, dtype=bool)
    truth[run.support] = True
    signal = run.real_max[truth]
    null = run.real_max[~truth]
    artificial = run.artificial_max
    threshold = run.stabl_threshold

    def summarize(prefix: str, values: np.ndarray) -> dict[str, float]:
        clean = np.asarray(values, dtype=float)
        clean = clean[np.isfinite(clean)]
        if clean.size == 0:
            return {
                f"{prefix}_mean": float("nan"),
                f"{prefix}_median": float("nan"),
                f"{prefix}_q75": float("nan"),
                f"{prefix}_q90": float("nan"),
                f"{prefix}_q95": float("nan"),
                f"{prefix}_max": float("nan"),
            }
        return {
            f"{prefix}_mean": float(np.mean(clean)),
            f"{prefix}_median": float(np.median(clean)),
            f"{prefix}_q75": float(np.quantile(clean, 0.75)),
            f"{prefix}_q90": float(np.quantile(clean, 0.90)),
            f"{prefix}_q95": float(np.quantile(clean, 0.95)),
            f"{prefix}_max": float(np.max(clean)),
        }

    return {
        "actual_p": run.p,
        "replicate": run.replicate,
        "generator_label": run.generator,
        "draw": run.draw,
        "stabl_threshold": threshold,
        **summarize("signal_score", signal),
        **summarize("null_score", null),
        **summarize("artificial_score", artificial),
        "signal_minus_null_mean": float(np.mean(signal) - np.mean(null)),
        "artificial_minus_null_mean": float(np.mean(artificial) - np.mean(null)),
        "signal_null_score_auc": finite_auc(
            np.concatenate([np.ones(signal.size), np.zeros(null.size)]),
            np.concatenate([signal, null]),
        ),
        "artificial_null_score_auc": finite_auc(
            np.concatenate([np.ones(artificial.size), np.zeros(null.size)]),
            np.concatenate([artificial, null]),
        ),
        "signal_null_ks_stat": float(ks_2samp(signal, null).statistic),
        "artificial_null_ks_stat": float(ks_2samp(artificial, null).statistic),
        "signal_exceedance_at_stabl": float(np.mean(signal >= threshold)),
        "null_exceedance_at_stabl": float(np.mean(null >= threshold)),
        "artificial_exceedance_at_stabl": float(np.mean(artificial >= threshold)),
        "n_artificial_selected_at_stabl": int(np.sum(artificial >= threshold)),
        "null_to_artificial_selected_ratio": float(
            np.sum(null >= threshold) / max(1, int(np.sum(artificial >= threshold)))
        ),
    }


def load_score_runs(g2_dir: Path, thresholds: np.ndarray) -> list[ScoreRun]:
    score_dir = g2_dir / "scores"
    if not score_dir.is_dir():
        raise FileNotFoundError(f"Saved score directory is missing: {score_dir}")
    files = sorted(score_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No score files found in {score_dir}")

    runs: list[ScoreRun] = []
    for path in files:
        match = SCORE_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unrecognized G2 score filename: {path.name}")
        meta = match.groupdict()
        with np.load(path, allow_pickle=True) as payload:
            real_scores = np.asarray(payload["real_scores"], dtype=float)
            artificial_scores = np.asarray(payload["knockoff_scores"], dtype=float)
            support = np.asarray(payload["support"], dtype=int)
            feature_names = np.asarray(payload["feature_names"], dtype=str)
        if real_scores.shape != artificial_scores.shape:
            raise ValueError(f"Real and artificial score shapes differ in {path}")
        if real_scores.shape[0] != feature_names.size:
            raise ValueError(f"Feature names do not match score rows in {path}")
        selected = select_stabl_min(real_scores, artificial_scores, thresholds)
        chosen = np.asarray(selected.selected, dtype=bool)
        runs.append(
            ScoreRun(
                p=int(meta["p"]),
                replicate=int(meta["rep"]),
                generator=meta["generator"],
                draw=int(meta["draw"]),
                real_scores=real_scores,
                artificial_scores=artificial_scores,
                real_max=max_scores(real_scores),
                artificial_max=max_scores(artificial_scores),
                support=support,
                feature_names=feature_names,
                stabl_threshold=float(selected.threshold),
                stabl_estimated_fdp=float(selected.estimated_fdp),
                stabl_selected=chosen,
                stabl_metrics=selection_metrics(chosen, support),
                source_path=path,
            )
        )
    return runs


def path_rows_for_run(run: ScoreRun, thresholds: np.ndarray) -> list[dict[str, Any]]:
    estimated = fdp_plus_curve(run.real_scores, run.artificial_scores, thresholds)
    truth = np.zeros(run.p, dtype=bool)
    truth[run.support] = True
    selected_matrix = run.real_max[:, None] >= thresholds[None, :]
    artificial_matrix = run.artificial_max[:, None] >= thresholds[None, :]
    n_selected = selected_matrix.sum(axis=0)
    tp = selected_matrix[truth].sum(axis=0)
    fp = selected_matrix[~truth].sum(axis=0)
    n_artificial = artificial_matrix.sum(axis=0)
    realized = fp / np.maximum(1, n_selected)
    power = tp / max(1, int(np.sum(truth)))
    union = n_selected + int(np.sum(truth)) - tp
    jaccard = tp / np.maximum(1, union)

    rows: list[dict[str, Any]] = []
    for index, threshold in enumerate(thresholds):
        rows.append(
            {
                "actual_p": run.p,
                "replicate": run.replicate,
                "generator_label": run.generator,
                "draw": run.draw,
                "threshold_index": index,
                "threshold": float(threshold),
                "is_stabl_min_threshold": int(np.isclose(threshold, run.stabl_threshold)),
                "estimated_fdp_plus": float(estimated[index]),
                "realized_fdp": float(realized[index]),
                "calibration_gap": float(realized[index] - estimated[index]),
                "n_selected": int(n_selected[index]),
                "true_positives": int(tp[index]),
                "false_positives": int(fp[index]),
                "n_artificial_selected": int(n_artificial[index]),
                "power": float(power[index]),
                "support_jaccard": float(jaccard[index]),
                "signal_exceedance_rate": float(tp[index] / max(1, int(np.sum(truth)))),
                "null_exceedance_rate": float(fp[index] / max(1, int(np.sum(~truth)))),
                "artificial_exceedance_rate": float(n_artificial[index] / max(1, run.p)),
                "null_minus_artificial_selected": int(fp[index] - n_artificial[index]),
            }
        )
    return rows


def metric_prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": float(metrics[name]) for name in METRIC_NAMES}


def matched_size_rows(
    runs: Sequence[ScoreRun],
    comparisons: Sequence[ContrastSpec],
    thresholds: np.ndarray,
) -> pd.DataFrame:
    lookup = {(run.p, run.replicate, run.draw, run.generator): run for run in runs}
    rows: list[dict[str, Any]] = []

    for spec in comparisons:
        baseline_runs = [run for run in runs if run.generator == spec.baseline]
        for baseline in baseline_runs:
            candidate = lookup.get((baseline.p, baseline.replicate, baseline.draw, spec.candidate))
            if candidate is None:
                continue
            if not np.array_equal(np.sort(candidate.support), np.sort(baseline.support)):
                raise ValueError(
                    f"Support mismatch for {spec.label}, p={baseline.p}, "
                    f"rep={baseline.replicate}, draw={baseline.draw}"
                )
            if not np.array_equal(candidate.feature_names, baseline.feature_names):
                raise ValueError(
                    f"Feature-name mismatch for {spec.label}, p={baseline.p}, "
                    f"rep={baseline.replicate}, draw={baseline.draw}"
                )

            baseline_raw = baseline.stabl_metrics
            candidate_raw = candidate.stabl_metrics
            target_k = int(baseline_raw["n_selected"])

            candidate_topk_selected = stable_top_k(candidate.real_max, target_k)
            candidate_topk = selection_metrics(candidate_topk_selected, candidate.support)

            candidate_at_baseline_threshold = selection_metrics(
                select_at_threshold(candidate.real_max, baseline.stabl_threshold),
                candidate.support,
            )
            baseline_at_candidate_threshold = selection_metrics(
                select_at_threshold(baseline.real_max, candidate.stabl_threshold),
                baseline.support,
            )

            candidate_counts = np.asarray(
                [np.sum(candidate.real_max >= threshold) for threshold in thresholds],
                dtype=int,
            )
            distances = np.abs(candidate_counts - target_k)
            threshold_distances = np.abs(thresholds - baseline.stabl_threshold)
            order = np.lexsort((-thresholds, threshold_distances, distances))
            nearest_index = int(order[0])
            nearest_threshold = float(thresholds[nearest_index])
            candidate_nearest = selection_metrics(
                select_at_threshold(candidate.real_max, nearest_threshold),
                candidate.support,
            )

            row: dict[str, Any] = {
                "comparison": spec.label,
                "candidate": spec.candidate,
                "baseline": spec.baseline,
                "actual_p": baseline.p,
                "replicate": baseline.replicate,
                "draw": baseline.draw,
                "baseline_stabl_threshold": baseline.stabl_threshold,
                "candidate_stabl_threshold": candidate.stabl_threshold,
                "threshold_delta_candidate_minus_baseline": (
                    candidate.stabl_threshold - baseline.stabl_threshold
                ),
                "baseline_stabl_estimated_fdp": baseline.stabl_estimated_fdp,
                "candidate_stabl_estimated_fdp": candidate.stabl_estimated_fdp,
                "target_k": target_k,
                "candidate_nearest_k_threshold": nearest_threshold,
                "candidate_nearest_k_size_difference": (
                    candidate_nearest["n_selected"] - target_k
                ),
                **metric_prefixed("baseline_raw", baseline_raw),
                **metric_prefixed("candidate_raw", candidate_raw),
                **metric_prefixed("candidate_topk", candidate_topk),
                **metric_prefixed(
                    "candidate_at_baseline_threshold", candidate_at_baseline_threshold
                ),
                **metric_prefixed(
                    "baseline_at_candidate_threshold", baseline_at_candidate_threshold
                ),
                **metric_prefixed("candidate_nearest_k", candidate_nearest),
            }
            for metric in METRIC_NAMES:
                row[f"raw_delta_{metric}"] = candidate_raw[metric] - baseline_raw[metric]
                row[f"matched_k_ranking_component_{metric}"] = (
                    candidate_topk[metric] - baseline_raw[metric]
                )
                row[f"selection_size_component_{metric}"] = (
                    candidate_raw[metric] - candidate_topk[metric]
                )
                row[f"common_threshold_score_component_{metric}"] = (
                    candidate_at_baseline_threshold[metric] - baseline_raw[metric]
                )
                row[f"threshold_choice_component_{metric}"] = (
                    candidate_raw[metric] - candidate_at_baseline_threshold[metric]
                )
                row[f"reverse_threshold_transfer_{metric}"] = (
                    baseline_at_candidate_threshold[metric] - candidate_raw[metric]
                )
                decomposition_error = (
                    row[f"matched_k_ranking_component_{metric}"]
                    + row[f"selection_size_component_{metric}"]
                    - row[f"raw_delta_{metric}"]
                )
                row[f"matched_k_decomposition_error_{metric}"] = float(decomposition_error)
                threshold_error = (
                    row[f"common_threshold_score_component_{metric}"]
                    + row[f"threshold_choice_component_{metric}"]
                    - row[f"raw_delta_{metric}"]
                )
                row[f"threshold_decomposition_error_{metric}"] = float(threshold_error)
            rows.append(row)

    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, clean.size, size=(samples, clean.size))
    means = clean[index].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def safe_wilcoxon(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2 or np.allclose(clean, 0.0):
        return 1.0
    try:
        return float(
            wilcoxon(
                clean,
                zero_method="pratt",
                correction=False,
                alternative="two-sided",
                method="approx",
            ).pvalue
        )
    except (ValueError, TypeError):
        return 1.0


def aggregate_matched_size(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    numeric_columns = [
        column
        for column in frame.columns
        if column not in {"comparison", "candidate", "baseline", "actual_p", "replicate", "draw"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    replicate = (
        frame.groupby(
            ["comparison", "candidate", "baseline", "actual_p", "replicate"],
            as_index=False,
        )[numeric_columns]
        .mean()
    )

    diagnostic_prefixes = (
        "raw_delta_",
        "matched_k_ranking_component_",
        "selection_size_component_",
        "common_threshold_score_component_",
        "threshold_choice_component_",
        "threshold_delta_",
        "candidate_nearest_k_size_difference",
    )
    metrics = [
        column
        for column in replicate.columns
        if column.startswith(diagnostic_prefixes)
    ]
    rows: list[dict[str, Any]] = []
    counter = 0
    for group_key, group in replicate.groupby(
        ["comparison", "candidate", "baseline", "actual_p"], sort=True
    ):
        meta = dict(zip(["comparison", "candidate", "baseline", "actual_p"], group_key))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            low, high = bootstrap_mean_ci(
                values,
                samples=bootstrap_samples,
                seed=seed + counter * 7919,
            )
            counter += 1
            rows.append(
                {
                    **meta,
                    "metric": metric,
                    "n_replicates": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "paired_wilcoxon_p": safe_wilcoxon(values),
                }
            )
    return replicate, pd.DataFrame(rows)


def summarize_threshold_paths(path_runs: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "estimated_fdp_plus",
        "realized_fdp",
        "calibration_gap",
        "n_selected",
        "true_positives",
        "false_positives",
        "n_artificial_selected",
        "power",
        "support_jaccard",
        "signal_exceedance_rate",
        "null_exceedance_rate",
        "artificial_exceedance_rate",
        "null_minus_artificial_selected",
    ]
    grouped = path_runs.groupby(
        ["actual_p", "generator_label", "threshold_index", "threshold"],
        sort=True,
    )[numeric]
    mean = grouped.mean().add_suffix("_mean")
    sd = grouped.std(ddof=1).add_suffix("_sd")
    return mean.join(sd).reset_index()


def summarize_scores(score_runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["actual_p", "generator_label"]
    numeric = [column for column in score_runs.columns if column not in keys + ["replicate", "draw"]]
    grouped = score_runs.groupby(keys, sort=True)[numeric]
    return grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_sd")).reset_index()


def reconstruct_real_matrices(
    config: dict[str, Any],
    runs: Sequence[ScoreRun],
) -> dict[int, pd.DataFrame]:
    dataset = load_real_dataset(
        dataset=config["dataset"],
        data_path=config["data_path"],
        omic=config.get("omic", "CyTOF"),
        x_csv=config.get("x_csv"),
        groups_csv=config.get("groups_csv"),
        subject_mode=config.get("subject_mode", "first"),
        complete_strategy=config.get("complete_strategy", "drop-columns"),
        seed=int(config.get("random_state", 0)),
    )
    result: dict[int, pd.DataFrame] = {}
    for p in sorted({run.p for run in runs}):
        reference = next(run for run in runs if run.p == p)
        subset = cap_features_by_variance(dataset.X, p)
        scaled, _ = standardize_completed_matrix(subset)
        missing = [name for name in reference.feature_names if name not in scaled.columns]
        if missing:
            raise ValueError(
                f"Could not reconstruct {len(missing)} G2 feature columns for p={p}. "
                f"Examples: {missing[:5]}"
            )
        scaled = scaled.loc[:, reference.feature_names.tolist()]
        if scaled.shape != (config["dataset_metadata"]["n_rows_final"], p):
            raise ValueError(
                f"Reconstructed X shape {scaled.shape} does not match expected "
                f"({config['dataset_metadata']['n_rows_final']}, {p})"
            )
        result[p] = scaled
    return result


def expanded_c2st_rows(
    g2_dir: Path,
    runs: Sequence[ScoreRun],
    config: dict[str, Any],
    *,
    seed: int,
) -> pd.DataFrame:
    real_by_p = reconstruct_real_matrices(config, runs)
    pair_dir = g2_dir / "pair_cache"
    if not pair_dir.is_dir():
        raise FileNotFoundError(f"Pair cache directory is missing: {pair_dir}")

    rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        pair_path = pair_dir / (
            f"p_{run.p:04d}__rep_{run.replicate:04d}__"
            f"{run.generator}__draw_{run.draw:03d}.npz"
        )
        if not pair_path.exists():
            raise FileNotFoundError(pair_path)
        with np.load(pair_path, allow_pickle=True) as payload:
            x_tilde = np.asarray(payload["X_tilde"], dtype=float)
            names = np.asarray(payload["feature_names"], dtype=str)
            metadata_json = str(payload["metadata_json"].item())
        if not np.array_equal(names, run.feature_names):
            raise ValueError(f"Pair-cache feature mismatch in {pair_path}")
        x = real_by_p[run.p].to_numpy(dtype=float)
        if x.shape != x_tilde.shape:
            raise ValueError(f"Real and knockoff shapes differ in {pair_path}")
        diagnostics = c2st_diagnostics(x, x_tilde, seed=seed + index * 17)
        rows.append(
            {
                "actual_p": run.p,
                "replicate": run.replicate,
                "generator_label": run.generator,
                "draw": run.draw,
                "pair_cache_path": str(pair_path),
                "metadata_json": metadata_json,
                **diagnostics,
            }
        )
    return pd.DataFrame(rows)



def validate_reconstruction(
    g2_dir: Path,
    run_metrics: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    selection_path = g2_dir / "selection_results.csv"
    if not selection_path.exists():
        return pd.DataFrame()
    original = pd.read_csv(selection_path)
    if "selection_rule" in original.columns:
        original = original[original["selection_rule"] == "stabl_min"].copy()
    if "error" in original.columns:
        original = original[original["error"].fillna("").eq("")].copy()

    keys = ["actual_p", "replicate", "generator_label", "draw"]
    expected = run_metrics.copy()
    keep = keys + [
        "stabl_threshold",
        "stabl_estimated_fdp",
        "stabl_n_selected",
        "stabl_true_positives",
        "stabl_false_positives",
        "stabl_realized_fdp",
        "stabl_power",
        "stabl_support_jaccard",
    ]
    expected = expected[keep]
    rename = {
        "threshold": "original_threshold",
        "estimated_fdp": "original_estimated_fdp",
        "n_selected": "original_n_selected",
        "true_positives": "original_true_positives",
        "false_positives": "original_false_positives",
        "fdp": "original_realized_fdp",
        "power": "original_power",
        "support_jaccard": "original_support_jaccard",
    }
    original = original[keys + list(rename)].rename(columns=rename)
    merged = expected.merge(original, on=keys, how="outer", indicator=True)
    comparisons = (
        ("threshold", "stabl_threshold", "original_threshold"),
        ("estimated_fdp", "stabl_estimated_fdp", "original_estimated_fdp"),
        ("n_selected", "stabl_n_selected", "original_n_selected"),
        ("true_positives", "stabl_true_positives", "original_true_positives"),
        ("false_positives", "stabl_false_positives", "original_false_positives"),
        ("realized_fdp", "stabl_realized_fdp", "original_realized_fdp"),
        ("power", "stabl_power", "original_power"),
        ("support_jaccard", "stabl_support_jaccard", "original_support_jaccard"),
    )
    pass_columns: list[str] = []
    for label, reconstructed, stored in comparisons:
        difference = pd.to_numeric(merged[reconstructed], errors="coerce") - pd.to_numeric(
            merged[stored], errors="coerce"
        )
        merged[f"difference_{label}"] = difference
        pass_column = f"pass_{label}"
        merged[pass_column] = difference.abs().le(tolerance)
        pass_columns.append(pass_column)
    merged["all_metrics_pass"] = (
        merged["_merge"].eq("both") & merged[pass_columns].all(axis=1)
    )
    return merged

def make_figures(
    path_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    for metric, ylabel, filename in (
        ("n_selected_mean", "Mean selected real features", "threshold_selected_count.png"),
        ("power_mean", "Mean power", "threshold_power.png"),
        ("realized_fdp_mean", "Mean realized FDP", "threshold_realized_fdp.png"),
        ("estimated_fdp_plus_mean", "Mean estimated FDP plus", "threshold_estimated_fdp_plus.png"),
        (
            "null_minus_artificial_selected_mean",
            "Mean selected real null minus artificial count",
            "threshold_null_minus_artificial.png",
        ),
    ):
        plt.figure(figsize=(9, 6))
        for label, block in path_summary.groupby("generator_label", sort=True):
            block = block.sort_values("threshold")
            plt.plot(block["threshold"], block[metric], label=label)
        plt.xlabel("Selection frequency threshold")
        plt.ylabel(ylabel)
        plt.title(ylabel + " across the reconstructed STABL path")
        plt.legend()
        plt.tight_layout()
        path = figure_dir / filename
        plt.savefig(path, dpi=180)
        plt.close()
        created.append(str(path))

    if not score_summary.empty:
        columns = [
            "signal_score_mean_mean",
            "null_score_mean_mean",
            "artificial_score_mean_mean",
        ]
        available = [column for column in columns if column in score_summary.columns]
        if available:
            plot = score_summary.set_index("generator_label")[available]
            plot.columns = [
                column.replace("_score_mean_mean", "") for column in plot.columns
            ]
            plt.figure(figsize=(9, 6))
            plot.plot(kind="bar", ax=plt.gca())
            plt.xlabel("Generator")
            plt.ylabel("Mean maximum selection frequency")
            plt.title("Signal, real null, and artificial score levels")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            path = figure_dir / "score_group_means.png"
            plt.savefig(path, dpi=180)
            plt.close()
            created.append(str(path))

    if not contrast_summary.empty:
        wanted = contrast_summary[
            contrast_summary["metric"].isin(
                [
                    "raw_delta_power",
                    "matched_k_ranking_component_power",
                    "selection_size_component_power",
                ]
            )
        ].copy()
        if not wanted.empty:
            pivot = wanted.pivot_table(
                index="comparison", columns="metric", values="mean", aggfunc="first"
            )
            plt.figure(figsize=(10, 6))
            pivot.plot(kind="bar", ax=plt.gca())
            plt.axhline(0.0, linewidth=1)
            plt.xlabel("Comparison")
            plt.ylabel("Mean power difference")
            plt.title("Raw power difference and matched-size decomposition")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            path = figure_dir / "power_decomposition.png"
            plt.savefig(path, dpi=180)
            plt.close()
            created.append(str(path))

    return created


def extract_contrast_metric(
    summary: pd.DataFrame,
    comparison: str,
    metric: str,
) -> dict[str, float] | None:
    selected = summary[
        (summary["comparison"] == comparison) & (summary["metric"] == metric)
    ]
    if selected.empty:
        return None
    row = selected.iloc[0]
    return {
        "mean": float(row["mean"]),
        "ci_low": float(row["bootstrap_ci_low"]),
        "ci_high": float(row["bootstrap_ci_high"]),
        "wilcoxon_p": float(row["paired_wilcoxon_p"]),
    }


def classify_comparison(
    summary: pd.DataFrame,
    comparison: str,
    c2st_summary: pd.DataFrame | None,
    candidate: str,
) -> dict[str, Any]:
    raw_power = extract_contrast_metric(summary, comparison, "raw_delta_power")
    matched_power = extract_contrast_metric(
        summary, comparison, "matched_k_ranking_component_power"
    )
    raw_fdp = extract_contrast_metric(summary, comparison, "raw_delta_realized_fdp")
    raw_selected = extract_contrast_metric(summary, comparison, "raw_delta_n_selected")
    matched_fdp = extract_contrast_metric(
        summary, comparison, "matched_k_ranking_component_realized_fdp"
    )

    label = "mixed_or_inconclusive"
    reasons: list[str] = []
    size_power = extract_contrast_metric(
        summary, comparison, "selection_size_component_power"
    )
    threshold_power = extract_contrast_metric(
        summary, comparison, "threshold_choice_component_power"
    )
    if raw_power and raw_power["ci_low"] > 0:
        matched_inconclusive = (
            matched_power is not None
            and matched_power["ci_low"] <= 0 <= matched_power["ci_high"]
        )
        size_supported = size_power is not None and size_power["ci_low"] > 0
        threshold_supported = (
            threshold_power is not None and threshold_power["ci_low"] > 0
        )
        selected_count_direction = (
            raw_selected is not None and raw_selected["mean"] > 0
        )
        if matched_inconclusive and (
            size_supported or threshold_supported or selected_count_direction
        ):
            label = "selection_size_or_threshold_interaction"
            reasons.append(
                "Raw power improved, exact matched-size power was inconclusive, "
                "and the size or threshold component was positive."
            )
    if matched_power and matched_fdp:
        if matched_power["ci_low"] > 0 and matched_fdp["ci_high"] <= 0.05:
            label = "ranking_support"
            reasons.append(
                "Exact matched-size power improved without a material matched-size FDP penalty."
            )
    if raw_power and raw_fdp and raw_power["ci_low"] > 0 and raw_fdp["ci_low"] > 0:
        label = "reliability_power_tradeoff"
        reasons.append("Power and realized FDP both increased.")

    c2st_warning = None
    if c2st_summary is not None and not c2st_summary.empty:
        row = c2st_summary[c2st_summary["generator_label"] == candidate]
        if not row.empty:
            swap_auc = float(row.iloc[0]["swap_c2st_auc_mean"])
            if swap_auc >= 0.80:
                c2st_warning = "severe"
            elif swap_auc >= 0.70:
                c2st_warning = "strong"
            elif swap_auc >= 0.60:
                c2st_warning = "moderate"
            else:
                c2st_warning = "low"
            reasons.append(f"Expanded mean swap C2ST AUC was {swap_auc:.3f}.")

    return {
        "comparison": comparison,
        "candidate": candidate,
        "diagnostic_label": label,
        "c2st_warning": c2st_warning,
        "reasons": reasons,
        "raw_power": raw_power,
        "matched_size_power": matched_power,
        "raw_fdp": raw_fdp,
        "matched_size_fdp": matched_fdp,
        "raw_selected_count": raw_selected,
        "selection_size_power_component": size_power,
        "threshold_choice_power_component": threshold_power,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="G2.1 saved-score threshold and matched-size diagnosis.",
    )
    parser.add_argument("--g2-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--comparisons",
        default=None,
        help=(
            "Comma-separated candidate:baseline comparisons. When omitted, "
            "common PLSKO comparisons are selected automatically."
        ),
    )
    parser.add_argument("--threshold-min", type=float, default=None)
    parser.add_argument("--threshold-max", type=float, default=None)
    parser.add_argument("--threshold-step", type=float, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--random-state", type=int, default=20260723)
    parser.add_argument(
        "--expanded-c2st",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute marginal and swap C2ST for every saved knockoff draw.",
    )
    parser.add_argument(
        "--save-full-threshold-path",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--make-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict-reconstruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when reconstructed stabl_min outputs disagree with G2 selection_results.csv.",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    g2_dir = Path(args.g2_dir).expanduser().resolve()
    if not g2_dir.is_dir():
        raise FileNotFoundError(g2_dir)
    config_path = g2_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else g2_dir / "G21_saved_score_diagnosis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.threshold_min is None and args.threshold_max is None and args.threshold_step is None:
        threshold_values = config.get("threshold_grid")
        if not threshold_values:
            raise ValueError("config.json does not contain threshold_grid")
        thresholds = np.asarray(threshold_values, dtype=float)
    else:
        threshold_min = 0.10 if args.threshold_min is None else args.threshold_min
        threshold_max = 0.99 if args.threshold_max is None else args.threshold_max
        threshold_step = 0.01 if args.threshold_step is None else args.threshold_step
        if threshold_step <= 0 or threshold_min >= threshold_max:
            raise ValueError("Invalid threshold grid")
        thresholds = np.arange(
            threshold_min,
            threshold_max + threshold_step / 2,
            threshold_step,
        )

    runs = load_score_runs(g2_dir, thresholds)
    generators = sorted({run.generator for run in runs})
    comparisons = parse_comparisons(args.comparisons, generators)

    run_rows = []
    score_rows = []
    threshold_rows = []
    for run_item in runs:
        run_rows.append(
            {
                "actual_p": run_item.p,
                "replicate": run_item.replicate,
                "generator_label": run_item.generator,
                "draw": run_item.draw,
                "stabl_threshold": run_item.stabl_threshold,
                "stabl_estimated_fdp": run_item.stabl_estimated_fdp,
                "stabl_calibration_gap": (
                    run_item.stabl_metrics["realized_fdp"]
                    - run_item.stabl_estimated_fdp
                ),
                **{f"stabl_{key}": value for key, value in run_item.stabl_metrics.items()},
                "source_path": str(run_item.source_path),
            }
        )
        score_rows.append(score_summary(run_item))
        threshold_rows.extend(path_rows_for_run(run_item, thresholds))

    run_metrics = pd.DataFrame(run_rows)
    score_distributions = pd.DataFrame(score_rows)
    threshold_runs = pd.DataFrame(threshold_rows)
    threshold_summary = summarize_threshold_paths(threshold_runs)
    score_distribution_summary = summarize_scores(score_distributions)

    matched_runs = matched_size_rows(runs, comparisons, thresholds)
    matched_replicates, contrast_summary = aggregate_matched_size(
        matched_runs,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.random_state,
    )

    atomic_write_csv(run_metrics, out_dir / "g21_stabl_min_run_metrics.csv")
    reconstruction = validate_reconstruction(g2_dir, run_metrics)
    reconstruction_passed = True
    if not reconstruction.empty:
        atomic_write_csv(
            reconstruction,
            out_dir / "g21_reconstruction_validation.csv",
        )
        reconstruction_passed = bool(reconstruction["all_metrics_pass"].all())
        if args.strict_reconstruction and not reconstruction_passed:
            failed = reconstruction[~reconstruction["all_metrics_pass"]]
            raise RuntimeError(
                f"Reconstructed STABL outputs disagree with {len(failed)} stored G2 rows. "
                f"See {out_dir / 'g21_reconstruction_validation.csv'}"
            )
    atomic_write_csv(score_distributions, out_dir / "g21_score_distribution_runs.csv")
    atomic_write_csv(
        score_distribution_summary,
        out_dir / "g21_score_distribution_summary.csv",
    )
    if args.save_full_threshold_path:
        atomic_write_csv(threshold_runs, out_dir / "g21_threshold_path_runs.csv")
    atomic_write_csv(threshold_summary, out_dir / "g21_threshold_path_summary.csv")
    atomic_write_csv(matched_runs, out_dir / "g21_matched_size_runs.csv")
    atomic_write_csv(matched_replicates, out_dir / "g21_matched_size_replicates.csv")
    atomic_write_csv(contrast_summary, out_dir / "g21_paired_contrast_summary.csv")

    c2st_runs: pd.DataFrame | None = None
    c2st_summary: pd.DataFrame | None = None
    if args.expanded_c2st:
        c2st_runs = expanded_c2st_rows(
            g2_dir,
            runs,
            config,
            seed=args.random_state + 500_000,
        )
        c2st_summary = (
            c2st_runs.groupby(["actual_p", "generator_label"], sort=True)[
                ["marginal_c2st_auc", "swap_c2st_auc"]
            ]
            .agg(["mean", "std", "min", "max"])
        )
        c2st_summary.columns = ["_".join(column) for column in c2st_summary.columns]
        c2st_summary = c2st_summary.reset_index()
        atomic_write_csv(c2st_runs, out_dir / "g21_c2st_all_draws.csv")
        atomic_write_csv(c2st_summary, out_dir / "g21_c2st_summary.csv")

    figures: list[str] = []
    if args.make_figures:
        figures = make_figures(
            threshold_summary,
            score_distribution_summary,
            contrast_summary,
            out_dir,
        )

    classifications = [
        classify_comparison(
            contrast_summary,
            spec.label,
            c2st_summary,
            spec.candidate,
        )
        for spec in comparisons
    ]

    status = {
        "experiment": "G2.1_saved_score_threshold_diagnosis",
        "source_g2_dir": str(g2_dir),
        "out_dir": str(out_dir),
        "n_score_files": len(runs),
        "n_generators": len(generators),
        "generators": generators,
        "n_replicates": len({run_item.replicate for run_item in runs}),
        "n_draws": len({run_item.draw for run_item in runs}),
        "n_thresholds": int(thresholds.size),
        "logical_threshold_path_rows": int(len(threshold_runs)),
        "full_threshold_path_saved": bool(args.save_full_threshold_path),
        "reconstruction_validation_rows": int(len(reconstruction)),
        "reconstruction_validation_passed": bool(reconstruction_passed),
        "expanded_c2st_completed": bool(args.expanded_c2st),
        "expanded_c2st_rows": 0 if c2st_runs is None else int(len(c2st_runs)),
        "comparisons": [spec.label for spec in comparisons],
        "comparison_diagnoses": classifications,
        "figures": figures,
        "bootstrap_samples": args.bootstrap_samples,
        "random_state": args.random_state,
        "notes": [
            "This experiment reuses saved score and pair-cache files only.",
            "Exact matched-size top-k selection uses no support labels to choose k or ranks.",
            "Diagnostic labels are descriptive and are not confirmatory decisions.",
        ],
    }
    atomic_write_json(status, out_dir / "g21_status.json")

    print(f"G2.1 completed: {out_dir}")
    print(f"Score files: {len(runs)}")
    print(f"Threshold path rows: {len(threshold_runs)}")
    print(f"Matched-size rows: {len(matched_runs)}")
    if args.expanded_c2st:
        print(f"Expanded C2ST rows: {len(c2st_runs) if c2st_runs is not None else 0}")
    return out_dir


def main() -> None:
    run()


if __name__ == "__main__":
    main()
