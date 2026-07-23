#!/usr/bin/env python3
"""V11.1 score-only threshold diagnosis across the V11 stress map.

This script reads saved score arrays from a completed V11 missingness-stress
pilot. It does not rerun completion, knockoff generation, or STABL.

For every dataset, missingness rate, mechanism, replicate, and paired run it:
1. reconstructs the original stabl_min threshold and full threshold path;
2. measures estimated FDP+ versus realized FDP calibration;
3. evaluates BR mean at a threshold chosen to match the Median selected count;
4. combines the matched-size results with threshold-independent ranking metrics;
5. labels each cell descriptively as ranking support, threshold interaction,
   ranking limitation, or inconclusive.

The diagnostic labels are descriptive. They are not confirmatory decisions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from research_common import atomic_write_csv, fdp_plus_curve, select_stabl_min
from v91_common import dataframe_to_markdown, paired_wilcoxon_p, percentile_bootstrap_mean_ci

FILE_PATTERN = re.compile(
    r"p_(?P<p>\d+)__rep_(?P<rep>\d+)__(?P<mechanism>MCAR|MAR|MNAR|BLOCK)__"
    r"(?P<completion>oracle_complete|median|bayesianridge_mean)__cdraw_(?P<draw>\d+)__"
    r"(?P<generator>[^.]+)\.npz$"
)
RATE_PATTERN = re.compile(r"rate_(?P<rate>\d+)p(?P<decimal>\d+)$")
COMPLETIONS = ("oracle_complete", "median", "bayesianridge_mean")


def max_scores(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return np.max(values, axis=1)
    raise ValueError(f"Unsupported score shape: {values.shape}")


def selection_metrics(selected: np.ndarray, support: np.ndarray) -> dict[str, float]:
    truth = np.zeros(selected.size, dtype=bool)
    truth[np.asarray(support, dtype=int)] = True
    tp = int(np.sum(selected & truth))
    fp = int(np.sum(selected & ~truth))
    count = int(selected.sum())
    return {
        "n_selected": count,
        "true_positives": tp,
        "false_positives": fp,
        "realized_fdp": float(fp / max(1, count)),
        "power": float(tp / max(1, int(truth.sum()))),
    }


def parse_rate(rate_dir: Path) -> float:
    match = RATE_PATTERN.match(rate_dir.name)
    if not match:
        raise ValueError(f"Unrecognized rate directory: {rate_dir.name}")
    return float(f"{int(match.group('rate'))}.{match.group('decimal')}")


def bootstrap_contrast(
    values: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
) -> tuple[float, float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    low, high = percentile_bootstrap_mean_ci(clean, seed=seed, n_bootstrap=n_bootstrap)
    return float(np.mean(clean)), low, high, paired_wilcoxon_p(clean)


def paired_cell_contrasts(
    frame: pd.DataFrame,
    *,
    index_cols: list[str],
    value_cols: Iterable[str],
    group_cols: list[str],
    seed: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed_offset = 0
    for group_key, group in frame.groupby(group_cols, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_meta = dict(zip(group_cols, group_key))
        for metric in value_cols:
            wide = group.pivot_table(
                index=index_cols,
                columns="completion",
                values=metric,
                aggfunc="first",
            )
            required = ["median", "bayesianridge_mean"]
            if not set(required).issubset(wide.columns):
                continue
            paired = wide[required].dropna()
            delta = paired["bayesianridge_mean"].to_numpy(dtype=float) - paired["median"].to_numpy(dtype=float)
            mean, low, high, pvalue = bootstrap_contrast(
                delta,
                seed=seed + seed_offset,
                n_bootstrap=n_bootstrap,
            )
            seed_offset += 1
            rows.append(
                {
                    **group_meta,
                    "metric": metric,
                    "n_pairs": int(len(paired)),
                    "median_mean": float(paired["median"].mean()) if len(paired) else float("nan"),
                    "br_mean": float(paired["bayesianridge_mean"].mean()) if len(paired) else float("nan"),
                    "mean_difference_br_minus_median": mean,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "paired_wilcoxon_p": pvalue,
                }
            )
    return pd.DataFrame(rows)


def metric_delta_table(contrasts: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keys = ["block_label", "missing_rate", "mechanism"]
    if contrasts.empty:
        return pd.DataFrame(columns=keys)
    wide = contrasts.pivot_table(
        index=keys,
        columns="metric",
        values="mean_difference_br_minus_median",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    rename = {column: f"{prefix}{column}" for column in wide.columns if column not in keys}
    return wide.rename(columns=rename)


def integrate_over_rates(frame: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["block_label", "mechanism", "replicate"]
    for group_key, group in frame.groupby(group_cols, sort=True):
        group = group.sort_values("missing_rate")
        x = group["missing_rate"].to_numpy(dtype=float)
        if np.unique(x).size < 2:
            continue
        row = dict(zip(group_cols, group_key))
        for metric in metric_cols:
            y = group[metric].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            row[f"auc_{metric}"] = float(integrator(y[valid], x[valid])) if valid.sum() >= 2 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="V11.1 score-only threshold and selection-size diagnosis.",
    )
    parser.add_argument("--v11-pilot-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--bootstrap-seed", type=int, default=80260730)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--fdp-worsening-guard", type=float, default=0.05)
    parser.add_argument("--power-loss-guard", type=float, default=0.05)
    parser.add_argument(
        "--save-full-path-runs",
        action="store_true",
        help="Also write the large per-run threshold-path table. The aggregated path summary is always written.",
    )
    args = parser.parse_args()

    pilot_dir = Path(args.v11_pilot_dir).expanduser().resolve()
    if not pilot_dir.exists():
        raise FileNotFoundError(pilot_dir)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else pilot_dir / "v111_threshold_diagnosis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2,
        args.threshold_step,
    )

    rate_dirs: list[tuple[str, float, Path]] = []
    for block_dir in sorted(path for path in pilot_dir.iterdir() if path.is_dir()):
        if block_dir.name.startswith("figures") or block_dir.name.startswith("v111"):
            continue
        for rate_dir in sorted(path for path in block_dir.iterdir() if path.is_dir() and path.name.startswith("rate_")):
            rate_dirs.append((block_dir.name, parse_rate(rate_dir), rate_dir))
    if not rate_dirs:
        raise RuntimeError(f"No block/rate directories found under {pilot_dir}")

    run_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    path_accum: dict[tuple[str, float, str, str, str], dict[str, Any]] = {}
    logical_path_rows = 0
    score_cache: dict[tuple[str, float, int, str, int, str], dict[str, Any]] = {}
    score_file_count = 0

    for block_label, missing_rate, rate_dir in rate_dirs:
        score_dir = rate_dir / "scores"
        score_files = sorted(score_dir.glob("*.npz"))
        if not score_files:
            raise RuntimeError(f"No score files found in {score_dir}")
        score_file_count += len(score_files)
        for path in score_files:
            match = FILE_PATTERN.match(path.name)
            if not match:
                raise ValueError(f"Unrecognized score filename: {path.name}")
            meta = match.groupdict()
            replicate = int(meta["rep"])
            draw = int(meta["draw"])
            mechanism = meta["mechanism"]
            completion = meta["completion"]
            generator = meta["generator"]
            with np.load(path, allow_pickle=True) as payload:
                real_scores = np.asarray(payload["real_scores"], dtype=float)
                knockoff_scores = np.asarray(payload["knockoff_scores"], dtype=float)
                support = np.asarray(payload["support"], dtype=int)
            real = max_scores(real_scores)
            knockoff = max_scores(knockoff_scores)
            estimated_curve = fdp_plus_curve(real_scores, knockoff_scores, thresholds)
            stabl = select_stabl_min(real_scores, knockoff_scores, thresholds)
            stabl_metrics = selection_metrics(np.asarray(stabl.selected, dtype=bool), support)
            run_rows.append(
                {
                    "block_label": block_label,
                    "missing_rate": missing_rate,
                    "replicate": replicate,
                    "mechanism": mechanism,
                    "completion_draw": draw,
                    "completion": completion,
                    "generator_label": generator,
                    "stabl_min_threshold": float(stabl.threshold),
                    "stabl_min_estimated_fdp": float(stabl.estimated_fdp),
                    "stabl_min_calibration_gap_realized_minus_estimated": float(
                        stabl_metrics["realized_fdp"] - stabl.estimated_fdp
                    ),
                    **{f"stabl_min_{key}": value for key, value in stabl_metrics.items()},
                }
            )
            score_cache[(block_label, missing_rate, replicate, mechanism, draw, completion)] = {
                "real": real,
                "knockoff": knockoff,
                "support": support,
                "threshold": float(stabl.threshold),
                "stabl_metrics": stabl_metrics,
            }
            truth = np.zeros(real.size, dtype=bool)
            truth[np.asarray(support, dtype=int)] = True
            selected_matrix = real[:, None] >= thresholds[None, :]
            n_selected = selected_matrix.sum(axis=0).astype(float)
            true_positives = selected_matrix[truth, :].sum(axis=0).astype(float)
            false_positives = selected_matrix[~truth, :].sum(axis=0).astype(float)
            realized_fdp = false_positives / np.maximum(1.0, n_selected)
            power = true_positives / max(1.0, float(truth.sum()))
            calibration_gap = realized_fdp - estimated_curve
            logical_path_rows += int(len(thresholds))

            accum_key = (block_label, missing_rate, mechanism, completion, generator)
            if accum_key not in path_accum:
                path_accum[accum_key] = {
                    "n_runs": 0,
                    "estimated_fdp_plus_sum": np.zeros(len(thresholds), dtype=float),
                    "realized_fdp_sum": np.zeros(len(thresholds), dtype=float),
                    "calibration_gap_sum": np.zeros(len(thresholds), dtype=float),
                    "n_selected_sum": np.zeros(len(thresholds), dtype=float),
                    "power_sum": np.zeros(len(thresholds), dtype=float),
                }
            accumulator = path_accum[accum_key]
            accumulator["n_runs"] += 1
            accumulator["estimated_fdp_plus_sum"] += estimated_curve
            accumulator["realized_fdp_sum"] += realized_fdp
            accumulator["calibration_gap_sum"] += calibration_gap
            accumulator["n_selected_sum"] += n_selected
            accumulator["power_sum"] += power

            if args.save_full_path_runs:
                for threshold_index, threshold in enumerate(thresholds):
                    path_rows.append(
                        {
                            "block_label": block_label,
                            "missing_rate": missing_rate,
                            "replicate": replicate,
                            "mechanism": mechanism,
                            "completion_draw": draw,
                            "completion": completion,
                            "generator_label": generator,
                            "threshold_index": threshold_index,
                            "threshold": float(threshold),
                            "estimated_fdp_plus": float(estimated_curve[threshold_index]),
                            "calibration_gap_realized_minus_estimated": float(calibration_gap[threshold_index]),
                            "n_selected": int(n_selected[threshold_index]),
                            "true_positives": int(true_positives[threshold_index]),
                            "false_positives": int(false_positives[threshold_index]),
                            "realized_fdp": float(realized_fdp[threshold_index]),
                            "power": float(power[threshold_index]),
                        }
                    )

    runs = pd.DataFrame(run_rows)
    atomic_write_csv(runs, out_dir / "v111_stabl_min_run_metrics.csv")
    if args.save_full_path_runs:
        atomic_write_csv(pd.DataFrame(path_rows), out_dir / "v111_threshold_path_run_metrics.csv")

    replicate_metrics = (
        runs.groupby(
            ["block_label", "missing_rate", "replicate", "mechanism", "completion"],
            as_index=False,
        ).mean(numeric_only=True)
    )
    atomic_write_csv(replicate_metrics, out_dir / "v111_stabl_min_replicate_metrics.csv")

    original_metrics = [
        "stabl_min_realized_fdp",
        "stabl_min_power",
        "stabl_min_n_selected",
        "stabl_min_true_positives",
        "stabl_min_false_positives",
        "stabl_min_threshold",
        "stabl_min_estimated_fdp",
        "stabl_min_calibration_gap_realized_minus_estimated",
    ]
    original_contrasts = paired_cell_contrasts(
        replicate_metrics,
        index_cols=["replicate"],
        value_cols=original_metrics,
        group_cols=["block_label", "missing_rate", "mechanism"],
        seed=args.bootstrap_seed,
        n_bootstrap=args.n_bootstrap,
    )
    atomic_write_csv(original_contrasts, out_dir / "v111_stabl_min_paired_contrasts.csv")

    matched_rows: list[dict[str, Any]] = []
    for key, median_info in score_cache.items():
        block_label, missing_rate, replicate, mechanism, draw, completion = key
        if completion != "median":
            continue
        br_key = (block_label, missing_rate, replicate, mechanism, draw, "bayesianridge_mean")
        if br_key not in score_cache:
            continue
        target_size = int(median_info["stabl_metrics"]["n_selected"])
        target_threshold = float(median_info["threshold"])
        br_info = score_cache[br_key]
        br_real = np.asarray(br_info["real"], dtype=float)
        selected_matrix = br_real[:, None] >= thresholds[None, :]
        selected_counts = selected_matrix.sum(axis=0)
        size_distance = np.abs(selected_counts - target_size)
        threshold_distance = np.abs(thresholds - target_threshold)
        # Lexicographic rule: selected-count distance, threshold distance, then
        # prefer the higher threshold. This rule uses no support labels.
        order = np.lexsort((-thresholds, threshold_distance, size_distance))
        chosen_index = int(order[0])
        chosen_threshold = float(thresholds[chosen_index])
        chosen_selected = selected_matrix[:, chosen_index]
        chosen_metrics = selection_metrics(chosen_selected, br_info["support"])
        median_metrics = median_info["stabl_metrics"]
        matched_rows.append(
            {
                "block_label": block_label,
                "missing_rate": missing_rate,
                "replicate": replicate,
                "mechanism": mechanism,
                "completion_draw": draw,
                "median_stabl_threshold": target_threshold,
                "median_n_selected": target_size,
                "median_realized_fdp": median_metrics["realized_fdp"],
                "median_power": median_metrics["power"],
                "median_true_positives": median_metrics["true_positives"],
                "median_false_positives": median_metrics["false_positives"],
                "br_matched_threshold": chosen_threshold,
                "br_n_selected": int(chosen_metrics["n_selected"]),
                "selection_size_difference": int(chosen_metrics["n_selected"] - target_size),
                "br_realized_fdp": float(chosen_metrics["realized_fdp"]),
                "br_power": float(chosen_metrics["power"]),
                "br_true_positives": int(chosen_metrics["true_positives"]),
                "br_false_positives": int(chosen_metrics["false_positives"]),
                "delta_fdp_br_minus_median": float(chosen_metrics["realized_fdp"] - median_metrics["realized_fdp"]),
                "delta_power_br_minus_median": float(chosen_metrics["power"] - median_metrics["power"]),
                "delta_tp_br_minus_median": float(chosen_metrics["true_positives"] - median_metrics["true_positives"]),
                "delta_fp_br_minus_median": float(chosen_metrics["false_positives"] - median_metrics["false_positives"]),
            }
        )
    matched_runs = pd.DataFrame(matched_rows)
    atomic_write_csv(matched_runs, out_dir / "v111_br_at_median_selection_size_runs.csv")

    matched_replicates = (
        matched_runs.groupby(
            ["block_label", "missing_rate", "replicate", "mechanism"],
            as_index=False,
        ).mean(numeric_only=True)
    )
    atomic_write_csv(matched_replicates, out_dir / "v111_br_at_median_selection_size_replicates.csv")

    matched_contrast_rows: list[dict[str, Any]] = []
    seed_offset = len(original_contrasts) + 1000
    matched_metrics = [
        "delta_fdp_br_minus_median",
        "delta_power_br_minus_median",
        "delta_tp_br_minus_median",
        "delta_fp_br_minus_median",
        "selection_size_difference",
    ]
    for group_key, group in matched_replicates.groupby(
        ["block_label", "missing_rate", "mechanism"], sort=True
    ):
        meta = dict(zip(["block_label", "missing_rate", "mechanism"], group_key))
        for metric in matched_metrics:
            values = group[metric].to_numpy(dtype=float)
            mean, low, high, pvalue = bootstrap_contrast(
                values,
                seed=args.bootstrap_seed + seed_offset,
                n_bootstrap=args.n_bootstrap,
            )
            seed_offset += 1
            matched_contrast_rows.append(
                {
                    **meta,
                    "metric": metric,
                    "n_replicates": int(np.isfinite(values).sum()),
                    "mean": mean,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "paired_wilcoxon_p": pvalue,
                }
            )
    matched_contrasts = pd.DataFrame(matched_contrast_rows)
    atomic_write_csv(matched_contrasts, out_dir / "v111_br_at_median_selection_size_contrasts.csv")

    argmin_variability = (
        runs.groupby(
            ["block_label", "missing_rate", "replicate", "mechanism", "completion"],
            as_index=False,
        ).agg(
            threshold_mean=("stabl_min_threshold", "mean"),
            threshold_sd=("stabl_min_threshold", "std"),
            selected_mean=("stabl_min_n_selected", "mean"),
            selected_sd=("stabl_min_n_selected", "std"),
            estimated_fdp_mean=("stabl_min_estimated_fdp", "mean"),
            realized_fdp_mean=("stabl_min_realized_fdp", "mean"),
            calibration_gap_mean=("stabl_min_calibration_gap_realized_minus_estimated", "mean"),
        )
    )
    atomic_write_csv(argmin_variability, out_dir / "v111_argmin_variability.csv")

    path_summary_rows: list[dict[str, Any]] = []
    for (block_label, missing_rate, mechanism, completion, generator), accumulator in sorted(path_accum.items()):
        n_runs = int(accumulator["n_runs"])
        for threshold_index, threshold in enumerate(thresholds):
            path_summary_rows.append(
                {
                    "block_label": block_label,
                    "missing_rate": missing_rate,
                    "mechanism": mechanism,
                    "completion": completion,
                    "generator_label": generator,
                    "threshold_index": threshold_index,
                    "threshold": float(threshold),
                    "n_runs": n_runs,
                    "estimated_fdp_plus_mean": float(accumulator["estimated_fdp_plus_sum"][threshold_index] / n_runs),
                    "realized_fdp_mean": float(accumulator["realized_fdp_sum"][threshold_index] / n_runs),
                    "calibration_gap_mean": float(accumulator["calibration_gap_sum"][threshold_index] / n_runs),
                    "n_selected_mean": float(accumulator["n_selected_sum"][threshold_index] / n_runs),
                    "power_mean": float(accumulator["power_sum"][threshold_index] / n_runs),
                }
            )
    path_summary = pd.DataFrame(path_summary_rows)
    atomic_write_csv(path_summary, out_dir / "v111_threshold_path_summary.csv")

    ranking_frames: list[pd.DataFrame] = []
    for block_label, missing_rate, rate_dir in rate_dirs:
        ranking_path = rate_dir / "v11_threshold_independent_ranking" / "ranking_replicate_metrics.csv"
        if not ranking_path.exists():
            raise FileNotFoundError(ranking_path)
        ranking = pd.read_csv(ranking_path)
        ranking.insert(0, "missing_rate", missing_rate)
        ranking.insert(0, "block_label", block_label)
        ranking_frames.append(ranking)
    ranking_replicates = pd.concat(ranking_frames, ignore_index=True)
    atomic_write_csv(ranking_replicates, out_dir / "v111_ranking_replicate_metrics.csv")
    ranking_metrics = [
        "precision_at_5",
        "precision_at_10",
        "precision_at_15",
        "precision_at_20",
        "average_precision",
        "median_signal_rank",
        "support_ranking_auroc",
        "mean_signal_score",
        "mean_null_score",
    ]
    ranking_contrasts = paired_cell_contrasts(
        ranking_replicates,
        index_cols=["replicate"],
        value_cols=ranking_metrics,
        group_cols=["block_label", "missing_rate", "mechanism"],
        seed=args.bootstrap_seed + 5000,
        n_bootstrap=args.n_bootstrap,
    )
    atomic_write_csv(ranking_contrasts, out_dir / "v111_ranking_paired_contrasts.csv")

    original_wide = metric_delta_table(original_contrasts, "original_delta_")
    ranking_wide = metric_delta_table(ranking_contrasts, "ranking_delta_")
    matched_wide = matched_contrasts.pivot_table(
        index=["block_label", "missing_rate", "mechanism"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    matched_wide.columns.name = None
    matched_wide = matched_wide.rename(
        columns={
            column: f"matched_{column}"
            for column in matched_wide.columns
            if column not in ["block_label", "missing_rate", "mechanism"]
        }
    )

    diagnosis = original_wide.merge(
        ranking_wide,
        on=["block_label", "missing_rate", "mechanism"],
        how="outer",
    ).merge(
        matched_wide,
        on=["block_label", "missing_rate", "mechanism"],
        how="outer",
    )

    labels: list[str] = []
    explanations: list[str] = []
    for _, row in diagnosis.iterrows():
        ap = float(row.get("ranking_delta_average_precision", np.nan))
        auroc = float(row.get("ranking_delta_support_ranking_auroc", np.nan))
        null_score = float(row.get("ranking_delta_mean_null_score", np.nan))
        orig_fdp = float(row.get("original_delta_stabl_min_realized_fdp", np.nan))
        orig_power = float(row.get("original_delta_stabl_min_power", np.nan))
        orig_selected = float(row.get("original_delta_stabl_min_n_selected", np.nan))
        matched_fdp = float(row.get("matched_delta_fdp_br_minus_median", np.nan))
        matched_power = float(row.get("matched_delta_power_br_minus_median", np.nan))
        ranking_favorable = bool(ap > 0 and auroc > 0 and null_score < 0)
        matched_favorable = bool(
            matched_fdp <= args.fdp_worsening_guard
            and matched_power >= -args.power_loss_guard
        )
        original_favorable = bool(
            orig_fdp <= args.fdp_worsening_guard
            and orig_power >= -args.power_loss_guard
        )
        size_mediation = bool(
            np.isfinite(orig_fdp)
            and np.isfinite(matched_fdp)
            and (orig_fdp - matched_fdp) >= 0.03
        )
        if ranking_favorable and matched_favorable and not original_favorable and size_mediation:
            label = "threshold_interaction_strong"
            explanation = "Ranking and matched-size results favor BR mean, while original stabl_min is unfavorable and the FDP gap improves materially after size matching."
        elif ranking_favorable and matched_favorable and size_mediation:
            label = "threshold_interaction_directional"
            explanation = "Ranking is favorable and size matching improves the selected-set comparison, consistent with a threshold or selected-count interaction."
        elif ranking_favorable and original_favorable:
            label = "ranking_and_stabl_min_support"
            explanation = "Threshold-independent ranking and the original stabl_min selected set are directionally favorable."
        elif ap <= 0 and auroc <= 0:
            label = "ranking_limitation"
            explanation = "Both average precision and ranking AUROC are non-favorable, so the limitation is not explained by thresholding alone."
        elif ranking_favorable and not matched_favorable:
            label = "ranking_support_selected_set_tradeoff"
            explanation = "Ranking favors BR mean, but matched-size FDP or power remains unfavorable, indicating a selected-set trade-off beyond count inflation alone."
        else:
            label = "mixed_inconclusive"
            explanation = "Ranking, matched-size, and original-threshold evidence do not support a single dominant explanation."
        labels.append(label)
        explanations.append(explanation)
    diagnosis["diagnostic_label"] = labels
    diagnosis["diagnostic_explanation"] = explanations
    atomic_write_csv(diagnosis, out_dir / "v111_cell_diagnosis.csv")

    matched_auc_source = matched_replicates.copy()
    matched_auc = integrate_over_rates(
        matched_auc_source,
        [
            "delta_fdp_br_minus_median",
            "delta_power_br_minus_median",
            "delta_tp_br_minus_median",
            "delta_fp_br_minus_median",
        ],
    )
    atomic_write_csv(matched_auc, out_dir / "v111_matched_size_auc_by_replicate.csv")
    auc_rows: list[dict[str, Any]] = []
    for group_key, group in matched_auc.groupby(["block_label", "mechanism"], sort=True):
        meta = dict(zip(["block_label", "mechanism"], group_key))
        for metric in [column for column in matched_auc.columns if column.startswith("auc_")]:
            values = group[metric].to_numpy(dtype=float)
            mean, low, high, pvalue = bootstrap_contrast(
                values,
                seed=args.bootstrap_seed + seed_offset,
                n_bootstrap=args.n_bootstrap,
            )
            seed_offset += 1
            auc_rows.append(
                {
                    **meta,
                    "metric": metric,
                    "n_replicates": int(np.isfinite(values).sum()),
                    "mean": mean,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "paired_wilcoxon_p": pvalue,
                }
            )
    auc_contrasts = pd.DataFrame(auc_rows)
    atomic_write_csv(auc_contrasts, out_dir / "v111_matched_size_auc_contrasts.csv")

    expected_rates = len(rate_dirs)
    expected_score_files = 0
    expected_matched_runs = 0
    for _, _, rate_dir in rate_dirs:
        count = len(list((rate_dir / "scores").glob("*.npz")))
        expected_score_files += count
        selection_path = rate_dir / "selection_results.csv"
        if selection_path.exists():
            selection = pd.read_csv(selection_path)
            expected_matched_runs += int(selection[selection["completion"].eq("median")].shape[0])
    audit = pd.DataFrame(
        [
            {
                "block_rate_directories": expected_rates,
                "score_files": score_file_count,
                "expected_score_files_from_directories": expected_score_files,
                "thresholds": len(thresholds),
                "threshold_path_rows": logical_path_rows,
                "expected_threshold_path_rows": score_file_count * len(thresholds),
                "matched_runs": len(matched_runs),
                "expected_matched_runs": expected_matched_runs,
                "diagnostic_cells": len(diagnosis),
                "expected_diagnostic_cells": len(rate_dirs) * 3,
                "complete": bool(
                    score_file_count == expected_score_files
                    and logical_path_rows == score_file_count * len(thresholds)
                    and len(matched_runs) == expected_matched_runs
                    and len(diagnosis) == len(rate_dirs) * 3
                ),
            }
        ]
    )
    atomic_write_csv(audit, out_dir / "v111_integrity_audit.csv")

    config = {
        "analysis": "V11.1 score-only threshold diagnosis",
        "source": str(pilot_dir),
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_step": args.threshold_step,
        "bootstrap_seed": args.bootstrap_seed,
        "n_bootstrap": args.n_bootstrap,
        "fdp_worsening_guard": args.fdp_worsening_guard,
        "power_loss_guard": args.power_loss_guard,
        "save_full_path_runs": bool(args.save_full_path_runs),
        "scientific_status": "descriptive_score_only_reanalysis",
    }
    (out_dir / "v111_analysis_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    label_counts = diagnosis["diagnostic_label"].value_counts().rename_axis("diagnostic_label").reset_index(name="n_cells")
    report_lines = [
        "# V11.1 threshold and selection-size diagnosis",
        "",
        "This is a score-only reanalysis of the completed V11 pilot. It does not rerun completion, knockoffs, or STABL.",
        "",
        "## Integrity",
        "",
        dataframe_to_markdown(audit, index=False),
        "",
        "## Descriptive diagnostic labels",
        "",
        dataframe_to_markdown(label_counts, index=False),
        "",
        "## Cell-level diagnosis",
        "",
        dataframe_to_markdown(
            diagnosis[
                [
                    "block_label",
                    "missing_rate",
                    "mechanism",
                    "diagnostic_label",
                    "original_delta_stabl_min_realized_fdp",
                    "original_delta_stabl_min_power",
                    "original_delta_stabl_min_n_selected",
                    "matched_delta_fdp_br_minus_median",
                    "matched_delta_power_br_minus_median",
                    "ranking_delta_average_precision",
                    "ranking_delta_support_ranking_auroc",
                    "ranking_delta_mean_null_score",
                ]
            ],
            index=False,
        ),
        "",
        "The matched-size threshold is chosen without support labels: minimize selected-count distance to Median, then threshold distance from Median's stabl_min threshold, then prefer the higher threshold.",
        "",
        "Diagnostic labels are descriptive and must not be interpreted as confirmatory statistical decisions.",
    ]
    (out_dir / "v111_threshold_diagnosis_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"[V11.1] read {score_file_count} score files")
    print(f"[V11.1] reconstructed {logical_path_rows} threshold-path rows")
    print(f"[V11.1] integrity complete={bool(audit.loc[0, 'complete'])}")
    print(f"[V11.1] outputs written to {out_dir}")


if __name__ == "__main__":
    main()
