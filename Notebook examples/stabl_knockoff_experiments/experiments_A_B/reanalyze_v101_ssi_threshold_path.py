#!/usr/bin/env python3
"""V10.1C: diagnose SSI Proteomics stabl_min threshold behavior.

This is a score-only reanalysis. It does not rerun completion, knockoffs, or
STABL. The script reconstructs the full threshold path from saved real and
knockoff selection-frequency arrays, compares calibration and selection size,
and evaluates BR mean at the Median stabl_min selection size using an
outcome-independent threshold matching rule.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_common import atomic_write_csv, fdp_plus_curve, select_stabl_min
from v91_common import dataframe_to_markdown, paired_wilcoxon_p, percentile_bootstrap_mean_ci

FILE_PATTERN = re.compile(
    r"p_(?P<p>\d+)__rep_(?P<rep>\d+)__(?P<mechanism>MCAR|MAR|MNAR)__"
    r"(?P<completion>oracle_complete|median|bayesianridge_mean)__cdraw_(?P<draw>\d+)__"
    r"(?P<generator>[^.]+)\.npz$"
)


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


def paired_contrast(
    frame: pd.DataFrame,
    *,
    metric: str,
    mechanism: str,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    subset = frame[frame["mechanism"].eq(mechanism)]
    wide = subset.pivot(index="replicate", columns="completion", values=metric)
    wide = wide[["median", "bayesianridge_mean"]].dropna()
    difference = wide["bayesianridge_mean"].to_numpy() - wide["median"].to_numpy()
    ci_low, ci_high = percentile_bootstrap_mean_ci(
        difference, seed=seed, n_bootstrap=n_bootstrap
    )
    return {
        "mechanism": mechanism,
        "metric": metric,
        "n_pairs": int(len(wide)),
        "median_mean": float(wide["median"].mean()),
        "br_mean": float(wide["bayesianridge_mean"].mean()),
        "mean_difference_br_minus_median": float(np.mean(difference)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "paired_wilcoxon_p": paired_wilcoxon_p(difference),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reanalyze SSI Proteomics V10 score paths.",
    )
    parser.add_argument("--ssi-pilot-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--bootstrap-seed", type=int, default=60260730)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    args = parser.parse_args()

    pilot_dir = Path(args.ssi_pilot_dir).expanduser().resolve()
    score_dir = pilot_dir / "scores"
    if not score_dir.exists():
        raise FileNotFoundError(score_dir)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else pilot_dir / "v101_ssi_threshold_path_diagnosis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2,
        args.threshold_step,
    )
    score_files = sorted(score_dir.glob("*.npz"))
    if not score_files:
        raise RuntimeError(f"No score files found in {score_dir}")

    run_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    score_cache: dict[tuple[int, str, int, str], dict[str, Any]] = {}
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
        run_row = {
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
        run_rows.append(run_row)
        score_cache[(replicate, mechanism, draw, completion)] = {
            "real": real,
            "knockoff": knockoff,
            "support": support,
            "threshold": float(stabl.threshold),
            "stabl_metrics": stabl_metrics,
        }
        for threshold_index, threshold in enumerate(thresholds):
            selected = real >= threshold
            metrics = selection_metrics(selected, support)
            path_rows.append(
                {
                    "replicate": replicate,
                    "mechanism": mechanism,
                    "completion_draw": draw,
                    "completion": completion,
                    "generator_label": generator,
                    "threshold_index": threshold_index,
                    "threshold": float(threshold),
                    "estimated_fdp_plus": float(estimated_curve[threshold_index]),
                    "calibration_gap_realized_minus_estimated": float(
                        metrics["realized_fdp"] - estimated_curve[threshold_index]
                    ),
                    **metrics,
                }
            )

    runs = pd.DataFrame(run_rows)
    paths = pd.DataFrame(path_rows)
    atomic_write_csv(runs, out_dir / "ssi_stabl_min_run_metrics.csv")
    atomic_write_csv(paths, out_dir / "ssi_threshold_path_run_metrics.csv")

    replicate_metrics = (
        runs.groupby(["replicate", "mechanism", "completion"], as_index=False)
        .mean(numeric_only=True)
    )
    atomic_write_csv(replicate_metrics, out_dir / "ssi_stabl_min_replicate_metrics.csv")

    contrasts: list[dict[str, Any]] = []
    metrics = [
        "stabl_min_realized_fdp", "stabl_min_power", "stabl_min_n_selected",
        "stabl_min_true_positives", "stabl_min_false_positives",
        "stabl_min_threshold", "stabl_min_estimated_fdp",
        "stabl_min_calibration_gap_realized_minus_estimated",
    ]
    seed_offset = 0
    for mechanism in ("MCAR", "MAR"):
        for metric in metrics:
            contrasts.append(
                paired_contrast(
                    replicate_metrics,
                    metric=metric,
                    mechanism=mechanism,
                    seed=args.bootstrap_seed + seed_offset,
                    n_bootstrap=args.n_bootstrap,
                )
            )
            seed_offset += 1
    contrast_frame = pd.DataFrame(contrasts)
    atomic_write_csv(contrast_frame, out_dir / "ssi_stabl_min_paired_contrasts.csv")

    matched_rows: list[dict[str, Any]] = []
    for key, median_info in score_cache.items():
        replicate, mechanism, draw, completion = key
        if completion != "median":
            continue
        br_key = (replicate, mechanism, draw, "bayesianridge_mean")
        if br_key not in score_cache:
            continue
        br_info = score_cache[br_key]
        target_size = int(median_info["stabl_metrics"]["n_selected"])
        target_threshold = float(median_info["threshold"])
        candidate_rows = paths[
            paths["replicate"].eq(replicate)
            & paths["mechanism"].eq(mechanism)
            & paths["completion_draw"].eq(draw)
            & paths["completion"].eq("bayesianridge_mean")
        ].copy()
        candidate_rows["size_distance"] = (
            candidate_rows["n_selected"] - target_size
        ).abs()
        candidate_rows["threshold_distance"] = (
            candidate_rows["threshold"] - target_threshold
        ).abs()
        chosen = candidate_rows.sort_values(
            ["size_distance", "threshold_distance", "threshold"],
            ascending=[True, True, False],
        ).iloc[0]
        median_metrics = median_info["stabl_metrics"]
        matched_rows.append(
            {
                "replicate": replicate,
                "mechanism": mechanism,
                "completion_draw": draw,
                "median_stabl_threshold": target_threshold,
                "median_n_selected": target_size,
                "median_realized_fdp": median_metrics["realized_fdp"],
                "median_power": median_metrics["power"],
                "median_true_positives": median_metrics["true_positives"],
                "median_false_positives": median_metrics["false_positives"],
                "br_matched_threshold": float(chosen["threshold"]),
                "br_n_selected": int(chosen["n_selected"]),
                "selection_size_difference": int(chosen["n_selected"] - target_size),
                "br_realized_fdp": float(chosen["realized_fdp"]),
                "br_power": float(chosen["power"]),
                "br_true_positives": int(chosen["true_positives"]),
                "br_false_positives": int(chosen["false_positives"]),
                "delta_fdp_br_minus_median": float(chosen["realized_fdp"] - median_metrics["realized_fdp"]),
                "delta_power_br_minus_median": float(chosen["power"] - median_metrics["power"]),
                "delta_tp_br_minus_median": float(chosen["true_positives"] - median_metrics["true_positives"]),
                "delta_fp_br_minus_median": float(chosen["false_positives"] - median_metrics["false_positives"]),
            }
        )
    matched = pd.DataFrame(matched_rows)
    atomic_write_csv(matched, out_dir / "ssi_br_at_median_selection_size_runs.csv")

    matched_rep = (
        matched.groupby(["replicate", "mechanism"], as_index=False)
        .mean(numeric_only=True)
    )
    atomic_write_csv(matched_rep, out_dir / "ssi_br_at_median_selection_size_replicates.csv")
    matched_contrast_rows: list[dict[str, Any]] = []
    for mechanism in ("MCAR", "MAR"):
        block = matched_rep[matched_rep["mechanism"].eq(mechanism)]
        for metric in (
            "delta_fdp_br_minus_median", "delta_power_br_minus_median",
            "delta_tp_br_minus_median", "delta_fp_br_minus_median",
            "selection_size_difference",
        ):
            values = block[metric].to_numpy(dtype=float)
            ci_low, ci_high = percentile_bootstrap_mean_ci(
                values,
                seed=args.bootstrap_seed + seed_offset,
                n_bootstrap=args.n_bootstrap,
            )
            seed_offset += 1
            matched_contrast_rows.append(
                {
                    "mechanism": mechanism,
                    "metric": metric,
                    "n_replicates": int(len(values)),
                    "mean": float(np.mean(values)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "paired_wilcoxon_p": paired_wilcoxon_p(values),
                }
            )
    matched_contrasts = pd.DataFrame(matched_contrast_rows)
    atomic_write_csv(matched_contrasts, out_dir / "ssi_br_at_median_selection_size_contrasts.csv")

    argmin_variability = (
        runs.groupby(["replicate", "mechanism", "completion"], as_index=False)
        .agg(
            threshold_mean=("stabl_min_threshold", "mean"),
            threshold_sd=("stabl_min_threshold", "std"),
            selected_mean=("stabl_min_n_selected", "mean"),
            selected_sd=("stabl_min_n_selected", "std"),
            estimated_fdp_mean=("stabl_min_estimated_fdp", "mean"),
            realized_fdp_mean=("stabl_min_realized_fdp", "mean"),
            calibration_gap_mean=("stabl_min_calibration_gap_realized_minus_estimated", "mean"),
        )
    )
    atomic_write_csv(argmin_variability, out_dir / "ssi_argmin_variability.csv")

    audit = pd.DataFrame(
        [
            {
                "score_files": len(score_files),
                "expected_score_files": 300,
                "path_rows": len(paths),
                "expected_path_rows": len(score_files) * len(thresholds),
                "matched_runs": len(matched),
                "expected_matched_runs": 10 * 2 * 5,
                "complete": bool(
                    len(score_files) == 300
                    and len(paths) == len(score_files) * len(thresholds)
                    and len(matched) == 100
                ),
            }
        ]
    )
    atomic_write_csv(audit, out_dir / "ssi_threshold_path_integrity_audit.csv")

    report_lines = [
        "# V10.1C SSI Proteomics threshold-path diagnosis",
        "",
        "This reanalysis uses saved V10 score arrays only. No completion, knockoff, or STABL fit was rerun.",
        "",
        "## Integrity",
        "",
        dataframe_to_markdown(audit, index=False),
        "",
        "## Original stabl_min contrasts",
        "",
        dataframe_to_markdown(contrast_frame, index=False),
        "",
        "## BR mean evaluated near the Median stabl_min selection size",
        "",
        dataframe_to_markdown(matched_contrasts, index=False),
        "",
        "The matched-size threshold is selected without using support labels: first minimize the absolute selected-count difference, then the threshold distance from Median's stabl_min threshold, then choose the higher threshold.",
    ]
    (out_dir / "v101_ssi_threshold_path_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"[V10.1C] read {len(score_files)} score files")
    print(f"[V10.1C] integrity complete={bool(audit.loc[0, 'complete'])}")
    print(f"[V10.1C] outputs written to {out_dir}")


if __name__ == "__main__":
    main()
