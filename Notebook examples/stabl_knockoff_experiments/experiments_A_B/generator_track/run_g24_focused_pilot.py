#!/usr/bin/env python3
"""G2.4 focused Gaussian knockoff pilot on SSI CyTOF.

This experiment follows the G2.3 smoke screen. It intentionally compares only
three preregistered methods:

1. ``equicorr``: current Ledoit Wolf plus equicorrelated Gaussian baseline.
2. ``mvr``: the same default covariance estimate with an MVR S matrix.
3. ``factor_r20_mvr``: rank 20 factor covariance with an MVR S matrix.

Stage 1 uses outcome-free paired C2ST diagnostics over repeated knockoff draws.
Logistic convergence is tracked fold by fold and retried with the same solver
and a larger iteration budget. Stage 2 uses paired semi-synthetic outcomes,
paired knockoff seeds, and paired STABL seeds to compare empirical FDP, power,
ranking quality, matched-size power, and across-draw Jaccard stability.

A result can nominate a confirmation candidate, but it cannot establish exact
exchangeability or authorize a final replacement without fresh-seed holdout
confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

EXPERIMENTS_AB_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENTS_AB_DIR.parent
ROOT = EXPERIMENTS_AB_DIR
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from g22_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    bootstrap_mean_ci,
    downstream_gate_decision,
    load_json,
    max_scores,
    paired_summary,
    pairwise_jaccard,
    selected_set_metrics,
    stable_top_k,
    swap_pair_matrices,
    validity_gate_decision,
)
from g24_c2st import repeated_paired_c2st_tracked  # noqa: E402
from research_common import (  # noqa: E402
    GeneratorConfig,
    apply_selection_rule,
    cap_features_by_variance,
    fit_stabl_scores,
    generate_sparse_outcome,
    load_or_generate_knockoff,
    load_real_dataset,
    pair_diagnostics,
    prepare_output_directory,
    ranking_metrics,
    selected_indices_string,
    selection_metrics,
    standardize_completed_matrix,
)


def _resolve_project_path(value: str | Path | None, default_relative: str) -> Path:
    """Resolve user paths against the project root, never against cwd."""
    raw = Path(default_relative if value is None else value).expanduser()
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw
    return raw.resolve()


def _resolve_artifact_path(out_dir: Path, value: str | Path) -> Path:
    """Resolve portable result references stored relative to an output folder."""
    path = Path(value)
    return path if path.is_absolute() else out_dir / path


BASELINE_LABELS = ("equicorr", "mvr")
ALLOWED_GENERATORS = {
    "gaussian_equicorrelated",
    "gaussian_mvr",
    "gaussian_factor_mvr",
}


def _load_grid(path: str | Path) -> list[GeneratorConfig]:
    raw = load_json(path)
    if not isinstance(raw, list) or not raw:
        raise ValueError("G2.4 generator grid JSON must contain a nonempty list")
    configs = [GeneratorConfig.from_mapping(dict(item)) for item in raw]
    invalid = [item.generator for item in configs if item.generator not in ALLOWED_GENERATORS]
    if invalid:
        raise ValueError(
            "G2.4 accepts only equicorr, MVR, and factor-MVR generators: "
            f"{sorted(set(invalid))}"
        )
    labels = [item.label for item in configs]
    if len(labels) != len(set(labels)):
        raise ValueError("G2.4 generator grid contains duplicate labels")
    required = {"equicorr", "mvr", "factor_r20_mvr"}
    if set(labels) != required:
        raise ValueError(
            "G2.4 focused grid must contain exactly equicorr, mvr, and "
            f"factor_r20_mvr; received {labels}"
        )
    by_label = {item.label: item for item in configs}
    factor = by_label["factor_r20_mvr"]
    if factor.generator != "gaussian_factor_mvr" or factor.factor_rank != 20:
        raise ValueError("factor_r20_mvr must use gaussian_factor_mvr with factor_rank=20")
    return [by_label["equicorr"], by_label["mvr"], by_label["factor_r20_mvr"]]


def _read_existing(path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _completed_keys(
    rows: list[dict[str, Any]], columns: Sequence[str]
) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for row in rows:
        if str(row.get("error", "")):
            continue
        keys.add(tuple(row.get(column) for column in columns))
    return keys


def _c2st_pair(
    X: np.ndarray,
    X_tilde: np.ndarray,
    *,
    seed: int,
    repeats: int,
    folds: int,
    classifier: str,
    rf_estimators: int,
    initial_max_iter: int,
    retry_max_iter: int,
) -> dict[str, float | int]:
    marginal = repeated_paired_c2st_tracked(
        X,
        X_tilde,
        classifier_kind=classifier,
        repeats=repeats,
        n_splits=folds,
        seed=seed,
        rf_estimators=rf_estimators,
        initial_max_iter=initial_max_iter,
        retry_max_iter=retry_max_iter,
    )
    original, swapped = swap_pair_matrices(X, X_tilde, np.arange(X.shape[1]))
    global_swap = repeated_paired_c2st_tracked(
        original,
        swapped,
        classifier_kind=classifier,
        repeats=repeats,
        n_splits=folds,
        seed=seed + 10000,
        rf_estimators=rf_estimators,
        initial_max_iter=initial_max_iter,
        retry_max_iter=retry_max_iter,
    )
    return {
        "marginal_auc": marginal.auc_mean,
        "marginal_auc_sd": marginal.auc_sd,
        "marginal_auc_max_repeat": marginal.auc_max,
        "marginal_c2st_converged": int(marginal.converged),
        "marginal_c2st_warning_count": marginal.convergence_warning_count,
        "marginal_c2st_retry_count": marginal.retry_count,
        "marginal_c2st_persistent_folds": marginal.persistent_nonconverged_folds,
        "marginal_c2st_max_n_iter": marginal.max_n_iter_observed,
        "global_swap_auc": global_swap.auc_mean,
        "global_swap_auc_sd": global_swap.auc_sd,
        "global_swap_auc_max_repeat": global_swap.auc_max,
        "global_swap_c2st_converged": int(global_swap.converged),
        "global_swap_c2st_warning_count": global_swap.convergence_warning_count,
        "global_swap_c2st_retry_count": global_swap.retry_count,
        "global_swap_c2st_persistent_folds": global_swap.persistent_nonconverged_folds,
        "global_swap_c2st_max_n_iter": global_swap.max_n_iter_observed,
    }


def _validity_summary(frame: pd.DataFrame, gate: dict[str, float]) -> pd.DataFrame:
    valid = frame[frame["error"].fillna("").eq("")].copy()
    if valid.empty:
        return pd.DataFrame()
    summary = (
        valid.groupby("generator_label", dropna=False)
        .agg(
            n_draws=("draw", "size"),
            marginal_auc_mean=("marginal_auc", "mean"),
            marginal_auc_sd=("marginal_auc", "std"),
            marginal_auc_max=("marginal_auc", "max"),
            global_swap_auc_mean=("global_swap_auc", "mean"),
            global_swap_auc_sd=("global_swap_auc", "std"),
            global_swap_auc_max=("global_swap_auc", "max"),
            pair_corr_mean=("pair_corr_mean", "mean"),
            s_relative_mean=("s_relative_mean", "mean"),
            cov_kk_fro_relative_mean=("cov_kk_fro_relative", "mean"),
            cov_xk_offdiag_rmse_mean=("cov_xk_offdiag_rmse", "mean"),
            near_constant_knockoff_columns_max=("near_constant_knockoff_columns", "max"),
            generation_seconds_mean=("generation_seconds", "mean"),
            marginal_c2st_converged_min=("marginal_c2st_converged", "min"),
            global_swap_c2st_converged_min=("global_swap_c2st_converged", "min"),
            c2st_warning_count_sum=("marginal_c2st_warning_count", "sum"),
            global_c2st_warning_count_sum=("global_swap_c2st_warning_count", "sum"),
            c2st_retry_count_sum=("marginal_c2st_retry_count", "sum"),
            global_c2st_retry_count_sum=("global_swap_c2st_retry_count", "sum"),
            persistent_nonconverged_folds_sum=("marginal_c2st_persistent_folds", "sum"),
            global_persistent_nonconverged_folds_sum=("global_swap_c2st_persistent_folds", "sum"),
        )
        .reset_index()
    )
    base_decisions = [validity_gate_decision(row, gate) for _, row in summary.iterrows()]
    passes: list[int] = []
    reasons_text: list[str] = []
    for (_, row), decision in zip(summary.iterrows(), base_decisions):
        reasons = list(decision.reasons)
        converged = bool(
            int(row["marginal_c2st_converged_min"]) == 1
            and int(row["global_swap_c2st_converged_min"]) == 1
            and int(row["persistent_nonconverged_folds_sum"]) == 0
            and int(row["global_persistent_nonconverged_folds_sum"]) == 0
        )
        if not converged:
            reasons.append("c2st_nonconvergence")
        passes.append(int(decision.passed and converged))
        reasons_text.append(";".join(reasons))
    summary["validity_pass"] = passes
    summary["validity_failure_reasons"] = reasons_text
    summary["validity_score"] = (
        summary["marginal_auc_mean"]
        + summary["global_swap_auc_mean"]
        + 0.25 * summary["cov_kk_fro_relative_mean"]
        + 0.25 * summary["cov_xk_offdiag_rmse_mean"]
    )
    return summary.sort_values(
        ["validity_pass", "validity_score", "generation_seconds_mean", "generator_label"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def _choose_stage2(
    summary: pd.DataFrame,
    configs: list[GeneratorConfig],
    *,
    max_factor_candidates: int,
    diagnostic_top_k_if_none: int,
) -> tuple[list[GeneratorConfig], bool]:
    del max_factor_candidates, diagnostic_top_k_if_none
    factor_rows = summary[summary["generator_label"].eq("factor_r20_mvr")]
    factor_valid = bool(
        not factor_rows.empty and int(factor_rows.iloc[0]["validity_pass"]) == 1
    )
    # G2.4 is a preregistered focused pilot. All three methods continue to the
    # paired downstream stage even if a validity gate fails, so a failure is
    # measured rather than hidden by adaptive candidate selection.
    return list(configs), factor_valid


def _stage1(
    *,
    X: pd.DataFrame,
    configs: list[GeneratorConfig],
    gate: dict[str, float],
    args: argparse.Namespace,
    out_dir: Path,
    pair_cache_dir: Path,
) -> tuple[pd.DataFrame, list[GeneratorConfig], bool]:
    rows_path = out_dir / "g24_stage1_validity_runs.csv"
    rows = _read_existing(rows_path, args.resume)
    done = _completed_keys(rows, ("generator_label", "draw"))
    cache_dir = pair_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    for draw in range(args.validity_draws):
        pair_seed = args.random_state + 100000 + draw
        for config in configs:
            key = (config.label, draw)
            if key in done:
                continue
            row: dict[str, Any] = {
                "generator_label": config.label,
                "draw": draw,
                "seed": pair_seed,
                "error": "",
                **asdict(config),
            }
            print(f"[G2.4 Stage1] draw={draw} generator={config.label}", flush=True)
            try:
                pair, cache_hit = load_or_generate_knockoff(
                    X,
                    config,
                    seed=pair_seed,
                    cache_path=cache_dir / f"{config.label}__draw_{draw:03d}.npz",
                    refresh=args.refresh_pairs,
                )
                row["cache_hit"] = int(cache_hit)
                row["generation_seconds"] = pair.generation_seconds
                row.update(pair_diagnostics(pair.X, pair.X_tilde))
                row.update(
                    _c2st_pair(
                        pair.X,
                        pair.X_tilde,
                        seed=pair_seed + 50000,
                        repeats=args.c2st_repeats,
                        folds=args.c2st_folds,
                        classifier=args.c2st_classifier,
                        rf_estimators=args.rf_estimators,
                        initial_max_iter=args.c2st_initial_max_iter,
                        retry_max_iter=args.c2st_retry_max_iter,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()
                if args.fail_fast:
                    raise
            rows.append(row)
            atomic_write_csv(pd.DataFrame(rows), rows_path)

    frame = pd.DataFrame(rows)
    summary = _validity_summary(frame, gate)
    if summary.empty:
        raise RuntimeError("G2.4 Stage 1 produced no successful generator draws")
    atomic_write_csv(summary, out_dir / "g24_stage1_validity_summary.csv")

    selected, any_factor_valid = _choose_stage2(
        summary,
        configs,
        max_factor_candidates=args.max_stage2_factor_candidates,
        diagnostic_top_k_if_none=args.diagnostic_top_k_if_none,
    )
    atomic_write_json(
        {
            "any_factor_validity_pass": any_factor_valid,
            "stage2_mode": "focused_preregistered",
            "stage2_configs": [asdict(item) for item in selected],
            "gate": gate,
        },
        out_dir / "g24_stage1_selection.json",
    )
    return summary, selected, any_factor_valid


def _score_path(score_dir: Path, replicate: int, label: str, draw: int) -> Path:
    return score_dir / f"rep_{replicate:03d}__{label}__draw_{draw:03d}.npz"


def _stage2(
    *,
    X: pd.DataFrame,
    configs: list[GeneratorConfig],
    validity_summary: pd.DataFrame,
    any_factor_valid: bool,
    args: argparse.Namespace,
    out_dir: Path,
    pair_cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows_path = out_dir / "g24_stage2_stabl_runs.csv"
    rows = _read_existing(rows_path, args.resume)
    done = _completed_keys(rows, ("replicate", "generator_label", "draw"))
    pair_dir = pair_cache_dir
    score_dir = out_dir / "stage2_scores"
    truth_dir = out_dir / "stage2_truth"
    for directory in (pair_dir, score_dir, truth_dir):
        directory.mkdir(parents=True, exist_ok=True)

    threshold_grid = np.arange(
        args.threshold_min,
        args.threshold_max + 0.5 * args.threshold_step,
        args.threshold_step,
    )

    for replicate in range(args.screen_replicates):
        outcome_seed = args.random_state + 1000000 + replicate
        y, support, beta = generate_sparse_outcome(
            X,
            n_signal=args.n_signal,
            task="classification",
            signal_strength=args.signal_strength,
            seed=outcome_seed,
        )
        truth_path = truth_dir / f"rep_{replicate:03d}.npz"
        if not truth_path.exists() or args.refresh_scores:
            np.savez_compressed(
                truth_path,
                y=y.to_numpy(),
                support=support,
                beta=beta,
                feature_names=np.asarray(X.columns.astype(str)),
            )

        for draw in range(args.screen_draws):
            pair_seed = args.random_state + 2000000 + 1000 * replicate + draw
            stabl_seed = args.random_state + 3000000 + 1000 * replicate + draw
            for config in configs:
                key = (replicate, config.label, draw)
                if key in done:
                    continue
                row: dict[str, Any] = {
                    "replicate": replicate,
                    "draw": draw,
                    "generator_label": config.label,
                    "outcome_seed": outcome_seed,
                    "knockoff_seed": pair_seed,
                    "stabl_seed": stabl_seed,
                    "error": "",
                    **asdict(config),
                }
                print(
                    f"[G2.4 Stage2] rep={replicate} draw={draw} generator={config.label}",
                    flush=True,
                )
                try:
                    pair, cache_hit = load_or_generate_knockoff(
                        X,
                        config,
                        seed=pair_seed,
                        cache_path=pair_dir
                        / f"rep_{replicate:03d}__{config.label}__draw_{draw:03d}.npz",
                        refresh=args.refresh_pairs,
                    )
                    stabl_started = time.perf_counter()
                    real_scores, artificial_scores = fit_stabl_scores(
                        pair.X,
                        pair.X_tilde,
                        y.to_numpy(),
                        task="classification",
                        groups=None,
                        n_bootstraps=args.n_bootstraps,
                        n_jobs=args.n_jobs,
                        random_state=stabl_seed,
                        threshold_grid=threshold_grid,
                        regularization_grid_size=args.stabl_grid_size,
                        classification_c_min=args.stabl_c_min,
                        classification_c_max=args.stabl_c_max,
                    )
                    selection = apply_selection_rule(
                        "stabl_min",
                        real_scores,
                        artificial_scores,
                        q=args.target_fdr,
                        threshold_grid=threshold_grid,
                    )
                    row["cache_hit"] = int(cache_hit)
                    row["generation_seconds"] = pair.generation_seconds
                    row["stabl_seconds"] = float(time.perf_counter() - stabl_started)
                    row["threshold"] = selection.threshold
                    row["estimated_fdp"] = selection.estimated_fdp
                    row["selected_indices"] = selected_indices_string(selection.selected)
                    row.update(
                        selection_metrics(
                            selection.selected,
                            support,
                            X.shape[1],
                            target_fdr=args.target_fdr,
                        )
                    )
                    row.update(ranking_metrics(real_scores, support, X.shape[1]))
                    row.update(pair_diagnostics(pair.X, pair.X_tilde))
                    score_path = _score_path(score_dir, replicate, config.label, draw)
                    np.savez_compressed(
                        score_path,
                        real_scores=real_scores,
                        artificial_scores=artificial_scores,
                        support=support,
                        selected=selection.selected,
                        threshold=np.asarray(selection.threshold),
                        estimated_fdp=np.asarray(selection.estimated_fdp),
                    )
                    row["score_file"] = str(score_path.relative_to(out_dir))
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["traceback"] = traceback.format_exc()
                    if args.fail_fast:
                        raise
                rows.append(row)
                atomic_write_csv(pd.DataFrame(rows), rows_path)
                if not row.get("error"):
                    print(
                        "[G2.4 Stage2 complete] "
                        f"rep={replicate} draw={draw} generator={config.label} "
                        f"seconds={float(row.get('stabl_seconds', float('nan'))):.1f} "
                        f"fdp={float(row.get('fdp', float('nan'))):.3f} "
                        f"power={float(row.get('power', float('nan'))):.3f}",
                        flush=True,
                    )

    frame = pd.DataFrame(rows)
    valid = frame[frame["error"].fillna("").eq("")].copy()
    if valid.empty:
        raise RuntimeError("G2.4 Stage 2 produced no successful STABL runs")

    stability_rows: list[dict[str, Any]] = []
    for (replicate, label), block in valid.groupby(["replicate", "generator_label"]):
        selected_sets: list[np.ndarray] = []
        for path in block["score_file"]:
            artifact_path = _resolve_artifact_path(out_dir, path)
            with np.load(artifact_path, allow_pickle=False) as loaded:
                selected_sets.append(np.asarray(loaded["selected"], dtype=bool))
        stability_rows.append(
            {
                "replicate": int(replicate),
                "generator_label": str(label),
                "draw_jaccard": pairwise_jaccard(selected_sets),
                "n_draws": int(len(selected_sets)),
            }
        )
    stability = pd.DataFrame(stability_rows)
    atomic_write_csv(stability, out_dir / "g24_stage2_draw_stability.csv")

    replicate_metrics = (
        valid.groupby(["replicate", "generator_label"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
        .merge(stability, on=["replicate", "generator_label"], how="left")
    )
    atomic_write_csv(replicate_metrics, out_dir / "g24_stage2_replicate_metrics.csv")

    labels = sorted(valid["generator_label"].unique())
    by_key: dict[tuple[int, int, str], pd.Series] = {}
    for _, row in valid.iterrows():
        by_key[(int(row["replicate"]), int(row["draw"]), str(row["generator_label"]))] = row

    matched_rows: list[dict[str, Any]] = []
    for baseline in ("equicorr", "mvr"):
        if baseline not in labels:
            continue
        for candidate in labels:
            if candidate == baseline:
                continue
            for replicate in range(args.screen_replicates):
                for draw in range(args.screen_draws):
                    candidate_row = by_key.get((replicate, draw, candidate))
                    baseline_row = by_key.get((replicate, draw, baseline))
                    if candidate_row is None or baseline_row is None:
                        continue
                    candidate_score_path = _resolve_artifact_path(
                        out_dir, candidate_row["score_file"]
                    )
                    with np.load(candidate_score_path, allow_pickle=False) as loaded:
                        candidate_scores = max_scores(
                            np.asarray(loaded["real_scores"], dtype=float)
                        )
                        support = np.asarray(loaded["support"], dtype=int)
                    k = int(round(float(baseline_row["n_selected"])))
                    selected = stable_top_k(candidate_scores, k)
                    metrics = selected_set_metrics(selected, support)
                    matched_rows.append(
                        {
                            "replicate": replicate,
                            "draw": draw,
                            "candidate": candidate,
                            "baseline": baseline,
                            "matched_k": k,
                            **{f"matched_{key}": value for key, value in metrics.items()},
                        }
                    )
    matched = pd.DataFrame(matched_rows)
    atomic_write_csv(matched, out_dir / "g24_stage2_matched_size_runs.csv")
    matched_replicates = (
        matched.groupby(["replicate", "candidate", "baseline"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
        if not matched.empty
        else pd.DataFrame()
    )
    atomic_write_csv(
        matched_replicates, out_dir / "g24_stage2_matched_size_replicates.csv"
    )

    metrics = (
        "fdp",
        "power",
        "support_jaccard",
        "average_precision",
        "ranking_auroc",
        "n_selected",
        "threshold",
        "mean_null_score",
        "draw_jaccard",
    )
    contrast_rows: list[dict[str, Any]] = []
    for baseline in ("equicorr", "mvr"):
        if baseline not in labels:
            continue
        for candidate in labels:
            if candidate == baseline:
                continue
            for metric_index, metric in enumerate(metrics):
                contrast_rows.append(
                    paired_summary(
                        replicate_metrics,
                        candidate=candidate,
                        baseline=baseline,
                        metric=metric,
                        key_columns=("replicate",),
                        seed=args.random_state
                        + 100000 * (0 if baseline == "equicorr" else 1)
                        + 1000 * labels.index(candidate)
                        + metric_index,
                        bootstrap_samples=args.bootstrap_samples,
                    )
                )
            if not matched_replicates.empty:
                subset = matched_replicates[
                    matched_replicates["candidate"].eq(candidate)
                    & matched_replicates["baseline"].eq(baseline)
                ]
                baseline_power = replicate_metrics[
                    replicate_metrics["generator_label"].eq(baseline)
                ][["replicate", "power"]].rename(columns={"power": "baseline_power"})
                merged = subset.merge(baseline_power, on="replicate", how="inner")
                if not merged.empty:
                    delta = (
                        merged["matched_power"].to_numpy(dtype=float)
                        - merged["baseline_power"].to_numpy(dtype=float)
                    )
                    low, high = bootstrap_mean_ci(
                        delta,
                        seed=args.random_state + 880000 + len(contrast_rows),
                        samples=args.bootstrap_samples,
                    )
                    contrast_rows.append(
                        {
                            "candidate": candidate,
                            "baseline": baseline,
                            "metric": "matched_power",
                            "n_pairs": int(len(delta)),
                            "candidate_mean": float(merged["matched_power"].mean()),
                            "baseline_mean": float(merged["baseline_power"].mean()),
                            "delta_mean": float(np.mean(delta)),
                            "ci_low": low,
                            "ci_high": high,
                            "wilcoxon_p": float("nan"),
                        }
                    )
    contrasts = pd.DataFrame(contrast_rows)
    atomic_write_csv(contrasts, out_dir / "g24_stage2_paired_contrasts.csv")

    validity_by_label = validity_summary.set_index("generator_label").to_dict(
        orient="index"
    )

    def _deltas(candidate: str, baseline: str) -> dict[str, float]:
        subset = contrasts[
            contrasts["candidate"].eq(candidate)
            & contrasts["baseline"].eq(baseline)
        ]
        return {
            str(row["metric"]): float(row["delta_mean"])
            for _, row in subset.iterrows()
        }

    def _gate(delta: dict[str, float]):
        return downstream_gate_decision(
            {
                "fdp_delta": delta.get("fdp", math.inf),
                "average_precision_delta": delta.get("average_precision", -math.inf),
                "matched_power_delta": delta.get("matched_power", -math.inf),
                "draw_jaccard_delta": delta.get("draw_jaccard", -math.inf),
            },
            max_fdp_delta=args.max_fdp_delta,
            min_average_precision_delta=args.min_average_precision_delta,
            min_matched_power_delta=args.min_matched_power_delta,
            min_draw_jaccard_delta=args.min_draw_jaccard_delta,
        )

    decision_rows: list[dict[str, Any]] = []
    for candidate in ("mvr", "factor_r20_mvr"):
        delta_eq = _deltas(candidate, "equicorr")
        gate_eq = _gate(delta_eq)
        delta_mvr: dict[str, float] = {}
        gate_mvr = None
        if candidate == "factor_r20_mvr":
            delta_mvr = _deltas(candidate, "mvr")
            gate_mvr = _gate(delta_mvr)

        validity_pass = bool(
            validity_by_label.get(candidate, {}).get("validity_pass", 0)
        )
        overall = validity_pass and gate_eq.passed
        reasons: list[str] = []
        if not validity_pass:
            reasons.append("validity_gate")
        reasons.extend(f"vs_equicorr:{item}" for item in gate_eq.reasons)
        if gate_mvr is not None:
            overall = overall and gate_mvr.passed
            reasons.extend(f"vs_mvr:{item}" for item in gate_mvr.reasons)

        candidate_metrics = replicate_metrics[
            replicate_metrics["generator_label"].eq(candidate)
        ]
        row: dict[str, Any] = {
            "generator_label": candidate,
            "validity_pass": int(validity_pass),
            "downstream_pass_vs_equicorr": int(gate_eq.passed),
            "downstream_pass_vs_mvr": (
                int(gate_mvr.passed) if gate_mvr is not None else float("nan")
            ),
            "overall_pass": int(overall),
            "failure_reasons": ";".join(reasons),
            "empirical_fdp_mean": float(candidate_metrics["fdp"].mean()),
            "power_mean": float(candidate_metrics["power"].mean()),
            "average_precision_mean": float(
                candidate_metrics["average_precision"].mean()
            ),
            "draw_jaccard_mean": float(candidate_metrics["draw_jaccard"].mean()),
            "global_swap_auc_mean": validity_by_label.get(candidate, {}).get(
                "global_swap_auc_mean", float("nan")
            ),
            "marginal_auc_mean": validity_by_label.get(candidate, {}).get(
                "marginal_auc_mean", float("nan")
            ),
        }
        for metric in (
            "fdp",
            "power",
            "average_precision",
            "matched_power",
            "draw_jaccard",
        ):
            row[f"{metric}_delta_vs_equicorr"] = delta_eq.get(metric, float("nan"))
            row[f"{metric}_delta_vs_mvr"] = delta_mvr.get(metric, float("nan"))
        decision_rows.append(row)

    decisions = pd.DataFrame(decision_rows)
    atomic_write_csv(decisions, out_dir / "g24_candidate_decisions.csv")

    eligible = decisions[decisions["overall_pass"].eq(1)].copy()
    recommendation: dict[str, Any]
    if eligible.empty:
        recommendation = {
            "status": "no_candidate",
            "reason": (
                "Neither MVR nor factor-r20-MVR passed the preregistered validity "
                "and downstream gates. Factor-r20-MVR must pass against both "
                "equicorr and MVR."
            ),
            "holdout_required": True,
            "final_replacement_authorized": False,
        }
    else:
        eligible = eligible.sort_values(
            [
                "global_swap_auc_mean",
                "empirical_fdp_mean",
                "average_precision_mean",
                "power_mean",
                "draw_jaccard_mean",
            ],
            ascending=[True, True, False, False, False],
        )
        winner_label = str(eligible.iloc[0]["generator_label"])
        winner = next(item for item in configs if item.label == winner_label)
        recommendation = {
            "status": "confirmation_candidate",
            "selection_rule": (
                "validity and downstream noninferiority gates, with factor-r20-MVR "
                "required to pass against both equicorr and MVR; ties use lower "
                "global swap AUC, lower empirical FDP, then ranking and stability"
            ),
            "generator_config": asdict(winner),
            "decision_metrics": eligible.iloc[0].to_dict(),
            "holdout_required": True,
            "final_replacement_authorized": False,
        }
    atomic_write_json(recommendation, out_dir / "g24_recommendation.json")
    return valid, replicate_metrics, contrasts, decisions, recommendation


def run(args: argparse.Namespace) -> Path:
    if args.p < 2:
        raise ValueError("p must be at least 2")
    if args.validity_draws < 1 or args.screen_replicates < 1 or args.screen_draws < 1:
        raise ValueError("Validity draws, screen replicates, and screen draws must be positive")
    if args.c2st_repeats < 1 or args.c2st_folds < 2:
        raise ValueError("C2ST requires at least one repeat and two folds")
    if args.c2st_initial_max_iter < 1 or args.c2st_retry_max_iter < args.c2st_initial_max_iter:
        raise ValueError("C2ST retry max_iter must be at least the initial max_iter")
    if args.n_bootstraps < 1 or args.n_jobs < 1:
        raise ValueError("n_bootstraps and n_jobs must be positive")
    if not 0.0 < args.target_fdr < 1.0:
        raise ValueError("target_fdr must lie in (0, 1)")
    if args.threshold_step <= 0.0 or args.threshold_min >= args.threshold_max:
        raise ValueError("Invalid STABL threshold grid")
    if args.resume and (args.refresh_pairs or args.refresh_scores):
        raise ValueError(
            "Use a fresh output directory or --overwrite when refreshing pairs or scores"
        )
    generator_grid_path = _resolve_project_path(
        args.generator_grid_json,
        "experiments_A_B/generator_track/configs/g24_focused_grid.json",
    )
    c2st_gate_path = _resolve_project_path(
        args.c2st_gate_json,
        "experiments_A_B/generator_track/configs/g24_c2st_gate.json",
    )
    configs = _load_grid(generator_grid_path)
    gate = load_json(c2st_gate_path)
    calibrated_classifier = str(gate.get("classifier", args.c2st_classifier))
    if calibrated_classifier != args.c2st_classifier:
        raise ValueError(
            "The C2ST gate was calibrated for "
            f"{calibrated_classifier!r}, not {args.c2st_classifier!r}"
        )
    dataset = load_real_dataset(
        dataset="ssi",
        data_path=args.ssi_data_path,
        omic=args.ssi_omic,
        x_csv=None,
        groups_csv=None,
        subject_mode="first",
        complete_strategy=args.complete_strategy,
        seed=args.random_state,
    )
    X = cap_features_by_variance(dataset.X, args.p)
    X, _ = standardize_completed_matrix(X)
    if args.n_signal >= X.shape[1]:
        raise ValueError(
            f"n_signal={args.n_signal} must be smaller than actual p={X.shape[1]}"
        )

    out_dir_requested = _resolve_project_path(
        args.out_dir, "experiment_B/G24_ssi_focused_pilot"
    )
    pair_cache_root = _resolve_project_path(
        args.pair_cache_dir, f"pair_cache/{out_dir_requested.name}"
    )

    scientific_args = dict(vars(args))
    scientific_args.pop("out_dir", None)
    scientific_args.pop("pair_cache_dir", None)
    scientific_args.pop("generator_grid_json", None)
    scientific_args.pop("c2st_gate_json", None)
    config = {
        "experiment": "G2.4_focused_gaussian_pilot",
        "version": 1,
        **scientific_args,
        "dataset_label": dataset.dataset_label,
        "dataset_metadata": dataset.metadata,
        "n": int(X.shape[0]),
        "actual_p": int(X.shape[1]),
        "c2st_gate": gate,
        "generator_grid": [asdict(item) for item in configs],
    }
    out_dir = prepare_output_directory(
        out_dir_requested,
        config,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    config_path = out_dir / "config.json"
    recorded_config = load_json(config_path)
    recorded_config.update(
        {
            "out_dir": str(out_dir),
            "pair_cache_dir": str(pair_cache_root),
            "project_root": str(PROJECT_ROOT),
            "generator_grid_json": str(generator_grid_path),
            "c2st_gate_json": str(c2st_gate_path),
        }
    )
    atomic_write_json(recorded_config, config_path)

    validity_summary, stage2_configs, any_factor_valid = _stage1(
        X=X,
        configs=configs,
        gate=gate,
        args=args,
        out_dir=out_dir,
        pair_cache_dir=pair_cache_root / "stage1",
    )
    _, replicate_metrics, contrasts, decisions, recommendation = _stage2(
        X=X,
        configs=stage2_configs,
        validity_summary=validity_summary,
        any_factor_valid=any_factor_valid,
        args=args,
        out_dir=out_dir,
        pair_cache_dir=pair_cache_root / "stage2",
    )
    atomic_write_json(
        {
            "status": "completed",
            "out_dir": str(out_dir),
            "stage1_factor_r20_valid": any_factor_valid,
            "stage2_configs": [asdict(item) for item in stage2_configs],
            "recommendation_status": recommendation.get("status"),
            "n_replicate_metric_rows": int(len(replicate_metrics)),
            "n_contrast_rows": int(len(contrasts)),
            "n_decision_rows": int(len(decisions)),
        },
        out_dir / "g24_status.json",
    )
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ssi-data-path", required=True)
    parser.add_argument("--ssi-omic", default="CyTOF")
    parser.add_argument(
        "--complete-strategy",
        choices=("drop-columns", "median", "error"),
        default="drop-columns",
    )
    parser.add_argument("--generator-grid-json", default=None)
    parser.add_argument("--c2st-gate-json", default=None)
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Result directory. Relative paths are resolved from the project root, "
            "not from the current working directory."
        ),
    )
    parser.add_argument(
        "--pair-cache-dir",
        default=None,
        help=(
            "Knockoff pair cache directory. Relative paths are resolved from the "
            "project root. Defaults to pair_cache/<output-name>."
        ),
    )
    parser.add_argument("--p", type=int, default=100)
    parser.add_argument("--validity-draws", type=int, default=10)
    parser.add_argument("--c2st-repeats", type=int, default=5)
    parser.add_argument("--c2st-folds", type=int, default=5)
    parser.add_argument(
        "--c2st-classifier", choices=("logistic", "extra_trees"), default="logistic"
    )
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--c2st-initial-max-iter", type=int, default=10000)
    parser.add_argument("--c2st-retry-max-iter", type=int, default=1000000)
    parser.add_argument("--max-stage2-factor-candidates", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--diagnostic-top-k-if-none", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--screen-replicates", type=int, default=10)
    parser.add_argument("--screen-draws", type=int, default=5)
    parser.add_argument("--n-signal", type=int, default=15)
    parser.add_argument("--signal-strength", type=float, default=4.0)
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--stabl-grid-size", type=int, default=30)
    parser.add_argument("--stabl-c-min", type=float, default=0.01)
    parser.add_argument("--stabl-c-max", type=float, default=1.0)
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--target-fdr", type=float, default=0.10)
    parser.add_argument("--max-fdp-delta", type=float, default=0.05)
    parser.add_argument("--min-average-precision-delta", type=float, default=-0.02)
    parser.add_argument("--min-matched-power-delta", type=float, default=-0.02)
    parser.add_argument("--min-draw-jaccard-delta", type=float, default=-0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=20260803)
    parser.add_argument("--refresh-pairs", action="store_true")
    parser.add_argument("--refresh-scores", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = run(args)
    print(f"G2.4 completed: {out_dir}")


if __name__ == "__main__":
    main()
