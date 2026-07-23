#!/usr/bin/env python3
"""G2.2B validity-first, STABL-aware PLSKO parameter screening.

Stage 1 audits the full PLSKO grid using marginal and global-swap C2ST before
any outcome is used. Only configurations passing the G2.2A calibrated gate are
eligible for confirmation. If no PLSKO configuration passes, the closest few
configurations may still enter Stage 2 as diagnostic-only candidates, but the
script will never freeze them as a replacement generator.

Stage 2 runs paired semi-synthetic STABL fits on SSI CyTOF. Candidate decisions
use a pre-registered gate on realized FDP, threshold-independent average
precision, exact matched-selection-size power, and across-draw Jaccard. Raw
power is considered only after those reliability gates pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from g22_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    downstream_gate_decision,
    load_json,
    max_scores,
    paired_summary,
    pairwise_jaccard,
    repeated_paired_c2st,
    selected_set_metrics,
    stable_top_k,
    swap_pair_matrices,
    validity_gate_decision,
)
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


def _load_grid(path: str | Path) -> list[GeneratorConfig]:
    raw = load_json(path)
    if not isinstance(raw, list) or not raw:
        raise ValueError("PLSKO grid JSON must contain a nonempty list")
    configs = [GeneratorConfig.from_mapping(dict(item)) for item in raw]
    if any(config.generator != "official_plsko" for config in configs):
        raise ValueError("G2.2B PLSKO grid may contain only official_plsko configs")
    labels = [config.label for config in configs]
    if len(labels) != len(set(labels)):
        raise ValueError("PLSKO grid contains duplicate labels")
    return configs


def _baselines() -> list[GeneratorConfig]:
    return [
        GeneratorConfig("equicorr", "gaussian_equicorrelated"),
        GeneratorConfig("mvr", "gaussian_mvr"),
    ]


def _c2st_pair(X: np.ndarray, X_tilde: np.ndarray, *, seed: int, repeats: int, folds: int) -> dict[str, float]:
    marginal = repeated_paired_c2st(
        X,
        X_tilde,
        classifier_kind="logistic",
        repeats=repeats,
        n_splits=folds,
        seed=seed,
    )
    original, swapped = swap_pair_matrices(X, X_tilde, np.arange(X.shape[1]))
    global_swap = repeated_paired_c2st(
        original,
        swapped,
        classifier_kind="logistic",
        repeats=repeats,
        n_splits=folds,
        seed=seed + 10000,
    )
    return {
        "marginal_auc": marginal.auc_mean,
        "marginal_auc_sd": marginal.auc_sd,
        "global_swap_auc": global_swap.auc_mean,
        "global_swap_auc_sd": global_swap.auc_sd,
    }


def _stage1_summary(frame: pd.DataFrame, gate: dict[str, float]) -> pd.DataFrame:
    valid = frame[frame["error"].fillna("").eq("")].copy()
    if valid.empty:
        return pd.DataFrame()
    grouped = valid.groupby("generator_label", dropna=False)
    summary = grouped.agg(
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
    ).reset_index()
    decisions = [validity_gate_decision(row, gate) for _, row in summary.iterrows()]
    summary["validity_pass"] = [int(item.passed) for item in decisions]
    summary["validity_failure_reasons"] = [";".join(item.reasons) for item in decisions]
    summary["validity_score"] = (
        summary["marginal_auc_mean"] + summary["global_swap_auc_mean"]
    )
    return summary.sort_values(
        ["validity_pass", "validity_score", "generator_label"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _select_stage2_configs(
    summary: pd.DataFrame,
    grid_by_label: dict[str, GeneratorConfig],
    *,
    max_configs: int,
    diagnostic_top_k_if_none: int,
) -> tuple[list[GeneratorConfig], bool]:
    plsko = summary[summary["generator_label"].isin(grid_by_label)].copy()
    passed = plsko[plsko["validity_pass"].eq(1)].sort_values("validity_score")
    if not passed.empty:
        labels = passed.head(max_configs)["generator_label"].tolist()
        return [grid_by_label[label] for label in labels], True
    labels = plsko.sort_values("validity_score").head(diagnostic_top_k_if_none)[
        "generator_label"
    ].tolist()
    return [grid_by_label[label] for label in labels], False


def _read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def _completed_keys(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for row in rows:
        if str(row.get("error", "")):
            continue
        keys.add(tuple(row.get(column) for column in columns))
    return keys


def _stage1(
    *,
    X: pd.DataFrame,
    configs: list[GeneratorConfig],
    args: argparse.Namespace,
    out_dir: Path,
    gate: dict[str, float],
) -> tuple[pd.DataFrame, list[GeneratorConfig], bool]:
    rows_path = out_dir / "g22b_stage1_validity_runs.csv"
    rows = _read_existing(rows_path) if args.resume else []
    done = _completed_keys(rows, ("generator_label", "draw"))
    cache_dir = out_dir / "stage1_pair_cache"
    cache_dir.mkdir(exist_ok=True)

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
            print(f"[G2.2B Stage1] draw={draw} generator={config.label}", flush=True)
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
    summary = _stage1_summary(frame, gate)
    atomic_write_csv(summary, out_dir / "g22b_stage1_validity_summary.csv")
    grid_by_label = {config.label: config for config in configs if config.generator == "official_plsko"}
    stage2_plsko, any_valid = _select_stage2_configs(
        summary,
        grid_by_label,
        max_configs=args.max_downstream_configs,
        diagnostic_top_k_if_none=args.diagnostic_top_k_if_none,
    )
    atomic_write_json(
        {
            "any_plsko_validity_pass": any_valid,
            "stage2_mode": "eligible" if any_valid else "diagnostic_only",
            "selected_plsko_configs": [asdict(item) for item in stage2_plsko],
            "gate": gate,
        },
        out_dir / "g22b_stage1_selection.json",
    )
    return summary, stage2_plsko, any_valid


def _stage2(
    *,
    X: pd.DataFrame,
    configs: list[GeneratorConfig],
    validity_summary: pd.DataFrame,
    any_valid: bool,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    results_path = out_dir / "g22b_stage2_stabl_runs.csv"
    rows = _read_existing(results_path) if args.resume else []
    done = _completed_keys(rows, ("replicate", "generator_label", "draw"))
    score_dir = out_dir / "stage2_scores"
    pair_dir = out_dir / "stage2_pair_cache"
    truth_dir = out_dir / "stage2_truth"
    for directory in (score_dir, pair_dir, truth_dir):
        directory.mkdir(exist_ok=True)

    threshold_grid = np.arange(args.threshold_min, args.threshold_max + 0.5 * args.threshold_step, args.threshold_step)

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
                    f"[G2.2B Stage2] rep={replicate} draw={draw} generator={config.label}",
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
                    score_path = score_dir / f"rep_{replicate:03d}__{config.label}__draw_{draw:03d}.npz"
                    np.savez_compressed(
                        score_path,
                        real_scores=real_scores,
                        artificial_scores=artificial_scores,
                        support=support,
                        selected=selection.selected,
                        threshold=np.asarray(selection.threshold),
                        estimated_fdp=np.asarray(selection.estimated_fdp),
                    )
                    row["score_file"] = str(score_path)
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["traceback"] = traceback.format_exc()
                    if args.fail_fast:
                        raise
                rows.append(row)
                atomic_write_csv(pd.DataFrame(rows), results_path)

    frame = pd.DataFrame(rows)
    valid = frame[frame["error"].fillna("").eq("")].copy()
    if valid.empty:
        raise RuntimeError("G2.2B Stage 2 produced no successful STABL runs")

    stability_rows: list[dict[str, Any]] = []
    for (replicate, generator), block in valid.groupby(["replicate", "generator_label"]):
        sets: list[np.ndarray] = []
        for path in block["score_file"]:
            with np.load(path, allow_pickle=False) as loaded:
                sets.append(np.asarray(loaded["selected"], dtype=bool))
        stability_rows.append(
            {
                "replicate": int(replicate),
                "generator_label": generator,
                "draw_jaccard": pairwise_jaccard(sets),
                "n_draws": int(len(sets)),
            }
        )
    stability = pd.DataFrame(stability_rows)
    atomic_write_csv(stability, out_dir / "g22b_stage2_draw_stability.csv")

    replicate_metrics = (
        valid.groupby(["replicate", "generator_label"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
        .merge(stability, on=["replicate", "generator_label"], how="left")
    )
    atomic_write_csv(replicate_metrics, out_dir / "g22b_stage2_replicate_metrics.csv")

    matched_rows: list[dict[str, Any]] = []
    baseline = "equicorr"
    by_key: dict[tuple[int, int, str], pd.Series] = {}
    for _, row in valid.iterrows():
        by_key[(int(row["replicate"]), int(row["draw"]), str(row["generator_label"]))] = row
    candidates = sorted(set(valid["generator_label"]) - {"equicorr", "mvr"})
    for candidate in candidates:
        for replicate in range(args.screen_replicates):
            for draw in range(args.screen_draws):
                candidate_row = by_key.get((replicate, draw, candidate))
                baseline_row = by_key.get((replicate, draw, baseline))
                if candidate_row is None or baseline_row is None:
                    continue
                with np.load(candidate_row["score_file"], allow_pickle=False) as loaded:
                    candidate_scores = max_scores(np.asarray(loaded["real_scores"], dtype=float))
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
    atomic_write_csv(matched, out_dir / "g22b_stage2_matched_size_runs.csv")
    matched_replicates = (
        matched.groupby(["replicate", "candidate", "baseline"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
        if not matched.empty
        else pd.DataFrame()
    )
    atomic_write_csv(matched_replicates, out_dir / "g22b_stage2_matched_size_replicates.csv")

    contrast_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for metric in (
            "fdp",
            "power",
            "support_jaccard",
            "average_precision",
            "ranking_auroc",
            "n_selected",
            "threshold",
            "mean_null_score",
            "draw_jaccard",
        ):
            contrast_rows.append(
                paired_summary(
                    replicate_metrics,
                    candidate=candidate,
                    baseline=baseline,
                    metric=metric,
                    key_columns=("replicate",),
                    seed=args.random_state + 1000 * candidates.index(candidate) + list((
                            "fdp",
                            "power",
                            "support_jaccard",
                            "average_precision",
                            "ranking_auroc",
                            "n_selected",
                            "threshold",
                            "mean_null_score",
                            "draw_jaccard",
                        )).index(metric),
                    bootstrap_samples=args.bootstrap_samples,
                )
            )
        if not matched_replicates.empty:
            subset = matched_replicates[matched_replicates["candidate"].eq(candidate)]
            baseline_power = replicate_metrics[
                replicate_metrics["generator_label"].eq(baseline)
            ][["replicate", "power"]].rename(columns={"power": "baseline_power"})
            merged = subset.merge(baseline_power, on="replicate", how="inner")
            if not merged.empty:
                delta = merged["matched_power"].to_numpy() - merged["baseline_power"].to_numpy()
                from g22_common import bootstrap_mean_ci

                low, high = bootstrap_mean_ci(
                    delta,
                    seed=args.random_state + 88000 + len(contrast_rows),
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
    atomic_write_csv(contrasts, out_dir / "g22b_stage2_paired_contrasts.csv")

    validity_by_label = validity_summary.set_index("generator_label").to_dict(orient="index")
    decisions: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = {
            row["metric"]: float(row["delta_mean"])
            for _, row in contrasts[contrasts["candidate"].eq(candidate)].iterrows()
        }
        validity_pass = bool(validity_by_label.get(candidate, {}).get("validity_pass", 0))
        downstream = downstream_gate_decision(
            {
                "fdp_delta": metrics.get("fdp", float("inf")),
                "average_precision_delta": metrics.get("average_precision", float("-inf")),
                "matched_power_delta": metrics.get("matched_power", float("-inf")),
                "draw_jaccard_delta": metrics.get("draw_jaccard", float("-inf")),
            },
            max_fdp_delta=args.max_fdp_delta,
            min_average_precision_delta=args.min_average_precision_delta,
            min_matched_power_delta=args.min_matched_power_delta,
            min_draw_jaccard_delta=args.min_draw_jaccard_delta,
        )
        reasons = []
        if not validity_pass:
            reasons.append("validity_gate")
        reasons.extend(downstream.reasons)
        decisions.append(
            {
                "generator_label": candidate,
                "validity_pass": int(validity_pass),
                "downstream_pass": int(downstream.passed),
                "overall_pass": int(validity_pass and downstream.passed),
                "failure_reasons": ";".join(reasons),
                "raw_power_delta": metrics.get("power", float("nan")),
                "raw_fdp_delta": metrics.get("fdp", float("nan")),
                "average_precision_delta": metrics.get("average_precision", float("nan")),
                "matched_power_delta": metrics.get("matched_power", float("nan")),
                "draw_jaccard_delta": metrics.get("draw_jaccard", float("nan")),
                "global_swap_auc_mean": validity_by_label.get(candidate, {}).get(
                    "global_swap_auc_mean", float("nan")
                ),
                "marginal_auc_mean": validity_by_label.get(candidate, {}).get(
                    "marginal_auc_mean", float("nan")
                ),
            }
        )
    decision_frame = pd.DataFrame(decisions)
    atomic_write_csv(decision_frame, out_dir / "g22b_candidate_decisions.csv")

    frozen: dict[str, Any] = {
        "status": "no_candidate",
        "reason": "No PLSKO configuration passed all validity and downstream gates",
        "stage2_mode": "eligible" if any_valid else "diagnostic_only",
    }
    passers = decision_frame[decision_frame["overall_pass"].eq(1)].copy()
    if not passers.empty:
        passers = passers.sort_values(
            ["raw_power_delta", "average_precision_delta", "global_swap_auc_mean"],
            ascending=[False, False, True],
        )
        winner_label = str(passers.iloc[0]["generator_label"])
        winner = next(config for config in configs if config.label == winner_label)
        frozen = {
            "status": "candidate_frozen",
            "selection_rule": "validity-first then reliability gates then raw power",
            "generator_config": asdict(winner),
            "decision_metrics": passers.iloc[0].to_dict(),
            "holdout_required": True,
        }
        atomic_write_json(frozen, out_dir / "frozen_plsko_stabl_candidate.json")
    else:
        atomic_write_json(frozen, out_dir / "g22b_no_candidate.json")

    return valid, replicate_metrics, contrasts, frozen


def run(args: argparse.Namespace) -> Path:
    grid = _load_grid(args.plsko_grid_json)
    gate = load_json(args.c2st_gate_json)
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

    config = {
        "experiment": "G2.2B_stabl_aware_plsko_screening",
        "version": 1,
        **vars(args),
        "dataset_label": dataset.dataset_label,
        "dataset_metadata": dataset.metadata,
        "n": int(X.shape[0]),
        "actual_p": int(X.shape[1]),
        "c2st_gate": gate,
        "plsk_grid": [asdict(item) for item in grid],
    }
    out_dir = prepare_output_directory(
        args.out_dir,
        config,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    stage1_configs = _baselines() + grid
    validity_summary, selected_plsko, any_valid = _stage1(
        X=X,
        configs=stage1_configs,
        args=args,
        out_dir=out_dir,
        gate=gate,
    )
    if not selected_plsko:
        atomic_write_json(
            {
                "status": "stopped_no_stage2_configs",
                "reason": "Stage 1 produced no PLSKO configurations for diagnostic screening",
            },
            out_dir / "g22b_status.json",
        )
        return out_dir

    stage2_configs = _baselines() + selected_plsko
    _, replicate_metrics, contrasts, frozen = _stage2(
        X=X,
        configs=stage2_configs,
        validity_summary=validity_summary,
        any_valid=any_valid,
        args=args,
        out_dir=out_dir,
    )
    status = {
        "status": "completed",
        "out_dir": str(out_dir),
        "stage1_any_valid_plsko": any_valid,
        "stage2_mode": "eligible" if any_valid else "diagnostic_only",
        "stage2_configs": [asdict(item) for item in stage2_configs],
        "candidate_status": frozen.get("status"),
        "n_replicate_metric_rows": int(len(replicate_metrics)),
        "n_contrast_rows": int(len(contrasts)),
    }
    atomic_write_json(status, out_dir / "g22b_status.json")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ssi-data-path", required=True)
    parser.add_argument("--ssi-omic", default="CyTOF")
    parser.add_argument("--complete-strategy", choices=("drop-columns", "median", "error"), default="drop-columns")
    parser.add_argument("--plsko-grid-json", required=True)
    parser.add_argument("--c2st-gate-json", required=True)
    parser.add_argument("--out-dir", default="generator_track_results/G22B_stabl_aware_screening")
    parser.add_argument("--p", type=int, default=100)
    parser.add_argument("--validity-draws", type=int, default=5)
    parser.add_argument("--c2st-repeats", type=int, default=3)
    parser.add_argument("--c2st-folds", type=int, default=5)
    parser.add_argument("--max-downstream-configs", type=int, default=6)
    parser.add_argument("--diagnostic-top-k-if-none", type=int, default=3)
    parser.add_argument("--screen-replicates", type=int, default=8)
    parser.add_argument("--screen-draws", type=int, default=2)
    parser.add_argument("--n-signal", type=int, default=15)
    parser.add_argument("--signal-strength", type=float, default=4.0)
    parser.add_argument("--n-bootstraps", type=int, default=50)
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
    parser.add_argument("--random-state", type=int, default=20260725)
    parser.add_argument("--refresh-pairs", action="store_true")
    parser.add_argument("--refresh-scores", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = run(args)
    print(f"G2.2B completed: {out_dir}")


if __name__ == "__main__":
    main()
