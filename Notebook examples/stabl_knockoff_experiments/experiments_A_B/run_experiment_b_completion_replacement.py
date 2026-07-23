#!/usr/bin/env python3
"""V9 Experiment B: real-X semisynthetic completion replacement benchmark.

This experiment keeps the original STABL procedure fixed and changes only the
missing-data completion method.  A complete reference matrix is obtained from a
real omics dataset by retaining features without native missingness.  Controlled
MCAR or MAR missingness is then injected, a sparse outcome is simulated from the
complete real-X reference, and ordinary STABL ``stabl_min`` is run after median imputation,
BayesianRidge conditional-mean imputation, or BayesianRidge posterior-sampling
imputation.

BayesianRidge mean and posterior completions are treated as ordinary STABL runs.
The mean completion is deterministic within a replicate/mechanism, whereas
posterior completion draws vary. Scores are never pooled and no new threshold
rule is introduced.
"""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "8")

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_common import (
    apply_mask,
    apply_selection_rule,
    atomic_savez_compressed,
    atomic_write_csv,
    cap_features_by_variance,
    complete_matrix,
    generate_sparse_outcome,
    generator_configs_to_json,
    imputation_recovery_metrics,
    load_generator_configs,
    load_or_generate_knockoff,
    load_real_dataset,
    make_missing_mask,
    pair_diagnostics,
    parse_csv_items,
    prepare_output_directory,
    selection_metrics,
    standardize_completed_matrix,
)

VALID_COMPLETIONS = {
    "oracle_complete",
    "median",
    "bayesianridge_mean",
    "bayesianridge_posterior",
}
VALID_GENERATORS = {
    "gaussian_equicorrelated",
    "gaussian_mvr",
    "official_plsko",
}


def parse_p_values(value: str) -> tuple[tuple[str, int | None], ...]:
    parsed: list[tuple[str, int | None]] = []
    for raw in parse_csv_items(value):
        if raw.lower() == "full":
            parsed.append(("full", None))
        else:
            p = int(raw)
            if p < 2:
                raise ValueError("Every p value must be at least 2")
            parsed.append((str(p), p))
    return tuple(parsed)


def validate_items(items: tuple[str, ...], allowed: set[str], label: str) -> tuple[str, ...]:
    invalid = sorted(set(items) - allowed)
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}. Allowed: {sorted(allowed)}")
    return items


def unique_feature_settings(
    X: pd.DataFrame,
    requested: tuple[tuple[str, int | None], ...],
) -> tuple[tuple[str, pd.DataFrame], ...]:
    settings: list[tuple[str, pd.DataFrame]] = []
    seen_actual_p: set[int] = set()
    for label, value in requested:
        subset = cap_features_by_variance(X, value)
        actual_p = int(subset.shape[1])
        if actual_p in seen_actual_p:
            print(
                f"[p setting skipped] requested={label} duplicates actual p={actual_p}",
                flush=True,
            )
            continue
        seen_actual_p.add(actual_p)
        settings.append((label, subset))
    if not settings:
        raise ValueError("No unique feature settings remain")
    return tuple(settings)



def completed_keys(frame: pd.DataFrame, key_columns: tuple[str, ...]) -> set[tuple[Any, ...]]:
    if frame.empty or not set(key_columns).issubset(frame.columns):
        return set()
    error_series = frame["error"] if "error" in frame.columns else pd.Series("", index=frame.index)
    valid = frame[error_series.fillna("").eq("")]
    return set(valid.loc[:, key_columns].itertuples(index=False, name=None))


def upsert(rows: list[dict[str, Any]], row: dict[str, Any], key_columns: tuple[str, ...]) -> None:
    key = tuple(row.get(column) for column in key_columns)
    rows[:] = [
        old for old in rows if tuple(old.get(column) for column in key_columns) != key
    ]
    rows.append(row)


def selected_indices_text(selected: np.ndarray) -> str:
    return ";".join(str(int(i)) for i in np.flatnonzero(selected))


def selected_names_text(selected: np.ndarray, columns: pd.Index) -> str:
    names = columns[np.asarray(selected, dtype=bool)].astype(str).tolist()
    return ";".join(names)


def completion_seed_for(
    random_state: int,
    p_index: int,
    replicate: int,
    mechanism_index: int,
    completion_draw: int,
) -> int:
    return (
        int(random_state)
        + p_index * 100_000_000
        + replicate * 1_000_000
        + mechanism_index * 100_000
        + completion_draw * 1_000
        + 17
    )


def downstream_seed_for(
    random_state: int,
    p_index: int,
    replicate: int,
    completion_draw: int,
    generator_index: int,
) -> int:
    """Common-random-number seed shared across mechanisms and completions."""
    return (
        int(random_state)
        + p_index * 100_000_000
        + replicate * 1_000_000
        + 500_000
        + completion_draw * 10_000
        + generator_index * 100
        + 31
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Real-X semisynthetic benchmark of median, BayesianRidge conditional "
            "mean, and BayesianRidge posterior completion under the original "
            "STABL stabl_min rule. It can also run oracle-only difficulty calibration."
        ),
    )
    parser.add_argument(
        "--experiment-mode",
        choices=("completion_replacement", "oracle_calibration", "mean_confirmation"),
        default="completion_replacement",
        help=(
            "completion_replacement compares imputation methods; oracle_calibration "
            "allows an oracle_complete-only run for difficulty calibration."
        ),
    )
    parser.add_argument("--dataset", choices=("ool", "ssi", "dream", "csv"), default="ssi")
    parser.add_argument("--data-path", default="../Sample Data/Biobank SSI")
    parser.add_argument("--omic", default="CyTOF")
    parser.add_argument("--x-csv", default=None)
    parser.add_argument("--groups-csv", default=None)
    parser.add_argument(
        "--subject-mode",
        choices=("first", "random", "all"),
        default="first",
        help="Use first for OOL primary analysis; all is a repeated-measures stress test.",
    )
    parser.add_argument(
        "--reference-complete-strategy",
        choices=("drop-columns", "error"),
        default="drop-columns",
        help=(
            "How to obtain the complete real-X reference before controlled missingness "
            "is injected. drop-columns retains only features with no native missingness."
        ),
    )
    parser.add_argument("--p-values", default="250,full")
    parser.add_argument("--out-dir", default="./experiment_B_completion_replacement_v9")
    parser.add_argument("--n-replicates", type=int, default=50)
    parser.add_argument("--n-completion-draws", type=int, default=5)
    parser.add_argument("--n-signal", type=int, default=15)
    parser.add_argument(
        "--task", choices=("classification", "regression"), default="classification"
    )
    parser.add_argument("--signal-strength", type=float, default=2.0)
    parser.add_argument("--mechanisms", default="MCAR,MAR")
    parser.add_argument("--missing-rate", type=float, default=0.20)
    parser.add_argument("--block-feature-blocks", type=int, default=5)
    parser.add_argument(
        "--completion-methods",
        default="oracle_complete,median,bayesianridge_mean",
    )
    parser.add_argument(
        "--generators",
        default="gaussian_equicorrelated",
    )
    parser.add_argument("--generator-configs-json", default=None)
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--stabl-grid-size", type=int, default=30)
    parser.add_argument("--stabl-c-min", type=float, default=0.01)
    parser.add_argument("--stabl-c-max", type=float, default=1.0)
    parser.add_argument("--stabl-alpha-min", type=float, default=1e-2)
    parser.add_argument("--stabl-alpha-max", type=float, default=1e2)
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--target-fdr", type=float, default=0.10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--iterative-max-iter", type=int, default=10)
    parser.add_argument("--iterative-nearest-features", type=int, default=50)
    parser.add_argument("--save-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pair-cache-dir", default=None)
    parser.add_argument("--refresh-pair-cache", action="store_true")
    parser.add_argument("--skip-precision-recovery", action="store_true")
    parser.add_argument("--random-state", type=int, default=20260718)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--plsko-threshold-abs", type=float, default=None)
    parser.add_argument("--plsko-threshold-q", type=float, default=0.8)
    parser.add_argument("--plsko-ncomp", type=int, default=None)
    parser.add_argument("--plsko-sparsity", type=float, default=1.0)
    args = parser.parse_args()

    if args.n_replicates < 1 or args.n_completion_draws < 1:
        raise ValueError("Replicate and completion-draw counts must be positive")
    if args.n_signal < 1:
        raise ValueError("n_signal must be positive")
    if not 0.0 < args.signal_strength:
        raise ValueError("signal_strength must be positive")
    if not 0.0 <= args.missing_rate < 1.0:
        raise ValueError("missing_rate must lie in [0, 1)")
    if args.block_feature_blocks < 1:
        raise ValueError("block_feature_blocks must be positive")
    if not 0.0 < args.target_fdr < 1.0:
        raise ValueError("target_fdr must lie in (0, 1)")
    if args.threshold_step <= 0 or args.threshold_min >= args.threshold_max:
        raise ValueError("Invalid STABL threshold grid")
    if args.stabl_grid_size < 2:
        raise ValueError("stabl_grid_size must be at least 2")

    mechanisms = tuple(item.upper() for item in parse_csv_items(args.mechanisms))
    invalid_mechanisms = sorted(set(mechanisms) - {"MCAR", "MAR", "MNAR", "BLOCK"})
    if invalid_mechanisms:
        raise ValueError(f"Unsupported mechanisms: {invalid_mechanisms}")
    completions = validate_items(
        parse_csv_items(args.completion_methods), VALID_COMPLETIONS, "completion methods"
    )
    if args.experiment_mode == "completion_replacement":
        if "median" not in completions:
            raise ValueError("Completion replacement requires median")
        if not ({"bayesianridge_mean", "bayesianridge_posterior"} & set(completions)):
            raise ValueError(
                "Completion replacement requires at least one model-based comparator"
            )
    elif args.experiment_mode == "mean_confirmation":
        required = {"oracle_complete", "median", "bayesianridge_mean"}
        if set(completions) != required:
            raise ValueError(
                "mean_confirmation requires exactly "
                "oracle_complete,median,bayesianridge_mean"
            )
    elif set(completions) != {"oracle_complete"}:
        raise ValueError(
            "oracle_calibration mode requires --completion-methods oracle_complete"
        )
    generators = validate_items(
        parse_csv_items(args.generators), VALID_GENERATORS, "generators"
    )
    requested_p_values = parse_p_values(args.p_values)
    generator_configs = load_generator_configs(
        config_json=args.generator_configs_json,
        generators=generators,
        plsko_threshold_abs=args.plsko_threshold_abs,
        plsko_threshold_q=args.plsko_threshold_q,
        plsko_ncomp=args.plsko_ncomp,
        plsko_sparsity=args.plsko_sparsity,
    )
    threshold_grid = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2.0,
        args.threshold_step,
    )

    dataset = load_real_dataset(
        dataset=args.dataset,
        data_path=args.data_path,
        omic=args.omic,
        x_csv=args.x_csv,
        groups_csv=args.groups_csv,
        subject_mode=args.subject_mode,
        complete_strategy=args.reference_complete_strategy,
        seed=args.random_state,
    )
    feature_settings = unique_feature_settings(dataset.X, requested_p_values)
    for requested_label, X_subset in feature_settings:
        print(
            f"[reference data] setting={requested_label} n={X_subset.shape[0]} "
            f"p={X_subset.shape[1]} native_missing=0",
            flush=True,
        )

    config = {
        "experiment": (
            "B_real_X_oracle_calibration"
            if args.experiment_mode == "oracle_calibration"
            else (
                "B_real_X_br_mean_confirmation"
                if args.experiment_mode == "mean_confirmation"
                else "B_real_X_completion_replacement"
            )
        ),
        "version": 9,
        **vars(args),
        "primary_selection_rule": "stabl_min",
        "completion_draw_interpretation": (
            "repeated ordinary STABL runs; BayesianRidge mean is fixed within "
            "replicate/mechanism; posterior draws vary; no score pooling"
        ),
        "dataset_label": dataset.dataset_label,
        "dataset_metadata": dataset.metadata,
        "mechanisms_resolved": mechanisms,
        "completion_methods_resolved": completions,
        "generator_configs_resolved": generator_configs_to_json(generator_configs),
        "p_values_resolved": [
            {"requested": label, "actual": int(X_subset.shape[1])}
            for label, X_subset in feature_settings
        ],
        "threshold_grid": threshold_grid.tolist(),
    }
    out_dir = prepare_output_directory(
        args.out_dir, config, resume=args.resume, overwrite=args.overwrite
    )
    truth_dir = out_dir / "truth"
    score_dir = out_dir / "scores"
    pair_cache_dir = (
        Path(args.pair_cache_dir).expanduser().resolve()
        if args.pair_cache_dir
        else out_dir / "pair_cache"
    )
    truth_dir.mkdir(parents=True, exist_ok=True)
    pair_cache_dir.mkdir(parents=True, exist_ok=True)
    if args.save_scores:
        score_dir.mkdir(parents=True, exist_ok=True)

    selection_path = out_dir / "selection_results.csv"
    diagnostic_path = out_dir / "draw_diagnostics.csv"
    recovery_path = out_dir / "imputation_recovery.csv"

    selection_frame = (
        pd.read_csv(selection_path)
        if args.resume and selection_path.exists()
        else pd.DataFrame()
    )
    diagnostic_frame = (
        pd.read_csv(diagnostic_path)
        if args.resume and diagnostic_path.exists()
        else pd.DataFrame()
    )
    recovery_frame = (
        pd.read_csv(recovery_path)
        if args.resume and recovery_path.exists()
        else pd.DataFrame()
    )

    selection_rows = selection_frame.to_dict("records")
    diagnostic_rows = diagnostic_frame.to_dict("records")
    recovery_rows = recovery_frame.to_dict("records")

    selection_key = (
        "p_setting",
        "actual_p",
        "replicate",
        "mechanism",
        "completion",
        "completion_draw",
        "generator_label",
    )
    diagnostic_key = selection_key
    recovery_key = (
        "p_setting",
        "actual_p",
        "replicate",
        "mechanism",
        "completion",
        "completion_draw",
    )
    done_selection = completed_keys(selection_frame, selection_key)
    done_diagnostics = completed_keys(diagnostic_frame, diagnostic_key)
    done_recovery = completed_keys(recovery_frame, recovery_key)

    for p_index, (p_label, X_reference) in enumerate(feature_settings):
        p = int(X_reference.shape[1])
        n = int(X_reference.shape[0])
        if args.n_signal >= p:
            raise ValueError(f"n_signal={args.n_signal} must be smaller than actual p={p}")

        X_reference_scaled, _ = standardize_completed_matrix(X_reference)
        group_array = (
            None
            if dataset.groups is None
            else dataset.groups.loc[X_reference.index].to_numpy()
        )

        for replicate in range(args.n_replicates):
            replicate_seed = args.random_state + p_index * 100_000_000 + replicate * 1_000_000
            y, support, beta = generate_sparse_outcome(
                X_reference_scaled,
                n_signal=args.n_signal,
                task=args.task,
                signal_strength=args.signal_strength,
                seed=replicate_seed + 7,
            )
            atomic_savez_compressed(
                truth_dir / f"p_{p:04d}__rep_{replicate:04d}.npz",
                y=y.to_numpy(),
                support=support,
                beta=beta,
                feature_names=np.asarray(X_reference.columns.astype(str)),
            )

            for mechanism_index, mechanism in enumerate(mechanisms):
                mask_seed = replicate_seed + mechanism_index * 100_000 + 101
                mask = make_missing_mask(
                    X_reference,
                    y,
                    mechanism,
                    args.missing_rate,
                    mask_seed,
                    block_feature_blocks=args.block_feature_blocks,
                )
                X_missing = apply_mask(X_reference, mask)
                actual_missing_rate = float(mask.mean())

                for completion_draw in range(args.n_completion_draws):
                    posterior_completion_seed = completion_seed_for(
                        args.random_state,
                        p_index,
                        replicate,
                        mechanism_index,
                        completion_draw,
                    )
                    fixed_completion_seed = completion_seed_for(
                        args.random_state,
                        p_index,
                        replicate,
                        mechanism_index,
                        0,
                    )

                    for completion in completions:
                        completion_seed = (
                            posterior_completion_seed
                            if completion == "bayesianridge_posterior"
                            else fixed_completion_seed
                        )
                        recovery_id = (
                            p_label,
                            p,
                            replicate,
                            mechanism,
                            completion,
                            completion_draw,
                        )
                        recovery_row: dict[str, Any] = {
                            "dataset": dataset.dataset_label,
                            "p_setting": p_label,
                            "actual_p": p,
                            "n": n,
                            "replicate": replicate,
                            "mechanism": mechanism,
                            "missing_rate_target": args.missing_rate,
                            "missing_rate_actual": actual_missing_rate,
                            "missing_rate_row_sd": float(mask.mean(axis=1).std()),
                            "missing_rate_column_sd": float(mask.mean(axis=0).std()),
                            "max_row_missing_fraction": float(mask.mean(axis=1).max()),
                            "max_column_missing_fraction": float(mask.mean(axis=0).max()),
                            "n_fully_observed_rows": int((~mask.any(axis=1)).sum()),
                            "n_fully_observed_columns": int((~mask.any(axis=0)).sum()),
                            "block_feature_blocks": int(args.block_feature_blocks),
                            "completion": completion,
                            "completion_draw": completion_draw,
                            "completion_seed": completion_seed,
                            "error": "",
                        }
                        try:
                            X_completed = complete_matrix(
                                completion,  # type: ignore[arg-type]
                                X_complete=X_reference,
                                X_missing=X_missing,
                                seed=completion_seed,
                                iterative_max_iter=args.iterative_max_iter,
                                iterative_nearest_features=args.iterative_nearest_features,
                            )
                            recovery_row.update(
                                imputation_recovery_metrics(
                                    X_reference,
                                    X_completed,
                                    mask,
                                    compute_precision=not args.skip_precision_recovery,
                                )
                            )
                            X_scaled, _ = standardize_completed_matrix(X_completed)
                        except Exception as exc:
                            recovery_row["error"] = repr(exc)
                            recovery_row["traceback"] = traceback.format_exc(limit=8)
                            upsert(recovery_rows, recovery_row, recovery_key)
                            atomic_write_csv(pd.DataFrame(recovery_rows), recovery_path)
                            print(
                                f"[completion failed] p={p} rep={replicate} mechanism={mechanism} "
                                f"completion={completion} draw={completion_draw}: {exc!r}",
                                flush=True,
                            )
                            if args.fail_fast:
                                raise
                            continue

                        if recovery_id not in done_recovery:
                            upsert(recovery_rows, recovery_row, recovery_key)
                            atomic_write_csv(pd.DataFrame(recovery_rows), recovery_path)

                        for generator_index, generator_config in enumerate(generator_configs):
                            result_id = (
                                p_label,
                                p,
                                replicate,
                                mechanism,
                                completion,
                                completion_draw,
                                generator_config.label,
                            )
                            if (
                                args.resume
                                and result_id in done_selection
                                and result_id in done_diagnostics
                            ):
                                print(f"[skip] {result_id}", flush=True)
                                continue

                            downstream_seed = downstream_seed_for(
                                args.random_state,
                                p_index,
                                replicate,
                                completion_draw,
                                generator_index,
                            )
                            pair_cache_path = pair_cache_dir / (
                                f"{dataset.dataset_label}__p_{p:04d}__rep_{replicate:04d}__"
                                f"{mechanism}__{completion}__cdraw_{completion_draw:02d}__"
                                f"{generator_config.label}.npz"
                            )
                            diagnostic_row: dict[str, Any] = {
                                "dataset": dataset.dataset_label,
                                "p_setting": p_label,
                                "actual_p": p,
                                "n": n,
                                "replicate": replicate,
                                "mechanism": mechanism,
                                "completion": completion,
                                "completion_draw": completion_draw,
                                "generator_label": generator_config.label,
                                "generator": generator_config.generator,
                                "completion_seed": completion_seed,
                                "knockoff_seed": downstream_seed,
                                "stabl_seed": downstream_seed + 1,
                                "pair_cache_path": str(pair_cache_path),
                                "error": "",
                            }
                            try:
                                print(
                                    f"[start] p={p} rep={replicate} mechanism={mechanism} "
                                    f"completion={completion} cdraw={completion_draw} "
                                    f"generator={generator_config.label}",
                                    flush=True,
                                )
                                pair, cache_hit = load_or_generate_knockoff(
                                    X_scaled,
                                    generator_config,
                                    seed=downstream_seed,
                                    cache_path=pair_cache_path,
                                    refresh=args.refresh_pair_cache,
                                )
                                diagnostic_row["pair_cache_hit"] = int(cache_hit)
                                diagnostic_row["generation_seconds"] = pair.generation_seconds
                                diagnostic_row.update(pair_diagnostics(pair.X, pair.X_tilde))

                                from research_common import fit_stabl_scores

                                real_scores, knockoff_scores = fit_stabl_scores(
                                    pair.X,
                                    pair.X_tilde,
                                    y.to_numpy(),
                                    task=args.task,
                                    groups=group_array,
                                    n_bootstraps=args.n_bootstraps,
                                    n_jobs=args.n_jobs,
                                    random_state=downstream_seed + 1,
                                    threshold_grid=threshold_grid,
                                    regularization_grid_size=args.stabl_grid_size,
                                    classification_c_min=args.stabl_c_min,
                                    classification_c_max=args.stabl_c_max,
                                    regression_alpha_min=args.stabl_alpha_min,
                                    regression_alpha_max=args.stabl_alpha_max,
                                )
                                selected = apply_selection_rule(
                                    "stabl_min",
                                    real_scores,
                                    knockoff_scores,
                                    q=args.target_fdr,
                                    threshold_grid=threshold_grid,
                                )
                                metrics = selection_metrics(
                                    selected.selected,
                                    support,
                                    p,
                                    target_fdr=args.target_fdr,
                                )
                                row = {
                                    "dataset": dataset.dataset_label,
                                    "p_setting": p_label,
                                    "actual_p": p,
                                    "n": n,
                                    "replicate": replicate,
                                    "mechanism": mechanism,
                                    "completion": completion,
                                    "completion_draw": completion_draw,
                                    "generator_label": generator_config.label,
                                    "generator": generator_config.generator,
                                    "selection_rule": "stabl_min",
                                    "threshold": selected.threshold,
                                    "estimated_fdp": selected.estimated_fdp,
                                    "n_signal": args.n_signal,
                                    "signal_strength": args.signal_strength,
                                    "target_fdr": args.target_fdr,
                                    "n_bootstraps": args.n_bootstraps,
                                    "stabl_grid_size": args.stabl_grid_size,
                                    "selected_indices": selected_indices_text(selected.selected),
                                    "selected_features": selected_names_text(
                                        selected.selected, X_reference.columns
                                    ),
                                    "error": "",
                                    **metrics,
                                }
                                if args.save_scores:
                                    atomic_savez_compressed(
                                        score_dir / (
                                            f"p_{p:04d}__rep_{replicate:04d}__{mechanism}__"
                                            f"{completion}__cdraw_{completion_draw:02d}__"
                                            f"{generator_config.label}.npz"
                                        ),
                                        real_scores=real_scores,
                                        knockoff_scores=knockoff_scores,
                                        selected=np.asarray(selected.selected, dtype=np.uint8),
                                        support=support,
                                        feature_names=np.asarray(X_reference.columns.astype(str)),
                                        completion_seed=np.asarray(completion_seed),
                                        knockoff_seed=np.asarray(downstream_seed),
                                        stabl_seed=np.asarray(downstream_seed + 1),
                                    )
                                upsert(selection_rows, row, selection_key)
                                print(
                                    f"[result] fdp={row['fdp']:.3f} power={row['power']:.3f} "
                                    f"selected={row['n_selected']}",
                                    flush=True,
                                )
                            except Exception as exc:
                                diagnostic_row["error"] = repr(exc)
                                diagnostic_row["traceback"] = traceback.format_exc(limit=8)
                                row = {
                                    "dataset": dataset.dataset_label,
                                    "p_setting": p_label,
                                    "actual_p": p,
                                    "n": n,
                                    "replicate": replicate,
                                    "mechanism": mechanism,
                                    "completion": completion,
                                    "completion_draw": completion_draw,
                                    "generator_label": generator_config.label,
                                    "generator": generator_config.generator,
                                    "selection_rule": "stabl_min",
                                    "error": repr(exc),
                                }
                                upsert(selection_rows, row, selection_key)
                                print(f"[failed] {exc!r}", flush=True)
                                if args.fail_fast:
                                    raise

                            upsert(diagnostic_rows, diagnostic_row, diagnostic_key)
                            atomic_write_csv(pd.DataFrame(selection_rows), selection_path)
                            atomic_write_csv(pd.DataFrame(diagnostic_rows), diagnostic_path)

    print(f"\nV9 experiment completed. Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
