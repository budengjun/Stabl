#!/usr/bin/env python3
"""Experiment A: posterior-sampling completion versus median imputation.

The primary scientific question is whether posterior-sampling completion better
preserves complete-data structure and the original STABL biomarker-selection
behavior than deterministic median imputation. The primary selection rule is
STABL's published FDP+ argmin rule (``stabl_min``). Additional target-q or LCD
rules remain available only as optional diagnostics.

The experiment compares the untouched complete matrix, median completion, an
exact Gaussian conditional posterior sample, and the current BayesianRidge
IterativeImputer posterior approximation.

Every branch uses the identical preprocessing order:

    complete -> fit StandardScaler on completed X -> generate knockoff

The completed matrix is fixed across repeated knockoff draws within a replicate.
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
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_common import (
    GeneratorConfig,
    apply_mask,
    apply_selection_rule,
    atomic_write_csv,
    atomic_savez_compressed,
    complete_matrix,
    load_or_generate_knockoff,
    generate_sparse_outcome,
    generator_configs_to_json,
    imputation_recovery_metrics,
    load_generator_configs,
    make_missing_mask,
    pair_diagnostics,
    parse_csv_items,
    prepare_output_directory,
    selection_metrics,
    simulate_gaussian_covariates,
    standardize_completed_matrix,
    transform_gaussian_parameters_with_scaler,
    summarize_results,
    fit_stabl_scores,
    fit_lcd_statistics,
)


VALID_COMPLETIONS = {
    "oracle_complete",
    "median",
    "exact_gaussian_posterior",
    "bayesianridge_posterior",
}
VALID_MECHANISMS = {"MCAR", "MAR", "MNAR"}
VALID_RULES = {
    "stabl_min",
    "stabl_q",
    "stabl_knockoff_plus",
    "knockoff_plus",
    "lcd_knockoff_plus",
}

MECHANISM_SEED_INDEX = {"MCAR": 0, "MAR": 1, "MNAR": 2}
COMPLETION_SEED_INDEX = {
    "oracle_complete": 0,
    "median": 1,
    "exact_gaussian_posterior": 2,
    "bayesianridge_posterior": 3,
}
VALID_GENERATORS = {
    "gaussian_equicorrelated",
    "gaussian_equicorrelated_true_sigma",
    "gaussian_mvr",
    "official_plsko",
}


def validate_items(items: tuple[str, ...], allowed: set[str], label: str) -> tuple[str, ...]:
    invalid = sorted(set(items) - allowed)
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}. Allowed: {sorted(allowed)}")
    return items


def upsert(rows: list[dict[str, Any]], row: dict[str, Any], key_columns: tuple[str, ...]) -> None:
    key = tuple(row.get(column) for column in key_columns)
    rows[:] = [
        old for old in rows if tuple(old.get(column) for column in key_columns) != key
    ]
    rows.append(row)


def completed_keys(
    frame: pd.DataFrame,
    key_columns: tuple[str, ...],
) -> set[tuple[Any, ...]]:
    if frame.empty or not set(key_columns).issubset(frame.columns):
        return set()
    valid = frame[frame.get("error", pd.Series("", index=frame.index)).fillna("").eq("")]
    return set(valid.loc[:, key_columns].itertuples(index=False, name=None))


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out-dir", default="./experiment_A_oracle_completion")
    parser.add_argument("--n-replicates", type=int, default=100)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--p", type=int, default=200)
    parser.add_argument("--n-signal", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--rho", type=float, default=0.60)
    parser.add_argument(
        "--task", choices=("classification", "regression"), default="classification"
    )
    parser.add_argument("--signal-strength", type=float, default=1.0)
    parser.add_argument("--mechanisms", default="MCAR,MAR")
    parser.add_argument("--missing-rate", type=float, default=0.20)
    parser.add_argument("--mar-driver-fraction", type=float, default=0.05)
    parser.add_argument(
        "--completion-methods",
        default=(
            "oracle_complete,median,exact_gaussian_posterior,"
            "bayesianridge_posterior"
        ),
    )
    parser.add_argument("--generators", default="gaussian_equicorrelated,gaussian_equicorrelated_true_sigma")
    parser.add_argument("--generator-configs-json", default=None)
    parser.add_argument("--n-knockoff-draws", type=int, default=1)
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--stabl-grid-size", type=int, default=30)
    parser.add_argument("--stabl-c-min", type=float, default=0.01)
    parser.add_argument("--stabl-c-max", type=float, default=1.0)
    parser.add_argument("--stabl-alpha-min", type=float, default=1e-2)
    parser.add_argument("--stabl-alpha-max", type=float, default=1e2)
    parser.add_argument("--lcd-cv-folds", type=int, default=5)
    parser.add_argument("--lcd-grid-size", type=int, default=30)
    parser.add_argument("--lcd-c-min", type=float, default=1e-3)
    parser.add_argument("--lcd-c-max", type=float, default=10.0)
    parser.add_argument("--lcd-alpha-min", type=float, default=1e-4)
    parser.add_argument("--lcd-alpha-max", type=float, default=10.0)
    parser.add_argument("--lcd-n-jobs", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--target-fdr", type=float, default=0.10)
    parser.add_argument(
        "--selection-rules",
        default="stabl_min"
    )
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--iterative-max-iter", type=int, default=10)
    parser.add_argument("--iterative-nearest-features", type=int, default=50)
    parser.add_argument("--precision-ridge", type=float, default=1e-3)
    parser.add_argument("--skip-precision-recovery", action="store_true")
    parser.add_argument("--save-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pair-cache-dir", default=None)
    parser.add_argument("--refresh-pair-cache", action="store_true")
    parser.add_argument(
        "--skip-misspecified-true-sigma",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip true-Sigma generator combinations that are not theoretically "
            "aligned. With the default setting, true Sigma is used only for "
            "oracle_complete and exact_gaussian_posterior under MCAR/MAR."
        ),
    )
    parser.add_argument("--random-state", type=int, default=20260716)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--plsko-threshold-abs", type=float, default=None)
    parser.add_argument("--plsko-threshold-q", type=float, default=0.8)
    parser.add_argument("--plsko-ncomp", type=int, default=None)
    parser.add_argument("--plsko-sparsity", type=float, default=1.0)
    args = parser.parse_args()

    if args.n_replicates < 1 or args.n_knockoff_draws < 1:
        raise ValueError("Replicate and knockoff draw counts must be positive")
    if not 1 <= args.n_signal < args.p:
        raise ValueError("n_signal must lie between 1 and p minus 1")
    if not 0.0 < args.target_fdr < 1.0:
        raise ValueError("target_fdr must lie in (0, 1)")
    if not 0.0 <= args.missing_rate < 1.0:
        raise ValueError("missing_rate must lie in [0, 1)")
    if args.threshold_step <= 0 or args.threshold_min >= args.threshold_max:
        raise ValueError("Invalid STABL threshold grid")
    if args.stabl_grid_size < 2 or args.lcd_grid_size < 2:
        raise ValueError("STABL and LCD regularization grids need at least 2 values")
    if args.lcd_cv_folds < 2:
        raise ValueError("lcd_cv_folds must be at least 2")

    mechanisms = validate_items(
        tuple(item.upper() for item in parse_csv_items(args.mechanisms)),
        VALID_MECHANISMS,
        "mechanisms",
    )
    completions = validate_items(
        parse_csv_items(args.completion_methods), VALID_COMPLETIONS, "completion methods"
    )
    rules = validate_items(parse_csv_items(args.selection_rules), VALID_RULES, "selection rules")
    generators = validate_items(
        parse_csv_items(args.generators), VALID_GENERATORS, "generators"
    )
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

    config = {
        "experiment": "A_posterior_vs_median_original_stabl",
        "version": 5,
        **vars(args),
        "mechanisms_resolved": mechanisms,
        "completion_methods_resolved": completions,
        "selection_rules_resolved": rules,
        "generator_configs_resolved": generator_configs_to_json(generator_configs),
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
    pair_cache_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(exist_ok=True)
    if args.save_scores:
        score_dir.mkdir(exist_ok=True)

    selection_path = out_dir / "selection_results.csv"
    diagnostic_path = out_dir / "draw_diagnostics.csv"
    recovery_path = out_dir / "imputation_recovery.csv"
    selection_frame = (
        pd.read_csv(selection_path) if args.resume and selection_path.exists() else pd.DataFrame()
    )
    diagnostic_frame = (
        pd.read_csv(diagnostic_path) if args.resume and diagnostic_path.exists() else pd.DataFrame()
    )
    recovery_frame = (
        pd.read_csv(recovery_path) if args.resume and recovery_path.exists() else pd.DataFrame()
    )
    selection_rows = selection_frame.to_dict("records")
    diagnostic_rows = diagnostic_frame.to_dict("records")
    recovery_rows = recovery_frame.to_dict("records")

    selection_key = (
        "replicate",
        "mechanism",
        "completion",
        "generator_label",
        "draw",
        "selection_rule",
    )
    diagnostic_key = (
        "replicate",
        "mechanism",
        "completion",
        "generator_label",
        "draw",
    )
    recovery_key = ("replicate", "mechanism", "completion")
    done_selection = completed_keys(selection_frame, selection_key)
    done_diagnostic = completed_keys(diagnostic_frame, diagnostic_key)
    done_recovery = completed_keys(recovery_frame, recovery_key)

    for replicate in range(args.n_replicates):
        dataset_seed = args.random_state + replicate * 1_000_000
        X_complete, oracle_mean, oracle_covariance = simulate_gaussian_covariates(
            args.n,
            args.p,
            args.block_size,
            args.rho,
            dataset_seed + 11,
        )
        y, support, beta = generate_sparse_outcome(
            X_complete,
            n_signal=args.n_signal,
            task=args.task,
            signal_strength=args.signal_strength,
            seed=dataset_seed + 23,
        )
        atomic_savez_compressed(
            truth_dir / f"rep_{replicate:04d}.npz",
            X_complete=X_complete.to_numpy(dtype=float),
            y=y.to_numpy(),
            support=support,
            beta=beta,
            oracle_mean=oracle_mean,
            oracle_covariance=oracle_covariance,
            feature_names=np.asarray(X_complete.columns.astype(str)),
        )

        for mechanism in mechanisms:
            mechanism_index = MECHANISM_SEED_INDEX[mechanism]
            mask_seed = dataset_seed + 1000 + mechanism_index * 100
            mask = make_missing_mask(
                X_complete,
                y,
                mechanism,
                args.missing_rate,
                mask_seed,
                mar_driver_fraction=args.mar_driver_fraction,
            )
            X_missing = apply_mask(X_complete, mask)
            realized_missing_rate = float(mask.mean())

            for completion in completions:
                completion_index = COMPLETION_SEED_INDEX[completion]
                completion_seed = dataset_seed + 10_000 + mechanism_index * 1000 + completion_index * 100
                try:
                    X_completed = complete_matrix(
                        completion,  # type: ignore[arg-type]
                        X_complete=X_complete,
                        X_missing=X_missing,
                        seed=completion_seed,
                        oracle_mean=oracle_mean,
                        oracle_covariance=oracle_covariance,
                        iterative_max_iter=args.iterative_max_iter,
                        iterative_nearest_features=args.iterative_nearest_features,
                    )
                    X_scaled, scaler = standardize_completed_matrix(X_completed)
                    oracle_mean_scaled, oracle_covariance_scaled = (
                        transform_gaussian_parameters_with_scaler(
                            oracle_mean, oracle_covariance, scaler
                        )
                    )
                except Exception as exc:
                    message = repr(exc)
                    print(
                        f"[completion failed] rep={replicate} mechanism={mechanism} "
                        f"completion={completion}: {message}",
                        flush=True,
                    )
                    if args.fail_fast:
                        raise
                    for config_item in generator_configs:
                        for draw in range(args.n_knockoff_draws):
                            for rule in rules:
                                row = {
                                    "replicate": replicate,
                                    "mechanism": mechanism,
                                    "completion": completion,
                                    "generator_label": config_item.label,
                                    "generator": config_item.generator,
                                    "draw": draw,
                                    "selection_rule": rule,
                                    "error": message,
                                }
                                upsert(selection_rows, row, selection_key)
                    atomic_write_csv(pd.DataFrame(selection_rows), selection_path)
                    continue

                recovery_id = (replicate, mechanism, completion)
                if not args.resume or recovery_id not in done_recovery:
                    recovery_row: dict[str, Any] = {
                        "replicate": replicate,
                        "mechanism": mechanism,
                        "completion": completion,
                        "n": args.n,
                        "p": args.p,
                        "target_missing_rate": args.missing_rate,
                        "realized_missing_rate": realized_missing_rate,
                        "completion_seed": completion_seed,
                        "missingness_assumption_compatible": int(
                            mechanism in {"MCAR", "MAR"}
                        ),
                        "error": "",
                    }
                    try:
                        recovery_row.update(
                            imputation_recovery_metrics(
                                X_complete,
                                X_completed,
                                mask,
                                true_covariance=oracle_covariance,
                                precision_ridge=args.precision_ridge,
                                compute_precision=not args.skip_precision_recovery,
                            )
                        )
                    except Exception as exc:
                        recovery_row["error"] = repr(exc)
                        recovery_row["traceback"] = traceback.format_exc(limit=8)
                        if args.fail_fast:
                            raise
                    upsert(recovery_rows, recovery_row, recovery_key)
                    atomic_write_csv(pd.DataFrame(recovery_rows), recovery_path)

                for config_index, generator_config in enumerate(generator_configs):
                    uses_true_sigma_config = (
                        generator_config.generator
                        == "gaussian_equicorrelated_true_sigma"
                    )
                    true_sigma_aligned = (
                        completion == "oracle_complete"
                        or (
                            completion == "exact_gaussian_posterior"
                            and mechanism in {"MCAR", "MAR"}
                        )
                    )
                    if (
                        args.skip_misspecified_true_sigma
                        and uses_true_sigma_config
                        and not true_sigma_aligned
                    ):
                        print(
                            f"[skip incompatible] rep={replicate} mechanism={mechanism} "
                            f"completion={completion} generator={generator_config.label}",
                            flush=True,
                        )
                        continue
                    for draw in range(args.n_knockoff_draws):
                        draw_id = (
                            replicate,
                            mechanism,
                            completion,
                            generator_config.label,
                            draw,
                        )
                        expected_rule_ids = {
                            draw_id + (rule,) for rule in rules
                        }
                        if (
                            args.resume
                            and draw_id in done_diagnostic
                            and expected_rule_ids.issubset(done_selection)
                        ):
                            print(f"[skip] {draw_id}", flush=True)
                            continue

                        # Pair generator and STABL randomness across completion
                        # methods. Completion itself may be stochastic, but all
                        # downstream random seeds are held fixed within each
                        # replicate x mechanism x generator x draw contrast.
                        knockoff_seed = (
                            dataset_seed
                            + 100_000
                            + mechanism_index * 10_000
                            + config_index * 100
                            + draw
                        )
                        print(
                            f"[start] rep={replicate} mechanism={mechanism} "
                            f"completion={completion} generator={generator_config.label} "
                            f"draw={draw}",
                            flush=True,
                        )
                        diagnostic_row: dict[str, Any] = {
                            "replicate": replicate,
                            "mechanism": mechanism,
                            "completion": completion,
                            "generator_label": generator_config.label,
                            "generator": generator_config.generator,
                            "draw": draw,
                            "n": args.n,
                            "p": args.p,
                            "completion_seed": completion_seed,
                            "knockoff_seed": knockoff_seed,
                            "realized_missing_rate": realized_missing_rate,
                            "error": "",
                        }
                        try:
                            pair_cache_path = pair_cache_dir / (
                                f"rep_{replicate:04d}__{mechanism}__{completion}__"
                                f"{generator_config.label}__draw_{draw:03d}.npz"
                            )
                            uses_true_sigma = (
                                generator_config.generator
                                == "gaussian_equicorrelated_true_sigma"
                            )
                            pair, cache_hit = load_or_generate_knockoff(
                                X_scaled,
                                generator_config,
                                seed=knockoff_seed,
                                known_mean=(
                                    oracle_mean_scaled if uses_true_sigma else None
                                ),
                                known_covariance=(
                                    oracle_covariance_scaled if uses_true_sigma else None
                                ),
                                cache_path=pair_cache_path,
                                refresh=args.refresh_pair_cache,
                            )
                            diagnostic_row["pair_cache_hit"] = int(cache_hit)
                            diagnostic_row["pair_cache_path"] = str(pair_cache_path)
                            diagnostic_row["generation_seconds"] = pair.generation_seconds
                            diagnostic_row["uses_true_sigma"] = int(uses_true_sigma)
                            diagnostic_row["true_sigma_exactly_applicable"] = int(
                                uses_true_sigma
                                and (
                                    completion == "oracle_complete"
                                    or (
                                        completion == "exact_gaussian_posterior"
                                        and mechanism in {"MCAR", "MAR"}
                                    )
                                )
                            )
                            diagnostic_row[
                                "oracle_scaled_cov_condition_number"
                            ] = float(np.linalg.cond(oracle_covariance_scaled))
                            diagnostic_row.update(pair_diagnostics(pair.X, pair.X_tilde))
                            real_scores, knockoff_scores = fit_stabl_scores(
                                pair.X,
                                pair.X_tilde,
                                y.to_numpy(),
                                task=args.task,
                                groups=None,
                                n_bootstraps=args.n_bootstraps,
                                n_jobs=args.n_jobs,
                                random_state=knockoff_seed + 1,
                                threshold_grid=threshold_grid,
                                regularization_grid_size=args.stabl_grid_size,
                                classification_c_min=args.stabl_c_min,
                                classification_c_max=args.stabl_c_max,
                                regression_alpha_min=args.stabl_alpha_min,
                                regression_alpha_max=args.stabl_alpha_max,
                            )
                            lcd = None
                            if "lcd_knockoff_plus" in rules:
                                lcd = fit_lcd_statistics(
                                    pair.X,
                                    pair.X_tilde,
                                    y.to_numpy(),
                                    task=args.task,
                                    random_state=knockoff_seed + 2,
                                    cv_folds=args.lcd_cv_folds,
                                    grid_size=args.lcd_grid_size,
                                    classification_c_min=args.lcd_c_min,
                                    classification_c_max=args.lcd_c_max,
                                    regression_alpha_min=args.lcd_alpha_min,
                                    regression_alpha_max=args.lcd_alpha_max,
                                    n_jobs=args.lcd_n_jobs,
                                )
                                diagnostic_row["lcd_model"] = lcd.model_name
                                diagnostic_row["lcd_best_parameter"] = lcd.best_parameter
                                diagnostic_row["lcd_fit_seconds"] = lcd.fit_seconds
                                diagnostic_row["lcd_w_positive"] = int(np.sum(lcd.W > 0))
                                diagnostic_row["lcd_w_negative"] = int(np.sum(lcd.W < 0))

                            if args.save_scores:
                                payload = {
                                    "real_scores": real_scores,
                                    "knockoff_scores": knockoff_scores,
                                    "support": support,
                                    "feature_names": np.asarray(
                                        X_complete.columns.astype(str)
                                    ),
                                }
                                if lcd is not None:
                                    payload["lcd_W"] = lcd.W
                                    payload["lcd_best_parameter"] = np.asarray(
                                        lcd.best_parameter
                                    )
                                atomic_savez_compressed(
                                    score_dir
                                    / (
                                        f"rep_{replicate:04d}__{mechanism}__{completion}__"
                                        f"{generator_config.label}__draw_{draw:03d}.npz"
                                    ),
                                    **payload,
                                )

                            for rule in rules:
                                result = apply_selection_rule(
                                    rule,  # type: ignore[arg-type]
                                    real_scores,
                                    knockoff_scores,
                                    q=args.target_fdr,
                                    threshold_grid=threshold_grid,
                                    lcd_statistics=(None if lcd is None else lcd.W),
                                )
                                row = {
                                    "replicate": replicate,
                                    "mechanism": mechanism,
                                    "completion": completion,
                                    "generator_label": generator_config.label,
                                    "generator": generator_config.generator,
                                    "draw": draw,
                                    "selection_rule": rule,
                                    "threshold": result.threshold,
                                    "estimated_fdp": result.estimated_fdp,
                                    "n": args.n,
                                    "p": args.p,
                                    "n_signal": args.n_signal,
                                    "target_fdr": args.target_fdr,
                                    "primary_analysis": int(rule == "stabl_min"),
                                    "n_bootstraps": args.n_bootstraps,
                                    "stabl_grid_size": args.stabl_grid_size,
                                    "lcd_grid_size": args.lcd_grid_size,
                                    "lcd_cv_folds": args.lcd_cv_folds,
                                    "realized_missing_rate": realized_missing_rate,
                                    "error": "",
                                    **selection_metrics(
                                        result.selected,
                                        support,
                                        args.p,
                                        target_fdr=args.target_fdr,
                                    ),
                                }
                                upsert(selection_rows, row, selection_key)
                                print(
                                    f"[result] rule={rule} fdp={row['fdp']:.3f} "
                                    f"power={row['power']:.3f} selected={row['n_selected']}",
                                    flush=True,
                                )
                        except Exception as exc:
                            diagnostic_row["error"] = repr(exc)
                            diagnostic_row["traceback"] = traceback.format_exc(limit=8)
                            for rule in rules:
                                row = {
                                    "replicate": replicate,
                                    "mechanism": mechanism,
                                    "completion": completion,
                                    "generator_label": generator_config.label,
                                    "generator": generator_config.generator,
                                    "draw": draw,
                                    "selection_rule": rule,
                                    "error": repr(exc),
                                }
                                upsert(selection_rows, row, selection_key)
                            print(f"[failed] {exc!r}", flush=True)
                            if args.fail_fast:
                                raise

                        upsert(diagnostic_rows, diagnostic_row, diagnostic_key)
                        atomic_write_csv(pd.DataFrame(diagnostic_rows), diagnostic_path)
                        atomic_write_csv(pd.DataFrame(selection_rows), selection_path)

        selection_frame = pd.DataFrame(selection_rows)
        summary = summarize_results(
            selection_frame,
            group_columns=(
                "mechanism",
                "completion",
                "generator_label",
                "selection_rule",
            ),
            target_fdr=args.target_fdr,
        )
        if not summary.empty:
            atomic_write_csv(summary, out_dir / "summary.csv")

    print(f"\nExperiment A completed. Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
