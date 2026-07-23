#!/usr/bin/env python3
"""Experiment A V6: nested multiple-completion pooling for original STABL.

The experiment addresses the main limitation identified by V5: one posterior
completion draw faithfully restores the data distribution, but its Monte Carlo
variation may prevent that advantage from translating into stable biomarker
selection.  V6 generates a nested bank of component pipelines and pools their
full STABL selection-frequency paths before applying the published ``stabl_min``
rule once.

Primary pooled estimator
------------------------
For component draw m and regularization value lambda, let Pi_j^(m)(lambda) be
the STABL score of feature j.  For pool size M, V6 computes

    Pi_bar_j(lambda) = M^{-1} sum_{m=1}^M Pi_j^(m)(lambda)

for both real and artificial features, then applies the original STABL FDP+
argmin threshold to those pooled paths.

Important control
-----------------
Oracle-complete and median branches are also repeated across M independent
knockoff draws.  They therefore measure generic knockoff derandomization.  A
posterior-specific pooling benefit must exceed the corresponding median gain.
The default ``shared`` STABL-seed mode holds bootstrap subsamples fixed across
component draws so that M primarily averages completion and knockoff variation.
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

from posterior_pooling_common import (
    encode_indices,
    max_score_dispersion,
    mean_pairwise_jaccard,
    parse_aggregators,
    parse_pool_sizes,
    pool_score_paths,
    validate_v5_reuse_config,
)
from research_common import (
    apply_mask,
    apply_selection_rule,
    atomic_savez_compressed,
    atomic_write_csv,
    complete_matrix,
    generate_sparse_outcome,
    generator_configs_to_json,
    imputation_recovery_metrics,
    load_generator_configs,
    load_or_generate_knockoff,
    make_missing_mask,
    pair_diagnostics,
    parse_csv_items,
    prepare_output_directory,
    selection_metrics,
    simulate_gaussian_covariates,
    standardize_completed_matrix,
    transform_gaussian_parameters_with_scaler,
    fit_stabl_scores,
)

VALID_COMPLETIONS = {
    "oracle_complete",
    "median",
    "exact_gaussian_posterior",
    "bayesianridge_posterior",
}
VALID_MECHANISMS = {"MCAR", "MAR", "MNAR"}
VALID_GENERATORS = {
    "gaussian_equicorrelated",
    "gaussian_equicorrelated_true_sigma",
    "gaussian_mvr",
    "official_plsko",
}
MECHANISM_SEED_INDEX = {"MCAR": 0, "MAR": 1, "MNAR": 2}
COMPLETION_SEED_INDEX = {
    "oracle_complete": 0,
    "median": 1,
    "exact_gaussian_posterior": 2,
    "bayesianridge_posterior": 3,
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


def _load_score_file(
    path: Path,
    *,
    p: int,
    grid_size: int,
    support: np.ndarray,
    feature_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        real = np.asarray(payload["real_scores"], dtype=float)
        artificial = np.asarray(payload["knockoff_scores"], dtype=float)
        cached_support = np.asarray(payload["support"], dtype=int)
        cached_names = np.asarray(payload["feature_names"]).astype(str)
    if real.ndim == 3 and real.shape[0] == 1:
        real = real[0]
    if artificial.ndim == 3 and artificial.shape[0] == 1:
        artificial = artificial[0]
    expected_shape = (p, grid_size)
    if real.shape != expected_shape or artificial.shape != expected_shape:
        raise RuntimeError(
            f"Incompatible score shape in {path}: real={real.shape}, artificial={artificial.shape}, "
            f"expected={expected_shape}"
        )
    if not np.array_equal(cached_support, support):
        raise RuntimeError(f"Support mismatch in reusable score file: {path}")
    if not np.array_equal(cached_names, feature_names.astype(str)):
        raise RuntimeError(f"Feature-order mismatch in reusable score file: {path}")
    if not np.isfinite(real).all() or not np.isfinite(artificial).all():
        raise RuntimeError(f"Nonfinite reusable score file: {path}")
    return real, artificial


def _load_optional_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _source_score_name(
    replicate: int,
    mechanism: str,
    completion: str,
    generator_label: str,
    component_draw: int,
) -> str:
    return (
        f"rep_{replicate:04d}__{mechanism}__{completion}__"
        f"{generator_label}__draw_{component_draw:03d}.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out-dir", default="./experiment_A_posterior_pooling_v6")
    parser.add_argument("--n-replicates", type=int, default=20)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--p", type=int, default=200)
    parser.add_argument("--n-signal", type=int, default=25)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--rho", type=float, default=0.60)
    parser.add_argument("--task", choices=("classification", "regression"), default="classification")
    parser.add_argument("--signal-strength", type=float, default=2.0)
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
    parser.add_argument(
        "--generators",
        default="gaussian_equicorrelated",
        help="Use sample-Sigma equicorrelated as the primary practical generator.",
    )
    parser.add_argument("--generator-configs-json", default=None)
    parser.add_argument("--pool-sizes", default="1,5,10")
    parser.add_argument("--pooling-aggregators", default="mean")
    parser.add_argument(
        "--stabl-seed-mode",
        choices=("shared", "independent"),
        default="shared",
        help=(
            "shared holds bootstrap subsamples fixed across component draws; "
            "independent also averages bootstrap randomness."
        ),
    )
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--stabl-grid-size", type=int, default=30)
    parser.add_argument("--stabl-c-min", type=float, default=0.01)
    parser.add_argument("--stabl-c-max", type=float, default=1.0)
    parser.add_argument("--stabl-alpha-min", type=float, default=1e-2)
    parser.add_argument("--stabl-alpha-max", type=float, default=1e2)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--target-fdr", type=float, default=0.10)
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--iterative-max-iter", type=int, default=10)
    parser.add_argument("--iterative-nearest-features", type=int, default=50)
    parser.add_argument("--precision-ridge", type=float, default=1e-3)
    parser.add_argument("--skip-precision-recovery", action="store_true")
    parser.add_argument("--pair-cache-dir", default=None)
    parser.add_argument("--refresh-pair-cache", action="store_true")
    parser.add_argument(
        "--reuse-v5-out-dir",
        default=None,
        help=(
            "Optional V5 output directory. Compatible component-draw 0 score, "
            "diagnostic and recovery records are imported instead of refitted."
        ),
    )
    parser.add_argument(
        "--skip-misspecified-true-sigma",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-component-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-pooled-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-state", type=int, default=20260716)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--plsko-threshold-abs", type=float, default=None)
    parser.add_argument("--plsko-threshold-q", type=float, default=0.8)
    parser.add_argument("--plsko-ncomp", type=int, default=None)
    parser.add_argument("--plsko-sparsity", type=float, default=1.0)
    args = parser.parse_args()

    if args.n_replicates < 1 or not 1 <= args.n_signal < args.p:
        raise ValueError("Invalid replicate count or n_signal")
    if args.n_bootstraps < 1 or args.stabl_grid_size < 2:
        raise ValueError("Invalid STABL path settings")
    if args.threshold_step <= 0 or args.threshold_min >= args.threshold_max:
        raise ValueError("Invalid STABL threshold grid")
    if not 0.0 < args.target_fdr < 1.0:
        raise ValueError("target_fdr must lie in (0, 1)")

    mechanisms = validate_items(
        tuple(item.upper() for item in parse_csv_items(args.mechanisms)),
        VALID_MECHANISMS,
        "mechanisms",
    )
    completions = validate_items(
        parse_csv_items(args.completion_methods), VALID_COMPLETIONS, "completion methods"
    )
    generators = validate_items(
        parse_csv_items(args.generators), VALID_GENERATORS, "generators"
    )
    pool_sizes = parse_pool_sizes(args.pool_sizes)
    aggregators = parse_aggregators(args.pooling_aggregators)
    max_pool_size = max(pool_sizes)
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

    config: dict[str, Any] = {
        "experiment": "A_multiple_posterior_pooling_original_stabl",
        "version": 6,
        **vars(args),
        "mechanisms_resolved": mechanisms,
        "completion_methods_resolved": completions,
        "pool_sizes_resolved": pool_sizes,
        "pooling_aggregators_resolved": aggregators,
        "max_pool_size": max_pool_size,
        "selection_rules_resolved": ("stabl_min",),
        "generator_configs_resolved": generator_configs_to_json(generator_configs),
        "threshold_grid": threshold_grid.tolist(),
    }

    # Reuse location is an operational acceleration, not a scientific setting.
    reuse_source_value = config.pop("reuse_v5_out_dir", None)
    out_dir = prepare_output_directory(
        args.out_dir, config, resume=args.resume, overwrite=args.overwrite
    )
    config["reuse_v5_out_dir"] = reuse_source_value
    config_path = out_dir / "config.json"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    saved["reuse_v5_out_dir"] = reuse_source_value
    config_path.write_text(json.dumps(saved, indent=2, sort_keys=True, default=str), encoding="utf-8")

    truth_dir = out_dir / "truth"
    component_score_dir = out_dir / "component_scores"
    pooled_score_dir = out_dir / "pooled_scores"
    pair_cache_dir = (
        Path(args.pair_cache_dir).expanduser().resolve()
        if args.pair_cache_dir
        else out_dir / "pair_cache"
    )
    for path in (truth_dir, component_score_dir, pooled_score_dir, pair_cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    reuse_score_dir: Path | None = None
    reuse_diag = pd.DataFrame()
    reuse_recovery = pd.DataFrame()
    if args.reuse_v5_out_dir:
        reuse_score_dir = validate_v5_reuse_config(args.reuse_v5_out_dir, config)
        reuse_root = reuse_score_dir.parent
        reuse_diag = _load_optional_frame(reuse_root / "draw_diagnostics.csv")
        reuse_recovery = _load_optional_frame(reuse_root / "imputation_recovery.csv")
        print(f"[reuse] validated V5 source: {reuse_root}", flush=True)

    pooled_path = out_dir / "pooled_selection_results.csv"
    diag_path = out_dir / "component_draw_diagnostics.csv"
    recovery_path = out_dir / "component_imputation_recovery.csv"
    pooled_rows = _load_optional_frame(pooled_path).to_dict("records") if args.resume else []
    diag_rows = _load_optional_frame(diag_path).to_dict("records") if args.resume else []
    recovery_rows = _load_optional_frame(recovery_path).to_dict("records") if args.resume else []

    pooled_key = (
        "replicate", "mechanism", "completion", "generator_label",
        "pool_size", "pooling_aggregator", "selection_rule",
    )
    diag_key = (
        "replicate", "mechanism", "completion", "generator_label", "component_draw",
    )
    recovery_key = ("replicate", "mechanism", "completion", "component_draw")

    for replicate in range(args.n_replicates):
        dataset_seed = args.random_state + replicate * 1_000_000
        X_complete, oracle_mean, oracle_covariance = simulate_gaussian_covariates(
            args.n, args.p, args.block_size, args.rho, dataset_seed + 11
        )
        y, support, beta = generate_sparse_outcome(
            X_complete,
            n_signal=args.n_signal,
            task=args.task,
            signal_strength=args.signal_strength,
            seed=dataset_seed + 23,
        )
        feature_names = np.asarray(X_complete.columns.astype(str))
        atomic_savez_compressed(
            truth_dir / f"rep_{replicate:04d}.npz",
            X_complete=X_complete.to_numpy(dtype=float),
            y=y.to_numpy(),
            support=support,
            beta=beta,
            oracle_mean=oracle_mean,
            oracle_covariance=oracle_covariance,
            feature_names=feature_names,
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
                deterministic_completion = completion in {"oracle_complete", "median"}
                deterministic_payload: tuple[pd.DataFrame, pd.DataFrame, Any, np.ndarray, np.ndarray] | None = None

                for config_index, generator_config in enumerate(generator_configs):
                    uses_true_sigma = generator_config.generator == "gaussian_equicorrelated_true_sigma"
                    true_sigma_aligned = (
                        completion == "oracle_complete"
                        or (
                            completion == "exact_gaussian_posterior"
                            and mechanism in {"MCAR", "MAR"}
                        )
                    )
                    if args.skip_misspecified_true_sigma and uses_true_sigma and not true_sigma_aligned:
                        print(
                            f"[skip incompatible] rep={replicate} mechanism={mechanism} "
                            f"completion={completion} generator={generator_config.label}",
                            flush=True,
                        )
                        continue

                    real_components: list[np.ndarray] = []
                    artificial_components: list[np.ndarray] = []
                    component_masks: list[np.ndarray] = []

                    for component_draw in range(max_pool_size):
                        score_name = _source_score_name(
                            replicate, mechanism, completion, generator_config.label, component_draw
                        )
                        component_score_path = component_score_dir / score_name
                        completion_seed = (
                            dataset_seed
                            + 10_000
                            + mechanism_index * 1000
                            + completion_index * 100
                            + component_draw
                        )
                        knockoff_seed = (
                            dataset_seed
                            + 100_000
                            + mechanism_index * 10_000
                            + config_index * 100
                            + component_draw
                        )
                        shared_stabl_seed = (
                            dataset_seed
                            + 100_000
                            + mechanism_index * 10_000
                            + config_index * 100
                            + 1
                        )
                        stabl_seed = (
                            shared_stabl_seed
                            if args.stabl_seed_mode == "shared"
                            else knockoff_seed + 1
                        )

                        # Fast resume from V6 component cache.
                        if component_score_path.exists() and args.resume:
                            real_scores, artificial_scores = _load_score_file(
                                component_score_path,
                                p=args.p,
                                grid_size=args.stabl_grid_size,
                                support=support,
                                feature_names=feature_names,
                            )
                            source_label = "v6_component_cache"
                        else:
                            real_scores = artificial_scores = None
                            source_label = "computed"

                            # Import the exact V5 M=1 baseline when available.
                            if component_draw == 0 and reuse_score_dir is not None:
                                source_path = reuse_score_dir / score_name
                                if source_path.exists():
                                    real_scores, artificial_scores = _load_score_file(
                                        source_path,
                                        p=args.p,
                                        grid_size=args.stabl_grid_size,
                                        support=support,
                                        feature_names=feature_names,
                                    )
                                    source_label = "v5_score_reuse"
                                    if args.save_component_scores:
                                        atomic_savez_compressed(
                                            component_score_path,
                                            real_scores=real_scores,
                                            knockoff_scores=artificial_scores,
                                            support=support,
                                            feature_names=feature_names,
                                            component_draw=np.asarray(component_draw),
                                            completion_seed=np.asarray(completion_seed),
                                            knockoff_seed=np.asarray(knockoff_seed),
                                            stabl_seed=np.asarray(stabl_seed),
                                            source=np.asarray(source_label),
                                        )

                            if real_scores is None or artificial_scores is None:
                                try:
                                    if deterministic_completion and deterministic_payload is not None:
                                        X_completed, X_scaled, scaler, oracle_mean_scaled, oracle_covariance_scaled = deterministic_payload
                                    else:
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
                                        oracle_mean_scaled, oracle_covariance_scaled = transform_gaussian_parameters_with_scaler(
                                            oracle_mean, oracle_covariance, scaler
                                        )
                                        if deterministic_completion:
                                            deterministic_payload = (
                                                X_completed, X_scaled, scaler,
                                                oracle_mean_scaled, oracle_covariance_scaled,
                                            )

                                    recovery_row: dict[str, Any] = {
                                        "replicate": replicate,
                                        "mechanism": mechanism,
                                        "completion": completion,
                                        "component_draw": component_draw,
                                        "completion_seed": completion_seed,
                                        "n": args.n,
                                        "p": args.p,
                                        "target_missing_rate": args.missing_rate,
                                        "realized_missing_rate": realized_missing_rate,
                                        "deterministic_completion": int(deterministic_completion),
                                        "error": "",
                                    }
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
                                    upsert(recovery_rows, recovery_row, recovery_key)
                                    atomic_write_csv(pd.DataFrame(recovery_rows), recovery_path)

                                    pair_cache_path = pair_cache_dir / score_name
                                    pair, cache_hit = load_or_generate_knockoff(
                                        X_scaled,
                                        generator_config,
                                        seed=knockoff_seed,
                                        known_mean=oracle_mean_scaled if uses_true_sigma else None,
                                        known_covariance=oracle_covariance_scaled if uses_true_sigma else None,
                                        cache_path=pair_cache_path,
                                        refresh=args.refresh_pair_cache,
                                    )
                                    diag_row: dict[str, Any] = {
                                        "replicate": replicate,
                                        "mechanism": mechanism,
                                        "completion": completion,
                                        "generator_label": generator_config.label,
                                        "generator": generator_config.generator,
                                        "component_draw": component_draw,
                                        "completion_seed": completion_seed,
                                        "knockoff_seed": knockoff_seed,
                                        "stabl_seed": stabl_seed,
                                        "stabl_seed_mode": args.stabl_seed_mode,
                                        "pair_cache_hit": int(cache_hit),
                                        "pair_cache_path": str(pair_cache_path),
                                        "generation_seconds": pair.generation_seconds,
                                        "realized_missing_rate": realized_missing_rate,
                                        "score_source": source_label,
                                        "error": "",
                                    }
                                    diag_row.update(pair_diagnostics(pair.X, pair.X_tilde))
                                    real_scores, artificial_scores = fit_stabl_scores(
                                        pair.X,
                                        pair.X_tilde,
                                        y.to_numpy(),
                                        task=args.task,
                                        groups=None,
                                        n_bootstraps=args.n_bootstraps,
                                        n_jobs=args.n_jobs,
                                        random_state=stabl_seed,
                                        threshold_grid=threshold_grid,
                                        regularization_grid_size=args.stabl_grid_size,
                                        classification_c_min=args.stabl_c_min,
                                        classification_c_max=args.stabl_c_max,
                                        regression_alpha_min=args.stabl_alpha_min,
                                        regression_alpha_max=args.stabl_alpha_max,
                                    )
                                    upsert(diag_rows, diag_row, diag_key)
                                    atomic_write_csv(pd.DataFrame(diag_rows), diag_path)
                                    if args.save_component_scores:
                                        atomic_savez_compressed(
                                            component_score_path,
                                            real_scores=real_scores,
                                            knockoff_scores=artificial_scores,
                                            support=support,
                                            feature_names=feature_names,
                                            component_draw=np.asarray(component_draw),
                                            completion_seed=np.asarray(completion_seed),
                                            knockoff_seed=np.asarray(knockoff_seed),
                                            stabl_seed=np.asarray(stabl_seed),
                                            source=np.asarray(source_label),
                                        )
                                except Exception as exc:
                                    print(
                                        f"[failed] rep={replicate} mechanism={mechanism} "
                                        f"completion={completion} generator={generator_config.label} "
                                        f"component={component_draw}: {exc!r}",
                                        flush=True,
                                    )
                                    if args.fail_fast:
                                        raise
                                    error_row = {
                                        "replicate": replicate,
                                        "mechanism": mechanism,
                                        "completion": completion,
                                        "generator_label": generator_config.label,
                                        "generator": generator_config.generator,
                                        "component_draw": component_draw,
                                        "completion_seed": completion_seed,
                                        "knockoff_seed": knockoff_seed,
                                        "stabl_seed": stabl_seed,
                                        "error": repr(exc),
                                        "traceback": traceback.format_exc(limit=8),
                                    }
                                    upsert(diag_rows, error_row, diag_key)
                                    atomic_write_csv(pd.DataFrame(diag_rows), diag_path)
                                    break
                            else:
                                # Imported V5 records preserve the already-computed diagnostics.
                                source_diag = reuse_diag[
                                    (reuse_diag.get("replicate") == replicate)
                                    & (reuse_diag.get("mechanism") == mechanism)
                                    & (reuse_diag.get("completion") == completion)
                                    & (reuse_diag.get("generator_label") == generator_config.label)
                                    & (reuse_diag.get("draw") == 0)
                                ] if not reuse_diag.empty else pd.DataFrame()
                                diag_row = source_diag.iloc[0].to_dict() if not source_diag.empty else {}
                                diag_row.update({
                                    "replicate": replicate,
                                    "mechanism": mechanism,
                                    "completion": completion,
                                    "generator_label": generator_config.label,
                                    "generator": generator_config.generator,
                                    "component_draw": component_draw,
                                    "completion_seed": completion_seed,
                                    "knockoff_seed": knockoff_seed,
                                    "stabl_seed": stabl_seed,
                                    "stabl_seed_mode": args.stabl_seed_mode,
                                    "score_source": source_label,
                                    "error": "",
                                })
                                diag_row.pop("draw", None)
                                upsert(diag_rows, diag_row, diag_key)
                                atomic_write_csv(pd.DataFrame(diag_rows), diag_path)

                                source_rec = reuse_recovery[
                                    (reuse_recovery.get("replicate") == replicate)
                                    & (reuse_recovery.get("mechanism") == mechanism)
                                    & (reuse_recovery.get("completion") == completion)
                                ] if not reuse_recovery.empty else pd.DataFrame()
                                if not source_rec.empty:
                                    recovery_row = source_rec.iloc[0].to_dict()
                                    recovery_row.update({
                                        "replicate": replicate,
                                        "mechanism": mechanism,
                                        "completion": completion,
                                        "component_draw": component_draw,
                                        "completion_seed": completion_seed,
                                        "error": "",
                                    })
                                    upsert(recovery_rows, recovery_row, recovery_key)
                                    atomic_write_csv(pd.DataFrame(recovery_rows), recovery_path)

                        if real_scores is None or artificial_scores is None:
                            break
                        real_components.append(np.asarray(real_scores, dtype=float))
                        artificial_components.append(np.asarray(artificial_scores, dtype=float))
                        component_result = apply_selection_rule(
                            "stabl_min",
                            real_scores,
                            artificial_scores,
                            q=args.target_fdr,
                            threshold_grid=threshold_grid,
                        )
                        component_masks.append(component_result.selected)
                        print(
                            f"[component] rep={replicate} {mechanism} {completion} "
                            f"{generator_config.label} m={component_draw + 1}/{max_pool_size} "
                            f"source={source_label}",
                            flush=True,
                        )

                    if len(real_components) < max_pool_size:
                        continue

                    for pool_size in pool_sizes:
                        for aggregator in aggregators:
                            pooled_real = pool_score_paths(
                                real_components, pool_size=pool_size, aggregator=aggregator
                            )
                            pooled_artificial = pool_score_paths(
                                artificial_components, pool_size=pool_size, aggregator=aggregator
                            )
                            result = apply_selection_rule(
                                "stabl_min",
                                pooled_real,
                                pooled_artificial,
                                q=args.target_fdr,
                                threshold_grid=threshold_grid,
                            )
                            row: dict[str, Any] = {
                                "replicate": replicate,
                                "mechanism": mechanism,
                                "completion": completion,
                                "generator_label": generator_config.label,
                                "generator": generator_config.generator,
                                "pool_size": pool_size,
                                "pooling_aggregator": aggregator,
                                "selection_rule": "stabl_min",
                                "threshold": result.threshold,
                                "estimated_fdp": result.estimated_fdp,
                                "selected_indices": encode_indices(result.selected),
                                "n": args.n,
                                "p": args.p,
                                "n_signal": args.n_signal,
                                "target_fdr": args.target_fdr,
                                "n_bootstraps_per_component": args.n_bootstraps,
                                "stabl_grid_size": args.stabl_grid_size,
                                "stabl_seed_mode": args.stabl_seed_mode,
                                "nested_prefix_pool": 1,
                                "realized_missing_rate": realized_missing_rate,
                                "component_selection_jaccard_mean": mean_pairwise_jaccard(
                                    component_masks[:pool_size]
                                ),
                                "error": "",
                                **selection_metrics(
                                    result.selected,
                                    support,
                                    args.p,
                                    target_fdr=args.target_fdr,
                                ),
                            }
                            real_disp = max_score_dispersion(real_components, pool_size=pool_size)
                            art_disp = max_score_dispersion(artificial_components, pool_size=pool_size)
                            row.update({f"real_{key}": value for key, value in real_disp.items()})
                            row.update({f"artificial_{key}": value for key, value in art_disp.items()})
                            upsert(pooled_rows, row, pooled_key)
                            atomic_write_csv(pd.DataFrame(pooled_rows), pooled_path)
                            if args.save_pooled_scores:
                                atomic_savez_compressed(
                                    pooled_score_dir
                                    / (
                                        f"rep_{replicate:04d}__{mechanism}__{completion}__"
                                        f"{generator_config.label}__M_{pool_size:03d}__{aggregator}.npz"
                                    ),
                                    real_scores=pooled_real,
                                    knockoff_scores=pooled_artificial,
                                    selected=result.selected,
                                    support=support,
                                    feature_names=feature_names,
                                    pool_size=np.asarray(pool_size),
                                    pooling_aggregator=np.asarray(aggregator),
                                    threshold=np.asarray(result.threshold),
                                    estimated_fdp=np.asarray(result.estimated_fdp),
                                )
                            print(
                                f"[pooled] rep={replicate} {mechanism} {completion} "
                                f"{generator_config.label} M={pool_size} {aggregator} "
                                f"fdp={row['fdp']:.3f} power={row['power']:.3f} "
                                f"selected={row['n_selected']}",
                                flush=True,
                            )

        pooled_frame = pd.DataFrame(pooled_rows)
        if not pooled_frame.empty:
            summary = (
                pooled_frame[pooled_frame.get("error", "").fillna("").eq("")]
                .groupby(
                    [
                        "mechanism", "completion", "generator_label",
                        "pool_size", "pooling_aggregator", "selection_rule",
                    ],
                    as_index=False,
                )
                .agg(
                    n_success=("fdp", "size"),
                    empirical_fdr=("fdp", "mean"),
                    fdp_sd=("fdp", "std"),
                    power_mean=("power", "mean"),
                    power_sd=("power", "std"),
                    selected_mean=("n_selected", "mean"),
                    true_positives_mean=("true_positives", "mean"),
                    false_positives_mean=("false_positives", "mean"),
                    estimated_fdp_mean=("estimated_fdp", "mean"),
                    component_jaccard_mean=("component_selection_jaccard_mean", "mean"),
                )
            )
            atomic_write_csv(summary, out_dir / "summary.csv")

    print(f"\nExperiment A V6 completed. Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
