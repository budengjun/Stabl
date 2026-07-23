#!/usr/bin/env python3
"""V10.1A: DREAM BayesianRidge max_iter convergence guard.

This prospective guard is run before the 50-replicate DREAM confirmation. It
uses fresh simulation seeds and pairs the outcome, support, missingness mask,
completion seed, knockoff seed, and STABL seed across max_iter budgets.
Completion and knockoff geometry are the primary diagnostics. Downstream
selection results are directional only because the default guard uses five
replicates.
"""

from __future__ import annotations

import os
for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "8")

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_common import (
    GeneratorConfig,
    apply_mask,
    apply_selection_rule,
    atomic_savez_compressed,
    atomic_write_csv,
    cap_features_by_variance,
    fit_stabl_scores,
    generate_sparse_outcome,
    imputation_recovery_metrics,
    load_or_generate_knockoff,
    load_real_dataset,
    make_missing_mask,
    pair_diagnostics,
    prepare_output_directory,
    selection_metrics,
    standardize_completed_matrix,
)
from v10_common import validate_mechanisms
from v91_common import fit_bayesianridge_mean_with_diagnostics


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not items or any(item < 1 for item in items):
        raise ValueError("max-iters must contain positive integers")
    return items


def completed_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> set[tuple[Any, ...]]:
    if frame.empty or not set(columns).issubset(frame.columns):
        return set()
    error = frame["error"] if "error" in frame.columns else pd.Series("", index=frame.index)
    valid = frame[error.fillna("").eq("")]
    return set(valid.loc[:, columns].itertuples(index=False, name=None))


def upsert(rows: list[dict[str, Any]], row: dict[str, Any], columns: tuple[str, ...]) -> None:
    key = tuple(row.get(column) for column in columns)
    rows[:] = [old for old in rows if tuple(old.get(column) for column in columns) != key]
    rows.append(row)


def selected_indices_text(selected: np.ndarray) -> str:
    return ";".join(str(int(index)) for index in np.flatnonzero(selected))


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    union = int(np.sum(a | b))
    return 1.0 if union == 0 else float(np.sum(a & b) / union)


def downstream_seed_for(
    random_state: int,
    replicate: int,
    mechanism_index: int,
    downstream_draw: int,
) -> int:
    return (
        int(random_state)
        + replicate * 1_000_000
        + mechanism_index * 100_000
        + 50_000
        + downstream_draw * 1_000
        + 31
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="V10.1A DREAM max_iter convergence guard.",
    )
    parser.add_argument("--data-path", default="../Sample Data/Dream")
    parser.add_argument("--out-dir", default="./experiment_B/v101_dream_max_iter_guard")
    parser.add_argument("--pair-cache-dir", default="./pair_cache/experiment_B_v101_guard")
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--n-downstream-draws", type=int, default=3)
    parser.add_argument("--max-iters", default="10,25,50")
    parser.add_argument("--mechanisms", default="MCAR,MAR")
    parser.add_argument("--missing-rate", type=float, default=0.20)
    parser.add_argument("--iterative-nearest-features", type=int, default=50)
    parser.add_argument("--iterative-tol", type=float, default=1e-3)
    parser.add_argument("--n-bootstraps", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=40260730)
    parser.add_argument("--save-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-downstream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-precision-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summarize", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both")
    if args.n_replicates < 1:
        raise ValueError("n-replicates must be positive")
    if args.n_downstream_draws < 2:
        raise ValueError("Use at least two downstream draws")
    if args.run_downstream and args.n_bootstraps < 20:
        raise ValueError("Use at least 20 STABL bootstraps")
    if not 0 < args.missing_rate < 1:
        raise ValueError("missing-rate must lie in (0, 1)")
    if args.iterative_tol <= 0:
        raise ValueError("iterative-tol must be positive")

    max_iters = parse_int_list(args.max_iters)
    mechanisms = validate_mechanisms(
        item.strip() for item in args.mechanisms.split(",") if item.strip()
    )
    if any(mechanism not in {"MCAR", "MAR"} for mechanism in mechanisms):
        raise ValueError("V10.1A guard is preregistered for MCAR and MAR only")

    config: dict[str, Any] = {
        "experiment": "V10.1A DREAM max_iter convergence guard",
        "version": "10.1A",
        "status": "prospective_directional_guard",
        "dataset": "DREAM Phylotype first specimen per participant",
        "p": 100,
        "n_signal": 15,
        "signal_strength": 4.0,
        "task": "classification",
        "mechanisms": list(mechanisms),
        "missing_rate": args.missing_rate,
        "completion": "bayesianridge_mean",
        "max_iters": list(max_iters),
        "generator": "gaussian_equicorrelated",
        "selection_rule": "stabl_min",
        "primary_diagnostics": [
            "masked_rmse", "sample_cov_fro_relative", "sample_corr_fro_relative",
            "masked_imputed_variance_ratio_median", "pair_corr_mean", "s_relative_mean",
            "cov_kk_fro_relative", "cov_xk_offdiag_rmse",
        ],
        "downstream_status": "exploratory_directional_only",
        "falsifiable_prediction": (
            "If additional iterations cause monotonic conditional-mean over-smoothing, "
            "imputed variance should decline, pair correlation should rise, s_relative "
            "should fall, and conditioning may worsen as max_iter increases."
        ),
        **vars(args),
    }
    config["max_iters"] = list(max_iters)
    config["mechanisms"] = list(mechanisms)

    out_dir = prepare_output_directory(
        args.out_dir, config, resume=args.resume, overwrite=args.overwrite
    )
    pair_cache_dir = Path(args.pair_cache_dir).expanduser().resolve()
    pair_cache_dir.mkdir(parents=True, exist_ok=True)
    truth_dir = out_dir / "truth"
    score_dir = out_dir / "scores"
    truth_dir.mkdir(exist_ok=True)
    if args.save_scores and args.run_downstream:
        score_dir.mkdir(exist_ok=True)

    completion_path = out_dir / "completion_diagnostics.csv"
    knockoff_path = out_dir / "knockoff_diagnostics.csv"
    selection_path = out_dir / "selection_results.csv"

    completion_frame = pd.read_csv(completion_path) if args.resume and completion_path.exists() else pd.DataFrame()
    knockoff_frame = pd.read_csv(knockoff_path) if args.resume and knockoff_path.exists() else pd.DataFrame()
    selection_frame = pd.read_csv(selection_path) if args.resume and selection_path.exists() else pd.DataFrame()

    completion_rows = completion_frame.to_dict("records")
    knockoff_rows = knockoff_frame.to_dict("records")
    selection_rows = selection_frame.to_dict("records")

    completion_key = ("replicate", "mechanism", "max_iter")
    result_key = ("replicate", "mechanism", "variant", "max_iter", "downstream_draw")
    done_completion = completed_keys(completion_frame, completion_key)
    done_knockoff = completed_keys(knockoff_frame, result_key)
    done_selection = completed_keys(selection_frame, result_key)

    dataset = load_real_dataset(
        dataset="dream",
        data_path=args.data_path,
        omic="Phylotype",
        x_csv=None,
        groups_csv=None,
        subject_mode="first",
        complete_strategy="drop-columns",
        seed=args.random_state,
    )
    X_reference = cap_features_by_variance(dataset.X, 100)
    if X_reference.shape[1] != 100:
        raise RuntimeError(f"Expected 100 features, obtained {X_reference.shape[1]}")
    X_reference_scaled, _ = standardize_completed_matrix(X_reference)
    n, p = X_reference.shape
    generator = GeneratorConfig(label="equicorr", generator="gaussian_equicorrelated")
    threshold_grid = np.arange(0.10, 0.99 + 0.005, 0.01)

    print(
        f"[V10.1A reference] n={n} p={p} reps={args.n_replicates} "
        f"mechanisms={mechanisms} max_iters={max_iters} downstream={args.run_downstream}",
        flush=True,
    )

    for replicate in range(args.n_replicates):
        replicate_seed = args.random_state + replicate * 1_000_000
        y, support, beta = generate_sparse_outcome(
            X_reference_scaled,
            n_signal=15,
            task="classification",
            signal_strength=4.0,
            seed=replicate_seed + 7,
        )
        atomic_savez_compressed(
            truth_dir / f"p_{p:04d}__rep_{replicate:04d}.npz",
            y=y.to_numpy(), support=support, beta=beta,
            feature_names=np.asarray(X_reference.columns.astype(str)),
        )

        for mechanism_index, mechanism in enumerate(mechanisms):
            mask_seed = replicate_seed + 101 + mechanism_index * 10_000
            mask = make_missing_mask(
                X_reference, y, mechanism, args.missing_rate, mask_seed
            )
            X_missing = apply_mask(X_reference, mask)
            completion_seed = replicate_seed + 17 + mechanism_index * 10_000

            scaled_by_iter: dict[int, pd.DataFrame] = {}
            for max_iter in max_iters:
                completion_id = (replicate, mechanism, max_iter)
                try:
                    fitted = fit_bayesianridge_mean_with_diagnostics(
                        X_reference,
                        X_missing,
                        mask,
                        seed=completion_seed,
                        max_iter=max_iter,
                        nearest_features=args.iterative_nearest_features,
                        tolerance=args.iterative_tol,
                    )
                    X_scaled, _ = standardize_completed_matrix(fitted.completed)
                    scaled_by_iter[max_iter] = X_scaled
                    recovery = imputation_recovery_metrics(
                        X_reference,
                        fitted.completed,
                        mask,
                        compute_precision=not args.skip_precision_recovery,
                    )
                    row = {
                        "dataset": dataset.dataset_label,
                        "n": n, "p": p, "replicate": replicate,
                        "mechanism": mechanism,
                        "completion": "bayesianridge_mean",
                        "max_iter": max_iter,
                        "mask_seed": mask_seed,
                        "completion_seed": completion_seed,
                        "missing_rate_actual": float(mask.mean()),
                        "n_iter": fitted.n_iter,
                        "convergence_warning_count": fitted.convergence_warning_count,
                        "converged_without_warning": int(fitted.converged_without_warning),
                        "masked_imputed_variance_ratio_median": fitted.masked_imputed_variance_ratio_median,
                        "masked_imputed_variance_ratio_mean": fitted.masked_imputed_variance_ratio_mean,
                        "completed_variance_ratio_median": fitted.completed_variance_ratio_median,
                        "completed_variance_ratio_mean": fitted.completed_variance_ratio_mean,
                        "corr_condition_number_effective": fitted.corr_condition_number_effective,
                        "corr_condition_number_ridge_1e3": fitted.corr_condition_number_ridge_1e3,
                        "corr_min_eigenvalue": fitted.corr_min_eigenvalue,
                        "corr_min_positive_eigenvalue": fitted.corr_min_positive_eigenvalue,
                        "corr_max_eigenvalue": fitted.corr_max_eigenvalue,
                        "error": "",
                        **recovery,
                    }
                except Exception as exc:
                    row = {
                        "dataset": dataset.dataset_label,
                        "n": n, "p": p, "replicate": replicate,
                        "mechanism": mechanism,
                        "completion": "bayesianridge_mean",
                        "max_iter": max_iter,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=10),
                    }
                    print(
                        f"[completion failed] rep={replicate} mechanism={mechanism} "
                        f"max_iter={max_iter}: {exc!r}", flush=True,
                    )
                    if args.fail_fast:
                        raise
                if completion_id not in done_completion or row.get("error"):
                    upsert(completion_rows, row, completion_key)
                    atomic_write_csv(pd.DataFrame(completion_rows), completion_path)

            if len(scaled_by_iter) != len(max_iters):
                continue

            variants: list[tuple[str, int, pd.DataFrame]] = [
                ("oracle_complete", 0, X_reference_scaled),
                *[("bayesianridge_mean", value, scaled_by_iter[value]) for value in max_iters],
            ]
            for downstream_draw in range(args.n_downstream_draws):
                downstream_seed = downstream_seed_for(
                    args.random_state, replicate, mechanism_index, downstream_draw
                )
                oracle_selected: np.ndarray | None = None
                if args.run_downstream:
                    for existing in selection_rows:
                        if (
                            int(existing.get("replicate", -1)) == replicate
                            and str(existing.get("mechanism", "")) == mechanism
                            and str(existing.get("variant", "")) == "oracle_complete"
                            and int(existing.get("max_iter", 0)) == 0
                            and int(existing.get("downstream_draw", -1)) == downstream_draw
                            and not str(existing.get("error", "")).strip()
                        ):
                            oracle_selected = np.zeros(p, dtype=bool)
                            raw = str(existing.get("selected_indices", "")).strip()
                            if raw and raw.lower() != "nan":
                                oracle_selected[[int(item) for item in raw.split(";") if item]] = True
                            break
                for variant, max_iter, X_scaled in variants:
                    result_id = (replicate, mechanism, variant, max_iter, downstream_draw)
                    suffix = "oracle_complete" if variant == "oracle_complete" else f"bayesianridge_mean__iter_{max_iter:04d}"
                    cache_path = pair_cache_dir / (
                        f"{dataset.dataset_label}__p_{p:04d}__rep_{replicate:04d}__"
                        f"{mechanism}__{suffix}__cdraw_{downstream_draw:02d}__equicorr.npz"
                    )
                    diagnostic_row: dict[str, Any] = {
                        "dataset": dataset.dataset_label,
                        "n": n, "p": p, "replicate": replicate,
                        "mechanism": mechanism,
                        "variant": variant, "max_iter": max_iter,
                        "downstream_draw": downstream_draw,
                        "knockoff_seed": downstream_seed,
                        "stabl_seed": downstream_seed + 1,
                        "pair_cache_path": str(cache_path),
                        "error": "",
                    }
                    try:
                        if result_id in done_knockoff and (
                            not args.run_downstream or result_id in done_selection
                        ):
                            print(f"[skip] {result_id}", flush=True)
                            continue
                        pair, cache_hit = load_or_generate_knockoff(
                            X_scaled, generator, seed=downstream_seed, cache_path=cache_path
                        )
                        diagnostic_row["pair_cache_hit"] = int(cache_hit)
                        diagnostic_row["generation_seconds"] = pair.generation_seconds
                        diagnostic_row.update(pair_diagnostics(pair.X, pair.X_tilde))
                        upsert(knockoff_rows, diagnostic_row, result_key)
                        atomic_write_csv(pd.DataFrame(knockoff_rows), knockoff_path)

                        if not args.run_downstream:
                            print(
                                f"[geometry] rep={replicate} mechanism={mechanism} "
                                f"iter={max_iter} draw={downstream_draw} "
                                f"pair_corr={diagnostic_row['pair_corr_mean']:.4f} "
                                f"s_rel={diagnostic_row['s_relative_mean']:.4f}",
                                flush=True,
                            )
                            continue

                        real_scores, knockoff_scores = fit_stabl_scores(
                            pair.X, pair.X_tilde, y.to_numpy(),
                            task="classification", groups=None,
                            n_bootstraps=args.n_bootstraps,
                            n_jobs=args.n_jobs,
                            random_state=downstream_seed + 1,
                            threshold_grid=threshold_grid,
                            regularization_grid_size=30,
                            classification_c_min=0.01,
                            classification_c_max=1.0,
                        )
                        selected_result = apply_selection_rule(
                            "stabl_min", real_scores, knockoff_scores,
                            q=0.10, threshold_grid=threshold_grid,
                        )
                        metrics = selection_metrics(
                            selected_result.selected, support, p, target_fdr=0.10
                        )
                        if variant == "oracle_complete":
                            oracle_selected = np.asarray(selected_result.selected, dtype=bool)
                        elif oracle_selected is None:
                            raise RuntimeError("Oracle selection must be computed before BR variants")

                        selection_row = {
                            "dataset": dataset.dataset_label,
                            "n": n, "p": p, "replicate": replicate,
                            "mechanism": mechanism,
                            "variant": variant, "max_iter": max_iter,
                            "downstream_draw": downstream_draw,
                            "selection_rule": "stabl_min",
                            "threshold": selected_result.threshold,
                            "estimated_fdp": selected_result.estimated_fdp,
                            "oracle_selection_jaccard": 1.0 if variant == "oracle_complete" else jaccard(selected_result.selected, oracle_selected),
                            "selected_indices": selected_indices_text(selected_result.selected),
                            "error": "",
                            **metrics,
                        }
                        if args.save_scores:
                            atomic_savez_compressed(
                                score_dir / (
                                    f"p_{p:04d}__rep_{replicate:04d}__{mechanism}__{suffix}__"
                                    f"cdraw_{downstream_draw:02d}__equicorr.npz"
                                ),
                                real_scores=real_scores,
                                knockoff_scores=knockoff_scores,
                                selected=np.asarray(selected_result.selected, dtype=np.uint8),
                                support=support,
                                feature_names=np.asarray(X_reference.columns.astype(str)),
                                max_iter=np.asarray(max_iter),
                                knockoff_seed=np.asarray(downstream_seed),
                                stabl_seed=np.asarray(downstream_seed + 1),
                            )
                        upsert(selection_rows, selection_row, result_key)
                        atomic_write_csv(pd.DataFrame(selection_rows), selection_path)
                        print(
                            f"[downstream] rep={replicate} mechanism={mechanism} "
                            f"variant={variant} iter={max_iter} draw={downstream_draw} "
                            f"fdp={selection_row['fdp']:.3f} power={selection_row['power']:.3f}",
                            flush=True,
                        )
                    except Exception as exc:
                        diagnostic_row["error"] = repr(exc)
                        diagnostic_row["traceback"] = traceback.format_exc(limit=10)
                        upsert(knockoff_rows, diagnostic_row, result_key)
                        atomic_write_csv(pd.DataFrame(knockoff_rows), knockoff_path)
                        if args.run_downstream:
                            selection_row = {
                                "dataset": dataset.dataset_label,
                                "n": n, "p": p, "replicate": replicate,
                                "mechanism": mechanism,
                                "variant": variant, "max_iter": max_iter,
                                "downstream_draw": downstream_draw,
                                "error": repr(exc),
                                "traceback": traceback.format_exc(limit=10),
                            }
                            upsert(selection_rows, selection_row, result_key)
                            atomic_write_csv(pd.DataFrame(selection_rows), selection_path)
                        print(f"[failed] {result_id}: {exc!r}", flush=True)
                        if args.fail_fast:
                            raise

    if args.summarize:
        summary = Path(__file__).with_name("summarize_v101_dream_guard.py")
        subprocess.run(
            [sys.executable, "-u", str(summary), "--out-dir", str(out_dir)],
            check=True,
        )

    print(f"V10.1A DREAM convergence guard completed: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
