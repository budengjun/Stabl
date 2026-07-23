#!/usr/bin/env python3
"""Experiment B: semi-synthetic generator and threshold benchmark.

The covariate matrix comes from a real omic dataset.  A sparse outcome is
simulated from that matrix, so the conditional support is known.  The same
knockoff score paths are evaluated with three rules:

1. Original STABL minimum FDP-plus threshold
2. Target-q STABL threshold
3. Knockoff-plus applied to paired STABL max-score differences

The experiment is intended to separate generator behavior from STABL threshold
behavior.  Missing-data completion is performed once before the benchmark and
is held fixed for every generator configuration.
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
    apply_selection_rule,
    atomic_write_csv,
    atomic_savez_compressed,
    cap_features_by_variance,
    c2st_diagnostics,
    fit_stabl_scores,
    fit_lcd_statistics,
    load_or_generate_knockoff,
    generate_sparse_outcome,
    generator_configs_to_json,
    load_generator_configs,
    load_real_dataset,
    pair_diagnostics,
    parse_csv_items,
    prepare_output_directory,
    ranking_metrics,
    selected_indices_string,
    selection_metrics,
    standardize_completed_matrix,
    summarize_results,
)


VALID_RULES = {
    "stabl_min",
    "stabl_q",
    "stabl_knockoff_plus",
    "knockoff_plus",
    "lcd_knockoff_plus",
}
VALID_GENERATORS = {
    "gaussian_equicorrelated",
    "gaussian_mvr",
    "official_plsko",
}


def validate_items(items: tuple[str, ...], allowed: set[str], label: str) -> tuple[str, ...]:
    invalid = sorted(set(items) - allowed)
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}. Allowed: {sorted(allowed)}")
    return items


def parse_p_values(value: str) -> tuple[tuple[str, int | None], ...]:
    parsed: list[tuple[str, int | None]] = []
    for raw in parse_csv_items(value):
        if raw.lower() == "full":
            parsed.append(("full", None))
        else:
            number = int(raw)
            if number < 2:
                raise ValueError("Every p value must be at least 2")
            parsed.append((str(number), number))
    return tuple(parsed)


def upsert(rows: list[dict[str, Any]], row: dict[str, Any], key_columns: tuple[str, ...]) -> None:
    key = tuple(row.get(column) for column in key_columns)
    rows[:] = [
        old for old in rows if tuple(old.get(column) for column in key_columns) != key
    ]
    rows.append(row)


def completed_keys(frame: pd.DataFrame, key_columns: tuple[str, ...]) -> set[tuple[Any, ...]]:
    if frame.empty or not set(key_columns).issubset(frame.columns):
        return set()
    valid = frame[frame.get("error", pd.Series("", index=frame.index)).fillna("").eq("")]
    return set(valid.loc[:, key_columns].itertuples(index=False, name=None))


def unique_feature_settings(
    X: pd.DataFrame,
    requested: tuple[tuple[str, int | None], ...],
) -> tuple[tuple[str, pd.DataFrame], ...]:
    settings: list[tuple[str, pd.DataFrame]] = []
    seen_actual_p: set[int] = set()
    for label, value in requested:
        subset = cap_features_by_variance(X, value)
        actual_p = subset.shape[1]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", choices=("ool", "ssi", "dream", "csv"), default="ool")
    parser.add_argument("--data-path", default="../Sample Data")
    parser.add_argument("--omic", default="CyTOF")
    parser.add_argument("--x-csv", default=None)
    parser.add_argument("--groups-csv", default=None)
    parser.add_argument(
        "--subject-mode",
        choices=("first", "random", "all"),
        default="first",
        help="Use first for the primary OOL iid-aligned benchmark; all is a stress test.",
    )
    parser.add_argument(
        "--complete-strategy",
        choices=("drop-columns", "median", "error"),
        default="drop-columns",
    )
    parser.add_argument("--p-values", default="250,500,1000,full")
    parser.add_argument("--out-dir", default="./experiment_B_semisynthetic")
    parser.add_argument("--n-replicates", type=int, default=50)
    parser.add_argument("--n-signal", type=int, default=10)
    parser.add_argument(
        "--task", choices=("classification", "regression"), default="classification"
    )
    parser.add_argument("--signal-strength", type=float, default=1.0)
    parser.add_argument(
        "--generators", default="gaussian_equicorrelated,gaussian_mvr"
    )
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
    parser.add_argument("--target-fdr", type=float, default=0.05)
    parser.add_argument(
        "--selection-rules",
        default="stabl_min,stabl_q,stabl_knockoff_plus,lcd_knockoff_plus"
    )
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--save-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pair-bank-scope",
        choices=("shared", "per-replicate"),
        default="shared",
        help=(
            "shared generates each X knockoff bank once and reuses it across "
            "independent synthetic outcomes; per-replicate also averages over "
            "new knockoff realizations but costs much more"
        ),
    )
    parser.add_argument("--pair-cache-dir", default=None)
    parser.add_argument("--refresh-pair-cache", action="store_true")
    parser.add_argument(
        "--c2st-every",
        type=int,
        default=0,
        help=(
            "Compute marginal and swap C2ST on draw zero every N replicates. "
            "Zero disables this relatively expensive diagnostic."
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
        raise ValueError("Replicate and draw counts must be positive")
    if args.n_signal < 1:
        raise ValueError("n_signal must be positive")
    if not 0.0 < args.target_fdr < 1.0:
        raise ValueError("target_fdr must lie in (0, 1)")
    if args.threshold_step <= 0 or args.threshold_min >= args.threshold_max:
        raise ValueError("Invalid STABL threshold grid")
    if args.stabl_grid_size < 2 or args.lcd_grid_size < 2:
        raise ValueError("STABL and LCD regularization grids need at least 2 values")
    if args.lcd_cv_folds < 2:
        raise ValueError("lcd_cv_folds must be at least 2")

    rules = validate_items(parse_csv_items(args.selection_rules), VALID_RULES, "selection rules")
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
    if (
        any(item.generator == "official_plsko" for item in generator_configs)
        and args.pair_bank_scope == "shared"
    ):
        print(
            "[warning] official PLSKO is being reused across synthetic outcomes. "
            "Use --pair-bank-scope per-replicate for the Generator Track primary analysis.",
            flush=True,
        )
    if any(
        item.generator == "gaussian_equicorrelated_true_sigma"
        for item in generator_configs
    ):
        raise ValueError(
            "gaussian_equicorrelated_true_sigma is available only in Experiment A, "
            "where the population covariance is known by construction"
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
        complete_strategy=args.complete_strategy,
        seed=args.random_state,
    )
    feature_settings = unique_feature_settings(dataset.X, requested_p_values)
    for requested_label, X_subset in feature_settings:
        print(
            f"[data] setting={requested_label} n={X_subset.shape[0]} p={X_subset.shape[1]}",
            flush=True,
        )
    if dataset.groups is not None:
        print(
            "[warning] repeated rows are retained. STABL subsampling is grouped, but "
            "the knockoff generator still treats rows as iid. Use --subject-mode first "
            "for the primary OOL analysis.",
            flush=True,
        )
        if "lcd_knockoff_plus" in rules:
            raise ValueError(
                "lcd_knockoff_plus in v4 requires iid-aligned rows. Use "
                "--subject-mode first for OOL, or omit lcd_knockoff_plus in the "
                "longitudinal stress test."
            )

    config = {
        "experiment": "B_real_X_semisynthetic",
        "version": 4,
        **vars(args),
        "dataset_metadata": dataset.metadata,
        "dataset_label": dataset.dataset_label,
        "p_values_resolved": [
            {"requested": label, "actual": int(X_subset.shape[1])}
            for label, X_subset in feature_settings
        ],
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

    result_path = out_dir / "selection_results.csv"
    diagnostic_path = out_dir / "draw_diagnostics.csv"
    result_frame = (
        pd.read_csv(result_path) if args.resume and result_path.exists() else pd.DataFrame()
    )
    diagnostic_frame = (
        pd.read_csv(diagnostic_path) if args.resume and diagnostic_path.exists() else pd.DataFrame()
    )
    result_rows = result_frame.to_dict("records")
    diagnostic_rows = diagnostic_frame.to_dict("records")
    result_key = (
        "p_setting",
        "actual_p",
        "replicate",
        "generator_label",
        "draw",
        "selection_rule",
    )
    diagnostic_key = (
        "p_setting",
        "actual_p",
        "replicate",
        "generator_label",
        "draw",
    )
    done_results = completed_keys(result_frame, result_key)
    done_diagnostics = completed_keys(diagnostic_frame, diagnostic_key)

    for p_index, (p_label, X_subset) in enumerate(feature_settings):
        X_scaled, _ = standardize_completed_matrix(X_subset)
        p = X_scaled.shape[1]
        if args.n_signal >= p:
            raise ValueError(
                f"n_signal={args.n_signal} must be smaller than actual p={p}"
            )
        group_array = (
            None
            if dataset.groups is None
            else dataset.groups.loc[X_scaled.index].to_numpy()
        )

        for replicate in range(args.n_replicates):
            replicate_seed = args.random_state + p_index * 10_000_000 + replicate * 100_000
            y, support, beta = generate_sparse_outcome(
                X_scaled,
                n_signal=args.n_signal,
                task=args.task,
                signal_strength=args.signal_strength,
                seed=replicate_seed + 11,
            )
            atomic_savez_compressed(
                truth_dir / f"p_{p:04d}__rep_{replicate:04d}.npz",
                y=y.to_numpy(),
                support=support,
                beta=beta,
                feature_names=np.asarray(X_scaled.columns.astype(str)),
            )

            for config_index, generator_config in enumerate(generator_configs):
                for draw in range(args.n_knockoff_draws):
                    draw_id = (
                        p_label,
                        p,
                        replicate,
                        generator_config.label,
                        draw,
                    )
                    expected_rules = {draw_id + (rule,) for rule in rules}
                    if (
                        args.resume
                        and draw_id in done_diagnostics
                        and expected_rules.issubset(done_results)
                    ):
                        print(f"[skip] {draw_id}", flush=True)
                        continue

                    # The numeric seeds are deliberately shared across generators.
                    # Generator identity changes the algorithm, not the paired random
                    # outcome or the downstream STABL bootstrap stream.
                    if args.pair_bank_scope == "shared":
                        knockoff_seed = (
                            args.random_state
                            + p_index * 10_000_000
                            + 5_000_000
                            + draw
                        )
                        pair_cache_path = pair_cache_dir / (
                            f"p_{p:04d}__{generator_config.label}__"
                            f"draw_{draw:03d}.npz"
                        )
                    else:
                        knockoff_seed = replicate_seed + 10_000 + draw
                        pair_cache_path = pair_cache_dir / (
                            f"p_{p:04d}__rep_{replicate:04d}__"
                            f"{generator_config.label}__draw_{draw:03d}.npz"
                        )
                    stabl_seed = replicate_seed + 20_000 + draw
                    lcd_seed = replicate_seed + 30_000 + draw
                    print(
                        f"[start] p={p} rep={replicate} "
                        f"generator={generator_config.label} draw={draw}",
                        flush=True,
                    )
                    diagnostic_row: dict[str, Any] = {
                        "dataset": dataset.dataset_label,
                        "p_setting": p_label,
                        "actual_p": p,
                        "n": X_scaled.shape[0],
                        "replicate": replicate,
                        "generator_label": generator_config.label,
                        "generator": generator_config.generator,
                        "draw": draw,
                        "knockoff_seed": knockoff_seed,
                        "stabl_seed": stabl_seed,
                        "lcd_seed": lcd_seed,
                        "threshold_abs": generator_config.threshold_abs,
                        "threshold_q": generator_config.threshold_q,
                        "ncomp": generator_config.ncomp,
                        "sparsity": generator_config.sparsity,
                        "error": "",
                    }
                    try:
                        pair, cache_hit = load_or_generate_knockoff(
                            X_scaled,
                            generator_config,
                            seed=knockoff_seed,
                            cache_path=pair_cache_path,
                            refresh=args.refresh_pair_cache,
                        )
                        diagnostic_row["pair_bank_scope"] = args.pair_bank_scope
                        diagnostic_row["pair_cache_hit"] = int(cache_hit)
                        diagnostic_row["pair_cache_path"] = str(pair_cache_path)
                        diagnostic_row["generation_seconds"] = pair.generation_seconds
                        diagnostic_row.update(pair_diagnostics(pair.X, pair.X_tilde))
                        if (
                            args.c2st_every > 0
                            and draw == 0
                            and replicate % args.c2st_every == 0
                        ):
                            diagnostic_row.update(
                                c2st_diagnostics(
                                    pair.X,
                                    pair.X_tilde,
                                    seed=replicate_seed + 40_000,
                                )
                            )
                        real_scores, knockoff_scores = fit_stabl_scores(
                            pair.X,
                            pair.X_tilde,
                            y.to_numpy(),
                            task=args.task,
                            groups=group_array,
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
                        lcd = None
                        if "lcd_knockoff_plus" in rules:
                            lcd = fit_lcd_statistics(
                                pair.X,
                                pair.X_tilde,
                                y.to_numpy(),
                                task=args.task,
                                random_state=lcd_seed,
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
                                    X_scaled.columns.astype(str)
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
                                    f"p_{p:04d}__rep_{replicate:04d}__"
                                    f"{generator_config.label}__draw_{draw:03d}.npz"
                                ),
                                **payload,
                            )

                        for rule in rules:
                            selection = apply_selection_rule(
                                rule,  # type: ignore[arg-type]
                                real_scores,
                                knockoff_scores,
                                q=args.target_fdr,
                                threshold_grid=threshold_grid,
                                lcd_statistics=(None if lcd is None else lcd.W),
                            )
                            row = {
                                "dataset": dataset.dataset_label,
                                "p_setting": p_label,
                                "actual_p": p,
                                "n": X_scaled.shape[0],
                                "replicate": replicate,
                                "generator_label": generator_config.label,
                                "generator": generator_config.generator,
                                "draw": draw,
                                "selection_rule": rule,
                                "threshold": selection.threshold,
                                "estimated_fdp": selection.estimated_fdp,
                                "n_signal": args.n_signal,
                                "target_fdr": args.target_fdr,
                                "n_bootstraps": args.n_bootstraps,
                                "stabl_grid_size": args.stabl_grid_size,
                                "lcd_grid_size": args.lcd_grid_size,
                                "lcd_cv_folds": args.lcd_cv_folds,
                                "knockoff_seed": knockoff_seed,
                                "stabl_seed": stabl_seed,
                                "selected_indices": selected_indices_string(
                                    selection.selected
                                ),
                                "error": "",
                                **selection_metrics(
                                    selection.selected,
                                    support,
                                    p,
                                    target_fdr=args.target_fdr,
                                ),
                                **ranking_metrics(real_scores, support, p),
                            }
                            row["calibration_gap"] = float(
                                row["estimated_fdp"] - row["fdp"]
                            )
                            upsert(result_rows, row, result_key)
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
                                "dataset": dataset.dataset_label,
                                "p_setting": p_label,
                                "actual_p": p,
                                "replicate": replicate,
                                "generator_label": generator_config.label,
                                "generator": generator_config.generator,
                                "draw": draw,
                                "selection_rule": rule,
                                "error": repr(exc),
                            }
                            upsert(result_rows, row, result_key)
                        print(f"[failed] {exc!r}", flush=True)
                        if args.fail_fast:
                            raise

                    upsert(diagnostic_rows, diagnostic_row, diagnostic_key)
                    atomic_write_csv(pd.DataFrame(diagnostic_rows), diagnostic_path)
                    atomic_write_csv(pd.DataFrame(result_rows), result_path)

            summary = summarize_results(
                pd.DataFrame(result_rows),
                group_columns=(
                    "dataset",
                    "p_setting",
                    "actual_p",
                    "generator_label",
                    "selection_rule",
                ),
                target_fdr=args.target_fdr,
            )
            if not summary.empty:
                atomic_write_csv(summary, out_dir / "summary.csv")

    print(f"\nExperiment B completed. Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
