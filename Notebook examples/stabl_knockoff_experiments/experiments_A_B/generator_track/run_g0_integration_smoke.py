#!/usr/bin/env python3
"""G0 smoke test for Gaussian MX knockoffs and official PLSKO.

This script deliberately reuses ``experiments_A_B/research_common.py`` so the
smoke test exercises exactly the same generator and STABL integration code as
Experiment B.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from research_common import (  # noqa: E402
    apply_selection_rule,
    atomic_savez_compressed,
    atomic_write_csv,
    atomic_write_json,
    c2st_diagnostics,
    cap_features_by_variance,
    fit_stabl_scores,
    generate_knockoff,
    generate_sparse_outcome,
    generator_configs_to_json,
    load_generator_configs,
    load_real_dataset,
    pair_diagnostics,
    prepare_output_directory,
    ranking_metrics,
    selected_indices_string,
    selection_metrics,
    standardize_completed_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", choices=("ssi", "dream", "csv"), default="ssi")
    parser.add_argument("--data-path", default="../Sample Data")
    parser.add_argument("--omic", default="CyTOF")
    parser.add_argument("--x-csv", default=None)
    parser.add_argument("--groups-csv", default=None)
    parser.add_argument("--subject-mode", choices=("first", "random", "all"), default="first")
    parser.add_argument(
        "--complete-strategy",
        choices=("drop-columns", "median", "error"),
        default="drop-columns",
    )
    parser.add_argument("--p", type=int, default=100)
    parser.add_argument(
        "--generator-configs-json",
        default=str(Path(__file__).with_name("configs") / "g0_generators.json"),
    )
    parser.add_argument("--out-dir", default="./generator_track_results/G0_ssi_smoke")
    parser.add_argument("--random-state", type=int, default=20260720)
    parser.add_argument("--n-bootstraps", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--stabl-grid-size", type=int, default=8)
    parser.add_argument("--n-signal", type=int, default=10)
    parser.add_argument("--signal-strength", type=float, default=4.0)
    parser.add_argument("--target-fdr", type=float, default=0.10)
    parser.add_argument("--run-stabl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    generator_configs = load_generator_configs(
        config_json=args.generator_configs_json,
        generators=(),
        plsko_threshold_abs=None,
        plsko_threshold_q=None,
        plsko_ncomp=None,
        plsko_sparsity=1.0,
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
    X_subset = cap_features_by_variance(dataset.X, args.p)
    X_scaled, _ = standardize_completed_matrix(X_subset)
    if args.n_signal >= X_scaled.shape[1]:
        raise ValueError("n_signal must be smaller than actual p")

    threshold_grid = np.arange(0.10, 1.00, 0.01)
    config = {
        "experiment": "G0_generator_integration_smoke",
        "version": 2,
        **vars(args),
        "dataset_label": dataset.dataset_label,
        "dataset_metadata": dataset.metadata,
        "n": int(X_scaled.shape[0]),
        "actual_p": int(X_scaled.shape[1]),
        "generator_configs": generator_configs_to_json(generator_configs),
    }
    out_dir = prepare_output_directory(
        args.out_dir,
        config,
        resume=False,
        overwrite=args.overwrite,
    )
    pair_dir = out_dir / "pairs"
    pair_dir.mkdir(exist_ok=True)

    y, support, beta = generate_sparse_outcome(
        X_scaled,
        n_signal=args.n_signal,
        task="classification",
        signal_strength=args.signal_strength,
        seed=args.random_state + 100,
    )
    atomic_savez_compressed(
        out_dir / "smoke_truth.npz",
        y=y.to_numpy(),
        support=support,
        beta=beta,
        feature_names=np.asarray(X_scaled.columns.astype(str)),
    )

    rows: list[dict[str, Any]] = []
    for generator_config in generator_configs:
        seed = args.random_state + 1000
        row: dict[str, Any] = {
            "dataset": dataset.dataset_label,
            "n": X_scaled.shape[0],
            "p": X_scaled.shape[1],
            "generator_label": generator_config.label,
            **asdict(generator_config),
            "seed": seed,
            "error": "",
        }
        print(f"[G0] generator={generator_config.label}", flush=True)
        try:
            first = generate_knockoff(X_scaled, generator_config, seed=seed)
            repeated = generate_knockoff(X_scaled, generator_config, seed=seed)
            alternate = generate_knockoff(X_scaled, generator_config, seed=seed + 1)
            row["generation_seconds"] = first.generation_seconds
            row["same_seed_max_abs_diff"] = float(
                np.max(np.abs(first.X_tilde - repeated.X_tilde))
            )
            row["different_seed_max_abs_diff"] = float(
                np.max(np.abs(first.X_tilde - alternate.X_tilde))
            )
            row["same_seed_reproducible"] = int(
                row["same_seed_max_abs_diff"] <= 1e-10
            )
            row.update(pair_diagnostics(first.X, first.X_tilde))
            row.update(c2st_diagnostics(first.X, first.X_tilde, seed=seed + 20))

            payload: dict[str, Any] = {
                "X": first.X,
                "X_tilde": first.X_tilde,
                "feature_names": np.asarray(X_scaled.columns.astype(str)),
            }
            if args.run_stabl:
                real_scores, artificial_scores = fit_stabl_scores(
                    first.X,
                    first.X_tilde,
                    y.to_numpy(),
                    task="classification",
                    groups=None,
                    n_bootstraps=args.n_bootstraps,
                    n_jobs=args.n_jobs,
                    random_state=args.random_state + 5000,
                    threshold_grid=threshold_grid,
                    regularization_grid_size=args.stabl_grid_size,
                    classification_c_min=0.01,
                    classification_c_max=1.0,
                    regression_alpha_min=1e-2,
                    regression_alpha_max=1e2,
                )
                selection = apply_selection_rule(
                    "stabl_min",
                    real_scores,
                    artificial_scores,
                    q=args.target_fdr,
                    threshold_grid=threshold_grid,
                )
                row["threshold"] = selection.threshold
                row["estimated_fdp"] = selection.estimated_fdp
                row.update(
                    selection_metrics(
                        selection.selected,
                        support,
                        X_scaled.shape[1],
                        target_fdr=args.target_fdr,
                    )
                )
                row.update(ranking_metrics(real_scores, support, X_scaled.shape[1]))
                row["selected_indices"] = selected_indices_string(selection.selected)
                payload.update(
                    real_scores=real_scores,
                    artificial_scores=artificial_scores,
                    support=support,
                    selected=selection.selected,
                )
            if args.save_pairs:
                atomic_savez_compressed(
                    pair_dir / f"{generator_config.label}.npz",
                    **payload,
                )
        except Exception as exc:
            row["error"] = repr(exc)
            row["traceback"] = traceback.format_exc(limit=12)
            print(f"[G0 failed] {generator_config.label}: {exc!r}", flush=True)
            if args.fail_fast:
                raise
        rows.append(row)
        atomic_write_csv(pd.DataFrame(rows), out_dir / "g0_results.csv")

    passed = [
        row
        for row in rows
        if row.get("error", "") == ""
        and row.get("same_seed_reproducible", 0) == 1
        and row.get("near_constant_knockoff_columns", 1) == 0
    ]
    atomic_write_json(
        {
            "n_configs": len(rows),
            "n_passed_basic_checks": len(passed),
            "all_basic_checks_passed": len(passed) == len(rows),
            "results": str(out_dir / "g0_results.csv"),
        },
        out_dir / "g0_status.json",
    )
    print(f"G0 completed: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
