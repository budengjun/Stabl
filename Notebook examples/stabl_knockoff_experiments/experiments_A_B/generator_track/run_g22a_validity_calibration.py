#!/usr/bin/env python3
"""G2.2A exchangeability calibration with valid and invalid controls.

The experiment runs two complementary designs:

* Gaussian synthetic X with a known population covariance, including an oracle
  true-Sigma equicorrelated knockoff as the positive validity control.
* SSI CyTOF real X, including sample-Sigma equicorrelated and MVR baselines.

PLSKO default and the G1 frozen configuration are audited beside deliberately
invalid controls. Marginal, global-swap, selected single-feature-swap, and block
swap C2STs are evaluated with paired grouped cross-validation.
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
    derive_c2st_gate,
    exchangeability_diagnostics,
    invalid_knockoff,
    load_json,
)
from research_common import (  # noqa: E402
    GeneratorConfig,
    cap_features_by_variance,
    generate_knockoff,
    load_real_dataset,
    pair_diagnostics,
    prepare_output_directory,
    simulate_gaussian_covariates,
    standardize_completed_matrix,
    transform_gaussian_parameters_with_scaler,
)


def _load_tuned_config(path: str | Path) -> GeneratorConfig:
    raw = load_json(path)
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError("Tuned PLSKO JSON list must contain exactly one configuration")
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ValueError("Tuned PLSKO JSON must contain an object")
    mapping = dict(raw)
    mapping["label"] = "plsko_tuned"
    mapping["generator"] = "official_plsko"
    return GeneratorConfig.from_mapping(mapping)


def _generator_specs(tuned: GeneratorConfig, *, include_invalid: bool) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "label": "oracle_true_sigma",
            "kind": "generator",
            "config": GeneratorConfig(
                "oracle_true_sigma", "gaussian_equicorrelated_true_sigma"
            ),
            "synthetic_only": True,
        },
        {
            "label": "equicorr",
            "kind": "generator",
            "config": GeneratorConfig("equicorr", "gaussian_equicorrelated"),
        },
        {
            "label": "mvr",
            "kind": "generator",
            "config": GeneratorConfig("mvr", "gaussian_mvr"),
        },
        {
            "label": "plsko_default",
            "kind": "generator",
            "config": GeneratorConfig(
                "plsko_default",
                "official_plsko",
                threshold_abs=None,
                threshold_q=0.8,
                ncomp=None,
                sparsity=1.0,
            ),
        },
        {"label": "plsko_tuned", "kind": "generator", "config": tuned},
    ]
    if include_invalid:
        specs.extend(
            [
                {
                    "label": "invalid_columnwise_permutation",
                    "kind": "invalid",
                    "invalid_kind": "columnwise_permutation",
                },
                {
                    "label": "invalid_mean_shift",
                    "kind": "invalid",
                    "invalid_kind": "mean_shift",
                },
            ]
        )
    return specs


def _save_pair(path: Path, X: np.ndarray, X_tilde: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=float),
        X_tilde=np.asarray(X_tilde, dtype=float),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _run_pair(
    *,
    X: pd.DataFrame,
    design: str,
    replicate: int,
    draw: int,
    spec: dict[str, Any],
    seed: int,
    known_mean: np.ndarray | None,
    known_covariance: np.ndarray | None,
    args: argparse.Namespace,
    pair_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label = str(spec["label"])
    pair_path = pair_dir / f"{design}__rep_{replicate:03d}__{label}__draw_{draw:03d}.npz"
    if pair_path.exists() and not args.refresh_pairs:
        with np.load(pair_path, allow_pickle=False) as cached:
            X_values = np.asarray(cached["X"], dtype=float)
            X_tilde = np.asarray(cached["X_tilde"], dtype=float)
        generation_seconds = float("nan")
    else:
        X_values = X.to_numpy(dtype=float)
        if spec["kind"] == "invalid":
            X_tilde = invalid_knockoff(
                X_values,
                kind=str(spec["invalid_kind"]),
                seed=seed,
            )
            generation_seconds = 0.0
        else:
            result = generate_knockoff(
                X,
                spec["config"],
                seed=seed,
                known_mean=known_mean,
                known_covariance=known_covariance,
            )
            X_tilde = result.X_tilde
            generation_seconds = result.generation_seconds
        _save_pair(
            pair_path,
            X_values,
            X_tilde,
            {
                "design": design,
                "replicate": replicate,
                "draw": draw,
                "generator_label": label,
                "seed": seed,
            },
        )

    local_swaps = ((replicate * args.draws + draw) % args.local_swap_every) == 0
    n_single = args.n_single_features if local_swaps else 0
    block_size = args.block_size if local_swaps else X.shape[1] + 1
    classifiers = ["logistic"]
    if args.include_extra_trees:
        classifiers.append("extra_trees")

    diagnostic_rows = exchangeability_diagnostics(
        X_values,
        X_tilde,
        seed=seed + 50000,
        repeats=args.c2st_repeats,
        n_splits=args.c2st_folds,
        null_permutations=args.null_permutations if local_swaps else 0,
        classifier_kinds=classifiers,
        n_single_features=n_single,
        block_size=block_size,
        rf_estimators=args.rf_estimators,
    )
    for row in diagnostic_rows:
        row.update(
            {
                "design": design,
                "replicate": replicate,
                "draw": draw,
                "generator_label": label,
                "generator_kind": spec["kind"],
                "seed": seed,
                "local_swap_audit": int(local_swaps),
                "error": "",
            }
        )

    pair_row: dict[str, Any] = {
        "design": design,
        "replicate": replicate,
        "draw": draw,
        "generator_label": label,
        "generator_kind": spec["kind"],
        "seed": seed,
        "generation_seconds": generation_seconds,
        "error": "",
        **pair_diagnostics(X_values, X_tilde),
    }
    return diagnostic_rows, pair_row


def _summarize_c2st(rows: pd.DataFrame) -> pd.DataFrame:
    valid = rows[rows["error"].fillna("").eq("")].copy()
    if valid.empty:
        return pd.DataFrame()
    group_columns = ["design", "generator_label", "test", "classifier"]
    summary = (
        valid.groupby(group_columns, dropna=False)["auc_mean"]
        .agg(
            auc_mean_mean="mean",
            auc_mean_sd="std",
            auc_mean_median="median",
            auc_mean_min="min",
            auc_mean_max="max",
            n="size",
        )
        .reset_index()
    )
    q90 = (
        valid.groupby(group_columns, dropna=False)["auc_mean"]
        .quantile(0.90)
        .rename("auc_mean_q90")
        .reset_index()
    )
    return summary.merge(q90, on=group_columns, how="left")


def _gate_input(summary: pd.DataFrame) -> pd.DataFrame:
    block = summary[summary["classifier"].eq("logistic")].copy()
    block["generator_label"] = np.where(
        block["design"].eq("gaussian") & block["generator_label"].eq("oracle_true_sigma"),
        "gaussian_oracle_true_sigma",
        np.where(
            block["design"].eq("ssi") & block["generator_label"].eq("equicorr"),
            "ssi_equicorr",
            np.where(
                block["design"].eq("ssi") & block["generator_label"].eq("mvr"),
                "ssi_mvr",
                block["generator_label"],
            ),
        ),
    )
    return block


def run(args: argparse.Namespace) -> Path:
    tuned = _load_tuned_config(args.tuned_config_json)
    specs = _generator_specs(tuned, include_invalid=args.include_invalid_controls)
    config = {
        "experiment": "G2.2A_validity_calibration",
        "version": 1,
        **vars(args),
        "generator_specs": [
            {
                key: (asdict(value) if isinstance(value, GeneratorConfig) else value)
                for key, value in spec.items()
            }
            for spec in specs
        ],
    }
    out_dir = prepare_output_directory(
        args.out_dir,
        config,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    pair_dir = out_dir / "pairs"
    pair_dir.mkdir(exist_ok=True)

    all_c2st: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []

    for replicate in range(args.synthetic_replicates):
        X_raw, population_mean, population_covariance = simulate_gaussian_covariates(
            n=args.synthetic_n,
            p=args.p,
            block_size=args.synthetic_block_size,
            rho=args.synthetic_rho,
            seed=args.random_state + 100000 + replicate,
        )
        X, scaler = standardize_completed_matrix(X_raw)
        scaled_mean, scaled_covariance = transform_gaussian_parameters_with_scaler(
            population_mean,
            population_covariance,
            scaler,
        )
        for draw in range(args.draws):
            pair_seed = args.random_state + 1000000 + 1000 * replicate + draw
            for spec in specs:
                if spec.get("ssi_only"):
                    continue
                try:
                    rows, pair_row = _run_pair(
                        X=X,
                        design="gaussian",
                        replicate=replicate,
                        draw=draw,
                        spec=spec,
                        seed=pair_seed,
                        known_mean=scaled_mean if spec["label"] == "oracle_true_sigma" else None,
                        known_covariance=(
                            scaled_covariance if spec["label"] == "oracle_true_sigma" else None
                        ),
                        args=args,
                        pair_dir=pair_dir,
                    )
                    all_c2st.extend(rows)
                    all_pairs.append(pair_row)
                    print(
                        f"[G2.2A] design=gaussian rep={replicate} draw={draw} generator={spec['label']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    all_pairs.append(
                        {
                            "design": "gaussian",
                            "replicate": replicate,
                            "draw": draw,
                            "generator_label": spec["label"],
                            "error": error,
                            "traceback": traceback.format_exc(),
                        }
                    )
                    if args.fail_fast:
                        raise

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
    X_ssi = cap_features_by_variance(dataset.X, args.p)
    X_ssi, _ = standardize_completed_matrix(X_ssi)
    for draw in range(args.ssi_draws):
        pair_seed = args.random_state + 2000000 + draw
        for spec in specs:
            if spec.get("synthetic_only"):
                continue
            try:
                rows, pair_row = _run_pair(
                    X=X_ssi,
                    design="ssi",
                    replicate=0,
                    draw=draw,
                    spec=spec,
                    seed=pair_seed,
                    known_mean=None,
                    known_covariance=None,
                    args=args,
                    pair_dir=pair_dir,
                )
                all_c2st.extend(rows)
                all_pairs.append(pair_row)
                print(
                    f"[G2.2A] design=ssi draw={draw} generator={spec['label']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                all_pairs.append(
                    {
                        "design": "ssi",
                        "replicate": 0,
                        "draw": draw,
                        "generator_label": spec["label"],
                        "error": error,
                        "traceback": traceback.format_exc(),
                    }
                )
                if args.fail_fast:
                    raise

    c2st_frame = pd.DataFrame(all_c2st)
    pair_frame = pd.DataFrame(all_pairs)
    atomic_write_csv(c2st_frame, out_dir / "g22a_c2st_runs.csv")
    atomic_write_csv(pair_frame, out_dir / "g22a_pair_diagnostics.csv")

    summary = _summarize_c2st(c2st_frame)
    atomic_write_csv(summary, out_dir / "g22a_c2st_summary.csv")
    pair_summary = (
        pair_frame[pair_frame["error"].fillna("").eq("")]
        .groupby(["design", "generator_label"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    atomic_write_csv(pair_summary, out_dir / "g22a_pair_summary.csv")

    gate = derive_c2st_gate(
        _gate_input(summary),
        synthetic_oracle_label="gaussian_oracle_true_sigma",
        ssi_equicorr_label="ssi_equicorr",
        ssi_mvr_label="ssi_mvr",
        absolute_cap=args.gate_absolute_cap,
        margin=args.gate_margin,
    )
    gate.update(
        {
            "source": "G2.2A calibrated valid controls only",
            "classifier": "logistic",
            "primary_tests": ["marginal", "global_swap"],
            "draw_max_slack": 0.05,
        }
    )
    atomic_write_json(gate, out_dir / "g22a_recommended_c2st_gate.json")

    errors = pair_frame[~pair_frame["error"].fillna("").eq("")]
    status = {
        "status": "completed" if errors.empty else "completed_with_errors",
        "out_dir": str(out_dir),
        "n_pair_rows": int(len(pair_frame)),
        "n_c2st_rows": int(len(c2st_frame)),
        "n_errors": int(len(errors)),
        "recommended_gate": gate,
        "ssi_dataset_label": dataset.dataset_label,
        "ssi_n": int(X_ssi.shape[0]),
        "ssi_p": int(X_ssi.shape[1]),
    }
    atomic_write_json(status, out_dir / "g22a_status.json")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ssi-data-path", required=True)
    parser.add_argument("--ssi-omic", default="CyTOF")
    parser.add_argument("--complete-strategy", choices=("drop-columns", "median", "error"), default="drop-columns")
    parser.add_argument("--tuned-config-json", required=True)
    parser.add_argument("--out-dir", default="generator_track_results/G22A_validity_calibration")
    parser.add_argument("--p", type=int, default=100)
    parser.add_argument("--synthetic-n", type=int, default=150)
    parser.add_argument("--synthetic-replicates", type=int, default=5)
    parser.add_argument("--synthetic-block-size", type=int, default=20)
    parser.add_argument("--synthetic-rho", type=float, default=0.60)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--ssi-draws", type=int, default=10)
    parser.add_argument("--c2st-repeats", type=int, default=3)
    parser.add_argument("--c2st-folds", type=int, default=5)
    parser.add_argument("--n-single-features", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--local-swap-every", type=int, default=2)
    parser.add_argument("--null-permutations", type=int, default=0)
    parser.add_argument("--include-extra-trees", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--include-invalid-controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gate-absolute-cap", type=float, default=0.80)
    parser.add_argument("--gate-margin", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=20260724)
    parser.add_argument("--refresh-pairs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = run(args)
    print(f"G2.2A completed: {out_dir}")


if __name__ == "__main__":
    main()
