#!/usr/bin/env python3
"""G1 tuning wrapper for the official ``PLSKO::plsko_tuning`` function."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    # Force one numerical-library thread per R PSOCK worker. Using setdefault
    # can preserve a large inherited value and cause severe oversubscription.
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from research_common import (  # noqa: E402
    atomic_write_json,
    cap_features_by_variance,
    load_real_dataset,
    matrix_fingerprint,
    parse_csv_items,
    prepare_output_directory,
    safe_label,
    standardize_completed_matrix,
)


def parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in parse_csv_items(value))


def parse_float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in parse_csv_items(value))


def first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    normalized = {str(key).lower().replace("_", "."): value for key, value in row.items()}
    for name in names:
        key = name.lower().replace("_", ".")
        if key in normalized and pd.notna(normalized[key]):
            return normalized[key]
    return None


def freeze_optimal_config(optimal: dict[str, Any], label: str) -> dict[str, Any]:
    ncomp = first_present(optimal, ("ncomp", "n.comp", "components"))
    threshold_abs = first_present(
        optimal,
        ("threshold.abs", "threshold_abs", "threshold", "neighbor.threshold"),
    )
    sparsity = first_present(optimal, ("sparsity", "sparse"))
    if ncomp is None or threshold_abs is None or sparsity is None:
        raise RuntimeError(
            "Could not identify ncomp, threshold.abs, and sparsity in optimal.csv. "
            f"Available columns: {sorted(optimal)}"
        )
    return {
        "label": safe_label(label),
        "generator": "official_plsko",
        "threshold_abs": float(threshold_abs),
        "threshold_q": None,
        "ncomp": int(round(float(ncomp))),
        "sparsity": float(sparsity),
    }


def run_tuning(
    X: pd.DataFrame,
    *,
    out_dir: Path,
    n_ko: int,
    p_s: int,
    q: float,
    ncomp: tuple[int, ...],
    threshold_abs: tuple[float, ...],
    sparsity: tuple[float, ...],
    fdp_measure: str,
    early_stop: bool,
    seed: int,
    n_cores: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript was not found")
    bridge = Path(__file__).with_name("official_plsko_tuning_bridge.R")
    if not bridge.exists():
        raise FileNotFoundError(bridge)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stabl_plsko_tuning_") as temporary:
        input_path = Path(temporary) / "X.csv"
        X.to_csv(input_path)
        command = [
            "Rscript",
            str(bridge),
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            str(seed),
            "--n-ko",
            str(n_ko),
            "--p-s",
            str(p_s),
            "--q",
            str(q),
            "--ncomp",
            ",".join(map(str, ncomp)),
            "--threshold-abs",
            ",".join(map(str, threshold_abs)),
            "--sparsity",
            ",".join(map(str, sparsity)),
            "--fdp-measure",
            fdp_measure,
            "--early-stop",
            str(bool(early_stop)).lower(),
            "--n-cores",
            str(n_cores),
        ]
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        (out_dir / "bridge_stdout.txt").write_text(result.stdout, encoding="utf-8")
        (out_dir / "bridge_stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                "Official PLSKO tuning failed.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
    optimal_path = out_dir / "optimal.csv"
    if not optimal_path.exists():
        raise RuntimeError("PLSKO tuning did not create optimal.csv")
    optimal = pd.read_csv(optimal_path).iloc[0].to_dict()
    return {"optimal": optimal, "command": command}


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
    parser.add_argument("--out-dir", default="./generator_track_results/G1_ssi_tuning")
    parser.add_argument("--n-ko", type=int, default=10)
    parser.add_argument("--p-s", type=int, default=20)
    parser.add_argument("--target-fdr", type=float, default=0.05)
    parser.add_argument("--ncomp", default="2,5,10")
    parser.add_argument("--threshold-abs", default="0,0.1,0.2,0.3")
    parser.add_argument("--sparsity", default="0.5,0.8,1.0")
    parser.add_argument(
        "--fdp-measure",
        choices=("median", "mean", "either", "both"),
        default="median",
    )
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--n-cores", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=20260721)
    parser.add_argument("--frozen-label", default="plsko_tuned")
    parser.add_argument("--timeout-seconds", type=int, default=172800)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ncomp = parse_int_csv(args.ncomp)
    threshold_abs = parse_float_csv(args.threshold_abs)
    sparsity = parse_float_csv(args.sparsity)
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
    if not 1 <= args.p_s < X_scaled.shape[1]:
        raise ValueError("p_s must lie between 1 and actual p minus 1")

    config = {
        "experiment": "G1_official_plsko_tuning",
        "version": 2,
        **vars(args),
        "ncomp_resolved": ncomp,
        "threshold_abs_resolved": threshold_abs,
        "sparsity_resolved": sparsity,
        "dataset_label": dataset.dataset_label,
        "dataset_metadata": dataset.metadata,
        "n": int(X_scaled.shape[0]),
        "actual_p": int(X_scaled.shape[1]),
        "x_scaled_hash": matrix_fingerprint(
            X_scaled.to_numpy(dtype=float),
            feature_names=X_scaled.columns,
        ),
        "number_of_configurations": len(ncomp) * len(threshold_abs) * len(sparsity),
    }
    out_dir = prepare_output_directory(
        args.out_dir,
        config,
        resume=False,
        overwrite=args.overwrite,
    )
    tuning_dir = out_dir / "official_tuning"
    result = run_tuning(
        X_scaled,
        out_dir=tuning_dir,
        n_ko=args.n_ko,
        p_s=args.p_s,
        q=args.target_fdr,
        ncomp=ncomp,
        threshold_abs=threshold_abs,
        sparsity=sparsity,
        fdp_measure=args.fdp_measure,
        early_stop=args.early_stop,
        seed=args.random_state,
        n_cores=args.n_cores,
        timeout_seconds=args.timeout_seconds,
    )
    frozen = freeze_optimal_config(result["optimal"], args.frozen_label)
    atomic_write_json([frozen], out_dir / "frozen_plsko_generator.json")
    atomic_write_json(
        {
            "dataset_label": dataset.dataset_label,
            "n": X_scaled.shape[0],
            "p": X_scaled.shape[1],
            "source_optimal": result["optimal"],
            "frozen_generator": frozen,
            "official_tuning_directory": str(tuning_dir),
        },
        out_dir / "g1_status.json",
    )
    print(json.dumps(frozen, indent=2), flush=True)
    print(f"G1 completed: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
