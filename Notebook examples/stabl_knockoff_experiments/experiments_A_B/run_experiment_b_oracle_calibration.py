#!/usr/bin/env python3
"""V8 SSI/OOL real-X oracle difficulty calibration grid.

This driver runs the existing real-X semisynthetic pipeline in oracle-only mode
across a grid of feature counts and signal strengths.  It does not compare
completion methods.  Its sole purpose is to identify a non-floor benchmark
before median, BayesianRidge conditional-mean, and BayesianRidge posterior
completion are compared.

Each grid cell uses ordinary STABL ``stabl_min``.  Common random numbers are
preserved across signal strengths, and compatible knockoff pairs are reused
because knockoff generation depends on X but not on y.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def strength_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Oracle-only real-X difficulty calibration for V8.",
    )
    parser.add_argument("--dataset", choices=("ool", "ssi", "dream", "csv"), default="ssi")
    parser.add_argument("--data-path", default="../Sample Data/Biobank SSI")
    parser.add_argument("--omic", default="CyTOF")
    parser.add_argument("--x-csv", default=None)
    parser.add_argument("--groups-csv", default=None)
    parser.add_argument("--subject-mode", choices=("first", "random", "all"), default="first")
    parser.add_argument("--reference-complete-strategy", choices=("drop-columns", "error"), default="drop-columns")
    parser.add_argument("--p-values", default="100,250")
    parser.add_argument("--signal-strengths", default="4,6")
    parser.add_argument("--out-dir", default="./experiment_B_oracle_calibration_v8")
    parser.add_argument("--pair-cache-dir", default=None)
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--n-completion-draws", type=int, default=3)
    parser.add_argument("--n-signal", type=int, default=15)
    parser.add_argument("--task", choices=("classification", "regression"), default="classification")
    parser.add_argument("--generators", default="gaussian_equicorrelated")
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
    parser.add_argument("--random-state", type=int, default=20260718)
    parser.add_argument("--save-scores", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-precision-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--summarize", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both")
    if args.n_replicates < 1 or args.n_completion_draws < 1:
        raise ValueError("Replicate and draw counts must be positive")

    p_values = parse_csv(args.p_values)
    strengths = tuple(float(item) for item in parse_csv(args.signal_strengths))
    if any(value <= 0 for value in strengths):
        raise ValueError("All signal strengths must be positive")

    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_cache = (
        Path(args.pair_cache_dir).expanduser().resolve()
        if args.pair_cache_dir
        else out_dir / "pair_cache"
    )
    pair_cache.mkdir(parents=True, exist_ok=True)

    grid_config = {
        "experiment": "B_real_X_oracle_difficulty_calibration",
        "version": 8,
        **vars(args),
        "p_values_resolved": p_values,
        "signal_strengths_resolved": strengths,
        "pair_cache_dir_resolved": str(pair_cache),
        "scientific_role": "difficulty calibration only; no completion-method inference",
    }
    (out_dir / "oracle_calibration_grid_config.json").write_text(
        json.dumps(grid_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    runner = Path(__file__).with_name("run_experiment_b_completion_replacement.py")
    for p_value in p_values:
        p_label = p_value.lower().replace(".", "p")
        for strength in strengths:
            cell_dir = out_dir / f"p_{p_label}__strength_{strength_label(strength)}"
            command = [
                sys.executable,
                "-u",
                str(runner),
                "--experiment-mode",
                "oracle_calibration",
                "--dataset",
                args.dataset,
                "--data-path",
                args.data_path,
                "--omic",
                args.omic,
                "--subject-mode",
                args.subject_mode,
                "--reference-complete-strategy",
                args.reference_complete_strategy,
                "--p-values",
                p_value,
                "--out-dir",
                str(cell_dir),
                "--pair-cache-dir",
                str(pair_cache),
                "--n-replicates",
                str(args.n_replicates),
                "--n-completion-draws",
                str(args.n_completion_draws),
                "--n-signal",
                str(args.n_signal),
                "--task",
                args.task,
                "--signal-strength",
                str(strength),
                "--mechanisms",
                "MCAR",
                "--missing-rate",
                "0.0",
                "--completion-methods",
                "oracle_complete",
                "--generators",
                args.generators,
                "--n-bootstraps",
                str(args.n_bootstraps),
                "--stabl-grid-size",
                str(args.stabl_grid_size),
                "--stabl-c-min",
                str(args.stabl_c_min),
                "--stabl-c-max",
                str(args.stabl_c_max),
                "--stabl-alpha-min",
                str(args.stabl_alpha_min),
                "--stabl-alpha-max",
                str(args.stabl_alpha_max),
                "--threshold-min",
                str(args.threshold_min),
                "--threshold-max",
                str(args.threshold_max),
                "--threshold-step",
                str(args.threshold_step),
                "--target-fdr",
                str(args.target_fdr),
                "--n-jobs",
                str(args.n_jobs),
                "--random-state",
                str(args.random_state),
            ]
            if args.x_csv:
                command.extend(["--x-csv", args.x_csv])
            if args.groups_csv:
                command.extend(["--groups-csv", args.groups_csv])
            if args.generator_configs_json:
                command.extend(["--generator-configs-json", args.generator_configs_json])
            command.append("--save-scores" if args.save_scores else "--no-save-scores")
            if args.skip_precision_recovery:
                command.append("--skip-precision-recovery")
            if args.resume:
                command.append("--resume")
            else:
                command.append("--overwrite")
            if args.fail_fast:
                command.append("--fail-fast")

            print(
                f"\n[oracle calibration cell] p={p_value} signal_strength={strength:g}",
                flush=True,
            )
            subprocess.run(command, check=True)

    if args.summarize:
        summarizer = Path(__file__).with_name("summarize_oracle_calibration.py")
        subprocess.run(
            [sys.executable, "-u", str(summarizer), "--out-dir", str(out_dir)],
            check=True,
        )
    print(f"\nV8 oracle calibration completed: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
