#!/usr/bin/env python3
"""V10 Stage 0: independent oracle difficulty calibration per external block.

Calibration is deliberately separated from the V10 evaluation pilot.  It uses
one random seed to select a signal strength that places complete-data STABL in
an informative power range.  The external generalization pilot must use a
different random seed, preventing the V8 winner's-curse problem from being
carried into the evaluation estimates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from research_common import atomic_write_csv
from v10_common import load_blocks


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="V10 independent oracle calibration across external omic blocks.",
    )
    parser.add_argument("--blocks-json", required=True)
    parser.add_argument("--out-dir", default="./experiment_B/v10_oracle_calibration")
    parser.add_argument("--pair-cache-dir", default="./pair_cache/experiment_B_v10_calibration")
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--n-completion-draws", type=int, default=3)
    parser.add_argument("--n-bootstraps", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--target-power-low", type=float, default=0.30)
    parser.add_argument("--target-power-high", type=float, default=0.60)
    parser.add_argument("--soft-fdp-ceiling", type=float, default=0.65)
    parser.add_argument("--random-state", type=int, default=20260720)
    parser.add_argument("--save-scores", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--delete-pair-cache-after-success",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both")
    if args.n_replicates < 2:
        raise ValueError("Calibration should use at least 2 replicates")
    if args.n_completion_draws < 1 or args.n_bootstraps < 20:
        raise ValueError("Use at least 1 draw and 20 STABL bootstraps")
    if not 0 < args.target_power_low < args.target_power_high < 1:
        raise ValueError("Invalid target power interval")

    blocks = load_blocks(args.blocks_json)
    out_dir = Path(args.out_dir).expanduser().resolve()
    pair_cache = Path(args.pair_cache_dir).expanduser().resolve()
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_cache.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "V10_external_omic_oracle_calibration",
        "version": 10,
        "scientific_role": "difficulty calibration only; no completion-method inference",
        "evaluation_seed_must_differ": True,
        "blocks": [block.to_mapping() for block in blocks],
        **vars(args),
    }
    (out_dir / "v10_calibration_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    calibration_runner = Path(__file__).with_name("run_experiment_b_oracle_calibration.py")
    recommendation_rows: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        block_dir = out_dir / block.label
        block_cache = pair_cache / block.label
        command = [
            sys.executable,
            "-u",
            str(calibration_runner),
            "--dataset",
            block.dataset,
            "--data-path",
            block.data_path,
            "--omic",
            block.omic,
            "--subject-mode",
            block.subject_mode,
            "--reference-complete-strategy",
            block.reference_complete_strategy,
            "--p-values",
            str(block.p),
            "--signal-strengths",
            ",".join(f"{value:g}" for value in block.signal_strengths),
            "--out-dir",
            str(block_dir),
            "--pair-cache-dir",
            str(block_cache),
            "--n-replicates",
            str(args.n_replicates),
            "--n-completion-draws",
            str(args.n_completion_draws),
            "--n-signal",
            str(block.n_signal),
            "--task",
            block.task,
            "--n-bootstraps",
            str(args.n_bootstraps),
            "--n-jobs",
            str(args.n_jobs),
            "--random-state",
            str(args.random_state + index * 10_000_000),
            "--target-fdr",
            "0.10",
            "--generators",
            "gaussian_equicorrelated",
        ]
        command.append("--save-scores" if args.save_scores else "--no-save-scores")
        command.append("--resume" if args.resume else "--overwrite")
        if args.fail_fast:
            command.append("--fail-fast")
        print(f"\n[V10 calibration] block={block.label}", flush=True)
        print("[command]", " ".join(command), flush=True)
        subprocess.run(command, check=True)

        # Re-run the small summary with V10's explicitly recorded target window.
        summarizer = Path(__file__).with_name("summarize_oracle_calibration.py")
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(summarizer),
                "--out-dir",
                str(block_dir),
                "--target-power-low",
                str(args.target_power_low),
                "--target-power-high",
                str(args.target_power_high),
                "--soft-fdp-ceiling",
                str(args.soft_fdp_ceiling),
            ],
            check=True,
        )
        recommendation = pd.read_csv(block_dir / "oracle_calibration_recommendation.csv").iloc[0]
        recommendation_rows.append(
            {
                "block_label": block.label,
                "dataset": block.dataset,
                "omic": block.omic,
                "n_signal": block.n_signal,
                "actual_p": int(recommendation["actual_p"]),
                "calibration_signal_strength": float(
                    recommendation["calibration_signal_strength"]
                ),
                "oracle_power_calibration": float(recommendation["power_mean"]),
                "oracle_fdp_calibration": float(recommendation["empirical_fdr"]),
                "oracle_selected_calibration": float(recommendation["selected_mean"]),
                "meets_power_window": (
                    bool(recommendation["meets_power_window"])
                    if isinstance(recommendation["meets_power_window"], (bool, np.bool_))
                    else str(recommendation["meets_power_window"]).strip().lower() == "true"
                ),
                "calibration_random_state": args.random_state + index * 10_000_000,
                "evaluation_status": "locked_for_fresh_seed_pilot",
            }
        )

    recommendations = pd.DataFrame(recommendation_rows)
    atomic_write_csv(recommendations, out_dir / "v10_calibration_recommendations.csv")
    (out_dir / "v10_calibration_recommendations.json").write_text(
        json.dumps(recommendation_rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.delete_pair_cache_after_success and pair_cache.exists():
        shutil.rmtree(pair_cache)
        print(f"[cleanup] deleted calibration pair cache: {pair_cache}", flush=True)

    print("\nV10 calibration recommendations")
    print(recommendations.to_string(index=False))
    print(f"\nV10 Stage 0 completed: {out_dir}")


if __name__ == "__main__":
    main()
