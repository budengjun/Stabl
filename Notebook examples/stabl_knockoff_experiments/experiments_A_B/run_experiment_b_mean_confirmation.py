#!/usr/bin/env python3
"""V9 confirmatory driver for BayesianRidge conditional mean vs median.

The scientific design is deliberately narrow.  It runs original STABL
``stabl_min`` on SSI CyTOF real-X semisynthetic outcomes and compares exactly
three completion branches: oracle complete, median, and BayesianRidge
conditional mean.  Posterior draws, score pooling, voting, and alternative
selection rules are excluded from the primary confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="V9 paired confirmation of BayesianRidge mean vs median.",
    )
    parser.add_argument("--data-path", default="../Sample Data/Biobank SSI")
    parser.add_argument("--out-dir", default="./experiment_B/ssi_br_mean_confirmation_v9")
    parser.add_argument("--pair-cache-dir", default="./pair_cache/experiment_B_v9")
    parser.add_argument("--n-replicates", type=int, default=50)
    parser.add_argument("--n-completion-draws", type=int, default=5)
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=20260718)
    parser.add_argument("--iterative-max-iter", type=int, default=10)
    parser.add_argument("--iterative-nearest-features", type=int, default=50)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--fdp-noninferiority-margin", type=float, default=0.0)
    parser.add_argument("--power-noninferiority-margin", type=float, default=0.0)
    parser.add_argument("--save-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-precision-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summarize", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both")
    if args.n_replicates < 1 or args.n_completion_draws < 2:
        raise ValueError("Use at least 1 replicate and 2 repeated downstream runs")
    if args.n_bootstraps < 20:
        raise ValueError("Use at least 20 STABL bootstraps")
    if args.fdp_noninferiority_margin < 0 or args.power_noninferiority_margin < 0:
        raise ValueError("Noninferiority margins must be nonnegative")

    out_dir = Path(args.out_dir).expanduser().resolve()
    pair_cache = Path(args.pair_cache_dir).expanduser().resolve()
    sidecar = out_dir.parent / f"{out_dir.name}.v9_preregistration.json"

    preregistration: dict[str, Any] = {
        "experiment": "SSI BayesianRidge conditional-mean confirmation",
        "version": 9,
        "status": "confirmatory",
        "dataset": "SSI CyTOF real-X",
        "subject_mode": "first",
        "reference_complete_strategy": "drop-columns",
        "p": 100,
        "n_signal": 15,
        "signal_strength": 4.0,
        "missingness_mechanisms": ["MCAR", "MAR"],
        "missing_rate": 0.20,
        "completion_methods": [
            "oracle_complete",
            "median",
            "bayesianridge_mean",
        ],
        "generator": "gaussian_equicorrelated",
        "selection_rule": "stabl_min",
        "stabl_grid_size": 30,
        "stabl_c_min": 0.01,
        "stabl_c_max": 1.0,
        "threshold_grid": {"min": 0.10, "max": 0.99, "step": 0.01},
        "target_fdr_field": 0.10,
        "n_replicates": args.n_replicates,
        "n_completion_draws": args.n_completion_draws,
        "n_bootstraps": args.n_bootstraps,
        "primary_comparison": "bayesianridge_mean minus median",
        "primary_endpoints": ["fdp", "power", "oracle_selection_jaccard"],
        "inference_unit": "simulation replicate after averaging downstream runs",
        "primary_ci": "paired percentile bootstrap 95% CI",
        "paired_test": "two-sided Wilcoxon signed-rank, Holm adjusted across primary tests",
        "decision_rule": {
            "fdp_nonworse": f"upper 95% CI of delta FDP <= {args.fdp_noninferiority_margin}",
            "power_nonworse": f"lower 95% CI of delta power >= {-args.power_noninferiority_margin}",
            "stable_improvement": (
                "at least one of: upper CI delta FDP < 0, lower CI delta power > 0, "
                "or lower CI delta Oracle Jaccard > 0"
            ),
        },
        "excluded_from_primary": [
            "BayesianRidge posterior draw",
            "multiple-imputation pooling",
            "voting",
            "stabl_q",
            "LCD knockoff+",
        ],
        "random_state": args.random_state,
        "bootstrap_seed": args.bootstrap_seed,
    }
    preregistration["preregistration_fingerprint"] = fingerprint(preregistration)

    if sidecar.exists() and args.resume:
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if previous.get("preregistration_fingerprint") != preregistration["preregistration_fingerprint"]:
            raise RuntimeError(
                "Resume preregistration does not match the existing sidecar. "
                "Use a new output directory."
            )
    else:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(preregistration, indent=2, sort_keys=True), encoding="utf-8"
        )

    runner = Path(__file__).with_name("run_experiment_b_completion_replacement.py")
    command = [
        sys.executable,
        "-u",
        str(runner),
        "--experiment-mode",
        "mean_confirmation",
        "--dataset",
        "ssi",
        "--data-path",
        args.data_path,
        "--omic",
        "CyTOF",
        "--subject-mode",
        "first",
        "--reference-complete-strategy",
        "drop-columns",
        "--p-values",
        "100",
        "--out-dir",
        str(out_dir),
        "--pair-cache-dir",
        str(pair_cache),
        "--n-replicates",
        str(args.n_replicates),
        "--n-completion-draws",
        str(args.n_completion_draws),
        "--n-signal",
        "15",
        "--task",
        "classification",
        "--signal-strength",
        "4",
        "--mechanisms",
        "MCAR,MAR",
        "--missing-rate",
        "0.20",
        "--completion-methods",
        "oracle_complete,median,bayesianridge_mean",
        "--generators",
        "gaussian_equicorrelated",
        "--n-bootstraps",
        str(args.n_bootstraps),
        "--stabl-grid-size",
        "30",
        "--stabl-c-min",
        "0.01",
        "--stabl-c-max",
        "1.0",
        "--threshold-min",
        "0.10",
        "--threshold-max",
        "0.99",
        "--threshold-step",
        "0.01",
        "--target-fdr",
        "0.10",
        "--n-jobs",
        str(args.n_jobs),
        "--iterative-max-iter",
        str(args.iterative_max_iter),
        "--iterative-nearest-features",
        str(args.iterative_nearest_features),
        "--random-state",
        str(args.random_state),
    ]
    command.append("--save-scores" if args.save_scores else "--no-save-scores")
    if args.skip_precision_recovery:
        command.append("--skip-precision-recovery")
    if args.resume:
        command.append("--resume")
    elif args.overwrite:
        command.append("--overwrite")
    if args.fail_fast:
        command.append("--fail-fast")

    print("[V9 preregistration]", sidecar, flush=True)
    print("[V9 runner]", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sidecar, out_dir / "v9_preregistration.json")

    if args.summarize:
        base_summary = Path(__file__).with_name("summarize_completion_replacement.py")
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(base_summary),
                "--out-dir",
                str(out_dir),
                "--bootstrap-seed",
                str(args.bootstrap_seed),
            ],
            check=True,
        )
        confirmatory_summary = Path(__file__).with_name("summarize_mean_confirmation.py")
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(confirmatory_summary),
                "--out-dir",
                str(out_dir),
                "--bootstrap-seed",
                str(args.bootstrap_seed),
                "--fdp-noninferiority-margin",
                str(args.fdp_noninferiority_margin),
                "--power-noninferiority-margin",
                str(args.power_noninferiority_margin),
            ],
            check=True,
        )

    print(f"V9 confirmation completed: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
