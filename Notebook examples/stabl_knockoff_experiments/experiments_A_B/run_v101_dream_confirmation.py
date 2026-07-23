#!/usr/bin/env python3
"""V10.1B: fresh DREAM 50-replicate confirmatory experiment.

The experiment keeps the original STABL stabl_min pipeline fixed and compares
Median with BayesianRidge conditional mean on DREAM Phylotype under MCAR and
MAR. The default confirmatory design uses fresh seeds that are disjoint from
V10 calibration, V10 pilot, and V10.1A guard seeds.
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
        description="V10.1B fresh DREAM confirmation for BR mean vs Median.",
    )
    parser.add_argument("--data-path", default="../Sample Data/Dream")
    parser.add_argument("--out-dir", default="./experiment_B/v101_dream_confirmation")
    parser.add_argument("--pair-cache-dir", default="./pair_cache/experiment_B_v101_confirmation")
    parser.add_argument("--n-replicates", type=int, default=50)
    parser.add_argument("--n-completion-draws", type=int, default=5)
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--iterative-max-iter", type=int, default=10)
    parser.add_argument("--iterative-nearest-features", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=50260730)
    parser.add_argument("--bootstrap-seed", type=int, default=50260730)
    parser.add_argument("--power-noninferiority-margin", type=float, default=0.03)
    parser.add_argument("--fdp-nonworsening-margin", type=float, default=0.03)
    parser.add_argument("--analysis-status", choices=("confirmatory", "smoke"), default="confirmatory")
    parser.add_argument("--save-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--delete-pair-cache-after-success", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both")
    if args.n_replicates < 2 or args.n_completion_draws < 2:
        raise ValueError("Use at least 2 replicates and 2 paired downstream runs")
    if args.n_bootstraps < 20:
        raise ValueError("Use at least 20 STABL bootstraps")
    if args.power_noninferiority_margin < 0 or args.fdp_nonworsening_margin < 0:
        raise ValueError("Noninferiority and non-worsening margins must be nonnegative")
    if args.analysis_status == "confirmatory":
        required = (50, 5, 100)
        observed = (args.n_replicates, args.n_completion_draws, args.n_bootstraps)
        if observed != required:
            raise ValueError(
                "Confirmatory status requires exactly 50 replicates, 5 paired runs, "
                f"and 100 bootstraps; observed {observed}. Use --analysis-status smoke "
                "for engineering runs."
            )

    out_dir = Path(args.out_dir).expanduser().resolve()
    pair_cache = Path(args.pair_cache_dir).expanduser().resolve()
    sidecar = out_dir.parent / f"{out_dir.name}.v101_preregistration.json"
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_cache.mkdir(parents=True, exist_ok=True)

    preregistration: dict[str, Any] = {
        "experiment": "V10.1B DREAM fresh confirmatory experiment",
        "version": "10.1B",
        "analysis_status": args.analysis_status,
        "scientific_question": (
            "Does BayesianRidge conditional mean provide a reliability-oriented "
            "replacement for Median in the original STABL stabl_min pipeline on "
            "an external DREAM Phylotype block?"
        ),
        "dataset": "DREAM Phylotype, first specimen per participant",
        "n": "resolved by loader; expected 580",
        "p": 100,
        "n_signal": 15,
        "signal_strength": 4.0,
        "missingness": ["MCAR", "MAR"],
        "missing_rate": 0.20,
        "completion_methods": ["oracle_complete", "median", "bayesianridge_mean"],
        "iterative_max_iter": args.iterative_max_iter,
        "generator": "gaussian_equicorrelated",
        "selection_rule": "original_stabl_min",
        "n_replicates": args.n_replicates,
        "n_completion_draws": args.n_completion_draws,
        "n_bootstraps": args.n_bootstraps,
        "primary_comparison": "bayesianridge_mean minus median",
        "inference_unit": "simulation replicate after averaging five paired STABL runs",
        "primary_superiority_endpoints": [
            "oracle_selection_jaccard",
            "average_precision",
        ],
        "supportive_ranking_endpoints": [
            "support_ranking_auroc",
            "precision_at_5",
            "precision_at_10",
            "precision_at_15",
            "precision_at_20",
            "mean_null_score",
        ],
        "guard_endpoints": {
            "power": {
                "criterion": "lower 95% paired bootstrap CI >= -margin",
                "margin": args.power_noninferiority_margin,
            },
            "fdp": {
                "criterion": "upper 95% paired bootstrap CI <= +margin",
                "margin": args.fdp_nonworsening_margin,
            },
        },
        "mechanism_level_success": (
            "Oracle Jaccard and average precision lower CI > 0, power lower CI "
            f">= -{args.power_noninferiority_margin:g}, and FDP upper CI "
            f"<= +{args.fdp_nonworsening_margin:g}."
        ),
        "overall_success": "both MCAR and MAR meet the mechanism-level gate",
        "fresh_seed_statement": (
            "random_state 50260730 is disjoint from V10 calibration 30260720, "
            "V10 pilot 30260730, and V10.1A guard 40260730"
        ),
        "random_state": args.random_state,
        "bootstrap_seed": args.bootstrap_seed,
    }
    preregistration["preregistration_fingerprint"] = fingerprint(preregistration)

    if args.resume and sidecar.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if previous.get("preregistration_fingerprint") != preregistration["preregistration_fingerprint"]:
            raise RuntimeError("Resume preregistration does not match the existing V10.1B sidecar")
    else:
        sidecar.write_text(json.dumps(preregistration, indent=2, sort_keys=True), encoding="utf-8")

    runner = Path(__file__).with_name("run_experiment_b_completion_replacement.py")
    base_summarizer = Path(__file__).with_name("summarize_completion_replacement.py")
    ranking_script = Path(__file__).with_name("reanalyze_v91_threshold_independent_ranking.py")
    confirmatory_summarizer = Path(__file__).with_name("summarize_v101_dream_confirmation.py")

    command = [
        sys.executable, "-u", str(runner),
        "--experiment-mode", "mean_confirmation",
        "--dataset", "dream",
        "--data-path", str(Path(args.data_path).expanduser()),
        "--omic", "Phylotype",
        "--subject-mode", "first",
        "--reference-complete-strategy", "drop-columns",
        "--p-values", "100",
        "--out-dir", str(out_dir),
        "--pair-cache-dir", str(pair_cache),
        "--n-replicates", str(args.n_replicates),
        "--n-completion-draws", str(args.n_completion_draws),
        "--n-signal", "15",
        "--task", "classification",
        "--signal-strength", "4.0",
        "--mechanisms", "MCAR,MAR",
        "--missing-rate", "0.2",
        "--completion-methods", "oracle_complete,median,bayesianridge_mean",
        "--generators", "gaussian_equicorrelated",
        "--n-bootstraps", str(args.n_bootstraps),
        "--stabl-grid-size", "30",
        "--stabl-c-min", "0.01",
        "--stabl-c-max", "1.0",
        "--threshold-min", "0.10",
        "--threshold-max", "0.99",
        "--threshold-step", "0.01",
        "--target-fdr", "0.10",
        "--n-jobs", str(args.n_jobs),
        "--iterative-max-iter", str(args.iterative_max_iter),
        "--iterative-nearest-features", str(args.iterative_nearest_features),
        "--random-state", str(args.random_state),
        "--skip-precision-recovery",
        "--resume" if args.resume else "--overwrite",
        "--save-scores" if args.save_scores else "--no-save-scores",
    ]
    if args.fail_fast:
        command.append("--fail-fast")
    print("[V10.1B command]", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    shutil.copy2(sidecar, out_dir / "v101_preregistration.json")
    subprocess.run(
        [
            sys.executable, "-u", str(base_summarizer),
            "--out-dir", str(out_dir),
            "--bootstrap-seed", str(args.bootstrap_seed),
        ],
        check=True,
    )
    if args.save_scores:
        subprocess.run(
            [
                sys.executable, "-u", str(ranking_script),
                "--v9-out-dir", str(out_dir),
                "--out-dir", str(out_dir / "v101_threshold_independent_ranking"),
                "--k-values", "5,10,15,20",
                "--mechanisms", "MCAR,MAR",
                "--bootstrap-seed", str(args.bootstrap_seed),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable, "-u", str(confirmatory_summarizer),
            "--out-dir", str(out_dir),
            "--bootstrap-seed", str(args.bootstrap_seed),
            "--power-noninferiority-margin", str(args.power_noninferiority_margin),
            "--fdp-nonworsening-margin", str(args.fdp_nonworsening_margin),
        ],
        check=True,
    )

    if args.delete_pair_cache_after_success and pair_cache.exists():
        shutil.rmtree(pair_cache)
        print(f"[cleanup] deleted pair cache: {pair_cache}", flush=True)

    print(f"V10.1B DREAM confirmation completed: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
