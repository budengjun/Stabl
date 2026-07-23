#!/usr/bin/env python3
"""V10 Stage 1: external omic generalization pilot for BR mean vs Median.

The pilot preserves the original STABL ``stabl_min`` pipeline and changes only
completion.  Each external block is evaluated with Oracle complete, Median, and
BayesianRidge conditional mean under paired MCAR/MAR missingness.  Signal
strengths are locked by the independent V10 Stage 0 calibration or specified
explicitly in the block configuration.
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

from v10_common import load_blocks, read_calibration_recommendations, validate_mechanisms


def fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="V10 external omic generalization pilot.",
    )
    parser.add_argument("--blocks-json", required=True)
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--out-dir", default="./experiment_B/v10_external_generalization_pilot")
    parser.add_argument("--pair-cache-dir", default="./pair_cache/experiment_B_v10_pilot")
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--n-completion-draws", type=int, default=5)
    parser.add_argument("--mechanisms", default="MCAR,MAR")
    parser.add_argument("--missing-rate", type=float, default=0.20)
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--iterative-max-iter", type=int, default=10)
    parser.add_argument("--iterative-nearest-features", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=20260730)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    parser.add_argument("--k-values", default="5,10,15,20")
    parser.add_argument("--pilot-power-loss-guard", type=float, default=0.05)
    parser.add_argument("--save-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-precision-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summarize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--delete-pair-cache-after-success",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both")
    if args.n_replicates < 2 or args.n_completion_draws < 2:
        raise ValueError("Use at least 2 replicates and 2 paired downstream runs")
    if args.n_bootstraps < 20:
        raise ValueError("Use at least 20 STABL bootstraps")
    if not 0 <= args.missing_rate < 1:
        raise ValueError("missing_rate must lie in [0, 1)")
    if args.pilot_power_loss_guard < 0:
        raise ValueError("pilot-power-loss-guard must be nonnegative")
    mechanisms = validate_mechanisms(
        item.strip() for item in args.mechanisms.split(",") if item.strip()
    )

    blocks = load_blocks(args.blocks_json)
    recommendations = (
        read_calibration_recommendations(args.calibration_dir)
        if args.calibration_dir
        else {}
    )
    resolved_blocks: list[dict[str, Any]] = []
    for block in blocks:
        if block.signal_strength is not None:
            strength = block.signal_strength
            p_value = block.p
            source = "blocks_json"
        elif block.label in recommendations:
            row = recommendations[block.label]
            strength = float(row["calibration_signal_strength"])
            p_value = int(row["actual_p"])
            source = "independent_v10_calibration"
        else:
            raise ValueError(
                f"Block {block.label!r} has no signal_strength and no calibration recommendation"
            )
        resolved_blocks.append(
            {
                **block.to_mapping(),
                "p": p_value,
                "signal_strength": strength,
                "strength_source": source,
            }
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    pair_cache = Path(args.pair_cache_dir).expanduser().resolve()
    sidecar = out_dir.parent / f"{out_dir.name}.v10_preregistration.json"
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_cache.mkdir(parents=True, exist_ok=True)

    preregistration: dict[str, Any] = {
        "experiment": "V10 external omic generalization pilot",
        "version": 10,
        "status": "prospective_pilot",
        "scientific_question": (
            "Does the reliability advantage of BayesianRidge conditional mean over "
            "Median generalize beyond SSI CyTOF while original STABL stabl_min remains fixed?"
        ),
        "blocks": resolved_blocks,
        "completion_methods": ["oracle_complete", "median", "bayesianridge_mean"],
        "missingness_mechanisms": list(mechanisms),
        "missing_rate": args.missing_rate,
        "generator": "gaussian_equicorrelated",
        "selection_rule": "stabl_min",
        "stabl_grid_size": 30,
        "threshold_grid": {"min": 0.10, "max": 0.99, "step": 0.01},
        "n_replicates": args.n_replicates,
        "n_completion_draws": args.n_completion_draws,
        "n_bootstraps": args.n_bootstraps,
        "primary_endpoints": [
            "realized_fdp",
            "power",
            "oracle_selection_jaccard",
            "precision_at_5",
            "precision_at_10",
            "precision_at_15",
            "precision_at_20",
            "average_precision",
        ],
        "reference_ranking_endpoints": [
            "support_ranking_auroc",
            "median_signal_rank",
            "mean_null_score",
            "mean_signal_score",
        ],
        "primary_comparison": "bayesianridge_mean minus median",
        "inference_unit": "simulation replicate after averaging paired downstream runs",
        "pilot_interpretation": (
            "directional external replication only; 10-replicate block results are not confirmatory"
        ),
        "progression_gate": {
            "fdp": "BR mean minus Median < 0",
            "oracle_jaccard": "BR mean minus Median > 0",
            "fixed_k_precision": "all configured Precision@k differences > 0",
            "power_guard": f"BR mean minus Median >= {-args.pilot_power_loss_guard}",
        },
        "random_state": args.random_state,
        "bootstrap_seed": args.bootstrap_seed,
        "calibration_seed_separation": (
            "Evaluation random_state must differ from V10 Stage 0 calibration random_state"
        ),
    }
    preregistration["preregistration_fingerprint"] = fingerprint(preregistration)

    if args.resume and sidecar.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if previous.get("preregistration_fingerprint") != preregistration["preregistration_fingerprint"]:
            raise RuntimeError("Resume preregistration does not match the existing V10 sidecar")
    else:
        sidecar.write_text(
            json.dumps(preregistration, indent=2, sort_keys=True), encoding="utf-8"
        )

    runner = Path(__file__).with_name("run_experiment_b_completion_replacement.py")
    base_summarizer = Path(__file__).with_name("summarize_completion_replacement.py")
    ranking_script = Path(__file__).with_name("reanalyze_v91_threshold_independent_ranking.py")

    for block_index, block in enumerate(resolved_blocks):
        block_label = str(block["label"])
        block_dir = out_dir / block_label
        block_cache = pair_cache / block_label
        block_seed = args.random_state + block_index * 10_000_000
        command = [
            sys.executable,
            "-u",
            str(runner),
            "--experiment-mode",
            "mean_confirmation",
            "--dataset",
            str(block["dataset"]),
            "--data-path",
            str(block["data_path"]),
            "--omic",
            str(block["omic"]),
            "--subject-mode",
            str(block["subject_mode"]),
            "--reference-complete-strategy",
            str(block["reference_complete_strategy"]),
            "--p-values",
            str(int(block["p"])),
            "--out-dir",
            str(block_dir),
            "--pair-cache-dir",
            str(block_cache),
            "--n-replicates",
            str(args.n_replicates),
            "--n-completion-draws",
            str(args.n_completion_draws),
            "--n-signal",
            str(int(block["n_signal"])),
            "--task",
            str(block["task"]),
            "--signal-strength",
            str(float(block["signal_strength"])),
            "--mechanisms",
            ",".join(mechanisms),
            "--missing-rate",
            str(args.missing_rate),
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
            str(block_seed),
        ]
        command.append("--save-scores" if args.save_scores else "--no-save-scores")
        if args.skip_precision_recovery:
            command.append("--skip-precision-recovery")
        command.append("--resume" if args.resume else "--overwrite")
        if args.fail_fast:
            command.append("--fail-fast")

        print(f"\n[V10 evaluation] block={block_label}", flush=True)
        print("[command]", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        (block_dir / "v10_block_config.json").write_text(
            json.dumps(
                {**block, "evaluation_random_state": block_seed},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(base_summarizer),
                "--out-dir",
                str(block_dir),
                "--bootstrap-seed",
                str(args.bootstrap_seed + block_index * 10_000_000),
            ],
            check=True,
        )
        if args.save_scores:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(ranking_script),
                    "--v9-out-dir",
                    str(block_dir),
                    "--out-dir",
                    str(block_dir / "v10_threshold_independent_ranking"),
                    "--k-values",
                    args.k_values,
                    "--mechanisms",
                    ",".join(mechanisms),
                    "--bootstrap-seed",
                    str(args.bootstrap_seed + block_index * 10_000_000),
                ],
                check=True,
            )

    shutil.copy2(sidecar, out_dir / "v10_preregistration.json")
    if args.summarize:
        summarizer = Path(__file__).with_name("summarize_v10_external_generalization.py")
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(summarizer),
                "--out-dir",
                str(out_dir),
                "--bootstrap-seed",
                str(args.bootstrap_seed),
                "--k-values",
                args.k_values,
                "--pilot-power-loss-guard",
                str(args.pilot_power_loss_guard),
            ],
            check=True,
        )

    if args.delete_pair_cache_after_success and pair_cache.exists():
        shutil.rmtree(pair_cache)
        print(f"[cleanup] deleted V10 pair cache: {pair_cache}", flush=True)

    print(f"\nV10 external generalization pilot completed: {out_dir}")


if __name__ == "__main__":
    main()
