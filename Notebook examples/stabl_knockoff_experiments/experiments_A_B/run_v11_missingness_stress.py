#!/usr/bin/env python3
"""V11 missingness-rate and block-missingness stress test.

The V11 map preserves the original STABL ``stabl_min`` pipeline and compares
Oracle complete, Median, and BayesianRidge conditional-mean completion on
SSI CyTOF and DREAM Phylotype.  Missingness rates are paired and nested within
replicate by reusing the same outcome, support, and mask seed at each rate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from v10_common import load_blocks
from v11_common import fingerprint, parse_float_list, parse_items, rate_label


STAGE_OFFSETS = {
    "smoke": 0,
    "pilot": 10_000_000,
    "stress": 20_000_000,
}


def validate_design(args: argparse.Namespace, rates: tuple[float, ...], mechanisms: tuple[str, ...]) -> None:
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both")
    if args.n_replicates < 1 or args.n_completion_draws < 1:
        raise ValueError("Replicate and paired-run counts must be positive")
    if args.n_bootstraps < 20:
        raise ValueError("Use at least 20 STABL bootstraps")
    if args.block_feature_blocks < 1:
        raise ValueError("block-feature-blocks must be positive")
    invalid = sorted(set(mechanisms) - {"MCAR", "MAR", "BLOCK"})
    if invalid:
        raise ValueError(f"V11 supports MCAR, MAR, and BLOCK only; invalid={invalid}")

    canonical_rates = (0.10, 0.20, 0.30, 0.40)
    canonical_mechanisms = ("MCAR", "MAR", "BLOCK")
    if args.analysis_status == "pilot":
        expected = (10, 3, 50)
        observed = (args.n_replicates, args.n_completion_draws, args.n_bootstraps)
        if observed != expected:
            raise ValueError(f"Pilot status requires {expected}; observed {observed}")
        if rates != canonical_rates or mechanisms != canonical_mechanisms:
            raise ValueError("Pilot status requires rates 0.1,0.2,0.3,0.4 and MCAR,MAR,BLOCK")
    elif args.analysis_status == "stress":
        expected = (20, 5, 100)
        observed = (args.n_replicates, args.n_completion_draws, args.n_bootstraps)
        if observed != expected:
            raise ValueError(f"Stress status requires {expected}; observed {observed}")
        if rates != canonical_rates or mechanisms != canonical_mechanisms:
            raise ValueError("Stress status requires rates 0.1,0.2,0.3,0.4 and MCAR,MAR,BLOCK")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="V11 missingness stress map for BR mean versus Median.",
    )
    parser.add_argument("--blocks-json", required=True)
    parser.add_argument("--out-dir", default="./experiment_B/v11_missingness_stress_pilot")
    parser.add_argument("--pair-cache-dir", default="./pair_cache/experiment_B_v11_stress")
    parser.add_argument("--analysis-status", choices=("smoke", "pilot", "stress"), default="pilot")
    parser.add_argument("--missing-rates", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--mechanisms", default="MCAR,MAR,BLOCK")
    parser.add_argument("--block-feature-blocks", type=int, default=5)
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--n-completion-draws", type=int, default=3)
    parser.add_argument("--n-bootstraps", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--iterative-max-iter", type=int, default=10)
    parser.add_argument("--iterative-nearest-features", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=60260730)
    parser.add_argument("--bootstrap-seed", type=int, default=60260730)
    parser.add_argument("--k-values", default="5,10,15,20")
    parser.add_argument("--power-loss-guard", type=float, default=0.05)
    parser.add_argument("--fdp-worsening-guard", type=float, default=0.05)
    parser.add_argument("--save-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--delete-pair-cache-after-success",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    rates = parse_float_list(args.missing_rates)
    mechanisms = tuple(item.upper() for item in parse_items(args.mechanisms))
    validate_design(args, rates, mechanisms)
    if args.power_loss_guard < 0 or args.fdp_worsening_guard < 0:
        raise ValueError("Guard margins must be nonnegative")

    blocks = load_blocks(args.blocks_json)
    resolved_blocks: list[dict[str, Any]] = []
    for block in blocks:
        if block.signal_strength is None:
            raise ValueError(f"V11 block {block.label!r} requires a fixed signal_strength")
        resolved_blocks.append(block.to_mapping())

    out_dir = Path(args.out_dir).expanduser().resolve()
    pair_cache_root = Path(args.pair_cache_dir).expanduser().resolve()
    sidecar = out_dir.parent / f"{out_dir.name}.v11_preregistration.json"
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_cache_root.mkdir(parents=True, exist_ok=True)

    effective_random_state = args.random_state + STAGE_OFFSETS[args.analysis_status]
    effective_bootstrap_seed = args.bootstrap_seed + STAGE_OFFSETS[args.analysis_status]
    preregistration: dict[str, Any] = {
        "experiment": "V11 missingness stress map",
        "version": 11,
        "analysis_status": args.analysis_status,
        "scientific_question": (
            "Across what missingness rates and mechanisms does BayesianRidge conditional "
            "mean retain a reliability advantage over Median while original STABL stabl_min remains fixed?"
        ),
        "blocks": resolved_blocks,
        "missing_rates": list(rates),
        "missingness_mechanisms": list(mechanisms),
        "block_missingness_definition": (
            "Seed-shuffled features are partitioned into fixed blocks; each feature block is "
            "masked for a seed-specific sample subset. Reusing the seed makes rate masks nested."
        ),
        "block_feature_blocks": args.block_feature_blocks,
        "completion_methods": ["oracle_complete", "median", "bayesianridge_mean"],
        "generator": "gaussian_equicorrelated",
        "selection_rule": "original_stabl_min",
        "p": 100,
        "n_signal": 15,
        "signal_strength": 4.0,
        "n_replicates": args.n_replicates,
        "n_completion_draws": args.n_completion_draws,
        "n_bootstraps": args.n_bootstraps,
        "primary_comparison": "bayesianridge_mean minus median",
        "inference_unit": "simulation replicate after averaging paired STABL runs",
        "primary_endpoints": [
            "realized_fdp",
            "power",
            "oracle_selection_jaccard",
            "average_precision",
            "support_ranking_auroc",
        ],
        "mechanism_diagnostics": [
            "mean_null_score",
            "masked_rmse",
            "sample_cov_fro_relative",
            "pair_corr_abs_error_to_oracle",
            "s_relative_abs_error_to_oracle",
        ],
        "directional_cell_gate": {
            "oracle_selection_jaccard": "difference > 0",
            "average_precision": "difference > 0",
            "power": f"difference >= -{args.power_loss_guard:g}",
            "fdp": f"difference <= +{args.fdp_worsening_guard:g}",
        },
        "strong_cell_gate": (
            "Oracle Jaccard and average-precision lower 95% CIs > 0, power lower CI above "
            f"-{args.power_loss_guard:g}, and FDP upper CI below +{args.fdp_worsening_guard:g}."
        ),
        "interpretation": (
            "Smoke has no scientific interpretation. Pilot is a directional boundary screen. "
            "Stress mode maps supported and mixed cells but is not a universal-superiority claim."
        ),
        "nested_rate_pairing": (
            "Within each dataset and replicate, outcome, support, and mask seed are shared across rates."
        ),
        "base_random_state": args.random_state,
        "effective_random_state": effective_random_state,
        "base_bootstrap_seed": args.bootstrap_seed,
        "effective_bootstrap_seed": effective_bootstrap_seed,
        "stage_seed_offset": STAGE_OFFSETS[args.analysis_status],
        "k_values": args.k_values,
        "save_scores": args.save_scores,
    }
    preregistration["preregistration_fingerprint"] = fingerprint(preregistration)

    if args.resume and sidecar.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if previous.get("preregistration_fingerprint") != preregistration["preregistration_fingerprint"]:
            raise RuntimeError("Resume preregistration does not match the existing V11 sidecar")
    else:
        sidecar.write_text(json.dumps(preregistration, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copy2(sidecar, out_dir / "v11_preregistration.json")

    runner = Path(__file__).with_name("run_experiment_b_completion_replacement.py")
    base_summarizer = Path(__file__).with_name("summarize_completion_replacement.py")
    ranking_script = Path(__file__).with_name("reanalyze_v91_threshold_independent_ranking.py")
    v11_summarizer = Path(__file__).with_name("summarize_v11_missingness_stress.py")

    for block_index, block in enumerate(resolved_blocks):
        block_label = str(block["label"])
        block_seed = effective_random_state + block_index * 100_000_000
        for rate in rates:
            label = rate_label(rate)
            cell_dir = out_dir / block_label / label
            cell_cache = pair_cache_root / block_label / label
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
                str(cell_dir),
                "--pair-cache-dir",
                str(cell_cache),
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
                str(rate),
                "--block-feature-blocks",
                str(args.block_feature_blocks),
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
                "--skip-precision-recovery",
                "--resume" if args.resume else "--overwrite",
                "--save-scores" if args.save_scores else "--no-save-scores",
            ]
            if args.fail_fast:
                command.append("--fail-fast")
            print(f"\n[V11 cell] block={block_label} rate={rate:g}", flush=True)
            print("[command]", " ".join(command), flush=True)
            subprocess.run(command, check=True)
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(base_summarizer),
                    "--out-dir",
                    str(cell_dir),
                    "--bootstrap-seed",
                    str(effective_bootstrap_seed + block_index * 100_000 + int(round(rate * 1000))),
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
                        str(cell_dir),
                        "--out-dir",
                        str(cell_dir / "v11_threshold_independent_ranking"),
                        "--k-values",
                        args.k_values,
                        "--mechanisms",
                        ",".join(mechanisms),
                        "--bootstrap-seed",
                        str(effective_bootstrap_seed + block_index * 100_000 + int(round(rate * 1000))),
                    ],
                    check=True,
                )

    if args.summarize:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(v11_summarizer),
                "--out-dir",
                str(out_dir),
                "--bootstrap-seed",
                str(effective_bootstrap_seed),
                "--power-loss-guard",
                str(args.power_loss_guard),
                "--fdp-worsening-guard",
                str(args.fdp_worsening_guard),
            ],
            check=True,
        )

    audit_path = out_dir / "v11_integrity_audit.csv"
    complete = False
    if audit_path.exists():
        import pandas as pd

        audit = pd.read_csv(audit_path)
        complete = bool(not audit.empty and audit["complete"].all())
    if args.delete_pair_cache_after_success and complete and pair_cache_root.exists():
        shutil.rmtree(pair_cache_root)
        print(f"[V11 cache] deleted {pair_cache_root}", flush=True)

    print(f"V11 missingness stress completed: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
