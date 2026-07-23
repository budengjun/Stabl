#!/usr/bin/env python3
"""Regression tests for the V10 external generalization package."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from research_common import load_real_dataset
from v10_common import ExternalBlock, load_blocks


HERE = Path(__file__).resolve().parent


def test_block_config() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "blocks.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "label": "ssi_proteomics",
                        "dataset": "ssi",
                        "data_path": "${HOME}/ssi",
                        "omic": "Proteomics",
                        "p": 100,
                        "n_signal": 15,
                        "signal_strengths": [0.5, 1, 2, 4],
                    },
                    {
                        "label": "dream_phylotype",
                        "dataset": "dream",
                        "data_path": "~/dream",
                        "omic": "Phylotype",
                        "p": 100,
                        "n_signal": 15,
                        "signal_strength": 1.0,
                    },
                ]
            ),
            encoding="utf-8",
        )
        blocks = load_blocks(path)
        assert len(blocks) == 2
        assert blocks[0].dataset == "ssi"
        assert blocks[1].dataset == "dream"
        assert blocks[1].signal_strength == 1.0
    print("[ok] V10 block configuration resolves SSI and DREAM")


def test_dream_loader() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        specimens = ["A-01", "A-02", "B-01", "C-01"]
        pd.DataFrame(
            {"participant_id": ["A", "A", "B", "C"]}, index=specimens
        ).to_csv(root / "Patients_id.csv")
        pd.DataFrame({"was_preterm": [0, 0, 1, 0]}, index=specimens).to_csv(
            root / "Preterm.csv"
        )
        X = pd.DataFrame(
            {
                "f1": [0.0, 0.2, 1.0, 2.0],
                "f2": [1.0, 1.2, 0.0, 3.0],
                "f3": [2.0, 2.2, 1.0, 0.0],
            },
            index=specimens,
        )
        X.to_csv(root / "Phylotype.csv")
        X.rename(columns={"f1": "t1", "f2": "t2", "f3": "t3"}).to_csv(
            root / "Taxonomy.csv"
        )
        dataset = load_real_dataset(
            dataset="dream",
            data_path=root,
            omic="Phylotype",
            x_csv=None,
            groups_csv=None,
            subject_mode="first",
            complete_strategy="drop-columns",
            seed=17,
        )
        assert dataset.X.shape == (3, 3)
        assert dataset.dataset_label == "DREAM_Phylotype_first"
        assert dataset.groups is None
        assert dataset.metadata["n_subjects_raw"] == 3
    print("[ok] DREAM loader enforces one row per participant")


def _fake_block(root: Path, label: str, br_gain: float) -> None:
    block = root / label
    block.mkdir(parents=True)
    rows = []
    agreement = []
    ranking = []
    recovery = []
    diagnostics = []
    for mechanism_index, mechanism in enumerate(["MCAR", "MAR"]):
        for replicate in range(2):
            for draw in range(2):
                oracle_selected = "0;1;2"
                for completion in ["oracle_complete", "median", "bayesianridge_mean"]:
                    if completion == "oracle_complete":
                        fdp, power, selected = 0.30, 0.40, oracle_selected
                    elif completion == "median":
                        fdp, power, selected = 0.55, 0.30, "0;3;4"
                    else:
                        fdp, power, selected = 0.55 - br_gain, 0.30, "0;1;4"
                    rows.append(
                        {
                            "dataset": f"TEST_{label}",
                            "p_setting": "100",
                            "actual_p": 100,
                            "n": 20,
                            "replicate": replicate,
                            "mechanism": mechanism,
                            "completion": completion,
                            "completion_draw": draw,
                            "generator_label": "equicorr",
                            "generator": "gaussian_equicorrelated",
                            "selection_rule": "stabl_min",
                            "fdp": fdp,
                            "power": power,
                            "n_selected": 3,
                            "true_positives": 2,
                            "false_positives": 1,
                            "estimated_fdp": 0.4,
                            "threshold": 0.5,
                            "selected_indices": selected,
                            "error": "",
                        }
                    )
                    recovery.append(
                        {
                            "dataset": f"TEST_{label}",
                            "p_setting": "100",
                            "actual_p": 100,
                            "replicate": replicate,
                            "mechanism": mechanism,
                            "completion": completion,
                            "completion_draw": draw,
                            "masked_rmse": 1.0,
                            "masked_mae": 0.8,
                            "sample_cov_fro_relative": 0.2,
                            "sample_corr_fro_relative": 0.2,
                            "error": "",
                        }
                    )
                    diagnostics.append(
                        {
                            "dataset": f"TEST_{label}",
                            "p_setting": "100",
                            "actual_p": 100,
                            "replicate": replicate,
                            "mechanism": mechanism,
                            "completion": completion,
                            "completion_draw": draw,
                            "generator_label": "equicorr",
                            "pair_corr_mean": 0.7,
                            "s_relative_mean": 0.3,
                            "cov_kk_fro_relative": 0.2,
                            "cov_xk_offdiag_rmse": 0.05,
                            "error": "",
                        }
                    )
                    if completion != "oracle_complete":
                        agreement.append(
                            {
                                "dataset": f"TEST_{label}",
                                "p_setting": "100",
                                "actual_p": 100,
                                "replicate": replicate,
                                "mechanism": mechanism,
                                "completion_draw": draw,
                                "generator_label": "equicorr",
                                "completion": completion,
                                "oracle_selection_jaccard": 0.3 if completion == "median" else 0.5,
                                "oracle_selected_count_abs_difference": 0,
                            }
                        )
                for completion in ["oracle_complete", "median", "bayesianridge_mean"]:
                    base = 0.3 if completion == "median" else 0.35
                    ranking.append(
                        {
                            "p": 100,
                            "replicate": replicate,
                            "mechanism": mechanism,
                            "completion": completion,
                            "generator_label": "equicorr",
                            "precision_at_5": base,
                            "tp_at_5": base * 5,
                            "recall_at_5": base * 5 / 15,
                            "precision_at_10": base,
                            "tp_at_10": base * 10,
                            "recall_at_10": base * 10 / 15,
                            "precision_at_15": base,
                            "tp_at_15": base * 15,
                            "recall_at_15": base,
                            "precision_at_20": base,
                            "tp_at_20": base * 20,
                            "recall_at_20": base * 20 / 15,
                            "average_precision": base,
                            "median_signal_rank": 40 if completion == "median" else 35,
                            "support_ranking_auroc": base + 0.2,
                            "mean_signal_score": 0.2,
                            "mean_null_score": 0.15 if completion == "median" else 0.10,
                            "random_average_precision_baseline": 0.15,
                        }
                    )
    pd.DataFrame(rows).to_csv(block / "selection_results.csv", index=False)
    pd.DataFrame(agreement).to_csv(block / "oracle_selection_agreement.csv", index=False)
    pd.DataFrame(recovery).to_csv(block / "imputation_recovery.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(block / "draw_diagnostics.csv", index=False)
    rank_dir = block / "v10_threshold_independent_ranking"
    rank_dir.mkdir()
    pd.DataFrame(ranking).to_csv(rank_dir / "ranking_replicate_metrics.csv", index=False)
    integrity = []
    for mechanism in ["MCAR", "MAR"]:
        for completion in ["oracle_complete", "median", "bayesianridge_mean"]:
            integrity.append(
                {
                    "mechanism": mechanism,
                    "completion": completion,
                    "n_score_files": 4,
                    "n_replicates": 2,
                    "min_draws_per_replicate": 2,
                    "max_draws_per_replicate": 2,
                    "complete_pairing": True,
                }
            )
    pd.DataFrame(integrity).to_csv(rank_dir / "ranking_integrity_audit.csv", index=False)


def test_cross_block_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prereg = {
            "n_replicates": 2,
            "n_completion_draws": 2,
            "missingness_mechanisms": ["MCAR", "MAR"],
        }
        (root / "v10_preregistration.json").write_text(json.dumps(prereg), encoding="utf-8")
        _fake_block(root, "block_a", 0.10)
        _fake_block(root, "block_b", 0.08)
        subprocess.run(
            [
                sys.executable,
                str(HERE / "summarize_v10_external_generalization.py"),
                "--out-dir",
                str(root),
                "--n-bootstrap",
                "200",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        audit = pd.read_csv(root / "v10_integrity_audit.csv")
        gate = pd.read_csv(root / "v10_pilot_progression_gate.csv")
        assert audit["complete"].all()
        assert gate["pilot_progression_gate"].all()
        assert (root / "v10_external_generalization_report.md").exists()
    print("[ok] V10 cross-block summary preserves pairing and pilot labels")


def test_command_line_interfaces() -> None:
    for script in [
        "run_v10_oracle_calibration.py",
        "run_v10_external_generalization.py",
        "summarize_v10_external_generalization.py",
    ]:
        subprocess.run(
            [sys.executable, str(HERE / script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
    print("[ok] V10 command-line interfaces")


def main() -> None:
    test_block_config()
    test_dream_loader()
    test_cross_block_summary()
    test_command_line_interfaces()
    print("[ok] all V10 update tests passed")


if __name__ == "__main__":
    main()
