#!/usr/bin/env python3
"""Lightweight tests for V8 oracle calibration and completion ablation."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from research_common import apply_mask, complete_matrix, make_missing_mask
from run_experiment_b_completion_replacement import (
    completion_seed_for,
    downstream_seed_for,
)


def test_completion_ablation() -> None:
    rng = np.random.default_rng(21)
    X = pd.DataFrame(rng.normal(size=(36, 10)), columns=[f"x{i}" for i in range(10)])
    y = pd.Series(rng.integers(0, 2, size=len(X)), index=X.index)
    mask = make_missing_mask(X, y, "MCAR", 0.2, 9)
    X_missing = apply_mask(X, mask)

    mean_a = complete_matrix(
        "bayesianridge_mean",
        X_complete=X,
        X_missing=X_missing,
        seed=17,
        iterative_max_iter=3,
        iterative_nearest_features=6,
    )
    mean_b = complete_matrix(
        "bayesianridge_mean",
        X_complete=X,
        X_missing=X_missing,
        seed=17,
        iterative_max_iter=3,
        iterative_nearest_features=6,
    )
    post_a = complete_matrix(
        "bayesianridge_posterior",
        X_complete=X,
        X_missing=X_missing,
        seed=17,
        iterative_max_iter=3,
        iterative_nearest_features=6,
    )
    post_b = complete_matrix(
        "bayesianridge_posterior",
        X_complete=X,
        X_missing=X_missing,
        seed=18,
        iterative_max_iter=3,
        iterative_nearest_features=6,
    )
    assert np.allclose(mean_a, mean_b)
    assert not np.allclose(post_a.to_numpy()[mask], post_b.to_numpy()[mask])
    assert not np.allclose(mean_a.to_numpy()[mask], post_a.to_numpy()[mask])
    print("[ok] BayesianRidge mean is fixed and posterior draws vary")


def test_seed_design() -> None:
    downstream = downstream_seed_for(20260718, 0, 2, 3, 0)
    assert downstream == downstream_seed_for(20260718, 0, 2, 3, 0)
    fixed = completion_seed_for(20260718, 0, 2, 1, 0)
    posterior_1 = completion_seed_for(20260718, 0, 2, 1, 1)
    assert fixed != posterior_1
    print("[ok] common downstream seeds and varying posterior completion seeds")


def test_runner_static() -> None:
    path = Path(__file__).with_name("run_experiment_b_completion_replacement.py")
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"bayesianridge_mean"' in source
    assert '"oracle_calibration"' in source
    assert '"stabl_min"' in source
    for forbidden in ("stabl_q", "lcd_knockoff_plus", "component_selection_vote"):
        assert forbidden not in source
    print("[ok] V8 runner retains original STABL and supports oracle calibration")


def test_replacement_summarizer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        selection_rows = []
        recovery_rows = []
        diagnostic_rows = []
        completions = [
            "oracle_complete",
            "median",
            "bayesianridge_mean",
            "bayesianridge_posterior",
        ]
        for rep in range(3):
            for mechanism in ["MCAR", "MAR"]:
                for draw in range(2):
                    for completion in completions:
                        tp = 3 if completion == "oracle_complete" else 2
                        fp = {
                            "oracle_complete": 1,
                            "median": 3,
                            "bayesianridge_mean": 2,
                            "bayesianridge_posterior": 2,
                        }[completion]
                        selection_rows.append(
                            {
                                "dataset": "SSI_CyTOF",
                                "p_setting": "20",
                                "actual_p": 20,
                                "replicate": rep,
                                "mechanism": mechanism,
                                "completion": completion,
                                "completion_draw": draw,
                                "generator_label": "equicorr",
                                "fdp": fp / (tp + fp),
                                "power": tp / 4,
                                "n_selected": tp + fp,
                                "true_positives": tp,
                                "false_positives": fp,
                                "estimated_fdp": 0.4,
                                "threshold": 0.3,
                                "selected_indices": "0;1;2" if completion == "oracle_complete" else "0;1",
                                "error": "",
                            }
                        )
                        recovery_rows.append(
                            {
                                "dataset": "SSI_CyTOF",
                                "p_setting": "20",
                                "actual_p": 20,
                                "replicate": rep,
                                "mechanism": mechanism,
                                "completion": completion,
                                "completion_draw": draw,
                                "masked_rmse": 0.0 if completion == "oracle_complete" else 1.0,
                                "masked_mae": 0.0 if completion == "oracle_complete" else 0.8,
                                "sample_cov_fro_relative": 0.0 if completion == "oracle_complete" else 0.2,
                                "sample_corr_fro_relative": 0.0 if completion == "oracle_complete" else 0.2,
                                "error": "",
                            }
                        )
                        diagnostic_rows.append(
                            {
                                "dataset": "SSI_CyTOF",
                                "p_setting": "20",
                                "actual_p": 20,
                                "replicate": rep,
                                "mechanism": mechanism,
                                "completion": completion,
                                "completion_draw": draw,
                                "generator_label": "equicorr",
                                "pair_corr_mean": 0.8,
                                "s_relative_mean": 0.2,
                                "cov_kk_fro_relative": 0.3,
                                "cov_xk_offdiag_rmse": 0.05,
                                "error": "",
                            }
                        )
        pd.DataFrame(selection_rows).to_csv(base / "selection_results.csv", index=False)
        pd.DataFrame(recovery_rows).to_csv(base / "imputation_recovery.csv", index=False)
        pd.DataFrame(diagnostic_rows).to_csv(base / "draw_diagnostics.csv", index=False)
        script = Path(__file__).with_name("summarize_completion_replacement.py")
        subprocess.run(
            [sys.executable, str(script), "--out-dir", str(base), "--bootstrap-seed", "1"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        contrasts = pd.read_csv(base / "paired_completion_contrasts.csv")
        assert ((contrasts.reference == "median") & (contrasts.comparator == "bayesianridge_mean")).any()
        assert ((contrasts.reference == "bayesianridge_mean") & (contrasts.comparator == "bayesianridge_posterior")).any()
    print("[ok] V8 summarizer separates conditional-model and sampling effects")


def test_oracle_calibration_summarizer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for p_value in [100, 250]:
            for strength in [4.0, 6.0]:
                cell = root / f"p_{p_value}__strength_{str(strength).replace('.', 'p')}"
                cell.mkdir(parents=True)
                config = {"signal_strength": strength}
                (cell / "config.json").write_text(json.dumps(config), encoding="utf-8")
                rows = []
                for rep in range(3):
                    for draw in range(2):
                        power = min(0.9, strength / 10 + (0.05 if p_value == 100 else 0.0))
                        fdp = 0.5 if p_value == 100 else 0.65
                        rows.append(
                            {
                                "dataset": "SSI_CyTOF",
                                "p_setting": str(p_value),
                                "actual_p": p_value,
                                "replicate": rep,
                                "completion_draw": draw,
                                "generator_label": "equicorr",
                                "fdp": fdp,
                                "power": power,
                                "n_selected": 10,
                                "true_positives": 5,
                                "false_positives": 5,
                                "estimated_fdp": 0.5,
                                "threshold": 0.3,
                                "fdp_exceeds_target": 1.0,
                                "selected_indices": "0;1;2",
                                "error": "",
                            }
                        )
                pd.DataFrame(rows).to_csv(cell / "selection_results.csv", index=False)
        script = Path(__file__).with_name("summarize_oracle_calibration.py")
        subprocess.run(
            [sys.executable, str(script), "--out-dir", str(root)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        summary = pd.read_csv(root / "oracle_calibration_summary.csv")
        assert len(summary) == 4
        assert summary["recommended_for_replacement_pilot"].sum() == 1
        assert (root / "oracle_calibration_recommendation.csv").exists()
    print("[ok] V8 oracle calibration grid summarizer")


def test_cli_help() -> None:
    for script_name in (
        "run_experiment_b_completion_replacement.py",
        "run_experiment_b_oracle_calibration.py",
        "summarize_oracle_calibration.py",
    ):
        script = Path(__file__).with_name(script_name)
        subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    print("[ok] V8 command-line interfaces")


def main() -> None:
    test_completion_ablation()
    test_seed_design()
    test_runner_static()
    test_replacement_summarizer()
    test_oracle_calibration_summarizer()
    test_cli_help()
    print("[ok] all V8 update tests passed")


if __name__ == "__main__":
    main()
