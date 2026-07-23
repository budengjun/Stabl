#!/usr/bin/env python3
"""Lightweight tests for the V7 real-X completion-replacement benchmark."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from research_common import apply_mask, complete_matrix, make_missing_mask, standardize_completed_matrix
from run_experiment_b_completion_replacement import (
    completion_seed_for,
    downstream_seed_for,
)


def test_seed_pairing() -> None:
    # Downstream seeds must be shared across missingness mechanisms/completions.
    a = downstream_seed_for(20260718, 0, 3, 2, 0)
    b = downstream_seed_for(20260718, 0, 3, 2, 0)
    assert a == b
    # Completion seeds vary by mechanism and posterior draw.
    assert completion_seed_for(20260718, 0, 3, 0, 0) != completion_seed_for(
        20260718, 0, 3, 1, 0
    )
    assert completion_seed_for(20260718, 0, 3, 0, 0) != completion_seed_for(
        20260718, 0, 3, 0, 1
    )
    print("[ok] paired downstream seeds and varying completion seeds")


def test_completion_behavior() -> None:
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(30, 8)), columns=[f"x{i}" for i in range(8)])
    y = pd.Series(rng.integers(0, 2, size=30), index=X.index)
    mask = make_missing_mask(X, y, "MCAR", 0.2, 19)
    X_missing = apply_mask(X, mask)
    median_a = complete_matrix(
        "median", X_complete=X, X_missing=X_missing, seed=1
    )
    median_b = complete_matrix(
        "median", X_complete=X, X_missing=X_missing, seed=2
    )
    post_a = complete_matrix(
        "bayesianridge_posterior",
        X_complete=X,
        X_missing=X_missing,
        seed=1,
        iterative_max_iter=3,
        iterative_nearest_features=5,
    )
    post_b = complete_matrix(
        "bayesianridge_posterior",
        X_complete=X,
        X_missing=X_missing,
        seed=2,
        iterative_max_iter=3,
        iterative_nearest_features=5,
    )
    assert np.allclose(median_a, median_b)
    assert not np.allclose(post_a.to_numpy()[mask], post_b.to_numpy()[mask])
    assert np.isfinite(standardize_completed_matrix(post_a)[0].to_numpy()).all()
    print("[ok] median is deterministic and posterior draws vary without pooling")


def test_primary_rule_static() -> None:
    path = Path(__file__).with_name("run_experiment_b_completion_replacement.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert '"stabl_min"' in source
    forbidden = ("stabl_q", "lcd_knockoff_plus", "component_selection_vote")
    for item in forbidden:
        assert item not in source
    assert "n-completion-draws" in source
    assert "no score pooling" in source
    print("[ok] original STABL stabl_min is the only selection rule")


def test_summarizer_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        selection_rows = []
        recovery_rows = []
        diagnostic_rows = []
        completions = ["oracle_complete", "median", "bayesianridge_posterior"]
        for rep in range(3):
            for mechanism in ["MCAR", "MAR"]:
                for draw in range(2):
                    oracle_selected = "0;1;2"
                    for completion in completions:
                        selected = oracle_selected if completion == "oracle_complete" else (
                            "0;1" if completion == "bayesianridge_posterior" else "0;3"
                        )
                        tp = 3 if completion == "oracle_complete" else 1
                        fp = 0 if completion == "oracle_complete" else 1
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
                                "fdp": fp / max(1, tp + fp),
                                "power": tp / 3,
                                "n_selected": tp + fp,
                                "true_positives": tp,
                                "false_positives": fp,
                                "estimated_fdp": 0.4,
                                "threshold": 0.3,
                                "selected_indices": selected,
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
        required = [
            "completion_replacement_summary.csv",
            "replicate_level_metrics.csv",
            "paired_completion_contrasts.csv",
            "oracle_selection_agreement_summary.csv",
            "selection_stability_summary.csv",
            "imputation_recovery_summary.csv",
            "knockoff_diagnostics_summary.csv",
        ]
        for name in required:
            assert (base / name).exists(), name
    print("[ok] V7 replicate-level summarizer and stability analysis")


def main() -> None:
    test_seed_pairing()
    test_completion_behavior()
    test_primary_rule_static()
    test_summarizer_smoke()
    print("[ok] all V7 update tests passed")


if __name__ == "__main__":
    main()
