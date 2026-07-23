#!/usr/bin/env python3
"""Lightweight tests for V10.1 DREAM confirmation and SSI diagnosis."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def test_scripts_parse() -> None:
    names = [
        "run_v101_dream_convergence_guard.py",
        "summarize_v101_dream_guard.py",
        "run_v101_dream_confirmation.py",
        "summarize_v101_dream_confirmation.py",
        "reanalyze_v101_ssi_threshold_path.py",
    ]
    for name in names:
        ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
    text = (ROOT / "run_v101_dream_confirmation.py").read_text(encoding="utf-8")
    assert "power-noninferiority-margin" in text
    assert "fdp-nonworsening-margin" in text
    assert "average_precision" in text
    assert "oracle_selection_jaccard" in text
    print("[ok] V10.1 scripts parse and preserve the preregistered gates")


def test_confirmation_summary() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        prereg = {"analysis_status": "smoke", "n_replicates": 2, "n_completion_draws": 2}
        (out / "v101_preregistration.json").write_text(json.dumps(prereg), encoding="utf-8")
        selection_rows = []
        replicate_rows = []
        oracle_rows = []
        ranking_rows = []
        for mechanism_index, mechanism in enumerate(("MCAR", "MAR")):
            for replicate in range(2):
                values = {
                    "oracle_complete": (0.10, 0.50),
                    "median": (0.20, 0.42),
                    "bayesianridge_mean": (0.15, 0.43),
                }
                for completion, (fdp, power) in values.items():
                    replicate_rows.append({
                        "replicate": replicate, "mechanism": mechanism,
                        "completion": completion, "generator_label": "equicorr",
                        "fdp": fdp + replicate * 0.01, "power": power,
                    })
                    for draw in range(2):
                        selection_rows.append({
                            "replicate": replicate, "mechanism": mechanism,
                            "completion": completion, "completion_draw": draw,
                            "generator_label": "equicorr", "error": "",
                        })
                    ranking_rows.append({
                        "replicate": replicate, "mechanism": mechanism,
                        "completion": completion, "generator_label": "equicorr",
                        "average_precision": 0.50 if completion == "median" else 0.60 if completion == "bayesianridge_mean" else 0.70,
                        "support_ranking_auroc": 0.60 if completion == "median" else 0.65 if completion == "bayesianridge_mean" else 0.75,
                        "precision_at_5": 0.5, "precision_at_10": 0.4,
                        "precision_at_15": 0.3, "precision_at_20": 0.25,
                        "mean_null_score": 0.2 if completion == "median" else 0.1,
                        "mean_signal_score": 0.6,
                    })
                for completion, value in (("median", 0.40), ("bayesianridge_mean", 0.55)):
                    for draw in range(2):
                        oracle_rows.append({
                            "replicate": replicate, "mechanism": mechanism,
                            "completion": completion, "completion_draw": draw,
                            "generator_label": "equicorr",
                            "oracle_selection_jaccard": value,
                            "oracle_selected_count_abs_difference": 1,
                        })
        pd.DataFrame(selection_rows).to_csv(out / "selection_results.csv", index=False)
        pd.DataFrame(replicate_rows).to_csv(out / "replicate_level_metrics.csv", index=False)
        pd.DataFrame(oracle_rows).to_csv(out / "oracle_selection_agreement.csv", index=False)
        ranking_dir = out / "v101_threshold_independent_ranking"
        ranking_dir.mkdir()
        pd.DataFrame(ranking_rows).to_csv(ranking_dir / "ranking_replicate_metrics.csv", index=False)
        subprocess.run(
            [sys.executable, str(ROOT / "summarize_v101_dream_confirmation.py"),
             "--out-dir", str(out), "--n-bootstrap", "200"],
            check=True, capture_output=True, text=True,
        )
        audit = pd.read_csv(out / "v101_integrity_audit.csv")
        decision = pd.read_csv(out / "v101_confirmatory_decision.csv")
        assert audit["complete"].all()
        assert set(decision["overall_decision"]) == {"smoke_only_no_scientific_decision"}
    print("[ok] V10.1 confirmation summary preserves pairing and smoke labels")


def test_guard_summary() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        config = {
            "max_iters": [10, 25], "mechanisms": ["MCAR", "MAR"],
            "n_replicates": 2, "n_downstream_draws": 2,
            "run_downstream": True,
        }
        (out / "config.json").write_text(json.dumps(config), encoding="utf-8")
        completion = []
        knockoff = []
        selection = []
        for mechanism in ("MCAR", "MAR"):
            for replicate in range(2):
                for max_iter in (10, 25):
                    completion.append({
                        "mechanism": mechanism, "replicate": replicate,
                        "max_iter": max_iter, "error": "", "masked_rmse": 1.0,
                        "masked_mae": 0.5, "sample_cov_fro_relative": 0.2,
                        "sample_corr_fro_relative": 0.2,
                        "masked_imputed_variance_ratio_median": 0.8,
                        "masked_imputed_variance_ratio_mean": 0.8,
                        "completed_variance_ratio_median": 0.95,
                        "completed_variance_ratio_mean": 0.95,
                        "corr_condition_number_effective": 10.0,
                        "corr_condition_number_ridge_1e3": 10.0,
                        "corr_min_eigenvalue": 0.0,
                        "corr_min_positive_eigenvalue": 0.1,
                        "corr_max_eigenvalue": 2.0,
                        "n_iter": max_iter, "convergence_warning_count": 1,
                        "converged_without_warning": 0,
                    })
                for draw in range(2):
                    for variant, max_iter in (("oracle_complete", 0), ("bayesianridge_mean", 10), ("bayesianridge_mean", 25)):
                        knockoff.append({
                            "mechanism": mechanism, "replicate": replicate,
                            "variant": variant, "max_iter": max_iter,
                            "downstream_draw": draw, "error": "",
                            "pair_corr_mean": 0.9, "s_relative_mean": 0.1,
                            "cov_kk_fro_relative": 0.05,
                            "cov_xk_offdiag_rmse": 0.02,
                        })
                        selection.append({
                            "mechanism": mechanism, "replicate": replicate,
                            "variant": variant, "max_iter": max_iter,
                            "downstream_draw": draw, "error": "",
                            "fdp": 0.2, "power": 0.4,
                            "oracle_selection_jaccard": 0.5 if variant != "oracle_complete" else 1.0,
                            "n_selected": 8, "true_positives": 6,
                            "false_positives": 2,
                        })
        pd.DataFrame(completion).to_csv(out / "completion_diagnostics.csv", index=False)
        pd.DataFrame(knockoff).to_csv(out / "knockoff_diagnostics.csv", index=False)
        pd.DataFrame(selection).to_csv(out / "selection_results.csv", index=False)
        subprocess.run(
            [sys.executable, str(ROOT / "summarize_v101_dream_guard.py"),
             "--out-dir", str(out), "--n-bootstrap", "200"],
            check=True, capture_output=True, text=True,
        )
        audit = pd.read_csv(out / "v101_guard_integrity_audit.csv")
        assert audit["complete"].all()
    print("[ok] V10.1 DREAM convergence guard summary preserves integrity")


def test_command_line_interfaces() -> None:
    for name in (
        "run_v101_dream_convergence_guard.py",
        "run_v101_dream_confirmation.py",
        "reanalyze_v101_ssi_threshold_path.py",
    ):
        subprocess.run(
            [sys.executable, str(ROOT / name), "--help"],
            check=True, capture_output=True, text=True,
        )
    print("[ok] V10.1 command-line interfaces")


def main() -> None:
    test_scripts_parse()
    test_confirmation_summary()
    test_guard_summary()
    test_command_line_interfaces()
    print("[ok] all V10.1 update tests passed")


if __name__ == "__main__":
    main()
