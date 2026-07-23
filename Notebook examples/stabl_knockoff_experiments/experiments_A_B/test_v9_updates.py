#!/usr/bin/env python3
"""Lightweight tests for the V9 BayesianRidge mean confirmation package."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def test_runner_static() -> None:
    path = Path(__file__).with_name("run_experiment_b_completion_replacement.py")
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"mean_confirmation"' in source
    assert '"version": 9' in source
    assert "mean_confirmation requires exactly" in source
    print("[ok] V9 runner has a locked mean-confirmation mode")


def test_driver_static() -> None:
    path = Path(__file__).with_name("run_experiment_b_mean_confirmation.py")
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert "oracle_complete,median,bayesianridge_mean" in source
    assert "bayesianridge_posterior" not in "\n".join(
        line for line in source.splitlines() if '"--completion-methods"' in line or 'oracle_complete,median' in line
    )
    assert '"n_replicates": args.n_replicates' in source
    print("[ok] V9 driver fixes the three primary completion branches")


def build_fake_results(base: Path) -> None:
    completions = ["oracle_complete", "median", "bayesianridge_mean"]
    selection_rows = []
    recovery_rows = []
    diagnostic_rows = []
    for rep in range(8):
        for mechanism in ["MCAR", "MAR"]:
            for draw in range(3):
                selected = {
                    "oracle_complete": "0;1;2;3",
                    "median": "0;1;5;6",
                    "bayesianridge_mean": "0;1;2;5",
                }
                for completion in completions:
                    tp = {"oracle_complete": 4, "median": 2, "bayesianridge_mean": 3}[completion]
                    fp = {"oracle_complete": 1, "median": 3, "bayesianridge_mean": 2}[completion]
                    selection_rows.append(
                        {
                            "dataset": "SSI_CyTOF",
                            "p_setting": "100",
                            "actual_p": 100,
                            "replicate": rep,
                            "mechanism": mechanism,
                            "completion": completion,
                            "completion_draw": draw,
                            "generator_label": "gaussian_equicorrelated",
                            "fdp": fp / (tp + fp),
                            "power": tp / 15,
                            "n_selected": tp + fp,
                            "true_positives": tp,
                            "false_positives": fp,
                            "estimated_fdp": 0.5,
                            "threshold": 0.3,
                            "selected_indices": selected[completion],
                            "error": "",
                        }
                    )
                    recovery_rows.append(
                        {
                            "dataset": "SSI_CyTOF",
                            "p_setting": "100",
                            "actual_p": 100,
                            "replicate": rep,
                            "mechanism": mechanism,
                            "completion": completion,
                            "completion_draw": draw,
                            "masked_rmse": {"oracle_complete": 0.0, "median": 2.0, "bayesianridge_mean": 1.3}[completion],
                            "masked_mae": {"oracle_complete": 0.0, "median": 1.0, "bayesianridge_mean": 0.6}[completion],
                            "sample_cov_fro_relative": {"oracle_complete": 0.0, "median": 0.4, "bayesianridge_mean": 0.2}[completion],
                            "sample_corr_fro_relative": {"oracle_complete": 0.0, "median": 0.4, "bayesianridge_mean": 0.2}[completion],
                            "error": "",
                        }
                    )
                    diagnostic_rows.append(
                        {
                            "dataset": "SSI_CyTOF",
                            "p_setting": "100",
                            "actual_p": 100,
                            "replicate": rep,
                            "mechanism": mechanism,
                            "completion": completion,
                            "completion_draw": draw,
                            "generator_label": "gaussian_equicorrelated",
                            "pair_corr_mean": 0.8,
                            "s_relative_mean": 0.2,
                            "cov_kk_fro_relative": {"oracle_complete": 0.2, "median": 0.4, "bayesianridge_mean": 0.25}[completion],
                            "cov_xk_offdiag_rmse": {"oracle_complete": 0.04, "median": 0.08, "bayesianridge_mean": 0.05}[completion],
                            "error": "",
                        }
                    )
    pd.DataFrame(selection_rows).to_csv(base / "selection_results.csv", index=False)
    pd.DataFrame(recovery_rows).to_csv(base / "imputation_recovery.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(base / "draw_diagnostics.csv", index=False)
    config = {
        "version": 9,
        "experiment_mode": "mean_confirmation",
        "n_replicates": 8,
        "n_completion_draws": 3,
        "mechanisms_resolved": ["MCAR", "MAR"],
        "completion_methods_resolved": completions,
    }
    (base / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_confirmatory_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        build_fake_results(base)
        generic = Path(__file__).with_name("summarize_completion_replacement.py")
        subprocess.run(
            [sys.executable, str(generic), "--out-dir", str(base), "--bootstrap-seed", "7"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        confirm = Path(__file__).with_name("summarize_mean_confirmation.py")
        subprocess.run(
            [
                sys.executable,
                str(confirm),
                "--out-dir",
                str(base),
                "--bootstrap-seed",
                "7",
                "--n-bootstrap",
                "1000",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        audit = pd.read_csv(base / "confirmatory_integrity_audit.csv")
        primary = pd.read_csv(base / "confirmatory_primary_contrasts.csv")
        decision = pd.read_csv(base / "confirmatory_decision.csv")
        assert audit["complete"].all()
        assert len(primary) == 6
        assert set(primary["metric"]) == {"fdp", "power", "oracle_selection_jaccard"}
        assert set(decision["mechanism_decision"]) == {"supported"}
        assert set(decision["overall_decision"]) == {"confirmed_across_mcar_and_mar"}
        assert (base / "confirmatory_report.md").exists()
    print("[ok] V9 confirmatory summary, integrity audit, and decision gate")


def test_cli_help() -> None:
    for script_name in (
        "run_experiment_b_completion_replacement.py",
        "run_experiment_b_mean_confirmation.py",
        "summarize_mean_confirmation.py",
    ):
        script = Path(__file__).with_name(script_name)
        subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    print("[ok] V9 command-line interfaces")


def main() -> None:
    test_runner_static()
    test_driver_static()
    test_confirmatory_summary()
    test_cli_help()
    print("[ok] all V9 update tests passed")


if __name__ == "__main__":
    main()
