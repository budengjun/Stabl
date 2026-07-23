#!/usr/bin/env python3
"""Lightweight tests for V11 missingness stress updates."""

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
sys.path.insert(0, str(ROOT))

from research_common import make_missing_mask
from reanalyze_v91_threshold_independent_ranking import SCORE_PATTERN


def test_scripts_parse() -> None:
    for name in (
        "run_v11_missingness_stress.py",
        "summarize_v11_missingness_stress.py",
        "v11_common.py",
    ):
        ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
    runner = (ROOT / "run_experiment_b_completion_replacement.py").read_text(encoding="utf-8")
    assert "BLOCK" in runner
    assert "block-feature-blocks" in runner
    print("[ok] V11 scripts parse and base runner accepts BLOCK")


def test_block_masks_are_nested_and_rectangular() -> None:
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(200, 100)))
    y = pd.Series(rng.integers(0, 2, size=200))
    masks = [
        make_missing_mask(X, y, "BLOCK", rate, 1234, block_feature_blocks=5)
        for rate in (0.10, 0.20, 0.40)
    ]
    assert np.all(masks[0] <= masks[1])
    assert np.all(masks[1] <= masks[2])
    # Columns within a generated block share a row-mask pattern.  There can be
    # no more unique column patterns than configured feature blocks.
    mask0 = masks[0]
    patterns = {np.packbits(mask0[:, j]).tobytes() for j in range(mask0.shape[1])}
    assert len(patterns) <= 5
    observed_rates = [float(mask.mean()) for mask in masks]
    assert all(abs(observed - target) < 0.05 for observed, target in zip(observed_rates, (0.10, 0.20, 0.40)))
    print("[ok] BLOCK masks are rectangular and nested across rates")


def test_ranking_pattern_accepts_block() -> None:
    name = "p_0100__rep_0001__BLOCK__bayesianridge_mean__cdraw_00__equicorr.npz"
    match = SCORE_PATTERN.match(name)
    assert match is not None and match.group("mechanism") == "BLOCK"
    print("[ok] ranking reanalysis accepts BLOCK score files")


def write_synthetic_cell(cell_dir: Path, *, mechanisms: tuple[str, ...], reps: int, draws: int) -> None:
    cell_dir.mkdir(parents=True)
    selection_rows = []
    replicate_rows = []
    oracle_rows = []
    ranking_rows = []
    recovery_rows = []
    diagnostic_rows = []
    values = {
        "oracle_complete": {"fdp": 0.10, "power": 0.50, "pair": 0.90, "s": 0.10},
        "median": {"fdp": 0.24, "power": 0.40, "pair": 0.70, "s": 0.25},
        "bayesianridge_mean": {"fdp": 0.18, "power": 0.41, "pair": 0.88, "s": 0.12},
    }
    for mechanism in mechanisms:
        for rep in range(reps):
            for completion, metric in values.items():
                replicate_rows.append(
                    {
                        "dataset": "synthetic",
                        "p_setting": 100,
                        "actual_p": 100,
                        "replicate": rep,
                        "mechanism": mechanism,
                        "completion": completion,
                        "generator_label": "equicorr",
                        "fdp": metric["fdp"] + rep * 0.001,
                        "power": metric["power"],
                        "n_selected": 8,
                        "true_positives": 6,
                        "false_positives": 2,
                        "estimated_fdp": 0.1,
                        "threshold": 0.8,
                    }
                )
                ranking_rows.append(
                    {
                        "p": 100,
                        "replicate": rep,
                        "mechanism": mechanism,
                        "completion": completion,
                        "generator_label": "equicorr",
                        "precision_at_5": 0.8 if completion != "median" else 0.7,
                        "precision_at_10": 0.7 if completion != "median" else 0.6,
                        "precision_at_15": 0.6 if completion != "median" else 0.5,
                        "precision_at_20": 0.5 if completion != "median" else 0.4,
                        "average_precision": 0.65 if completion == "bayesianridge_mean" else 0.55 if completion == "median" else 0.70,
                        "median_signal_rank": 10 if completion == "bayesianridge_mean" else 15 if completion == "median" else 8,
                        "support_ranking_auroc": 0.80 if completion == "bayesianridge_mean" else 0.72 if completion == "median" else 0.85,
                        "mean_signal_score": 0.7,
                        "mean_null_score": 0.2 if completion == "bayesianridge_mean" else 0.3 if completion == "median" else 0.15,
                    }
                )
                for draw in range(draws):
                    selection_rows.append(
                        {
                            "replicate": rep,
                            "mechanism": mechanism,
                            "completion": completion,
                            "completion_draw": draw,
                            "generator_label": "equicorr",
                            "error": "",
                        }
                    )
                    recovery_rows.append(
                        {
                            "replicate": rep,
                            "mechanism": mechanism,
                            "completion": completion,
                            "completion_draw": draw,
                            "error": "",
                            "missing_rate_actual": 0.2,
                            "missing_rate_row_sd": 0.1,
                            "missing_rate_column_sd": 0.1,
                            "max_row_missing_fraction": 0.4,
                            "max_column_missing_fraction": 0.4,
                            "n_fully_observed_rows": 2,
                            "n_fully_observed_columns": 0,
                            "masked_rmse": 0.0 if completion == "oracle_complete" else 1.0 if completion == "bayesianridge_mean" else 2.0,
                            "masked_mae": 0.0 if completion == "oracle_complete" else 0.5 if completion == "bayesianridge_mean" else 1.0,
                            "sample_cov_fro_relative": 0.0 if completion == "oracle_complete" else 0.1 if completion == "bayesianridge_mean" else 0.3,
                            "sample_corr_fro_relative": 0.0 if completion == "oracle_complete" else 0.1 if completion == "bayesianridge_mean" else 0.3,
                        }
                    )
                    diagnostic_rows.append(
                        {
                            "replicate": rep,
                            "mechanism": mechanism,
                            "completion": completion,
                            "completion_draw": draw,
                            "generator_label": "equicorr",
                            "error": "",
                            "pair_corr_mean": metric["pair"],
                            "s_relative_mean": metric["s"],
                            "cov_kk_fro_relative": 0.05,
                            "cov_xk_offdiag_rmse": 0.01,
                        }
                    )
            for completion, jaccard in (("median", 0.40), ("bayesianridge_mean", 0.60)):
                for draw in range(draws):
                    oracle_rows.append(
                        {
                            "replicate": rep,
                            "mechanism": mechanism,
                            "completion": completion,
                            "completion_draw": draw,
                            "generator_label": "equicorr",
                            "oracle_selection_jaccard": jaccard,
                            "oracle_selected_count_abs_difference": 1,
                        }
                    )
    pd.DataFrame(selection_rows).to_csv(cell_dir / "selection_results.csv", index=False)
    pd.DataFrame(replicate_rows).to_csv(cell_dir / "replicate_level_metrics.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(cell_dir / "oracle_selection_agreement.csv", index=False)
    pd.DataFrame(recovery_rows).to_csv(cell_dir / "imputation_recovery.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(cell_dir / "draw_diagnostics.csv", index=False)
    ranking_dir = cell_dir / "v11_threshold_independent_ranking"
    ranking_dir.mkdir()
    pd.DataFrame(ranking_rows).to_csv(ranking_dir / "ranking_replicate_metrics.csv", index=False)


def test_v11_summary() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        prereg = {
            "analysis_status": "smoke",
            "blocks": [{"label": "synthetic_block"}],
            "missing_rates": [0.2],
            "missingness_mechanisms": ["MCAR", "MAR", "BLOCK"],
            "n_replicates": 2,
            "n_completion_draws": 2,
        }
        (out / "v11_preregistration.json").write_text(json.dumps(prereg), encoding="utf-8")
        write_synthetic_cell(
            out / "synthetic_block" / "rate_0p2",
            mechanisms=("MCAR", "MAR", "BLOCK"),
            reps=2,
            draws=2,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "summarize_v11_missingness_stress.py"),
                "--out-dir",
                str(out),
                "--n-bootstrap",
                "200",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        audit = pd.read_csv(out / "v11_integrity_audit.csv")
        gate = pd.read_csv(out / "v11_cell_progression_gate.csv")
        decision = pd.read_csv(out / "v11_mapping_decision.csv")
        auc = pd.read_csv(out / "v11_auc_over_missing_rates.csv")
        auc_contrasts = pd.read_csv(out / "v11_auc_contrasts.csv")
        slopes = pd.read_csv(out / "v11_replicate_rate_slopes.csv")
        slope_contrasts = pd.read_csv(out / "v11_rate_slope_contrasts.csv")
        assert audit["complete"].all()
        assert gate["directional_gate_passed"].all()
        assert decision.iloc[0]["overall_decision"] == "smoke_only_no_scientific_decision"
        assert auc.empty and auc_contrasts.empty
        assert slopes.empty and slope_contrasts.empty
    print("[ok] V11 one-rate smoke preserves integrity and writes empty trajectory tables safely")



def test_command_line_interfaces() -> None:
    for name in (
        "run_v11_missingness_stress.py",
        "summarize_v11_missingness_stress.py",
    ):
        subprocess.run(
            [sys.executable, str(ROOT / name), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
    print("[ok] V11 command-line interfaces")


def main() -> None:
    test_scripts_parse()
    test_block_masks_are_nested_and_rectangular()
    test_ranking_pattern_accepts_block()
    test_command_line_interfaces()
    test_v11_summary()
    print("[ok] all V11 update tests passed")


if __name__ == "__main__":
    main()
