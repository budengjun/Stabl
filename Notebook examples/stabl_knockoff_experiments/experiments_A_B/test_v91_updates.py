#!/usr/bin/env python3
"""Lightweight tests for the V9.1 convergence and ranking package."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from v91_common import (
    fit_bayesianridge_mean_with_diagnostics,
    ranking_metrics,
    tie_aware_precision_at_k,
)



def run_subprocess_checked(command: list[str]) -> None:
    """Run a test subprocess and surface captured output on failure."""
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print("[child stdout]", file=sys.stderr)
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print("[child stderr]", file=sys.stderr)
            print(completed.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )

def test_static_files() -> None:
    required = (
        "run_v91_convergence_sensitivity.py",
        "summarize_v91_convergence.py",
        "reanalyze_v91_threshold_independent_ranking.py",
        "v91_common.py",
    )
    for name in required:
        path = Path(__file__).with_name(name)
        ast.parse(path.read_text(encoding="utf-8"))
    runner = Path(__file__).with_name("run_v91_convergence_sensitivity.py").read_text(encoding="utf-8")
    assert 'default="10,25,50"' in runner
    assert '"missingness": "MCAR"' in runner
    assert '"stage1b_status": "exploratory_descriptive_only_at_10_replicates"' in runner
    print("[ok] V9.1 scripts parse and preserve the preregistered scope")


def test_completion_diagnostics() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(40, 8))
    values[:, 1] = 0.7 * values[:, 0] + rng.normal(scale=0.3, size=40)
    X = pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])
    mask = rng.random(values.shape) < 0.2
    X_missing = X.mask(mask)
    fitted = fit_bayesianridge_mean_with_diagnostics(
        X,
        X_missing,
        mask,
        seed=11,
        max_iter=3,
        nearest_features=5,
    )
    assert fitted.completed.shape == X.shape
    assert np.isfinite(fitted.completed.to_numpy()).all()
    assert fitted.n_iter >= 1
    assert np.isfinite(fitted.masked_imputed_variance_ratio_median)
    assert np.isfinite(fitted.corr_condition_number_ridge_1e3)
    print("[ok] BayesianRidge mean completion records convergence and variance diagnostics")


def test_tie_aware_ranking() -> None:
    scores = np.asarray([1.0, 0.8, 0.8, 0.8, 0.2])
    truth = np.asarray([True, True, False, False, False])
    precision, tp = tie_aware_precision_at_k(scores, truth, 2)
    assert np.isclose(tp, 1.0 + 1.0 / 3.0)
    assert np.isclose(precision, tp / 2.0)
    metrics = ranking_metrics(scores, np.asarray([0, 1]), k_values=(2, 4))
    assert metrics["precision_at_2"] > metrics["precision_at_4"]
    print("[ok] fixed-k ranking is tie-aware and feature-order invariant")


def write_fake_score_files(root: Path) -> None:
    score_dir = root / "scores"
    score_dir.mkdir(parents=True)
    p = 20
    support = np.asarray([0, 1, 2])
    feature_names = np.asarray([f"x{i}" for i in range(p)], dtype=object)
    rng = np.random.default_rng(17)
    for replicate in range(4):
        for mechanism in ("MCAR", "MAR"):
            for draw in range(2):
                for completion in ("oracle_complete", "median", "bayesianridge_mean"):
                    base = rng.uniform(0.0, 0.3, size=(p, 4))
                    if completion == "oracle_complete":
                        base[support] += 0.65
                    elif completion == "bayesianridge_mean":
                        base[support] += 0.55
                    else:
                        base[support] += 0.20
                        base[[5, 6, 7]] += 0.35
                    knockoff = rng.uniform(0.0, 0.3, size=(p, 4))
                    path = score_dir / (
                        f"p_{p:04d}__rep_{replicate:04d}__{mechanism}__{completion}__"
                        f"cdraw_{draw:02d}__equicorr.npz"
                    )
                    np.savez_compressed(
                        path,
                        real_scores=base,
                        knockoff_scores=knockoff,
                        selected=np.zeros(p, dtype=np.uint8),
                        support=support,
                        feature_names=feature_names,
                    )


def test_ranking_reanalysis() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "v9"
        write_fake_score_files(root)
        output = Path(tmp) / "ranking"
        script = Path(__file__).with_name("reanalyze_v91_threshold_independent_ranking.py")
        run_subprocess_checked(
            [
                sys.executable,
                str(script),
                "--v9-out-dir",
                str(root),
                "--out-dir",
                str(output),
                "--k-values",
                "5,10,15,20",
                "--n-bootstrap",
                "150",
            ]
        )
        primary = pd.read_csv(output / "ranking_primary_contrasts.csv")
        ap = primary[primary["metric"].eq("average_precision")]
        assert len(ap) == 2
        assert (ap["mean_difference"] > 0).all()
        precision = primary[primary["metric"].str.startswith("precision_at_")]
        assert (precision.groupby("mechanism")["mean_difference"].mean() > 0).all()
        audit = pd.read_csv(output / "ranking_integrity_audit.csv")
        assert audit["complete_pairing"].all()
    print("[ok] threshold-independent ranking reanalysis produces paired contrasts")


def build_fake_stage1(root: Path) -> None:
    config = {
        "max_iters": [10, 25, 50],
        "n_replicates": 4,
        "n_downstream_draws": 2,
        "run_downstream": True,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    completion_rows = []
    knockoff_rows = []
    selection_rows = []
    for replicate in range(4):
        for max_iter in (10, 25, 50):
            completion_rows.append(
                {
                    "replicate": replicate,
                    "max_iter": max_iter,
                    "masked_rmse": 2.0 - 0.01 * max_iter,
                    "masked_mae": 1.0 - 0.005 * max_iter,
                    "sample_cov_fro_relative": 0.3 - 0.001 * max_iter,
                    "sample_corr_fro_relative": 0.3 - 0.001 * max_iter,
                    "masked_imputed_variance_ratio_median": 0.8 - 0.002 * max_iter,
                    "masked_imputed_variance_ratio_mean": 0.8 - 0.002 * max_iter,
                    "completed_variance_ratio_median": 0.9 - 0.001 * max_iter,
                    "completed_variance_ratio_mean": 0.9 - 0.001 * max_iter,
                    "corr_condition_number_effective": 100 + max_iter,
                    "corr_condition_number_ridge_1e3": 200 + max_iter,
                    "corr_min_eigenvalue": -1e-12 * max_iter,
                    "corr_min_positive_eigenvalue": 0.01 - 0.00005 * max_iter,
                    "corr_max_eigenvalue": 3.0,
                    "n_iter": max_iter,
                    "convergence_warning_count": 1,
                    "converged_without_warning": 0,
                    "error": "",
                }
            )
        for draw in range(2):
            for variant, max_iter in [("oracle_complete", 0), ("bayesianridge_mean", 10), ("bayesianridge_mean", 25), ("bayesianridge_mean", 50)]:
                knockoff_rows.append(
                    {
                        "replicate": replicate,
                        "variant": variant,
                        "max_iter": max_iter,
                        "downstream_draw": draw,
                        "pair_corr_mean": 0.7 + 0.001 * max_iter,
                        "s_relative_mean": 0.3 - 0.001 * max_iter,
                        "cov_kk_fro_relative": 0.25,
                        "cov_xk_offdiag_rmse": 0.07,
                        "error": "",
                    }
                )
                selection_rows.append(
                    {
                        "replicate": replicate,
                        "variant": variant,
                        "max_iter": max_iter,
                        "downstream_draw": draw,
                        "fdp": 0.5 - 0.0005 * max_iter,
                        "power": 0.2 - 0.0001 * max_iter,
                        "oracle_selection_jaccard": 1.0 if variant == "oracle_complete" else 0.4,
                        "n_selected": 8,
                        "true_positives": 3,
                        "false_positives": 5,
                        "error": "",
                    }
                )
    pd.DataFrame(completion_rows).to_csv(root / "completion_diagnostics.csv", index=False)
    pd.DataFrame(knockoff_rows).to_csv(root / "knockoff_diagnostics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(root / "selection_results.csv", index=False)


def test_stage1_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_fake_stage1(root)
        script = Path(__file__).with_name("summarize_v91_convergence.py")
        run_subprocess_checked(
            [sys.executable, str(script), "--out-dir", str(root), "--n-bootstrap", "150"]
        )
        audit = pd.read_csv(root / "v91_integrity_audit.csv")
        assert audit["complete"].all()
        contrasts = pd.read_csv(root / "stage1a_directional_contrasts.csv")
        pair = contrasts[
            contrasts["metric"].eq("pair_corr_mean")
            & contrasts["to_max_iter"].eq(50)
        ]
        assert float(pair.iloc[0]["mean_difference_to_minus_from"]) > 0
        exploratory = pd.read_csv(root / "stage1b_exploratory_summary.csv")
        assert set(exploratory["analysis_status"]) == {"exploratory_descriptive_only"}
    print("[ok] Stage 1 summary preserves integrity and exploratory labels")


def test_cli_help() -> None:
    for script_name in (
        "run_v91_convergence_sensitivity.py",
        "summarize_v91_convergence.py",
        "reanalyze_v91_threshold_independent_ranking.py",
    ):
        script = Path(__file__).with_name(script_name)
        subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    print("[ok] V9.1 command-line interfaces")


def main() -> None:
    test_static_files()
    test_completion_diagnostics()
    test_tie_aware_ranking()
    test_ranking_reanalysis()
    test_stage1_summary()
    test_cli_help()
    print("[ok] all V9.1 update tests passed")


if __name__ == "__main__":
    main()
