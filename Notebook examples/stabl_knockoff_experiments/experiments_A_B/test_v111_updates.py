#!/usr/bin/env python3
"""Regression tests for the V11.1 score-only threshold diagnosis."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def write_score(path: Path, *, completion: str, support: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed)
    p = 8
    grid = 4
    real = rng.uniform(0.05, 0.65, size=(p, grid))
    knockoff = rng.uniform(0.02, 0.45, size=(p, grid))
    # Add a predictable signal advantage. BR also suppresses null scores.
    real[support] += 0.25
    if completion == "median":
        real[np.setdiff1d(np.arange(p), support)] += 0.10
    elif completion == "bayesianridge_mean":
        real[np.setdiff1d(np.arange(p), support)] -= 0.02
    real = np.clip(real, 0.0, 1.0)
    selected = (real.max(axis=1) >= 0.5).astype(np.uint8)
    np.savez_compressed(
        path,
        real_scores=real,
        knockoff_scores=knockoff,
        selected=selected,
        support=support,
        feature_names=np.array([f"x{i}" for i in range(p)], dtype=object),
        completion_seed=seed,
        knockoff_seed=seed + 1,
        stabl_seed=seed + 2,
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    script = script_dir / "reanalyze_v111_threshold_diagnosis.py"
    subprocess.run([sys.executable, str(script), "--help"], check=True, capture_output=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pilot"
        support = np.array([0, 1], dtype=int)
        score_count = 0
        median_count = 0
        for rate in ("0p1", "0p2"):
            rate_dir = root / "toy_block" / f"rate_{rate}"
            score_dir = rate_dir / "scores"
            ranking_dir = rate_dir / "v11_threshold_independent_ranking"
            score_dir.mkdir(parents=True)
            ranking_dir.mkdir(parents=True)
            ranking_rows = []
            selection_rows = []
            for rep in range(2):
                for mechanism in ("MCAR", "MAR", "BLOCK"):
                    for completion in ("oracle_complete", "median", "bayesianridge_mean"):
                        for draw in range(2):
                            filename = (
                                f"p_0008__rep_{rep:04d}__{mechanism}__{completion}__"
                                f"cdraw_{draw:02d}__equicorr.npz"
                            )
                            write_score(
                                score_dir / filename,
                                completion=completion,
                                support=support,
                                seed=1000 + rep * 100 + draw * 10 + len(completion) + len(mechanism),
                            )
                            score_count += 1
                            selection_rows.append({"completion": completion})
                            if completion == "median":
                                median_count += 1
                    # Ranking metrics are already aggregated over draws.
                    for completion, ap, auc, null in (
                        ("oracle_complete", 0.80, 0.85, 0.20),
                        ("median", 0.55, 0.65, 0.40),
                        ("bayesianridge_mean", 0.65, 0.75, 0.30),
                    ):
                        ranking_rows.append(
                            {
                                "p": 8,
                                "replicate": rep,
                                "mechanism": mechanism,
                                "completion": completion,
                                "generator_label": "equicorr",
                                "precision_at_5": ap,
                                "precision_at_10": ap,
                                "precision_at_15": ap,
                                "precision_at_20": ap,
                                "average_precision": ap,
                                "median_signal_rank": 2.0 if completion != "median" else 3.0,
                                "support_ranking_auroc": auc,
                                "mean_signal_score": 0.70,
                                "mean_null_score": null,
                            }
                        )
            pd.DataFrame(ranking_rows).to_csv(ranking_dir / "ranking_replicate_metrics.csv", index=False)
            pd.DataFrame(selection_rows).to_csv(rate_dir / "selection_results.csv", index=False)

        out_dir = root / "v111_threshold_diagnosis"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--v11-pilot-dir",
                str(root),
                "--out-dir",
                str(out_dir),
                "--threshold-min",
                "0.2",
                "--threshold-max",
                "0.8",
                "--threshold-step",
                "0.2",
                "--n-bootstrap",
                "100",
            ],
            check=True,
            cwd=script_dir,
        )
        audit = pd.read_csv(out_dir / "v111_integrity_audit.csv").iloc[0]
        assert bool(audit["complete"])
        assert int(audit["score_files"]) == score_count
        assert int(audit["matched_runs"]) == median_count
        diagnosis = pd.read_csv(out_dir / "v111_cell_diagnosis.csv")
        assert len(diagnosis) == 2 * 3
        assert diagnosis["diagnostic_label"].notna().all()
        assert (out_dir / "v111_threshold_path_summary.csv").exists()
        assert not (out_dir / "v111_threshold_path_run_metrics.csv").exists()

    print("[ok] V11.1 CLI parses")
    print("[ok] V11.1 reconstructs threshold paths and matched-size comparisons")
    print("[ok] V11.1 preserves pairing and writes descriptive cell diagnoses")
    print("[ok] all V11.1 update tests passed")


if __name__ == "__main__":
    main()
