#!/usr/bin/env python3
"""Small dependency-light self-test for the G2.1 decomposition code."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from run_g21_saved_score_diagnosis import (
    ContrastSpec,
    ScoreRun,
    matched_size_rows,
    selection_metrics,
    stable_top_k,
)


def make_run(label: str, real: np.ndarray, threshold: float) -> ScoreRun:
    p = real.size
    support = np.array([0, 1], dtype=int)
    selected = real >= threshold
    artificial = np.linspace(0.1, 0.4, p)
    return ScoreRun(
        p=p,
        replicate=0,
        generator=label,
        draw=0,
        real_scores=real[:, None],
        artificial_scores=artificial[:, None],
        real_max=real,
        artificial_max=artificial,
        support=support,
        feature_names=np.asarray([f"x{i}" for i in range(p)]),
        stabl_threshold=threshold,
        stabl_estimated_fdp=0.2,
        stabl_selected=selected,
        stabl_metrics=selection_metrics(selected, support),
        source_path=Path("synthetic_test.npz"),
    )


def main() -> None:
    selected = stable_top_k(np.array([0.8, 0.8, 0.7, 0.1]), 2)
    assert selected.tolist() == [True, True, False, False]

    baseline = make_run("baseline", np.array([0.9, 0.8, 0.4, 0.3]), 0.75)
    candidate = make_run("candidate", np.array([0.95, 0.85, 0.7, 0.6]), 0.55)
    frame = matched_size_rows(
        [baseline, candidate],
        [ContrastSpec(candidate="candidate", baseline="baseline")],
        np.arange(0.1, 1.0, 0.01),
    )
    row = frame.iloc[0]
    assert row["baseline_raw_n_selected"] == 2
    assert row["candidate_raw_n_selected"] == 4
    assert row["candidate_topk_n_selected"] == 2
    for metric in (
        "n_selected",
        "realized_fdp",
        "power",
        "support_jaccard",
    ):
        assert abs(row[f"matched_k_decomposition_error_{metric}"]) < 1e-12
        assert abs(row[f"threshold_decomposition_error_{metric}"]) < 1e-12
    print("G2.1 self-test passed")


if __name__ == "__main__":
    main()
