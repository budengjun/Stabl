from __future__ import annotations

import numpy as np

from generator_track.run_g21_saved_score_diagnosis import (
    ScoreRun,
    ContrastSpec,
    matched_size_rows,
    selection_metrics,
    stable_top_k,
)


def make_run(label: str, real: np.ndarray, threshold: float) -> ScoreRun:
    p = real.size
    support = np.array([0, 1], dtype=int)
    selected = real >= threshold
    return ScoreRun(
        p=p,
        replicate=0,
        generator=label,
        draw=0,
        real_scores=real[:, None],
        artificial_scores=np.linspace(0.1, 0.4, p)[:, None],
        real_max=real,
        artificial_max=np.linspace(0.1, 0.4, p),
        support=support,
        feature_names=np.asarray([f"x{i}" for i in range(p)]),
        stabl_threshold=threshold,
        stabl_estimated_fdp=0.2,
        stabl_selected=selected,
        stabl_metrics=selection_metrics(selected, support),
        source_path=None,  # type: ignore[arg-type]
    )


def test_stable_top_k_is_exact_and_deterministic() -> None:
    selected = stable_top_k(np.array([0.8, 0.8, 0.7, 0.1]), 2)
    assert selected.tolist() == [True, True, False, False]


def test_matched_size_decomposition_is_exact() -> None:
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
