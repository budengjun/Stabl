#!/usr/bin/env python3
"""Lightweight tests for V6.1 validity-aware posterior pooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from posterior_pooling_common import pool_score_paths
from research_common import apply_selection_rule
from validity_aware_pooling import (
    parse_pooling_rules,
    parse_vote_fractions,
    select_component_vote,
    select_mean_real_component_artificial_count,
)


def test_parsers() -> None:
    assert parse_pooling_rules("naive,validity_aware,component_vote") == (
        "mean_path_naive",
        "mean_real_component_artificial_count",
        "component_selection_vote",
    )
    assert parse_vote_fractions("0.8,0.5,0.8") == (0.5, 0.8)
    print("[ok] V6.1 rule and vote parsing")


def test_m1_identity() -> None:
    rng = np.random.default_rng(17)
    real = rng.uniform(0, 1, size=(12, 5))
    artificial = rng.uniform(0, 1, size=(12, 5))
    grid = np.arange(0.1, 1.0, 0.05)
    original = apply_selection_rule(
        "stabl_min", real, artificial, q=0.1, threshold_grid=grid
    )
    validity = select_mean_real_component_artificial_count(
        real,
        [artificial],
        pool_size=1,
        threshold_grid=grid,
        naive_pooled_artificial_scores=artificial,
    ).result
    assert np.array_equal(original.selected, validity.selected)
    assert original.threshold == validity.threshold
    assert np.isclose(original.estimated_fdp, validity.estimated_fdp)
    print("[ok] M=1 validity-aware rule reproduces original stabl_min")


def test_artificial_dilution_guard() -> None:
    # Each draw contains one strong artificial feature, but the identity rotates.
    M = 5
    p = 5
    real_components = [np.array([[0.8], [0.7], [0.1], [0.1], [0.1]]) for _ in range(M)]
    artificial_components = []
    for draw in range(M):
        scores = np.full((p, 1), 0.1)
        scores[draw, 0] = 0.8
        artificial_components.append(scores)
    pooled_real = pool_score_paths(real_components, pool_size=M, aggregator="mean")
    pooled_artificial = pool_score_paths(
        artificial_components, pool_size=M, aggregator="mean"
    )
    grid = np.array([0.5])
    validity = select_mean_real_component_artificial_count(
        pooled_real,
        artificial_components,
        pool_size=M,
        threshold_grid=grid,
        naive_pooled_artificial_scores=pooled_artificial,
    )
    assert validity.naive_artificial_count_curve[0] == 0
    assert validity.artificial_count_curve[0] == 1
    assert np.isclose(validity.fdp_curve[0], 1.0)
    print("[ok] component-wise artificial counts prevent identity dilution")


def test_component_vote() -> None:
    masks = [
        np.array([1, 1, 0, 0], dtype=bool),
        np.array([1, 0, 1, 0], dtype=bool),
        np.array([1, 1, 1, 0], dtype=bool),
        np.array([0, 1, 0, 0], dtype=bool),
        np.array([1, 0, 0, 0], dtype=bool),
    ]
    majority = select_component_vote(masks, pool_size=5, vote_fraction=0.5)
    strict = select_component_vote(masks, pool_size=5, vote_fraction=0.8)
    assert np.array_equal(majority.selected, np.array([1, 1, 0, 0], dtype=bool))
    assert np.array_equal(strict.selected, np.array([1, 0, 0, 0], dtype=bool))
    assert np.isnan(majority.estimated_fdp)
    print("[ok] component-selection vote sensitivity rules")


def test_files_present() -> None:
    root = Path(__file__).resolve().parent
    for name in [
        "run_experiment_a_validity_aware_pooling.py",
        "reanalyze_v6_validity_aware_pooling.py",
        "summarize_validity_aware_pooling.py",
    ]:
        assert (root / name).exists(), name
    print("[ok] V6.1 runner, reanalysis, and summarizer available")


if __name__ == "__main__":
    test_parsers()
    test_m1_identity()
    test_artificial_dilution_guard()
    test_component_vote()
    test_files_present()
    print("[ok] all V6.1 update tests passed")
