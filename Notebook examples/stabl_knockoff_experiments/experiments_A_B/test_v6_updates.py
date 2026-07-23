#!/usr/bin/env python3
"""Lightweight V6 tests that do not require fitting STABL."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from posterior_pooling_common import (
    decode_indices,
    encode_indices,
    mean_pairwise_jaccard,
    parse_aggregators,
    parse_pool_sizes,
    pool_score_paths,
)
from research_common import apply_selection_rule


def main() -> None:
    assert parse_pool_sizes("10,1,5,5") == (1, 5, 10)
    assert parse_aggregators("mean,median") == ("mean", "median")
    print("[ok] pool-size and aggregator parsing")

    a = np.array([[0.2, 0.4], [0.1, 0.3]])
    b = np.array([[0.4, 0.6], [0.3, 0.5]])
    c = np.array([[0.6, 0.8], [0.5, 0.7]])
    mean2 = pool_score_paths([a, b, c], pool_size=2, aggregator="mean")
    median3 = pool_score_paths([a, b, c], pool_size=3, aggregator="median")
    np.testing.assert_allclose(mean2, (a + b) / 2)
    np.testing.assert_allclose(median3, b)
    np.testing.assert_allclose(
        pool_score_paths([a, b], pool_size=1, aggregator="mean"), a
    )
    print("[ok] nested score-path pooling and M=1 identity")

    mask = np.array([True, False, True, False])
    assert np.array_equal(decode_indices(encode_indices(mask), 4), mask)
    assert mean_pairwise_jaccard([mask]) == 1.0
    assert np.isclose(mean_pairwise_jaccard([mask, mask]), 1.0)
    print("[ok] selected-set serialization and stability metric")

    real = np.array([[0.1, 0.9], [0.2, 0.3], [0.8, 0.8]])
    artificial = np.array([[0.1, 0.1], [0.1, 0.1], [0.2, 0.2]])
    grid = np.array([0.1, 0.5, 0.8])
    direct = apply_selection_rule(
        "stabl_min", real, artificial, q=0.1, threshold_grid=grid
    )
    pooled = apply_selection_rule(
        "stabl_min",
        pool_score_paths([real], pool_size=1, aggregator="mean"),
        pool_score_paths([artificial], pool_size=1, aggregator="mean"),
        q=0.1,
        threshold_grid=grid,
    )
    assert np.array_equal(direct.selected, pooled.selected)
    assert direct.threshold == pooled.threshold
    print("[ok] pooled M=1 reproduces original STABL stabl_min")

    source = inspect.getsource(
        __import__("run_experiment_a_posterior_pooling")
    )
    assert 'default="shared"' in source
    assert '"selection_rule": "stabl_min"' in source
    assert "component_draw == 0 and reuse_score_dir is not None" in source
    assert "deterministic_completion" in source
    print("[ok] V6 original-STABL primary rule, shared seeds, and controls")

    assert Path(__file__).with_name("summarize_posterior_pooling.py").exists()
    print("[ok] V6 pooling summarizer available")
    print("[ok] all V6 update tests passed")


if __name__ == "__main__":
    main()
