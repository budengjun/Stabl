#!/usr/bin/env python3
"""Fast self-test for the Python-only G2.2 utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from g22_common import (
    atomic_write_csv,
    atomic_write_json,
    downstream_gate_decision,
    invalid_knockoff,
    load_json,
    repeated_paired_c2st,
    stable_top_k,
    swap_pair_matrices,
    validity_gate_decision,
)


def main() -> None:
    rng = np.random.default_rng(20260724)
    X = rng.normal(size=(500, 12))
    X_valid = rng.normal(size=(500, 12))
    X_invalid = invalid_knockoff(X, kind="mean_shift", seed=7)

    valid = repeated_paired_c2st(
        X,
        X_valid,
        repeats=3,
        n_splits=5,
        seed=11,
    )
    invalid = repeated_paired_c2st(
        X,
        X_invalid,
        repeats=3,
        n_splits=5,
        seed=13,
    )
    assert valid.auc_mean < 0.65, valid
    assert invalid.auc_mean > 0.75, invalid

    original, swapped = swap_pair_matrices(X, X_valid, [0, 1, 2])
    assert original.shape == swapped.shape == (500, 24)

    selected = stable_top_k(np.asarray([0.2, 0.9, 0.9, 0.1]), 2)
    assert selected.tolist() == [False, True, True, False]

    validity = validity_gate_decision(
        {
            "marginal_auc_mean": 0.60,
            "global_swap_auc_mean": 0.62,
            "near_constant_knockoff_columns_max": 0,
        },
        {"marginal_auc_max": 0.70, "global_swap_auc_max": 0.70},
    )
    assert validity.passed

    downstream = downstream_gate_decision(
        {
            "fdp_delta": 0.01,
            "average_precision_delta": 0.00,
            "matched_power_delta": 0.01,
            "draw_jaccard_delta": 0.00,
        },
        max_fdp_delta=0.05,
        min_average_precision_delta=-0.02,
        min_matched_power_delta=-0.02,
        min_draw_jaccard_delta=-0.05,
    )
    assert downstream.passed

    with tempfile.TemporaryDirectory(prefix="g22_self_test_") as temporary:
        root = Path(temporary)
        atomic_write_json({"ok": True}, root / "status.json")
        atomic_write_csv(pd.DataFrame([{"ok": 1}]), root / "rows.csv")
        assert load_json(root / "status.json")["ok"] is True

    print(
        "G2.2 self-test passed | "
        f"valid AUC={valid.auc_mean:.3f} | invalid AUC={invalid.auc_mean:.3f}"
    )


if __name__ == "__main__":
    main()
