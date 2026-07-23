#!/usr/bin/env python3
"""Fast code-level self-test for the G2.4 focused pilot."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from g24_c2st import repeated_paired_c2st_tracked
from run_g24_focused_pilot import PROJECT_ROOT, _load_grid, _resolve_project_path


def main() -> int:
    grid_path = _resolve_project_path(
        None, "experiments_A_B/generator_track/configs/g24_focused_grid.json"
    )
    configs = _load_grid(grid_path)
    labels = [item.label for item in configs]
    assert labels == ["equicorr", "mvr", "factor_r20_mvr"]
    assert grid_path.is_relative_to(PROJECT_ROOT)

    rng = np.random.default_rng(20260803)
    left = rng.normal(size=(240, 12))
    valid_like = rng.normal(size=(240, 12))
    invalid = valid_like + 1.25

    valid_result = repeated_paired_c2st_tracked(
        left,
        valid_like,
        repeats=3,
        n_splits=5,
        seed=41,
        initial_max_iter=10000,
        retry_max_iter=1000000,
    )
    invalid_result = repeated_paired_c2st_tracked(
        left,
        invalid,
        repeats=3,
        n_splits=5,
        seed=43,
        initial_max_iter=10000,
        retry_max_iter=1000000,
    )
    if not valid_result.converged or not invalid_result.converged:
        raise RuntimeError("Tracked C2ST did not converge in the self-test")
    if valid_result.auc_mean >= 0.70:
        raise RuntimeError(
            f"Valid-like C2ST unexpectedly high: {valid_result.auc_mean:.3f}"
        )
    if invalid_result.auc_mean <= 0.90:
        raise RuntimeError(
            f"Invalid C2ST unexpectedly low: {invalid_result.auc_mean:.3f}"
        )

    print(
        "G2.4 self-test passed | "
        f"project_root={PROJECT_ROOT} | "
        f"valid-like AUC={valid_result.auc_mean:.3f} | "
        f"invalid AUC={invalid_result.auc_mean:.3f} | "
        f"persistent convergence failures="
        f"{valid_result.persistent_nonconverged_folds + invalid_result.persistent_nonconverged_folds}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
