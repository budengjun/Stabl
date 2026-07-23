from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g24_c2st import repeated_paired_c2st_tracked
from run_g24_focused_pilot import _load_grid, _resolve_project_path


def test_focused_grid_is_exact() -> None:
    path = _resolve_project_path(
        None, "experiments_A_B/generator_track/configs/g24_focused_grid.json"
    )
    configs = _load_grid(path)
    assert [item.label for item in configs] == [
        "equicorr",
        "mvr",
        "factor_r20_mvr",
    ]
    assert configs[-1].factor_rank == 20


def test_tracked_c2st_separates_shift() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(size=(180, 8))
    right = rng.normal(size=(180, 8))
    shifted = right + 1.5
    valid = repeated_paired_c2st_tracked(
        left, right, repeats=2, n_splits=4, seed=11
    )
    invalid = repeated_paired_c2st_tracked(
        left, shifted, repeats=2, n_splits=4, seed=13
    )
    assert valid.converged
    assert invalid.converged
    assert valid.auc_mean < 0.75
    assert invalid.auc_mean > 0.90
