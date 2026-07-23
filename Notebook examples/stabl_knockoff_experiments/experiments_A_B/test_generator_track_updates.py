from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from generator_track.run_g1_plsko_tuning import freeze_optimal_config
from research_common import (
    GeneratorConfig,
    c2st_diagnostics,
    mean_pairwise_jaccard,
    pair_diagnostics,
    ranking_metrics,
    selection_metrics,
)


def test_tuned_generator_config_accepts_disabled_quantile_threshold() -> None:
    config = GeneratorConfig.from_mapping(
        {
            "label": "plsko_tuned",
            "generator": "official_plsko",
            "threshold_abs": 0.1,
            "threshold_q": None,
            "ncomp": 5,
            "sparsity": 0.8,
        }
    )
    assert config.threshold_abs == 0.1
    assert config.threshold_q is None
    assert config.ncomp == 5


def test_support_and_ranking_metrics() -> None:
    selected = np.array([True, False, True, False])
    support = np.array([0, 1])
    metrics = selection_metrics(selected, support, 4, target_fdr=0.1)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["support_jaccard"] == 1 / 3

    scores = np.array([[0.9, 0.8], [0.8, 0.7], [0.2, 0.1], [0.1, 0.0]])
    ranking = ranking_metrics(scores, support, 4)
    assert ranking["average_precision"] == 1.0
    assert ranking["ranking_auroc"] == 1.0


def test_pair_diagnostics_and_c2st_are_finite() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 5))
    X_tilde = X + rng.normal(scale=0.5, size=X.shape)
    diagnostics = pair_diagnostics(X, X_tilde)
    assert diagnostics["near_constant_knockoff_columns"] == 0
    c2st = c2st_diagnostics(X, X_tilde, seed=7)
    assert 0.5 <= c2st["marginal_c2st_auc"] <= 1.0
    assert 0.5 <= c2st["swap_c2st_auc"] <= 1.0


def test_draw_jaccard() -> None:
    value = mean_pairwise_jaccard([{1, 2}, {2, 3}, {1, 2}])
    assert 0.0 <= value <= 1.0


def test_freeze_optimal_config() -> None:
    frozen = freeze_optimal_config(
        {"ncomp": 10, "threshold.abs": 0.2, "sparsity": 0.5},
        "plsko_tuned",
    )
    assert frozen == {
        "label": "plsko_tuned",
        "generator": "official_plsko",
        "threshold_abs": 0.2,
        "threshold_q": None,
        "ncomp": 10,
        "sparsity": 0.5,
    }


def test_generator_track_files_exist() -> None:
    root = Path(__file__).with_name("generator_track")
    required = [
        root / "run_g0_integration_smoke.py",
        root / "run_g1_plsko_tuning.py",
        root / "official_plsko_tuning_bridge.R",
        root / "scripts" / "run_g2_ssi_pilot.sh",
    ]
    assert all(path.exists() for path in required)
    configs = json.loads((root / "configs" / "g0_generators.json").read_text())
    assert any(item["generator"] == "official_plsko" for item in configs)
