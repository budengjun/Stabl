#!/usr/bin/env python3
"""Fast self-test for the G2.3 Gaussian-family additions.

The test does not require knockpy or STABL. It validates configuration parsing,
factor covariance construction, and the C2ST helper used by the screening
runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from g22_common import repeated_paired_c2st
from research_common import GeneratorConfig, estimate_factor_covariance


def main() -> None:
    rng = np.random.default_rng(20260727)
    latent = rng.normal(size=(120, 4))
    loadings = rng.normal(size=(4, 30))
    X = latent @ loadings + rng.normal(scale=0.7, size=(120, 30))
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    Sigma = estimate_factor_covariance(
        X,
        rank=5,
        diag_floor=1e-3,
        shrinkage=0.05,
    )
    eigenvalues = np.linalg.eigvalsh(Sigma)
    assert Sigma.shape == (30, 30)
    assert np.isfinite(Sigma).all()
    assert float(eigenvalues.min()) > 0.0

    config = GeneratorConfig.from_mapping(
        {
            "label": "factor_r5_mvr",
            "generator": "gaussian_factor_mvr",
            "factor_rank": 5,
            "factor_diag_floor": 0.001,
            "factor_shrinkage": 0.05,
        }
    )
    assert config.factor_rank == 5
    assert config.generator == "gaussian_factor_mvr"

    valid_partner = rng.multivariate_normal(np.zeros(30), Sigma, size=120)
    invalid_partner = X + 0.5
    valid = repeated_paired_c2st(
        X,
        valid_partner,
        repeats=2,
        n_splits=4,
        seed=17,
    )
    invalid = repeated_paired_c2st(
        X,
        invalid_partner,
        repeats=2,
        n_splits=4,
        seed=19,
    )
    assert 0.5 <= valid.auc_mean <= 1.0
    assert invalid.auc_mean > valid.auc_mean

    print(
        "G2.3 self-test passed | "
        f"factor covariance min eigenvalue={eigenvalues.min():.3e} | "
        f"valid-like AUC={valid.auc_mean:.3f} | invalid AUC={invalid.auc_mean:.3f}"
    )


if __name__ == "__main__":
    main()
