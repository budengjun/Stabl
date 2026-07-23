#!/usr/bin/env python3
"""Lightweight checks for Experiment A v3 updates.

These checks do not require knockpy. They verify the known-covariance Gaussian
sampler and the STABL >= threshold semantics before a server pilot is launched.
"""

from __future__ import annotations

import numpy as np

from research_common import (
    block_covariance,
    fdp_plus_curve,
    gaussian_knockoff_known_covariance,
    select_stabl_min,
    select_stabl_q,
)


def main() -> None:
    rng = np.random.default_rng(7)
    p = 8
    Sigma = block_covariance(p, block_size=4, rho=0.4)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=30_000)
    X_tilde, _, s_correlation = gaussian_knockoff_known_covariance(
        X,
        mean=np.zeros(p),
        covariance=Sigma,
        seed=11,
    )

    joint_covariance = np.cov(np.concatenate([X, X_tilde], axis=1), rowvar=False)
    knockoff_covariance = joint_covariance[p:, p:]
    cross_covariance = joint_covariance[:p, p:]
    S = np.diag(s_correlation * np.diag(Sigma))

    kk_error = np.linalg.norm(knockoff_covariance - Sigma, ord="fro") / np.linalg.norm(
        Sigma, ord="fro"
    )
    xk_error = np.linalg.norm(
        cross_covariance - (Sigma - S), ord="fro"
    ) / np.linalg.norm(Sigma - S, ord="fro")
    if kk_error >= 0.04 or xk_error >= 0.06:
        raise AssertionError(
            f"Known Sigma sampler check failed: kk_error={kk_error}, xk_error={xk_error}"
        )

    real = np.asarray([0.50, 0.49])
    knockoff = np.asarray([0.49, 0.10])
    grid = np.asarray([0.50])
    curve = fdp_plus_curve(real, knockoff, grid)
    if curve[0] != 1.0:
        raise AssertionError(f">= threshold check failed: curve={curve}")
    minimum = select_stabl_min(real, knockoff, grid)
    if minimum.selected.tolist() != [True, False]:
        raise AssertionError(f"stabl_min >= check failed: {minimum.selected}")
    target = select_stabl_q(real, knockoff, grid, q=0.99)
    if target.selected.any():
        raise AssertionError("stabl_q should abstain when no FDP+ threshold meets q")

    print("[ok] known Sigma Gaussian sampler")
    print(f"[ok] covariance checks: kk={kk_error:.4f}, xk={xk_error:.4f}")
    print("[ok] STABL threshold comparisons use >=")


if __name__ == "__main__":
    main()
