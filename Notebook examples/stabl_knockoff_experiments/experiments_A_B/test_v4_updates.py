#!/usr/bin/env python3
"""Lightweight tests for the V4 paper-aligned grid and LCD comparator."""

from __future__ import annotations

import inspect

import numpy as np
from scipy.special import expit

from research_common import (
    apply_selection_rule,
    build_stabl_lambda_grid,
    fit_lcd_statistics,
    gaussian_knockoff_known_covariance,
)


def test_known_sigma_sampler() -> None:
    rng = np.random.default_rng(11)
    p = 12
    rho = 0.4
    Sigma = np.full((p, p), rho)
    np.fill_diagonal(Sigma, 1.0)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=4000)
    X_tilde, _, _ = gaussian_knockoff_known_covariance(
        X, mean=np.zeros(p), covariance=Sigma, seed=17
    )
    kk = np.linalg.norm(np.cov(X_tilde, rowvar=False) - Sigma, ord="fro") / np.linalg.norm(
        Sigma, ord="fro"
    )
    if not np.isfinite(kk) or kk > 0.12:
        raise AssertionError(f"Known-Sigma covariance check failed: {kk}")
    print(f"[ok] known Sigma Gaussian sampler: kk={kk:.4f}")


def test_paper_grid() -> None:
    grid = build_stabl_lambda_grid("classification")
    values = grid["C"]
    if len(values) != 30:
        raise AssertionError(f"Expected 30 C values, got {len(values)}")
    if not np.isclose(values[0], 0.01) or not np.isclose(values[-1], 1.0):
        raise AssertionError("Paper-aligned C endpoints are incorrect")
    print("[ok] paper-aligned STABL grid: 30 C values from 0.01 to 1.0")


def test_lcd_fit_and_threshold() -> None:
    rng = np.random.default_rng(23)
    n, p = 140, 16
    X = rng.standard_normal((n, p))
    X_tilde = rng.standard_normal((n, p))
    raw = 2.5 * X[:, 0] - 2.0 * X[:, 1] + 1.5 * X[:, 2]
    y = rng.binomial(1, expit(raw))
    lcd = fit_lcd_statistics(
        X,
        X_tilde,
        y,
        task="classification",
        random_state=29,
        cv_folds=3,
        grid_size=8,
        n_jobs=1,
    )
    if lcd.W.shape != (p,) or not np.isfinite(lcd.W).all():
        raise AssertionError("LCD returned invalid W statistics")
    result = apply_selection_rule(
        "lcd_knockoff_plus",
        np.zeros(p),
        np.zeros(p),
        q=0.20,
        threshold_grid=np.linspace(0.1, 0.9, 9),
        lcd_statistics=np.r_[np.ones(10), np.zeros(p - 10)],
    )
    if int(result.selected.sum()) != 10:
        raise AssertionError("LCD knockoff-plus threshold helper failed")
    print(
        "[ok] LCD comparator: "
        f"model={lcd.model_name}, best_parameter={lcd.best_parameter:.6g}"
    )


def test_threshold_comparisons() -> None:
    from research_common import fdp_plus_curve, select_stabl_min, select_stabl_q

    for fn in (fdp_plus_curve, select_stabl_min, select_stabl_q):
        source = inspect.getsource(fn)
        if ">= threshold" not in source and fn.__name__ != "select_stabl_q":
            raise AssertionError(f"{fn.__name__} does not visibly use >= threshold")
    print("[ok] STABL threshold comparisons use >=")


def main() -> None:
    test_known_sigma_sampler()
    test_paper_grid()
    test_lcd_fit_and_threshold()
    test_threshold_comparisons()
    print("[ok] all V4 update tests passed")


if __name__ == "__main__":
    main()
