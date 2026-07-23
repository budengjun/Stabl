"""Shared utilities for STABL knockoff timing experiments."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

from official_plsko_bridge import OfficialPLSKOSampler


GeneratorName = Literal[
    "gaussian_equicorrelated",
    "gaussian_mvr",
    "official_plsko",
]
TimingName = Literal[
    "post_median",
    "posterior_then_knockoff",
    "posterior_remask_negative_control",
]


@dataclass(frozen=True)
class GeneratedPair:
    X: NDArray[np.float64]
    X_tilde: NDArray[np.float64]
    Sigma_used: NDArray[np.float64] | None
    completion_seed: int
    knockoff_seed: int
    generation_seconds: float
    timing: str
    generator: str


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def cap_features_by_variance(X: pd.DataFrame, max_features: int | None) -> pd.DataFrame:
    if max_features is None or X.shape[1] <= max_features:
        return X
    variance = X.var(axis=0, skipna=True).sort_values(ascending=False)
    chosen = set(variance.index[:max_features])
    # Preserve original order so feature identity remains stable across runs.
    columns = [column for column in X.columns if column in chosen]
    return X.loc[:, columns]


def make_missing_mask(
    X: pd.DataFrame,
    y: pd.Series,
    mechanism: str,
    rate: float,
    rng: np.random.RandomState,
) -> NDArray[np.bool_]:
    """Mask generator copied from imputation_effect_on_stabl.py.

    The MNAR branch is an outcome dependent stress test.  It is useful for
    robustness experiments but does not satisfy the assumptions of every
    missing value knockoff theorem.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must lie in [0, 1)")
    n, p = X.shape
    mask = np.zeros((n, p), dtype=bool)

    if mechanism == "MCAR":
        mask = rng.rand(n, p) < rate
    elif mechanism == "MAR":
        values = X.to_numpy(dtype=float)
        for j in range(p):
            driver = rng.randint(p)
            while driver == j and p > 1:
                driver = rng.randint(p)
            rank = pd.Series(values[:, driver]).rank(pct=True).to_numpy()
            probability = np.clip(rate * 2.0 * rank, 0.0, 1.0)
            mask[:, j] = rng.rand(n) < probability
    elif mechanism == "MNAR":
        values = X.to_numpy(dtype=float)
        y_rank = y.rank(pct=True).to_numpy()
        for j in range(p):
            column_rank = pd.Series(values[:, j]).rank(pct=True).to_numpy()
            score = 0.5 * column_rank + 0.5 * y_rank
            probability = np.clip(rate * 2.0 * score, 0.0, 1.0)
            mask[:, j] = rng.rand(n) < probability
    else:
        raise ValueError(f"Unknown missingness mechanism: {mechanism}")

    min_observed = max(5, int(0.20 * n))
    for j in range(p):
        observed = np.flatnonzero(~mask[:, j])
        if len(observed) < min_observed:
            missing = np.flatnonzero(mask[:, j])
            return_count = min_observed - len(observed)
            give_back = rng.choice(missing, size=return_count, replace=False)
            mask[give_back, j] = False
    return mask


def apply_mask(X: pd.DataFrame, mask: NDArray[np.bool_]) -> pd.DataFrame:
    if mask.shape != X.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match X shape {X.shape}")
    values = X.to_numpy(dtype=float, copy=True)
    values[mask] = np.nan
    return pd.DataFrame(values, index=X.index, columns=X.columns)


def median_complete(X: pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(X)
    return pd.DataFrame(values, index=X.index, columns=X.columns)


def posterior_complete(
    X: pd.DataFrame,
    seed: int,
    *,
    max_iter: int = 10,
    n_nearest_features: int = 50,
) -> pd.DataFrame:
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        n_nearest_features=min(n_nearest_features, X.shape[1]),
        initial_strategy="median",
        skip_complete=True,
        imputation_order="ascending",
        sample_posterior=True,
        random_state=seed,
    )
    values = imputer.fit_transform(X)
    return pd.DataFrame(values, index=X.index, columns=X.columns)


def jointly_standardize(
    X: ArrayLike, X_tilde: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    X_arr = np.asarray(X, dtype=float)
    K_arr = np.asarray(X_tilde, dtype=float)
    if X_arr.shape != K_arr.shape:
        raise ValueError("X and X_tilde must have the same shape")
    p = X_arr.shape[1]
    scaled = StandardScaler().fit_transform(np.concatenate([X_arr, K_arr], axis=1))
    return scaled[:, :p], scaled[:, p:]


def generate_knockoff(
    X_complete: pd.DataFrame,
    generator: GeneratorName,
    seed: int,
    *,
    plsko_threshold_abs: float | None = None,
    plsko_threshold_q: float | None = 0.8,
    plsko_ncomp: int | None = None,
    plsko_sparsity: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    values = X_complete.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Knockoff generators require a complete finite matrix")
    set_all_seeds(seed)

    if generator in {"gaussian_equicorrelated", "gaussian_mvr"}:
        try:
            from knockpy.knockoffs import GaussianSampler
        except ImportError as exc:
            raise RuntimeError(
                "knockpy is required for Gaussian generators. Activate the STABL environment."
            ) from exc
        method = "equicorrelated" if generator.endswith("equicorrelated") else "mvr"
        sampler = GaussianSampler(values, method=method)
        knockoff = sampler.sample_knockoffs()
        Sigma = getattr(sampler, "Sigma", None)
        return np.asarray(knockoff, dtype=float), None if Sigma is None else np.asarray(Sigma, dtype=float)

    if generator == "official_plsko":
        sampler = OfficialPLSKOSampler(
            X_complete,
            threshold_abs=plsko_threshold_abs,
            threshold_q=plsko_threshold_q,
            ncomp=plsko_ncomp,
            sparsity=plsko_sparsity,
            random_state=seed,
        )
        return sampler.sample_knockoffs(), None

    raise ValueError(f"Unknown generator: {generator}")


def construct_pair(
    X_missing: pd.DataFrame,
    generator: GeneratorName,
    timing: TimingName,
    *,
    completion_seed: int,
    knockoff_seed: int,
    completed_override: pd.DataFrame | None = None,
    plsko_threshold_abs: float | None = None,
    plsko_threshold_q: float | None = 0.8,
    plsko_ncomp: int | None = None,
    plsko_sparsity: float = 1.0,
) -> GeneratedPair:
    """Construct one paired real and knockoff matrix for a timing cell."""
    start = time.perf_counter()
    missing_mask = X_missing.isna().to_numpy()

    if completed_override is not None:
        if not completed_override.index.equals(X_missing.index):
            raise ValueError("completed_override index does not match X_missing")
        if not completed_override.columns.equals(X_missing.columns):
            raise ValueError("completed_override columns do not match X_missing")
        if not np.isfinite(completed_override.to_numpy(dtype=float)).all():
            raise ValueError("completed_override must be complete and finite")

    if timing == "post_median":
        completed = (
            median_complete(X_missing)
            if completed_override is None
            else completed_override.copy()
        )
        # Match the current pipeline: standardize real data before Gaussian sampling.
        real_scaled = pd.DataFrame(
            StandardScaler().fit_transform(completed),
            index=completed.index,
            columns=completed.columns,
        )
        knockoff, Sigma = generate_knockoff(
            real_scaled,
            generator,
            knockoff_seed,
            plsko_threshold_abs=plsko_threshold_abs,
            plsko_threshold_q=plsko_threshold_q,
            plsko_ncomp=plsko_ncomp,
            plsko_sparsity=plsko_sparsity,
        )
        X_final, K_final = real_scaled.to_numpy(dtype=float), knockoff

    elif timing == "posterior_then_knockoff":
        completed = (
            posterior_complete(X_missing, completion_seed)
            if completed_override is None
            else completed_override.copy()
        )
        knockoff, Sigma = generate_knockoff(
            completed,
            generator,
            knockoff_seed,
            plsko_threshold_abs=plsko_threshold_abs,
            plsko_threshold_q=plsko_threshold_q,
            plsko_ncomp=plsko_ncomp,
            plsko_sparsity=plsko_sparsity,
        )
        X_final, K_final = jointly_standardize(completed.to_numpy(), knockoff)

    elif timing == "posterior_remask_negative_control":
        # This deliberately reproduces the problematic remasking idea as a
        # negative control.  It is not presented as a valid missing value method.
        completed = (
            posterior_complete(X_missing, completion_seed)
            if completed_override is None
            else completed_override.copy()
        )
        knockoff, Sigma = generate_knockoff(
            completed,
            generator,
            knockoff_seed,
            plsko_threshold_abs=plsko_threshold_abs,
            plsko_threshold_q=plsko_threshold_q,
            plsko_ncomp=plsko_ncomp,
            plsko_sparsity=plsko_sparsity,
        )
        knockoff_masked = np.asarray(knockoff, dtype=float).copy()
        knockoff_masked[missing_mask] = np.nan
        both = np.concatenate([completed.to_numpy(dtype=float), knockoff_masked], axis=1)
        both_imputed = SimpleImputer(strategy="median").fit_transform(both)
        p = completed.shape[1]
        X_final, K_final = jointly_standardize(
            both_imputed[:, :p], both_imputed[:, p:]
        )
    else:
        raise ValueError(f"Unknown timing: {timing}")

    if X_final.shape != K_final.shape or not np.isfinite(X_final).all() or not np.isfinite(K_final).all():
        raise RuntimeError("Pair construction produced invalid matrices")
    return GeneratedPair(
        X=np.asarray(X_final, dtype=float),
        X_tilde=np.asarray(K_final, dtype=float),
        Sigma_used=Sigma,
        completion_seed=completion_seed,
        knockoff_seed=knockoff_seed,
        generation_seconds=float(time.perf_counter() - start),
        timing=timing,
        generator=generator,
    )


def fit_stabl_scores(
    X: ArrayLike,
    X_tilde: ArrayLike,
    y: ArrayLike,
    *,
    task: Literal["regression", "classification"],
    groups: Sequence | None,
    n_bootstraps: int,
    n_jobs: int,
    random_state: int,
    lambda_grid: dict | list[dict] | None = None,
    model_name: str = "lasso",
) -> tuple[NDArray[np.float64], NDArray[np.float64], object]:
    """Fit STABL with precomputed paired knockoffs and return full score paths."""
    try:
        from stabl.stabl import Stabl
    except ImportError as exc:
        raise RuntimeError("The local stabl package must be importable") from exc

    X_arr = np.asarray(X, dtype=float)
    K_arr = np.asarray(X_tilde, dtype=float)
    y_arr = np.asarray(y)
    if X_arr.shape != K_arr.shape:
        raise ValueError("X and X_tilde must have the same shape")

    if task == "regression":
        if model_name != "lasso":
            raise ValueError("Regression experiments currently support model_name='lasso' only")
        estimator = Lasso(max_iter=int(1e6), random_state=random_state)
        grid = {"alpha": np.logspace(-2, 2, 12)} if lambda_grid is None else lambda_grid
    elif task == "classification":
        if model_name == "lasso":
            estimator = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                max_iter=int(1e6),
                random_state=random_state,
            )
            default_grid = {"C": np.linspace(0.01, 1.0, 6)}
        elif model_name == "alasso":
            try:
                from stabl.adaptive import ALogitLasso
            except ImportError as exc:
                raise RuntimeError("stabl.adaptive.ALogitLasso is required for alasso") from exc
            estimator = ALogitLasso(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                max_iter=int(1e6),
                random_state=random_state,
            )
            default_grid = {"C": np.linspace(0.01, 10.0, 6)}
        elif model_name == "elasticnet":
            estimator = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                class_weight="balanced",
                max_iter=int(1e5),
                random_state=random_state,
            )
            default_grid = [
                {"C": np.logspace(-2, 0, 4), "l1_ratio": [0.5]},
                {"C": np.logspace(-2, 0, 4), "l1_ratio": [0.7]},
                {"C": np.logspace(-2, 0, 4), "l1_ratio": [0.9]},
            ]
        else:
            raise ValueError(f"Unknown classification model_name: {model_name}")
        grid = default_grid if lambda_grid is None else lambda_grid
    else:
        raise ValueError(f"Unknown task: {task}")

    model = Stabl(
        base_estimator=estimator,
        lambda_grid=grid,
        n_bootstraps=n_bootstraps,
        artificial_type="knockoff",
        artificial_proportion=1.0,
        sample_fraction=0.5,
        replace=False,
        fdr_threshold_range=np.arange(0.10, 1.00, 0.01),
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )
    group_arr = None if groups is None else np.asarray(groups)
    model.fit(X_arr, y_arr, groups=group_arr, X_artificial=K_arr)
    return (
        np.asarray(model.stabl_scores_, dtype=float),
        np.asarray(model.stabl_scores_artificial_, dtype=float),
        model,
    )


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    os.replace(temporary, path)
