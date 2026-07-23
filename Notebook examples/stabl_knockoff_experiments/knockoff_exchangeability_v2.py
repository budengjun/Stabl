"""Knockoff validity diagnostics for STABL experiments.

The diagnostics in this module are deliberately separated from STABL's FDP+
selection rule.  They answer whether a proposed pair ``(X, X_tilde)`` exhibits
obvious asymmetry before downstream feature selection is run.  Both the
marginal X versus X_tilde classifier test from the nonexchangeability paper and
a paired augmented swap classifier test are provided.

Important interpretation
------------------------
A non significant classifier test does not prove exchangeability.  It only means
that the chosen classifier did not detect a violation at the available sample
size.  In small n, large p omics data, use the classifier test together with the
second moment report, synthetic null calibration, and empirical FDP in a
half synthetic benchmark.

Pairing requirement
-------------------
Column ``j`` of ``X_tilde`` must be the knockoff paired with column ``j`` of
``X``.  Any column shuffling or chunking that destroys this pairing invalidates
pairwise diagnostics and antisymmetric statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PermutationMode = Literal["fast", "refit", "none"]


def _as_2d_float(X: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two dimensional, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _validate_pair(X: ArrayLike, X_tilde: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    Xa = _as_2d_float(X, "X")
    Xk = _as_2d_float(X_tilde, "X_tilde")
    if Xa.shape != Xk.shape:
        raise ValueError(f"X and X_tilde must have identical shapes, got {Xa.shape} and {Xk.shape}")
    if Xa.shape[0] < 8:
        raise ValueError("At least 8 rows are required for a classifier diagnostic")
    return Xa, Xk


def _random_swap(
    Z: NDArray[np.float64],
    p: int,
    rng: np.random.Generator,
    per_row: bool,
    swap_probability: float,
) -> NDArray[np.float64]:
    if not 0.0 < swap_probability < 1.0:
        raise ValueError("swap_probability must lie strictly between 0 and 1")
    n = Z.shape[0]
    if per_row:
        mask = rng.random((n, p)) < swap_probability
    else:
        mask = np.broadcast_to(rng.random(p) < swap_probability, (n, p))
    left = Z[:, :p]
    right = Z[:, p:]
    return np.concatenate(
        [np.where(mask, right, left), np.where(mask, left, right)], axis=1
    )


def _labels_by_group(
    n: int,
    groups: NDArray | None,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    if groups is None:
        labels = rng.integers(0, 2, size=n, dtype=np.int64)
    else:
        unique_groups = np.unique(groups)
        group_labels = rng.integers(0, 2, size=len(unique_groups), dtype=np.int64)
        mapping = dict(zip(unique_groups.tolist(), group_labels.tolist()))
        labels = np.asarray([mapping[g] for g in groups], dtype=np.int64)

    # Extremely small datasets can receive only one class by chance.
    if np.unique(labels).size < 2:
        if groups is None:
            labels[: n // 2] = 0
            labels[n // 2 :] = 1
            rng.shuffle(labels)
        else:
            unique_groups = np.unique(groups)
            half = max(1, len(unique_groups) // 2)
            mapping = {g: int(i >= half) for i, g in enumerate(unique_groups)}
            labels = np.asarray([mapping[g] for g in groups], dtype=np.int64)
    return labels


def _effective_splits(labels: NDArray, groups: NDArray | None, requested: int) -> int:
    if groups is None:
        class_counts = np.bincount(labels, minlength=2)
        return max(2, min(requested, int(class_counts.min())))

    counts = []
    for cls in (0, 1):
        counts.append(np.unique(groups[labels == cls]).size)
    return max(2, min(requested, min(counts)))


def _cross_fitted_auc(
    A: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray | None,
    n_splits: int,
    random_state: int,
    C: float,
) -> tuple[float, NDArray[np.float64]]:
    splits = _effective_splits(labels, groups, n_splits)
    if groups is None:
        splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
        iterator = splitter.split(A, labels)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=splits, shuffle=True, random_state=random_state
        )
        iterator = splitter.split(A, labels, groups=groups)

    oof = np.full(A.shape[0], np.nan, dtype=float)
    for train_idx, test_idx in iterator:
        if np.unique(labels[train_idx]).size < 2 or np.unique(labels[test_idx]).size < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=C,
                solver="liblinear",
                max_iter=4000,
                random_state=random_state,
            ),
        )
        model.fit(A[train_idx], labels[train_idx])
        oof[test_idx] = model.predict_proba(A[test_idx])[:, 1]

    keep = np.isfinite(oof)
    if keep.sum() < max(8, int(0.6 * len(labels))) or np.unique(labels[keep]).size < 2:
        raise RuntimeError("Too few valid out of fold predictions for the classifier test")
    return float(roc_auc_score(labels[keep], oof[keep])), oof


def _permute_labels(
    labels: NDArray[np.int64], groups: NDArray | None, rng: np.random.Generator
) -> NDArray[np.int64]:
    if groups is None:
        return rng.permutation(labels)

    unique_groups = np.unique(groups)
    label_by_group = {g: int(labels[groups == g][0]) for g in unique_groups}
    permuted_values = rng.permutation([label_by_group[g] for g in unique_groups])
    mapping = dict(zip(unique_groups.tolist(), permuted_values.tolist()))
    return np.asarray([mapping[g] for g in groups], dtype=np.int64)


@dataclass(frozen=True)
class CTSTResult:
    swap_auc_mean: float
    swap_auc_oriented_mean: float
    swap_deviation_mean: float
    swap_auc_sd: float
    permutation_pvalue: float | None
    n_repeats: int
    n_permutations: int
    permutation_mode: str
    n: int
    p: int
    grouped: bool

    def to_dict(self) -> dict:
        return asdict(self)


def swap_classifier_test(
    X: ArrayLike,
    X_tilde: ArrayLike,
    *,
    groups: Sequence | None = None,
    n_repeats: int = 5,
    n_splits: int = 5,
    n_permutations: int = 100,
    permutation_mode: PermutationMode = "fast",
    per_row_swap: bool = True,
    swap_probability: float = 0.5,
    C: float = 0.05,
    random_state: int = 0,
) -> CTSTResult:
    """Classifier two sample test based on random pair swaps.

    Each sample contributes exactly one vector, either the original augmented
    vector or its swapped version.  For repeated measures data, one label is
    sampled per subject and cross validation is grouped by subject.

    ``permutation_mode='fast'`` permutes labels against the first repeat's
    cross fitted scores.  It is quick and useful for screening.  ``'refit'``
    refits the classifier for every permutation and is slower but better
    calibrated for a final report.
    """
    Xa, Xk = _validate_pair(X, X_tilde)
    n, p = Xa.shape
    group_array = None if groups is None else np.asarray(groups)
    if group_array is not None and len(group_array) != n:
        raise ValueError("groups must have one value per row")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")

    rng = np.random.default_rng(random_state)
    Z = np.concatenate([Xa, Xk], axis=1)

    aucs: list[float] = []
    first_A = first_labels = first_oof = None
    for repeat in range(n_repeats):
        Z_swapped = _random_swap(Z, p, rng, per_row_swap, swap_probability)
        labels = _labels_by_group(n, group_array, rng)
        A = np.where(labels[:, None] == 1, Z_swapped, Z)
        auc, oof = _cross_fitted_auc(
            A, labels, group_array, n_splits, random_state + repeat, C
        )
        aucs.append(auc)
        if repeat == 0:
            first_A, first_labels, first_oof = A, labels, oof

    aucs_arr = np.asarray(aucs)
    deviations = np.abs(aucs_arr - 0.5)
    oriented = 0.5 + deviations

    pvalue: float | None = None
    if permutation_mode != "none" and n_permutations > 0:
        assert first_A is not None and first_labels is not None and first_oof is not None
        null_deviations = []
        for b in range(n_permutations):
            perm_labels = _permute_labels(first_labels, group_array, rng)
            if permutation_mode == "fast":
                keep = np.isfinite(first_oof)
                null_auc = roc_auc_score(perm_labels[keep], first_oof[keep])
            elif permutation_mode == "refit":
                null_auc, _ = _cross_fitted_auc(
                    first_A,
                    perm_labels,
                    group_array,
                    n_splits,
                    random_state + 10000 + b,
                    C,
                )
            else:
                raise ValueError(f"Unknown permutation_mode: {permutation_mode}")
            null_deviations.append(abs(null_auc - 0.5))
        observed = float(deviations.mean())
        null_arr = np.asarray(null_deviations)
        pvalue = float((1 + np.sum(null_arr >= observed)) / (1 + len(null_arr)))

    return CTSTResult(
        swap_auc_mean=float(aucs_arr.mean()),
        swap_auc_oriented_mean=float(oriented.mean()),
        swap_deviation_mean=float(deviations.mean()),
        swap_auc_sd=float(aucs_arr.std(ddof=1)) if len(aucs_arr) > 1 else 0.0,
        permutation_pvalue=pvalue,
        n_repeats=n_repeats,
        n_permutations=n_permutations if permutation_mode != "none" else 0,
        permutation_mode=permutation_mode,
        n=n,
        p=p,
        grouped=group_array is not None,
    )



@dataclass(frozen=True)
class MarginalCTSTResult:
    marginal_auc_mean: float
    marginal_auc_oriented_mean: float
    marginal_deviation_mean: float
    marginal_auc_sd: float
    marginal_permutation_pvalue: float | None
    marginal_n_repeats: int
    marginal_n_permutations: int
    marginal_permutation_mode: str
    marginal_n: int
    marginal_p: int
    marginal_grouped: bool

    def to_dict(self) -> dict:
        return asdict(self)


def marginal_classifier_test(
    X: ArrayLike,
    X_tilde: ArrayLike,
    *,
    groups: Sequence | None = None,
    n_repeats: int = 5,
    n_splits: int = 5,
    n_permutations: int = 100,
    permutation_mode: PermutationMode = "fast",
    C: float = 0.05,
    random_state: int = 0,
) -> MarginalCTSTResult:
    """Classifier test of the marginal equality of X and X_tilde.

    This is the closest analogue of the C2ST diagnostic in *When Knockoffs
    Fail*.  Each sample contributes either its original row or its knockoff row,
    selected by a random label.  Using both rows from every sample in the same
    classifier dataset can create an artificial paired structure when n is much
    smaller than p, so this implementation uses one row per sample.

    Marginal equality is necessary but not sufficient for pairwise
    exchangeability.  Run this together with ``swap_classifier_test``.
    """
    Xa, Xk = _validate_pair(X, X_tilde)
    n, p = Xa.shape
    group_array = None if groups is None else np.asarray(groups)
    if group_array is not None and len(group_array) != n:
        raise ValueError("groups must have one value per row")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")

    rng = np.random.default_rng(random_state)
    aucs: list[float] = []
    first_A = first_labels = first_oof = None
    for repeat in range(n_repeats):
        labels = _labels_by_group(n, group_array, rng)
        A = np.where(labels[:, None] == 1, Xk, Xa)
        auc, oof = _cross_fitted_auc(
            A, labels, group_array, n_splits, random_state + repeat, C
        )
        aucs.append(auc)
        if repeat == 0:
            first_A, first_labels, first_oof = A, labels, oof

    aucs_arr = np.asarray(aucs)
    deviations = np.abs(aucs_arr - 0.5)
    oriented = 0.5 + deviations

    pvalue: float | None = None
    if permutation_mode != "none" and n_permutations > 0:
        assert first_A is not None and first_labels is not None and first_oof is not None
        null_deviations = []
        for b in range(n_permutations):
            perm_labels = _permute_labels(first_labels, group_array, rng)
            if permutation_mode == "fast":
                keep = np.isfinite(first_oof)
                null_auc = roc_auc_score(perm_labels[keep], first_oof[keep])
            elif permutation_mode == "refit":
                null_auc, _ = _cross_fitted_auc(
                    first_A,
                    perm_labels,
                    group_array,
                    n_splits,
                    random_state + 20000 + b,
                    C,
                )
            else:
                raise ValueError(f"Unknown permutation mode: {permutation_mode}")
            null_deviations.append(abs(null_auc - 0.5))
        observed = float(deviations.mean())
        pvalue = float(
            (1 + np.sum(np.asarray(null_deviations) >= observed))
            / (1 + len(null_deviations))
        )

    return MarginalCTSTResult(
        marginal_auc_mean=float(aucs_arr.mean()),
        marginal_auc_oriented_mean=float(oriented.mean()),
        marginal_deviation_mean=float(deviations.mean()),
        marginal_auc_sd=float(aucs_arr.std(ddof=1) if len(aucs_arr) > 1 else 0.0),
        marginal_permutation_pvalue=pvalue,
        marginal_n_repeats=n_repeats,
        marginal_n_permutations=n_permutations if permutation_mode != "none" else 0,
        marginal_permutation_mode=permutation_mode,
        marginal_n=n,
        marginal_p=p,
        marginal_grouped=group_array is not None,
    )

def second_moment_report(
    X: ArrayLike,
    X_tilde: ArrayLike,
    *,
    Sigma_used: ArrayLike | None = None,
) -> dict[str, float]:
    """Report empirical second moment discrepancies and pair separability.

    The normalized errors make comparisons across omics more meaningful.  For
    non Gaussian generators, small errors are useful evidence but are not a
    proof of full exchangeability.
    """
    Xa, Xk = _validate_pair(X, X_tilde)
    n, p = Xa.shape
    Xc = Xa - Xa.mean(axis=0, keepdims=True)
    Kc = Xk - Xk.mean(axis=0, keepdims=True)

    Sxx = Xc.T @ Xc / n
    Skk = Kc.T @ Kc / n
    Sxk = Xc.T @ Kc / n
    scale = max(float(np.linalg.norm(Sxx, ord="fro")), np.finfo(float).eps)

    offdiag = ~np.eye(p, dtype=bool)
    err_kk = Skk - Sxx
    err_xk_off = (Sxk - Sxx)[offdiag]

    var_x = np.diag(Sxx)
    cov_pair = np.diag(Sxk)
    s_implied = var_x - cov_pair
    s_relative = s_implied / np.maximum(var_x, np.finfo(float).eps)

    denom = np.sqrt(
        np.maximum(np.diag(Sxx), np.finfo(float).eps)
        * np.maximum(np.diag(Skk), np.finfo(float).eps)
    )
    pair_corr = cov_pair / denom

    result = {
        "n": float(n),
        "p": float(p),
        "cov_kk_fro_relative": float(np.linalg.norm(err_kk, ord="fro") / scale),
        "cov_kk_abs_mean": float(np.mean(np.abs(err_kk))),
        "cov_xk_offdiag_abs_mean": float(np.mean(np.abs(err_xk_off))),
        "cov_xk_offdiag_rmse": float(np.sqrt(np.mean(err_xk_off**2))),
        "pair_corr_mean": float(np.nanmean(pair_corr)),
        "pair_corr_median": float(np.nanmedian(pair_corr)),
        "pair_corr_q90": float(np.nanquantile(pair_corr, 0.90)),
        "pair_corr_frac_above_0_9": float(np.mean(pair_corr > 0.9)),
        "s_implied_mean": float(np.nanmean(s_implied)),
        "s_relative_mean": float(np.nanmean(s_relative)),
        "s_relative_median": float(np.nanmedian(s_relative)),
        "s_relative_frac_below_0_01": float(np.mean(s_relative < 0.01)),
    }

    if Sigma_used is not None:
        Sigma = _as_2d_float(Sigma_used, "Sigma_used")
        if Sigma.shape != (p, p):
            raise ValueError(f"Sigma_used must have shape {(p, p)}, got {Sigma.shape}")
        Sigma = (Sigma + Sigma.T) / 2
        eigvals = np.linalg.eigvalsh(Sigma)
        result.update(
            {
                "Sigma_min_eig": float(eigvals.min()),
                "Sigma_max_eig": float(eigvals.max()),
                "Sigma_condition_number": float(
                    eigvals.max() / max(eigvals.min(), np.finfo(float).eps)
                ),
                "Sigma_vs_empirical_fro_relative": float(
                    np.linalg.norm(Sigma - Sxx, ord="fro") / scale
                ),
            }
        )
    return result


def null_w_symmetry_report(W: ArrayLike, null_mask: ArrayLike | None = None) -> dict[str, float]:
    """Outcome aware symmetry diagnostic for known null features.

    Use this only in synthetic or half synthetic experiments where the null set
    is known.  ``W`` must be an antisymmetric original versus knockoff statistic.
    """
    w = np.asarray(W, dtype=float).reshape(-1)
    if null_mask is not None:
        mask = np.asarray(null_mask, dtype=bool).reshape(-1)
        if mask.shape != w.shape:
            raise ValueError("null_mask must have the same length as W")
        w = w[mask]
    w = w[np.isfinite(w)]
    nonzero = w[w != 0]
    if len(nonzero) == 0:
        return {
            "null_n": float(len(w)),
            "null_nonzero_n": 0.0,
            "null_positive_fraction": np.nan,
            "null_sign_imbalance": np.nan,
            "null_mean_W": float(np.nanmean(w)) if len(w) else np.nan,
        }
    positive_fraction = float(np.mean(nonzero > 0))
    return {
        "null_n": float(len(w)),
        "null_nonzero_n": float(len(nonzero)),
        "null_positive_fraction": positive_fraction,
        "null_sign_imbalance": float(abs(positive_fraction - 0.5)),
        "null_mean_W": float(np.mean(w)),
        "null_median_W": float(np.median(w)),
    }


def diagnose_pair(
    X: ArrayLike,
    X_tilde: ArrayLike,
    *,
    label: str = "",
    groups: Sequence | None = None,
    Sigma_used: ArrayLike | None = None,
    random_state: int = 0,
    permutation_mode: PermutationMode = "fast",
) -> pd.DataFrame:
    row: dict[str, object] = {"label": label}
    row.update(
        marginal_classifier_test(
            X,
            X_tilde,
            groups=groups,
            random_state=random_state + 1,
            permutation_mode=permutation_mode,
        ).to_dict()
    )
    row.update(
        swap_classifier_test(
            X,
            X_tilde,
            groups=groups,
            random_state=random_state,
            permutation_mode=permutation_mode,
        ).to_dict()
    )
    row.update(second_moment_report(X, X_tilde, Sigma_used=Sigma_used))
    return pd.DataFrame([row])


if __name__ == "__main__":
    # A self contained smoke test that does not require knockpy.
    rng = np.random.default_rng(7)
    n, p = 180, 30
    X = rng.normal(size=(n, p))
    # Independent copies are valid knockoffs for independent Gaussian columns.
    X_valid = rng.normal(size=(n, p))
    # A deliberately invalid near copy.
    X_invalid = X + 0.05 * rng.normal(size=(n, p))

    out = pd.concat(
        [
            diagnose_pair(X, X_valid, label="independent_valid", random_state=1),
            diagnose_pair(X, X_invalid, label="near_copy_invalid", random_state=2),
        ],
        ignore_index=True,
    )
    print(
        out[
            [
                "label",
                "marginal_auc_oriented_mean",
                "marginal_permutation_pvalue",
                "swap_auc_oriented_mean",
                "permutation_pvalue",
                "pair_corr_mean",
                "s_relative_mean",
                "cov_kk_fro_relative",
            ]
        ].to_string(index=False)
    )
