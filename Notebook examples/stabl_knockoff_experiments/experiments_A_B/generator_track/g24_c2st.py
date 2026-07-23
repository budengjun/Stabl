#!/usr/bin/env python3
"""Tracked paired classifier two-sample tests for G2.4.

The implementation preserves the G2.2 and G2.3 paired orientation design and
logistic classifier family, but it makes convergence observable. A logistic
fold that emits ``ConvergenceWarning`` is retried once with a much larger
iteration budget. Persistent warnings are recorded instead of being silently
ignored.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrackedC2STResult:
    auc_mean: float
    auc_sd: float
    auc_min: float
    auc_max: float
    n_repeats: int
    n_folds_total: int
    convergence_warning_count: int
    retry_count: int
    persistent_nonconverged_folds: int
    max_n_iter_observed: int
    converged: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _paired_orientation_data(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("C2ST inputs must be finite 2D matrices of equal shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("C2ST inputs must be finite")
    n = a.shape[0]
    features = np.vstack([a, b])
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    groups = np.concatenate([np.arange(n), np.arange(n)])
    return features, labels, groups


def _classifier(kind: str, seed: int, rf_estimators: int, max_iter: int):
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=int(max_iter),
                random_state=int(seed),
            ),
        )
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(rf_estimators),
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=1,
        )
    raise ValueError("classifier kind must be logistic or extra_trees")


def _fit_with_tracking(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    classifier_kind: str,
    retry_max_iter: int,
) -> tuple[object, int, int, int, int]:
    """Fit one fold and return warning, retry, persistent, and n_iter counts."""
    warning_count = 0
    retry_count = 0
    persistent = 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y_train)
    first_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    warning_count += len(first_warnings)

    if first_warnings and classifier_kind == "logistic":
        retry_count = 1
        retry = clone(model)
        retry.named_steps["logisticregression"].set_params(max_iter=int(retry_max_iter))
        with warnings.catch_warnings(record=True) as caught_retry:
            warnings.simplefilter("always", ConvergenceWarning)
            retry.fit(X_train, y_train)
        retry_warnings = [
            w for w in caught_retry if issubclass(w.category, ConvergenceWarning)
        ]
        warning_count += len(retry_warnings)
        model = retry
        if retry_warnings:
            persistent = 1

    max_n_iter = 0
    if classifier_kind == "logistic":
        estimator = model.named_steps["logisticregression"]
        if hasattr(estimator, "n_iter_"):
            max_n_iter = int(np.max(np.asarray(estimator.n_iter_, dtype=int)))
    return model, warning_count, retry_count, persistent, max_n_iter


def _grouped_oof_auc(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    classifier_kind: str,
    seed: int,
    n_splits: int,
    rf_estimators: int,
    initial_max_iter: int,
    retry_max_iter: int,
) -> tuple[float, int, int, int, int, int]:
    unique_groups = np.unique(groups)
    if unique_groups.size < 4:
        raise ValueError("Paired C2ST requires at least four paired observations")
    splits = min(int(n_splits), int(unique_groups.size))
    if splits < 2:
        raise ValueError("Paired C2ST requires at least two folds")

    predictions = np.full(labels.size, np.nan, dtype=float)
    warning_count = 0
    retry_count = 0
    persistent_count = 0
    max_n_iter = 0
    fold_count = 0

    splitter = KFold(n_splits=splits, shuffle=True, random_state=int(seed))
    for fold_index, (train_group_idx, test_group_idx) in enumerate(
        splitter.split(unique_groups)
    ):
        train_groups = unique_groups[train_group_idx]
        test_groups = unique_groups[test_group_idx]
        train_rows = np.flatnonzero(np.isin(groups, train_groups))
        test_rows = np.flatnonzero(np.isin(groups, test_groups))
        model = _classifier(
            classifier_kind,
            seed=int(seed) + 10007 * fold_index,
            rf_estimators=rf_estimators,
            max_iter=initial_max_iter,
        )
        model, warn_n, retry_n, persistent_n, n_iter = _fit_with_tracking(
            model,
            features[train_rows],
            labels[train_rows],
            classifier_kind=classifier_kind,
            retry_max_iter=retry_max_iter,
        )
        warning_count += warn_n
        retry_count += retry_n
        persistent_count += persistent_n
        max_n_iter = max(max_n_iter, n_iter)
        fold_count += 1
        predictions[test_rows] = model.predict_proba(features[test_rows])[:, 1]

    if not np.isfinite(predictions).all():
        raise RuntimeError("C2ST failed to obtain complete out-of-fold predictions")
    auc = float(roc_auc_score(labels, predictions))
    return (
        max(auc, 1.0 - auc),
        warning_count,
        retry_count,
        persistent_count,
        max_n_iter,
        fold_count,
    )


def repeated_paired_c2st_tracked(
    left: np.ndarray,
    right: np.ndarray,
    *,
    classifier_kind: str = "logistic",
    repeats: int = 5,
    n_splits: int = 5,
    seed: int = 0,
    rf_estimators: int = 300,
    initial_max_iter: int = 10000,
    retry_max_iter: int = 1000000,
) -> TrackedC2STResult:
    features, labels, groups = _paired_orientation_data(left, right)
    aucs: list[float] = []
    warning_count = 0
    retry_count = 0
    persistent_count = 0
    max_n_iter = 0
    fold_count = 0

    for repeat in range(int(repeats)):
        auc, warn_n, retry_n, persistent_n, n_iter, folds = _grouped_oof_auc(
            features,
            labels,
            groups,
            classifier_kind=classifier_kind,
            seed=int(seed) + 1009 * repeat,
            n_splits=n_splits,
            rf_estimators=rf_estimators,
            initial_max_iter=initial_max_iter,
            retry_max_iter=retry_max_iter,
        )
        aucs.append(auc)
        warning_count += warn_n
        retry_count += retry_n
        persistent_count += persistent_n
        max_n_iter = max(max_n_iter, n_iter)
        fold_count += folds

    values = np.asarray(aucs, dtype=float)
    return TrackedC2STResult(
        auc_mean=float(np.mean(values)),
        auc_sd=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        auc_min=float(np.min(values)),
        auc_max=float(np.max(values)),
        n_repeats=int(values.size),
        n_folds_total=int(fold_count),
        convergence_warning_count=int(warning_count),
        retry_count=int(retry_count),
        persistent_nonconverged_folds=int(persistent_count),
        max_n_iter_observed=int(max_n_iter),
        converged=bool(persistent_count == 0),
    )
