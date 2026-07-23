"""Shared helpers for the V9.1 convergence and ranking analyses.

V9.1 has two deliberately separate components:

1. A prospective max_iter sensitivity experiment for BayesianRidge
   conditional-mean completion.
2. A threshold-independent reanalysis of already-saved V9 STABL score paths.

The helpers in this file avoid changing the V9 primary experiment code and
therefore preserve the original V9 results exactly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata, wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class CompletionDiagnostics:
    completed: pd.DataFrame
    n_iter: int
    convergence_warning_count: int
    converged_without_warning: bool
    masked_imputed_variance_ratio_median: float
    masked_imputed_variance_ratio_mean: float
    completed_variance_ratio_median: float
    completed_variance_ratio_mean: float
    corr_condition_number_effective: float
    corr_condition_number_ridge_1e3: float
    corr_min_eigenvalue: float
    corr_min_positive_eigenvalue: float
    corr_max_eigenvalue: float


def _safe_featurewise_variance_ratios(
    truth: NDArray[np.float64],
    estimate: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    minimum_count: int = 3,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    ratios: list[float] = []
    for column in range(truth.shape[1]):
        keep = mask[:, column]
        if int(keep.sum()) < minimum_count:
            continue
        truth_variance = float(np.var(truth[keep, column], ddof=1))
        estimate_variance = float(np.var(estimate[keep, column], ddof=1))
        if np.isfinite(truth_variance) and truth_variance > epsilon:
            ratio = estimate_variance / truth_variance
            if np.isfinite(ratio):
                ratios.append(float(ratio))
    return np.asarray(ratios, dtype=float)


def _all_row_variance_ratios(
    truth: NDArray[np.float64],
    estimate: NDArray[np.float64],
    *,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    truth_variance = np.var(truth, axis=0, ddof=1)
    estimate_variance = np.var(estimate, axis=0, ddof=1)
    keep = np.isfinite(truth_variance) & np.isfinite(estimate_variance) & (truth_variance > epsilon)
    ratios = estimate_variance[keep] / truth_variance[keep]
    return np.asarray(ratios[np.isfinite(ratios)], dtype=float)


def _correlation_spectrum_diagnostics(
    completed: NDArray[np.float64],
    *,
    positive_tolerance: float = 1e-10,
    ridge: float = 1e-3,
) -> dict[str, float]:
    centered = completed - np.mean(completed, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-12):
        raise ValueError("Completed matrix contains a constant or invalid column")
    standardized = centered / scale
    correlation = np.cov(standardized, rowvar=False)
    correlation = 0.5 * (correlation + correlation.T)
    eigenvalues = np.linalg.eigvalsh(correlation)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    threshold = max(positive_tolerance * max(maximum, 1.0), 1e-12)
    positive = eigenvalues[eigenvalues > threshold]
    minimum_positive = float(positive[0]) if positive.size else float("nan")
    effective_condition = (
        float(maximum / minimum_positive)
        if positive.size and minimum_positive > 0
        else float("inf")
    )
    ridge_condition = float((maximum + ridge) / max(minimum + ridge, 1e-12))
    return {
        "corr_condition_number_effective": effective_condition,
        "corr_condition_number_ridge_1e3": ridge_condition,
        "corr_min_eigenvalue": minimum,
        "corr_min_positive_eigenvalue": minimum_positive,
        "corr_max_eigenvalue": maximum,
    }


def fit_bayesianridge_mean_with_diagnostics(
    X_complete: pd.DataFrame,
    X_missing: pd.DataFrame,
    mask: NDArray[np.bool_],
    *,
    seed: int,
    max_iter: int,
    nearest_features: int = 50,
    tolerance: float = 1e-3,
) -> CompletionDiagnostics:
    """Fit deterministic IterativeImputer and retain convergence diagnostics."""
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if X_complete.shape != X_missing.shape or mask.shape != X_complete.shape:
        raise ValueError("Complete, missing, and mask shapes must agree")

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        tol=tolerance,
        n_nearest_features=min(nearest_features, X_missing.shape[1]),
        initial_strategy="median",
        skip_complete=True,
        imputation_order="ascending",
        sample_posterior=False,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        values = np.asarray(imputer.fit_transform(X_missing), dtype=float)
    warning_count = sum(
        1 for warning in caught if issubclass(warning.category, ConvergenceWarning)
    )
    if values.shape != X_complete.shape or not np.isfinite(values).all():
        raise RuntimeError("BayesianRidge mean completion returned an invalid matrix")

    completed = pd.DataFrame(values, index=X_complete.index, columns=X_complete.columns)
    truth = X_complete.to_numpy(dtype=float)
    masked_ratios = _safe_featurewise_variance_ratios(truth, values, mask)
    completed_ratios = _all_row_variance_ratios(truth, values)
    spectrum = _correlation_spectrum_diagnostics(values)

    def safe_mean(array: NDArray[np.float64]) -> float:
        return float(np.mean(array)) if array.size else float("nan")

    def safe_median(array: NDArray[np.float64]) -> float:
        return float(np.median(array)) if array.size else float("nan")

    return CompletionDiagnostics(
        completed=completed,
        n_iter=int(getattr(imputer, "n_iter_", max_iter)),
        convergence_warning_count=int(warning_count),
        converged_without_warning=bool(warning_count == 0),
        masked_imputed_variance_ratio_median=safe_median(masked_ratios),
        masked_imputed_variance_ratio_mean=safe_mean(masked_ratios),
        completed_variance_ratio_median=safe_median(completed_ratios),
        completed_variance_ratio_mean=safe_mean(completed_ratios),
        **spectrum,
    )


def max_feature_scores(scores: ArrayLike, p: int) -> NDArray[np.float64]:
    array = np.asarray(scores, dtype=float)
    if array.ndim == 1:
        if array.shape[0] != p:
            raise ValueError("One-dimensional score vector has the wrong length")
        return array
    if array.ndim != 2:
        raise ValueError("STABL scores must be one- or two-dimensional")
    if array.shape[0] == p:
        return np.max(array, axis=1)
    if array.shape[1] == p:
        return np.max(array, axis=0)
    raise ValueError(f"Cannot identify feature axis in score shape {array.shape} for p={p}")


def tie_aware_precision_at_k(
    scores: ArrayLike,
    truth: ArrayLike,
    k: int,
    *,
    tolerance: float = 1e-12,
) -> tuple[float, float]:
    """Expected Precision@k under uniform random tie-breaking.

    The returned true-positive count can be fractional only when the kth rank
    cuts through a tied score block. This avoids feature-index-dependent
    conclusions while retaining an exact fixed selection size.
    """
    score = np.asarray(scores, dtype=float)
    labels = np.asarray(truth, dtype=bool)
    if score.shape != labels.shape or score.ndim != 1:
        raise ValueError("scores and truth must be equal-length vectors")
    if not 1 <= k <= score.size:
        raise ValueError("k must lie between 1 and p")

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_truth = labels[order]
    boundary = sorted_score[k - 1]
    above = sorted_score > boundary + tolerance
    tied = np.isclose(sorted_score, boundary, rtol=0.0, atol=tolerance)
    slots = k - int(above.sum())
    tp_above = float(sorted_truth[above].sum())
    tied_count = int(tied.sum())
    tied_tp = float(sorted_truth[tied].sum())
    expected_tp_from_ties = slots * tied_tp / max(1, tied_count)
    expected_tp = tp_above + expected_tp_from_ties
    return float(expected_tp / k), float(expected_tp)


def ranking_metrics(
    feature_scores: ArrayLike,
    support: ArrayLike,
    *,
    k_values: Iterable[int] = (5, 10, 15, 20),
) -> dict[str, float]:
    score = np.asarray(feature_scores, dtype=float)
    p = int(score.size)
    support_array = np.asarray(support, dtype=int)
    truth = np.zeros(p, dtype=bool)
    truth[support_array] = True
    metrics: dict[str, float] = {}
    n_signal = int(truth.sum())
    for k in k_values:
        precision, tp = tie_aware_precision_at_k(score, truth, int(k))
        metrics[f"precision_at_{int(k)}"] = precision
        metrics[f"tp_at_{int(k)}"] = tp
        metrics[f"recall_at_{int(k)}"] = float(tp / max(1, n_signal))
    metrics["average_precision"] = float(average_precision_score(truth.astype(int), score))
    metrics["support_ranking_auroc"] = float(roc_auc_score(truth.astype(int), score))
    ranks = rankdata(-score, method="average")
    metrics["median_signal_rank"] = float(np.median(ranks[truth]))
    metrics["mean_signal_score"] = float(np.mean(score[truth]))
    metrics["mean_null_score"] = float(np.mean(score[~truth]))
    return metrics



def dataframe_to_markdown(frame: pd.DataFrame, *, index: bool = False) -> str:
    """Render a small DataFrame as Markdown without requiring ``tabulate``.

    ``pandas.DataFrame.to_markdown`` depends on the optional ``tabulate``
    package, which is not installed in every STABL environment.  We first use
    pandas' implementation when available and otherwise emit a standards-
    compliant pipe table directly.  This helper is for human-readable reports
    only and does not alter any CSV result.
    """
    try:
        return frame.to_markdown(index=index)
    except ImportError:
        table = frame.copy()
        if index:
            index_name = table.index.name or "index"
            table.insert(0, index_name, table.index)

        def format_value(value: Any) -> str:
            if pd.isna(value):
                rendered = ""
            elif isinstance(value, (float, np.floating)):
                rendered = f"{float(value):.6g}"
            else:
                rendered = str(value)
            return rendered.replace("|", "\\|").replace("\n", "<br>")

        headers = [format_value(column) for column in table.columns]
        rows = [
            [format_value(value) for value in row]
            for row in table.itertuples(index=False, name=None)
        ]
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join("---" for _ in headers) + " |"
        body_lines = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join([header_line, separator_line, *body_lines])

def percentile_bootstrap_mean_ci(
    differences: ArrayLike,
    *,
    seed: int,
    n_bootstrap: int = 20_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(x) for x in np.quantile(means, [alpha, 1.0 - alpha]))


def paired_wilcoxon_p(differences: ArrayLike) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    try:
        return float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def holm_adjust(p_values: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(p[finite])]
    running = 0.0
    m = order.size
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted
