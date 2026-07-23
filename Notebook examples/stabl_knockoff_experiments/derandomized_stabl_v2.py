"""Derandomised aggregation for paired STABL knockoff scores.

This module implements the e value aggregation idea of Ren and Barber for a
sequence of *independent knockoff copies generated for the same observed
training dataset*.

It does not make an arbitrary STABL score automatically valid.  The following
conditions remain essential:

1. Every knockoff copy is valid, or sufficiently accurate for the intended use.
2. Artificial score j is paired with real score j.
3. The statistic is antisymmetric under swapping a real feature with its
   knockoff.  This module uses ``W_j = max_lambda f_j - max_lambda f_tilde_j``.
4. Preprocessing, penalties, feature filtering, and bootstrap indices treat the
   original and knockoff members symmetrically.

Do not treat different outer cross validation folds as repeated knockoff draws.
The draws being aggregated must condition on the same observed dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def _flatten_score(score: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(score, dtype=float)
    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2:
        out = np.max(arr, axis=1)
    else:
        raise ValueError(f"Score array must be one or two dimensional, got {arr.shape}")
    if not np.isfinite(out).all():
        raise ValueError("Score array contains NaN or infinite values")
    return out


def paired_stabl_statistic(real_score: ArrayLike, knockoff_score: ArrayLike) -> NDArray[np.float64]:
    """Return the antisymmetric paired statistic W = max(f) minus max(f_tilde)."""
    real = _flatten_score(real_score)
    knockoff = _flatten_score(knockoff_score)
    if real.shape != knockoff.shape:
        raise ValueError(f"Paired scores must have identical shape, got {real.shape} and {knockoff.shape}")
    return real - knockoff


def knockoff_plus_threshold(W: ArrayLike, alpha_kn: float, offset: int = 1) -> float:
    """Standard knockoff threshold used to create one set of knockoff e values.

    The threshold is the smallest observed positive magnitude t such that

        (offset + number of W_j <= -t) / max(1, number of W_j >= t) <= alpha_kn.

    ``offset=1`` gives knockoff+.  The theoretical FDR interpretation in the
    derandomised procedure should use the settings justified by the paper.
    """
    if not 0.0 < alpha_kn < 1.0:
        raise ValueError("alpha_kn must lie strictly between 0 and 1")
    if offset not in (0, 1):
        raise ValueError("offset must be 0 or 1")
    w = np.asarray(W, dtype=float).reshape(-1)
    if not np.isfinite(w).all():
        raise ValueError("W contains NaN or infinite values")
    candidates = np.sort(np.unique(np.abs(w[w != 0])))
    for threshold in candidates:
        numerator = offset + np.sum(w <= -threshold)
        denominator = max(1, int(np.sum(w >= threshold)))
        if numerator / denominator <= alpha_kn:
            return float(threshold)
    return float("inf")


def knockoff_evalues(
    W: ArrayLike,
    alpha_kn: float,
    *,
    offset: int = 1,
) -> tuple[NDArray[np.float64], float]:
    """Convert one vector of knockoff statistics into knockoff e values."""
    w = np.asarray(W, dtype=float).reshape(-1)
    p = len(w)
    threshold = knockoff_plus_threshold(w, alpha_kn, offset=offset)
    evalues = np.zeros(p, dtype=float)
    if np.isinf(threshold):
        return evalues, threshold
    denominator = offset + int(np.sum(w <= -threshold))
    if denominator <= 0:
        raise RuntimeError("Nonpositive e value denominator")
    evalues[w >= threshold] = p / denominator
    return evalues, threshold


def e_bh(evalues: ArrayLike, alpha_ebh: float) -> NDArray[np.bool_]:
    """Base e BH procedure under arbitrary dependence."""
    if not 0.0 < alpha_ebh < 1.0:
        raise ValueError("alpha_ebh must lie strictly between 0 and 1")
    e = np.asarray(evalues, dtype=float).reshape(-1)
    if not np.isfinite(e).all() or np.any(e < 0):
        raise ValueError("e values must be finite and nonnegative")
    p = len(e)
    order = np.argsort(-e, kind="stable")
    ordered = e[order]
    eligible = ordered >= p / (alpha_ebh * np.arange(1, p + 1))
    selection = np.zeros(p, dtype=bool)
    if np.any(eligible):
        k_star = int(np.max(np.flatnonzero(eligible)) + 1)
        selection[order[:k_star]] = True
    return selection


@dataclass(frozen=True)
class DerandomizedSummary:
    n_draws: int
    p: int
    alpha_kn: float
    alpha_ebh: float
    offset: int
    n_selected: int
    finite_threshold_fraction: float


def derandomized_stabl_select(
    real_scores: Sequence[ArrayLike],
    knockoff_scores: Sequence[ArrayLike],
    *,
    alpha_ebh: float = 0.10,
    alpha_kn: float | None = None,
    offset: int = 1,
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, DerandomizedSummary]:
    """Aggregate paired STABL scores from repeated knockoff copies.

    ``alpha_kn`` defaults to half of ``alpha_ebh``, following the empirical
    recommendation in the paper.  Its effect on power is not monotonic for all
    datasets, so it should be included in sensitivity analyses.
    """
    if len(real_scores) == 0:
        raise ValueError("At least one knockoff draw is required")
    if len(real_scores) != len(knockoff_scores):
        raise ValueError("real_scores and knockoff_scores must contain the same number of draws")
    if alpha_kn is None:
        alpha_kn = alpha_ebh / 2.0

    W_rows: list[NDArray[np.float64]] = []
    E_rows: list[NDArray[np.float64]] = []
    thresholds: list[float] = []
    for real, knockoff in zip(real_scores, knockoff_scores):
        W = paired_stabl_statistic(real, knockoff)
        evalues, threshold = knockoff_evalues(W, alpha_kn, offset=offset)
        W_rows.append(W)
        E_rows.append(evalues)
        thresholds.append(threshold)

    W_matrix = np.vstack(W_rows)
    E_matrix = np.vstack(E_rows)
    p = W_matrix.shape[1]
    if any(row.shape[0] != p for row in W_rows):
        raise ValueError("All knockoff draws must contain the same number of paired features")

    mean_evalue = E_matrix.mean(axis=0)
    selected = e_bh(mean_evalue, alpha_ebh)
    names = list(feature_names) if feature_names is not None else [f"feature_{j}" for j in range(p)]
    if len(names) != p:
        raise ValueError("feature_names has the wrong length")

    table = pd.DataFrame(
        {
            "feature": names,
            "W_mean": W_matrix.mean(axis=0),
            "W_median": np.median(W_matrix, axis=0),
            "W_positive_fraction": np.mean(W_matrix > 0, axis=0),
            "mean_evalue": mean_evalue,
            "nonzero_evalue_fraction": np.mean(E_matrix > 0, axis=0),
            "selected_ebh": selected,
        }
    ).sort_values(["selected_ebh", "mean_evalue", "W_mean"], ascending=[False, False, False])

    finite = np.isfinite(np.asarray(thresholds))
    table.attrs["thresholds"] = thresholds
    summary = DerandomizedSummary(
        n_draws=len(real_scores),
        p=p,
        alpha_kn=float(alpha_kn),
        alpha_ebh=float(alpha_ebh),
        offset=offset,
        n_selected=int(selected.sum()),
        finite_threshold_fraction=float(finite.mean()),
    )
    return table, summary


def stabl_fdp_plus_select(
    real_score: ArrayLike,
    artificial_score: ArrayLike,
    *,
    grid: ArrayLike | None = None,
    strict_greater: bool = True,
) -> tuple[NDArray[np.bool_], float, float, pd.DataFrame]:
    """Reference implementation of STABL's classic FDP+ minimisation."""
    real = _flatten_score(real_score)
    artificial = _flatten_score(artificial_score)
    if grid is None:
        grid_arr = np.arange(0.10, 1.00, 0.01)
    else:
        grid_arr = np.asarray(grid, dtype=float)
    if np.any((grid_arr < 0) | (grid_arr > 1)):
        raise ValueError("grid values must lie between 0 and 1")

    records = []
    for threshold in grid_arr:
        if strict_greater:
            n_real = int(np.sum(real > threshold))
            n_artificial = int(np.sum(artificial > threshold))
        else:
            n_real = int(np.sum(real >= threshold))
            n_artificial = int(np.sum(artificial >= threshold))
        fdp_plus = (1 + n_artificial) / max(1, n_real)
        records.append((float(threshold), n_real, n_artificial, float(fdp_plus)))
    curve = pd.DataFrame(records, columns=["threshold", "n_real", "n_artificial", "fdp_plus"])
    best = int(curve["fdp_plus"].to_numpy().argmin())
    theta = float(curve.iloc[best]["threshold"])
    q_plus = float(curve.iloc[best]["fdp_plus"])
    selection = real > theta if strict_greater else real >= theta
    return selection, theta, q_plus, curve


def compare_score_aggregators(
    real_scores: Sequence[ArrayLike],
    knockoff_scores: Sequence[ArrayLike],
    *,
    alpha_ebh: float = 0.10,
    alpha_kn: float | None = None,
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare single draw, mean score FDP+, and e value aggregation.

    This function is descriptive.  Mean score FDP+ is included because it is
    used by the current repaired pre impute branch, not because it carries a
    knockoff theorem.
    """
    flattened_real = [_flatten_score(x) for x in real_scores]
    flattened_knockoff = [_flatten_score(x) for x in knockoff_scores]
    p = len(flattened_real[0])
    names = list(feature_names) if feature_names is not None else [f"feature_{j}" for j in range(p)]

    single, theta_single, q_single, _ = stabl_fdp_plus_select(
        flattened_real[0], flattened_knockoff[0]
    )
    mean_real = np.mean(np.vstack(flattened_real), axis=0)
    mean_knockoff = np.mean(np.vstack(flattened_knockoff), axis=0)
    mean_sel, theta_mean, q_mean, _ = stabl_fdp_plus_select(mean_real, mean_knockoff)
    derand, summary = derandomized_stabl_select(
        flattened_real,
        flattened_knockoff,
        alpha_ebh=alpha_ebh,
        alpha_kn=alpha_kn,
        feature_names=names,
    )
    derand_mask = derand.set_index("feature").loc[names, "selected_ebh"].to_numpy(dtype=bool)

    feature_table = pd.DataFrame(
        {
            "feature": names,
            "single_draw_selected": single,
            "mean_score_fdp_selected": mean_sel,
            "derandomized_ebh_selected": derand_mask,
            "mean_real_score": mean_real,
            "mean_knockoff_score": mean_knockoff,
            "mean_W": mean_real - mean_knockoff,
        }
    )
    method_table = pd.DataFrame(
        [
            {
                "method": "single_draw_stabl_fdp_plus",
                "n_selected": int(single.sum()),
                "theta": theta_single,
                "q_plus": q_single,
            },
            {
                "method": "mean_score_then_stabl_fdp_plus",
                "n_selected": int(mean_sel.sum()),
                "theta": theta_mean,
                "q_plus": q_mean,
            },
            {
                "method": "derandomized_evalue_ebh",
                "n_selected": summary.n_selected,
                "theta": np.nan,
                "q_plus": np.nan,
            },
        ]
    )
    return feature_table, method_table


def save_score_draws(
    path: str | Path,
    real_scores: Sequence[ArrayLike],
    knockoff_scores: Sequence[ArrayLike],
    feature_names: Sequence[str],
    **metadata: object,
) -> None:
    real = np.stack([np.asarray(x, dtype=float) for x in real_scores])
    knockoff = np.stack([np.asarray(x, dtype=float) for x in knockoff_scores])
    np.savez_compressed(
        path,
        real_scores=real,
        knockoff_scores=knockoff,
        feature_names=np.asarray(feature_names, dtype=str),
        metadata=np.asarray([metadata], dtype=object),
    )


def load_score_draws(path: str | Path) -> tuple[list[NDArray], list[NDArray], list[str], dict]:
    with np.load(path, allow_pickle=True) as data:
        real = [row for row in data["real_scores"]]
        knockoff = [row for row in data["knockoff_scores"]]
        names = data["feature_names"].astype(str).tolist()
        metadata = dict(data["metadata"].item())
    return real, knockoff, names, metadata


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    p, n_signal, n_draws = 500, 35, 20
    real_draws, knockoff_draws = [], []
    for _ in range(n_draws):
        real = rng.beta(1.2, 8.0, size=p)
        knockoff = rng.beta(1.2, 8.0, size=p)
        real[:n_signal] = rng.beta(7.0, 2.5, size=n_signal)
        real_draws.append(real)
        knockoff_draws.append(knockoff)

    feature_table, method_table = compare_score_aggregators(
        real_draws, knockoff_draws, alpha_ebh=0.1
    )
    print(method_table.to_string(index=False))
    print(feature_table.head(10).to_string(index=False))
