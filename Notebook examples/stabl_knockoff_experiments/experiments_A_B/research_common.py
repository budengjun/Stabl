"""Shared utilities for STABL Experiments A and B.

The module is deliberately independent of the older timing runner.  Every
completion method follows the same pipeline:

    completion -> fit scaler on completed X -> scale X -> generate knockoff

No data-dependent transformation is fitted separately to the knockoff block
after generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from scipy.special import expit
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import (
    BayesianRidge,
    Lasso,
    LassoCV,
    LogisticRegression,
    LogisticRegressionCV,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from official_plsko_bridge import OfficialPLSKOSampler


TaskName = Literal["classification", "regression"]
GeneratorName = Literal[
    "gaussian_equicorrelated",
    "gaussian_equicorrelated_true_sigma",
    "gaussian_mvr",
    "gaussian_sdp",
    "gaussian_factor_equicorrelated",
    "gaussian_factor_mvr",
    "gaussian_factor_sdp",
    "official_plsko",
]
CompletionName = Literal[
    "oracle_complete",
    "median",
    "exact_gaussian_posterior",
    "bayesianridge_mean",
    "bayesianridge_posterior",
]
SelectionRule = Literal[
    "stabl_min",
    "stabl_q",
    "stabl_knockoff_plus",
    "knockoff_plus",
    "lcd_knockoff_plus",
]


@dataclass(frozen=True)
class GeneratorConfig:
    label: str
    generator: GeneratorName
    threshold_abs: float | None = None
    threshold_q: float | None = 0.8
    ncomp: int | None = None
    sparsity: float = 1.0
    factor_rank: int | None = None
    factor_diag_floor: float = 1e-3
    factor_shrinkage: float = 0.05

    @staticmethod
    def from_mapping(mapping: dict[str, Any]) -> "GeneratorConfig":
        allowed = {
            "label",
            "generator",
            "threshold_abs",
            "threshold_q",
            "ncomp",
            "sparsity",
            "factor_rank",
            "factor_diag_floor",
            "factor_shrinkage",
        }
        unknown = set(mapping) - allowed
        if unknown:
            raise ValueError(f"Unknown generator config fields: {sorted(unknown)}")
        label = str(mapping["label"])
        generator = str(mapping["generator"])
        if generator not in {
            "gaussian_equicorrelated",
            "gaussian_equicorrelated_true_sigma",
            "gaussian_mvr",
            "gaussian_sdp",
            "gaussian_factor_equicorrelated",
            "gaussian_factor_mvr",
            "gaussian_factor_sdp",
            "official_plsko",
        }:
            raise ValueError(f"Unsupported generator: {generator}")
        sparsity = float(mapping.get("sparsity", 1.0))
        if not 0.0 < sparsity <= 1.0:
            raise ValueError("PLSKO sparsity must lie in (0, 1]")
        factor_rank = _optional_int(mapping.get("factor_rank"))
        factor_diag_floor = float(mapping.get("factor_diag_floor", 1e-3))
        factor_shrinkage = float(mapping.get("factor_shrinkage", 0.05))
        factor_generators = {
            "gaussian_factor_equicorrelated",
            "gaussian_factor_mvr",
            "gaussian_factor_sdp",
        }
        if generator in factor_generators and (factor_rank is None or factor_rank < 1):
            raise ValueError("Factor Gaussian generators require factor_rank >= 1")
        if factor_diag_floor <= 0.0:
            raise ValueError("factor_diag_floor must be positive")
        if not 0.0 <= factor_shrinkage < 1.0:
            raise ValueError("factor_shrinkage must lie in [0, 1)")
        return GeneratorConfig(
            label=safe_label(label),
            generator=generator,  # type: ignore[arg-type]
            threshold_abs=_optional_float(mapping.get("threshold_abs")),
            threshold_q=_optional_float(mapping.get("threshold_q", 0.8)),
            ncomp=_optional_int(mapping.get("ncomp")),
            sparsity=sparsity,
            factor_rank=factor_rank,
            factor_diag_floor=factor_diag_floor,
            factor_shrinkage=factor_shrinkage,
        )


@dataclass(frozen=True)
class KnockoffPair:
    X: NDArray[np.float64]
    X_tilde: NDArray[np.float64]
    generation_seconds: float
    Sigma_used: NDArray[np.float64] | None


@dataclass(frozen=True)
class SelectionResult:
    selected: NDArray[np.bool_]
    threshold: float
    estimated_fdp: float
    rule: str


@dataclass(frozen=True)
class LCDStatistics:
    W: NDArray[np.float64]
    best_parameter: float
    fit_seconds: float
    model_name: str


@dataclass(frozen=True)
class RealDataset:
    X: pd.DataFrame
    groups: pd.Series | None
    dataset_label: str
    metadata: dict[str, Any]


def _optional_float(value: Any) -> float | None:
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value)


def safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.]+", "_", str(value)).strip("_")
    if not cleaned:
        raise ValueError("A nonempty safe label is required")
    return cleaned


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)
    os.replace(temporary, path)




def atomic_savez_compressed(path: str | Path, **arrays: Any) -> None:
    """Atomically write a compressed NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def matrix_fingerprint(
    X: ArrayLike,
    *,
    feature_names: Sequence[Any] | None = None,
) -> str:
    """Return a stable SHA256 fingerprint for a design matrix and column order."""
    values = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    if feature_names is not None:
        names = [str(item) for item in feature_names]
        digest.update("\x1f".join(names).encode("utf-8"))
    return digest.hexdigest()

def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_output_directory(
    out_dir: str | Path,
    config: dict[str, Any],
    *,
    resume: bool,
    overwrite: bool,
) -> Path:
    path = Path(out_dir).expanduser().resolve()
    config = dict(config)
    # Operational flags change between a fresh run and resume but do not alter
    # the scientific experiment.  Exclude them from the compatibility hash.
    fingerprint_payload = dict(config)
    for operational_key in ("resume", "overwrite", "fail_fast", "pair_cache_dir", "refresh_pair_cache"):
        fingerprint_payload.pop(operational_key, None)
    config["config_fingerprint"] = config_fingerprint(fingerprint_payload)
    config_path = path / "config.json"

    if path.exists() and any(path.iterdir()):
        if overwrite:
            import shutil

            shutil.rmtree(path)
        elif resume:
            if not config_path.exists():
                raise RuntimeError(f"Cannot resume without {config_path}")
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous.get("config_fingerprint") != config["config_fingerprint"]:
                raise RuntimeError(
                    "Resume configuration does not match the existing output. "
                    "Use a new output directory or pass --overwrite."
                )
            return path
        else:
            raise RuntimeError(
                f"Output directory is not empty: {path}. Use --resume or --overwrite."
            )

    path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config, config_path)
    return path


def parse_csv_items(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not items:
        raise ValueError("At least one comma-separated value is required")
    return items


def block_covariance(p: int, block_size: int, rho: float) -> NDArray[np.float64]:
    if p < 2:
        raise ValueError("p must be at least 2")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    Sigma = np.eye(p, dtype=float)
    for start in range(0, p, block_size):
        stop = min(start + block_size, p)
        Sigma[start:stop, start:stop] = rho
        np.fill_diagonal(Sigma[start:stop, start:stop], 1.0)
    return Sigma


def simulate_gaussian_covariates(
    n: int,
    p: int,
    block_size: int,
    rho: float,
    seed: int,
) -> tuple[pd.DataFrame, NDArray[np.float64], NDArray[np.float64]]:
    """Draw exact N(0, Sigma) covariates without sample standardization.

    Avoiding sample standardization is essential because Experiment A later
    uses ``Sigma`` as the oracle distribution in the exact conditional sampler.
    """
    Sigma = block_covariance(p, block_size, rho)
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(Sigma)
    values = rng.standard_normal((n, p)) @ chol.T
    columns = [f"x{j:04d}" for j in range(p)]
    index = [f"sample_{i:04d}" for i in range(n)]
    return pd.DataFrame(values, index=index, columns=columns), np.zeros(p), Sigma


def generate_sparse_outcome(
    X: pd.DataFrame,
    *,
    n_signal: int,
    task: TaskName,
    signal_strength: float,
    seed: int,
) -> tuple[pd.Series, NDArray[np.int64], NDArray[np.float64]]:
    if not 1 <= n_signal < X.shape[1]:
        raise ValueError("n_signal must lie between 1 and p minus 1")
    if signal_strength <= 0:
        raise ValueError("signal_strength must be positive")
    rng = np.random.default_rng(seed)
    support = np.sort(rng.choice(X.shape[1], size=n_signal, replace=False))
    beta = np.zeros(X.shape[1], dtype=float)
    beta[support] = rng.choice([-1.0, 1.0], size=n_signal)
    raw = X.to_numpy(dtype=float) @ beta
    raw = signal_strength * raw / max(float(raw.std()), 1e-12)

    if task == "classification":
        probability = expit(raw)
        y = rng.binomial(1, probability).astype(int)
        if np.unique(y).size < 2:
            y = (raw > np.median(raw)).astype(int)
    elif task == "regression":
        y = raw + rng.normal(scale=1.0, size=len(raw))
    else:
        raise ValueError(f"Unsupported task: {task}")
    return pd.Series(y, index=X.index, name="y"), support, beta


def make_missing_mask(
    X: pd.DataFrame,
    y: pd.Series,
    mechanism: str,
    rate: float,
    seed: int,
    *,
    mar_driver_fraction: float = 0.05,
    block_feature_blocks: int = 5,
) -> NDArray[np.bool_]:
    """Generate controlled MCAR, MAR, MNAR, or rectangular block missingness.

    ``BLOCK`` first partitions a seed-shuffled feature order into a fixed number
    of feature blocks.  Each feature block is then masked for a seed-specific
    subset of samples.  Reusing the same seed at increasing rates produces
    nested rectangular masks, which is required by the V11 rate stress test.

    For MAR, a small protected set of driver variables is never masked.  Every
    other variable's missingness probability depends only on one protected
    driver.  This makes the missingness mechanism explicitly observable and
    keeps the exact posterior benchmark theory-aligned.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError("missing rate must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    values = X.to_numpy(dtype=float)
    n, p = values.shape
    mechanism = mechanism.upper()

    if mechanism == "MCAR":
        mask = rng.random((n, p)) < rate
    elif mechanism == "MAR":
        n_drivers = min(p - 1, max(1, int(round(p * mar_driver_fraction))))
        drivers = np.arange(n_drivers)
        mask = np.zeros((n, p), dtype=bool)
        maskable = p - n_drivers
        adjusted_rate = min(0.95, rate * p / max(1, maskable))
        for j in range(n_drivers, p):
            driver = int(rng.choice(drivers))
            ranks = pd.Series(values[:, driver]).rank(pct=True).to_numpy()
            probability = np.clip(2.0 * adjusted_rate * ranks, 0.0, 1.0)
            mask[:, j] = rng.random(n) < probability
    elif mechanism == "MNAR":
        mask = np.zeros((n, p), dtype=bool)
        y_rank = y.rank(pct=True).to_numpy()
        for j in range(p):
            x_rank = pd.Series(values[:, j]).rank(pct=True).to_numpy()
            score = 0.5 * x_rank + 0.5 * y_rank
            probability = np.clip(2.0 * rate * score, 0.0, 1.0)
            mask[:, j] = rng.random(n) < probability
    elif mechanism in {"BLOCK", "BLOCK_MISSING"}:
        if block_feature_blocks < 1:
            raise ValueError("block_feature_blocks must be positive")
        n_blocks = min(int(block_feature_blocks), p)
        feature_order = rng.permutation(p)
        feature_blocks = np.array_split(feature_order, n_blocks)
        mask = np.zeros((n, p), dtype=bool)
        # One independent sample score vector per feature block.  Because a row
        # is masked when score < rate, calls with the same seed and larger rates
        # are nested by construction.
        for columns in feature_blocks:
            sample_scores = rng.random(n)
            masked_rows = sample_scores < rate
            mask[np.ix_(masked_rows, np.asarray(columns, dtype=int))] = True
    else:
        raise ValueError(f"Unsupported missingness mechanism: {mechanism}")

    min_observed = max(5, int(np.ceil(0.20 * n)))
    for j in range(p):
        observed_count = int((~mask[:, j]).sum())
        if observed_count < min_observed:
            missing_rows = np.flatnonzero(mask[:, j])
            restore = rng.choice(
                missing_rows, size=min_observed - observed_count, replace=False
            )
            mask[restore, j] = False
    return mask


def apply_mask(X: pd.DataFrame, mask: NDArray[np.bool_]) -> pd.DataFrame:
    if mask.shape != X.shape:
        raise ValueError("Mask shape does not match X")
    values = X.to_numpy(dtype=float, copy=True)
    values[mask] = np.nan
    return pd.DataFrame(values, index=X.index, columns=X.columns)


def exact_gaussian_posterior_complete(
    X_missing: pd.DataFrame,
    mean: ArrayLike,
    covariance: ArrayLike,
    *,
    seed: int,
    jitter: float = 1e-9,
) -> pd.DataFrame:
    """Sample every row from the exact Gaussian conditional distribution.

    The precision-matrix form is used, so each row only requires solving a
    system whose dimension is the number of missing variables rather than the
    number of observed variables.
    """
    values = X_missing.to_numpy(dtype=float, copy=True)
    mu = np.asarray(mean, dtype=float)
    Sigma = np.asarray(covariance, dtype=float)
    n, p = values.shape
    if mu.shape != (p,) or Sigma.shape != (p, p):
        raise ValueError("Oracle mean or covariance has incompatible shape")
    if not np.allclose(Sigma, Sigma.T, atol=1e-8):
        raise ValueError("Oracle covariance must be symmetric")

    precision = linalg.cho_solve(
        linalg.cho_factor(Sigma + jitter * np.eye(p), lower=True, check_finite=True),
        np.eye(p),
        check_finite=True,
    )
    rng = np.random.default_rng(seed)

    for i in range(n):
        missing = np.flatnonzero(np.isnan(values[i]))
        if missing.size == 0:
            continue
        observed = np.flatnonzero(~np.isnan(values[i]))
        if observed.size == 0:
            values[i, missing] = rng.multivariate_normal(mu, Sigma)[missing]
            continue

        Omega_mm = precision[np.ix_(missing, missing)]
        Omega_mo = precision[np.ix_(missing, observed)]
        rhs = Omega_mo @ (values[i, observed] - mu[observed])
        factor = _stable_cho_factor(Omega_mm, jitter=jitter)
        conditional_mean = mu[missing] - linalg.cho_solve(factor, rhs)
        conditional_cov = linalg.cho_solve(factor, np.eye(missing.size))
        conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)
        cond_chol = _stable_cholesky(conditional_cov, jitter=jitter)
        values[i, missing] = conditional_mean + cond_chol @ rng.standard_normal(
            missing.size
        )

    if not np.isfinite(values).all():
        raise RuntimeError("Exact posterior completion produced nonfinite values")
    return pd.DataFrame(values, index=X_missing.index, columns=X_missing.columns)


def _stable_cho_factor(
    matrix: NDArray[np.float64], *, jitter: float
) -> tuple[NDArray[np.float64], bool]:
    eye = np.eye(matrix.shape[0])
    current = jitter
    for _ in range(8):
        try:
            return linalg.cho_factor(
                0.5 * (matrix + matrix.T) + current * eye,
                lower=True,
                check_finite=True,
            )
        except linalg.LinAlgError:
            current *= 10.0
    raise linalg.LinAlgError("Unable to stabilize a positive definite system")


def _stable_cholesky(
    matrix: NDArray[np.float64], *, jitter: float
) -> NDArray[np.float64]:
    eye = np.eye(matrix.shape[0])
    current = jitter
    for _ in range(8):
        try:
            return np.linalg.cholesky(0.5 * (matrix + matrix.T) + current * eye)
        except np.linalg.LinAlgError:
            current *= 10.0
    raise np.linalg.LinAlgError("Unable to stabilize conditional covariance")


def complete_matrix(
    method: CompletionName,
    *,
    X_complete: pd.DataFrame,
    X_missing: pd.DataFrame,
    seed: int,
    oracle_mean: ArrayLike | None = None,
    oracle_covariance: ArrayLike | None = None,
    iterative_max_iter: int = 10,
    iterative_nearest_features: int = 50,
) -> pd.DataFrame:
    if method == "oracle_complete":
        completed = X_complete.copy()
    elif method == "median":
        completed = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(X_missing),
            index=X_missing.index,
            columns=X_missing.columns,
        )
    elif method == "exact_gaussian_posterior":
        if oracle_mean is None or oracle_covariance is None:
            raise ValueError("Exact posterior requires oracle mean and covariance")
        completed = exact_gaussian_posterior_complete(
            X_missing,
            oracle_mean,
            oracle_covariance,
            seed=seed,
        )
    elif method in {"bayesianridge_mean", "bayesianridge_posterior"}:
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=iterative_max_iter,
            n_nearest_features=min(iterative_nearest_features, X_missing.shape[1]),
            initial_strategy="median",
            skip_complete=True,
            imputation_order="ascending",
            sample_posterior=(method == "bayesianridge_posterior"),
            random_state=seed,
        )
        completed = pd.DataFrame(
            imputer.fit_transform(X_missing),
            index=X_missing.index,
            columns=X_missing.columns,
        )
    else:
        raise ValueError(f"Unsupported completion method: {method}")

    array = completed.to_numpy(dtype=float)
    if array.shape != X_complete.shape or not np.isfinite(array).all():
        raise RuntimeError(f"Completion method {method} returned an invalid matrix")
    return completed


def standardize_completed_matrix(
    X_completed: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    values = scaler.fit_transform(X_completed)
    if not np.isfinite(values).all():
        raise RuntimeError("Standardization produced nonfinite values")
    return (
        pd.DataFrame(values, index=X_completed.index, columns=X_completed.columns),
        scaler,
    )


def transform_gaussian_parameters_with_scaler(
    mean: ArrayLike,
    covariance: ArrayLike,
    scaler: StandardScaler,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transform population Gaussian parameters into a fitted scaler's space.

    ``StandardScaler`` applies ``(X - mean_) / scale_`` featurewise.  The
    resulting population parameters are therefore transformed by the same
    affine map.  This is used only by Experiment A, where the data-generating
    mean and covariance are known.
    """
    mu = np.asarray(mean, dtype=float)
    Sigma = np.asarray(covariance, dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    center = np.asarray(scaler.mean_, dtype=float)
    p = scale.size
    if mu.shape != (p,) or Sigma.shape != (p, p):
        raise ValueError("Gaussian parameters and fitted scaler have incompatible shapes")
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("Fitted scaler contains invalid scales")
    scaled_mean = (mu - center) / scale
    scaled_covariance = Sigma / np.outer(scale, scale)
    scaled_covariance = 0.5 * (scaled_covariance + scaled_covariance.T)
    return scaled_mean, scaled_covariance


def _equicorrelated_s_matrix(
    covariance: ArrayLike,
    *,
    eigenvalue_floor: float = 1e-10,
    safety_factor: float = 0.999,
) -> tuple[NDArray[np.float64], float]:
    """Return a valid equicorrelated diagonal S matrix for general covariance.

    The usual equicorrelated construction is defined on a correlation matrix.
    We normalize the supplied covariance, choose
    ``s = min(1, 2 * lambda_min(R))``, and transform the diagonal S matrix back
    to the supplied scale.  A tiny safety factor avoids numerical boundary
    failures in the conditional covariance.
    """
    Sigma = np.asarray(covariance, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Covariance must be square")
    Sigma = 0.5 * (Sigma + Sigma.T)
    variances = np.diag(Sigma)
    if np.any(variances <= 0) or not np.isfinite(Sigma).all():
        raise ValueError("Covariance must be finite with positive diagonal")
    std = np.sqrt(variances)
    correlation = Sigma / np.outer(std, std)
    correlation = 0.5 * (correlation + correlation.T)
    lambda_min = float(np.linalg.eigvalsh(correlation).min())
    if lambda_min <= eigenvalue_floor:
        raise ValueError(
            "Known covariance is not sufficiently positive definite for the "
            f"equicorrelated construction: lambda_min={lambda_min:.3e}"
        )
    s_correlation = min(1.0, 2.0 * lambda_min) * safety_factor
    S = np.diag(s_correlation * variances)
    return S, float(s_correlation)


def gaussian_knockoff_known_covariance(
    X: ArrayLike,
    *,
    mean: ArrayLike,
    covariance: ArrayLike,
    seed: int,
    jitter: float = 1e-10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Generate exact second-order Gaussian equicorrelated knockoffs.

    This implementation uses the known population mean and covariance and is
    independent of ``knockpy`` covariance estimation.  For row-vector data,

    ``E[X_tilde | X] = mu + (X - mu) @ (I - Sigma^{-1} S)``

    and

    ``Cov[X_tilde | X] = 2S - S Sigma^{-1} S``.
    """
    values = np.asarray(X, dtype=float)
    mu = np.asarray(mean, dtype=float)
    Sigma = np.asarray(covariance, dtype=float)
    if values.ndim != 2:
        raise ValueError("X must be two dimensional")
    n, p = values.shape
    if mu.shape != (p,) or Sigma.shape != (p, p):
        raise ValueError("Known Gaussian parameters have incompatible shapes")
    if not np.isfinite(values).all() or not np.isfinite(mu).all():
        raise ValueError("Known-covariance knockoff input must be finite")
    Sigma = 0.5 * (Sigma + Sigma.T)
    factor = _stable_cho_factor(Sigma, jitter=jitter)
    S, s_correlation = _equicorrelated_s_matrix(Sigma)
    sigma_inverse_s = linalg.cho_solve(factor, S, check_finite=True)
    conditional_mean = mu + (values - mu) @ (np.eye(p) - sigma_inverse_s)
    conditional_covariance = 2.0 * S - S @ sigma_inverse_s
    conditional_covariance = 0.5 * (
        conditional_covariance + conditional_covariance.T
    )
    conditional_cholesky = _stable_cholesky(
        conditional_covariance, jitter=jitter
    )
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, p)) @ conditional_cholesky.T
    knockoff = conditional_mean + noise
    if knockoff.shape != values.shape or not np.isfinite(knockoff).all():
        raise RuntimeError("Known-covariance Gaussian knockoff generation failed")
    return knockoff, Sigma, s_correlation




def estimate_factor_covariance(
    X: ArrayLike,
    *,
    rank: int,
    diag_floor: float = 1e-3,
    shrinkage: float = 0.05,
) -> NDArray[np.float64]:
    """Estimate a low-rank plus diagonal covariance for small-n omics data.

    The estimator uses a probabilistic-PCA style decomposition of the empirical
    covariance. The leading eigenvectors form the common-factor block. The
    remaining feature-specific variance is placed on the diagonal. A small
    identity shrinkage and eigenvalue clipping make the matrix suitable for
    Gaussian knockoff construction. The outcome is never used.
    """
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError("Factor covariance requires a two-dimensional matrix with n >= 3 and p >= 2")
    if not np.isfinite(values).all():
        raise ValueError("Factor covariance requires finite values")
    n, p = values.shape
    max_rank = min(n - 1, p - 1)
    if not 1 <= int(rank) <= max_rank:
        raise ValueError(f"factor_rank must lie in [1, {max_rank}] for shape {(n, p)}")
    if diag_floor <= 0.0:
        raise ValueError("diag_floor must be positive")
    if not 0.0 <= shrinkage < 1.0:
        raise ValueError("shrinkage must lie in [0, 1)")

    centered = values - values.mean(axis=0, keepdims=True)
    empirical = centered.T @ centered / float(max(n - 1, 1))
    empirical = 0.5 * (empirical + empirical.T)
    eigenvalues, eigenvectors = linalg.eigh(empirical, check_finite=True)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    rank = int(rank)
    tail = eigenvalues[rank:]
    noise_level = float(np.mean(tail)) if tail.size else float(diag_floor)
    noise_level = max(noise_level, float(diag_floor))
    common_values = np.maximum(eigenvalues[:rank] - noise_level, 0.0)
    loadings = eigenvectors[:, :rank] * np.sqrt(common_values)[None, :]
    low_rank = loadings @ loadings.T

    uniqueness = np.diag(empirical - low_rank).copy()
    uniqueness = np.maximum(uniqueness, float(diag_floor))
    Sigma = low_rank + np.diag(uniqueness)

    empirical_diag = np.maximum(np.diag(empirical), float(diag_floor))
    current_diag = np.maximum(np.diag(Sigma), float(diag_floor))
    scale = np.sqrt(empirical_diag / current_diag)
    Sigma = scale[:, None] * Sigma * scale[None, :]

    target = np.diag(empirical_diag)
    Sigma = (1.0 - shrinkage) * Sigma + shrinkage * target
    Sigma = 0.5 * (Sigma + Sigma.T)
    minimum = float(np.linalg.eigvalsh(Sigma).min())
    if minimum < diag_floor:
        Sigma = Sigma + np.eye(p) * (diag_floor - minimum + 1e-10)
    if not np.isfinite(Sigma).all():
        raise RuntimeError("Factor covariance estimation produced non-finite values")
    return Sigma


def generate_knockoff(
    X_scaled: pd.DataFrame,
    config: GeneratorConfig,
    *,
    seed: int,
    known_mean: ArrayLike | None = None,
    known_covariance: ArrayLike | None = None,
) -> KnockoffPair:
    values = X_scaled.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Knockoff generation requires complete finite X")
    set_all_seeds(seed)
    start = time.perf_counter()
    Sigma_used: NDArray[np.float64] | None = None

    gaussian_methods = {
        "gaussian_equicorrelated": "equicorrelated",
        "gaussian_mvr": "mvr",
        "gaussian_sdp": "sdp",
    }
    factor_methods = {
        "gaussian_factor_equicorrelated": "equicorrelated",
        "gaussian_factor_mvr": "mvr",
        "gaussian_factor_sdp": "sdp",
    }
    if config.generator in gaussian_methods or config.generator in factor_methods:
        try:
            from knockpy.knockoffs import GaussianSampler
        except ImportError as exc:
            raise RuntimeError(
                "knockpy is required. Activate the existing STABL environment."
            ) from exc
        if config.generator in gaussian_methods:
            method = gaussian_methods[config.generator]
            sampler = GaussianSampler(values, method=method)
        else:
            if config.factor_rank is None:
                raise ValueError("Factor Gaussian generator requires factor_rank")
            Sigma_used = estimate_factor_covariance(
                values,
                rank=config.factor_rank,
                diag_floor=config.factor_diag_floor,
                shrinkage=config.factor_shrinkage,
            )
            method = factor_methods[config.generator]
            sampler = GaussianSampler(values, Sigma=Sigma_used, method=method)
        X_tilde = np.asarray(sampler.sample_knockoffs(), dtype=float)
        raw_sigma = getattr(sampler, "Sigma", None)
        if raw_sigma is not None:
            Sigma_used = np.asarray(raw_sigma, dtype=float)
    elif config.generator == "gaussian_equicorrelated_true_sigma":
        if known_mean is None or known_covariance is None:
            raise ValueError(
                "gaussian_equicorrelated_true_sigma requires known_mean and "
                "known_covariance in the same scaled space as X"
            )
        X_tilde, Sigma_used, _ = gaussian_knockoff_known_covariance(
            values,
            mean=known_mean,
            covariance=known_covariance,
            seed=seed,
        )
    elif config.generator == "official_plsko":
        sampler = OfficialPLSKOSampler(
            X_scaled,
            threshold_abs=config.threshold_abs,
            threshold_q=config.threshold_q,
            ncomp=config.ncomp,
            sparsity=config.sparsity,
            random_state=seed,
        )
        X_tilde = np.asarray(sampler.sample_knockoffs(), dtype=float)
    else:
        raise ValueError(f"Unsupported generator: {config.generator}")

    if X_tilde.shape != values.shape or not np.isfinite(X_tilde).all():
        raise RuntimeError("Knockoff generator returned an invalid matrix")
    return KnockoffPair(
        X=values,
        X_tilde=X_tilde,
        generation_seconds=float(time.perf_counter() - start),
        Sigma_used=Sigma_used,
    )


def _array_fingerprint(value: ArrayLike | None) -> str | None:
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_or_generate_knockoff(
    X_scaled: pd.DataFrame,
    config: GeneratorConfig,
    *,
    seed: int,
    known_mean: ArrayLike | None = None,
    known_covariance: ArrayLike | None = None,
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> tuple[KnockoffPair, bool]:
    """Load a validated knockoff pair from cache or generate and cache it.

    Returns ``(pair, cache_hit)``.  The cache is accepted only when the exact
    scaled matrix, column order, generator configuration, seed, and any known
    Gaussian parameters match.
    """
    values = X_scaled.to_numpy(dtype=float)
    feature_names = X_scaled.columns.astype(str).tolist()
    x_hash = matrix_fingerprint(values, feature_names=feature_names)
    uses_known_parameters = known_mean is not None or known_covariance is not None
    expected = {
        # Version 1 remains compatible with previously cached sample-Sigma and
        # PLSKO pairs. Version 2 additionally fingerprints known Gaussian
        # parameters for the true-Sigma oracle generator.
        "cache_version": 2 if uses_known_parameters else 1,
        "x_fingerprint": x_hash,
        "generator_config": asdict(config),
        "seed": int(seed),
        "shape": list(values.shape),
        "known_mean_fingerprint": _array_fingerprint(known_mean),
        "known_covariance_fingerprint": _array_fingerprint(known_covariance),
    }
    path = None if cache_path is None else Path(cache_path)

    if path is not None and path.exists() and not refresh:
        try:
            with np.load(path, allow_pickle=False) as cached:
                metadata = json.loads(str(cached["metadata_json"].item()))
                core_keys = (
                    "cache_version",
                    "x_fingerprint",
                    "generator_config",
                    "seed",
                )
                for key in core_keys:
                    if metadata.get(key) != expected.get(key):
                        raise ValueError(f"cache {key} mismatch")
                if expected["cache_version"] >= 2:
                    for key in (
                        "known_mean_fingerprint",
                        "known_covariance_fingerprint",
                    ):
                        if metadata.get(key) != expected.get(key):
                            raise ValueError(f"cache {key} mismatch")
                X_tilde = np.asarray(cached["X_tilde"], dtype=float)
                if X_tilde.shape != values.shape or not np.isfinite(X_tilde).all():
                    raise ValueError("cached knockoff matrix is invalid")
                sigma = None
                if "Sigma_used" in cached.files:
                    sigma_array = np.asarray(cached["Sigma_used"], dtype=float)
                    if sigma_array.size:
                        sigma = sigma_array
                pair = KnockoffPair(
                    X=values,
                    X_tilde=X_tilde,
                    generation_seconds=float(metadata.get("generation_seconds", 0.0)),
                    Sigma_used=sigma,
                )
                return pair, True
        except Exception as exc:
            print(f"[cache invalid] {path}: {exc}; regenerating", flush=True)

    pair = generate_knockoff(
        X_scaled,
        config,
        seed=seed,
        known_mean=known_mean,
        known_covariance=known_covariance,
    )
    if path is not None:
        metadata = dict(expected)
        metadata["generation_seconds"] = pair.generation_seconds
        atomic_savez_compressed(
            path,
            X_tilde=pair.X_tilde,
            Sigma_used=(
                np.asarray(pair.Sigma_used, dtype=float)
                if pair.Sigma_used is not None
                else np.empty((0, 0), dtype=float)
            ),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            feature_names=np.asarray(feature_names),
        )
    return pair, False


def build_stabl_lambda_grid(
    task: TaskName,
    *,
    grid_size: int = 30,
    classification_c_min: float = 0.01,
    classification_c_max: float = 1.0,
    regression_alpha_min: float = 1e-2,
    regression_alpha_max: float = 1e2,
) -> dict[str, NDArray[np.float64]]:
    """Build the explicit paper-aligned regularization grid used by STABL.

    The original binary synthetic benchmark uses the package default, which is
    a 30-value linear C grid from 0.01 to 1.0.  V4 makes that grid explicit so
    the experiment remains reproducible if package defaults later change.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if task == "classification":
        if not 0.0 < classification_c_min < classification_c_max:
            raise ValueError("Invalid classification C range")
        return {
            "C": np.linspace(
                classification_c_min, classification_c_max, grid_size, dtype=float
            )
        }
    if task == "regression":
        if not 0.0 < regression_alpha_min < regression_alpha_max:
            raise ValueError("Invalid regression alpha range")
        return {
            "alpha": np.logspace(
                np.log10(regression_alpha_min),
                np.log10(regression_alpha_max),
                grid_size,
                dtype=float,
            )
        }
    raise ValueError(f"Unsupported task: {task}")


def fit_stabl_scores(
    X: ArrayLike,
    X_tilde: ArrayLike,
    y: ArrayLike,
    *,
    task: TaskName,
    groups: Sequence[Any] | None,
    n_bootstraps: int,
    n_jobs: int,
    random_state: int,
    threshold_grid: NDArray[np.float64],
    regularization_grid_size: int = 30,
    classification_c_min: float = 0.01,
    classification_c_max: float = 1.0,
    regression_alpha_min: float = 1e-2,
    regression_alpha_max: float = 1e2,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    try:
        from stabl.stabl import Stabl
    except ImportError as exc:
        raise RuntimeError(
            "The local stabl package is not importable. Run from the STABL repository "
            "root or install it in editable mode."
        ) from exc

    if task == "classification":
        estimator = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=int(1e6),
            random_state=random_state,
        )
    elif task == "regression":
        estimator = Lasso(max_iter=int(1e6), random_state=random_state)
    else:
        raise ValueError(f"Unsupported task: {task}")

    lambda_grid = build_stabl_lambda_grid(
        task,
        grid_size=regularization_grid_size,
        classification_c_min=classification_c_min,
        classification_c_max=classification_c_max,
        regression_alpha_min=regression_alpha_min,
        regression_alpha_max=regression_alpha_max,
    )

    model = Stabl(
        base_estimator=estimator,
        lambda_grid=lambda_grid,
        n_bootstraps=n_bootstraps,
        artificial_type="knockoff",
        artificial_proportion=1.0,
        sample_fraction=0.5,
        replace=False,
        fdr_threshold_range=np.asarray(threshold_grid, dtype=float),
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )
    group_array = None if groups is None else np.asarray(groups)
    model.fit(
        np.asarray(X, dtype=float),
        np.asarray(y),
        groups=group_array,
        X_artificial=np.asarray(X_tilde, dtype=float),
    )
    real = np.asarray(model.stabl_scores_, dtype=float)
    knockoff = np.asarray(model.stabl_scores_artificial_, dtype=float)
    if real.shape != knockoff.shape:
        raise RuntimeError("Real and knockoff STABL score paths have different shapes")
    return real, knockoff


def fit_lcd_statistics(
    X: ArrayLike,
    X_tilde: ArrayLike,
    y: ArrayLike,
    *,
    task: TaskName,
    random_state: int,
    cv_folds: int = 5,
    grid_size: int = 30,
    classification_c_min: float = 1e-3,
    classification_c_max: float = 10.0,
    regression_alpha_min: float = 1e-4,
    regression_alpha_max: float = 10.0,
    n_jobs: int = 1,
) -> LCDStatistics:
    """Fit a paired lasso model and return LCD knockoff statistics.

    The design is the symmetric augmented matrix ``[X, X_tilde]``.  For each
    feature pair, ``W_j = |beta_j| - |beta_tilde_j|``.  Hyperparameters are
    selected by deterministic shuffled cross-validation over a fixed grid.
    Inputs are already standardized jointly by the experiment pipeline; no
    separate transformation is fitted to either block.
    """
    values = np.asarray(X, dtype=float)
    knockoff = np.asarray(X_tilde, dtype=float)
    outcome = np.asarray(y)
    if values.ndim != 2 or knockoff.shape != values.shape:
        raise ValueError("X and X_tilde must be finite matrices with equal shape")
    if not np.isfinite(values).all() or not np.isfinite(knockoff).all():
        raise ValueError("LCD requires finite X and X_tilde")
    if cv_folds < 2 or grid_size < 2:
        raise ValueError("LCD cv_folds and grid_size must be at least 2")
    augmented = np.column_stack([values, knockoff])
    p = values.shape[1]
    start = time.perf_counter()

    if task == "classification":
        classes, counts = np.unique(outcome, return_counts=True)
        if classes.size != 2:
            raise ValueError("LCD classification currently requires a binary outcome")
        n_splits = min(int(cv_folds), int(counts.min()))
        if n_splits < 2:
            raise ValueError("Too few observations in one class for LCD cross-validation")
        if not 0.0 < classification_c_min < classification_c_max:
            raise ValueError("Invalid LCD classification C range")
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        model = LogisticRegressionCV(
            Cs=np.logspace(
                np.log10(classification_c_min),
                np.log10(classification_c_max),
                grid_size,
            ),
            cv=cv,
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            scoring="neg_log_loss",
            max_iter=int(1e6),
            random_state=random_state,
            n_jobs=n_jobs,
            refit=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=FutureWarning, module=r"sklearn\.linear_model\._logistic"
            )
            model.fit(augmented, outcome.astype(int))
        coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
        best_parameter = float(np.asarray(model.C_).reshape(-1)[0])
        model_name = "logistic_l1_cv_lcd"
    elif task == "regression":
        if not 0.0 < regression_alpha_min < regression_alpha_max:
            raise ValueError("Invalid LCD regression alpha range")
        n_splits = min(int(cv_folds), int(len(outcome)))
        if n_splits < 2:
            raise ValueError("Too few observations for LCD cross-validation")
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        model = LassoCV(
            alphas=np.logspace(
                np.log10(regression_alpha_min),
                np.log10(regression_alpha_max),
                grid_size,
            ),
            cv=cv,
            max_iter=int(1e6),
            n_jobs=n_jobs,
            random_state=random_state,
            selection="cyclic",
        )
        model.fit(augmented, outcome.astype(float))
        coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
        best_parameter = float(model.alpha_)
        model_name = "linear_lasso_cv_lcd"
    else:
        raise ValueError(f"Unsupported task: {task}")

    if coefficients.shape != (2 * p,) or not np.isfinite(coefficients).all():
        raise RuntimeError("LCD fit returned invalid coefficients")
    W = np.abs(coefficients[:p]) - np.abs(coefficients[p:])
    return LCDStatistics(
        W=np.asarray(W, dtype=float),
        best_parameter=best_parameter,
        fit_seconds=float(time.perf_counter() - start),
        model_name=model_name,
    )

def _max_scores(scores: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(scores, dtype=float)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return np.max(array, axis=1)
    raise ValueError("Scores must be one or two dimensional")


def fdp_plus_curve(
    real_scores: ArrayLike,
    knockoff_scores: ArrayLike,
    threshold_grid: ArrayLike,
) -> NDArray[np.float64]:
    real = _max_scores(real_scores)
    knockoff = _max_scores(knockoff_scores)
    grid = np.asarray(threshold_grid, dtype=float)
    return np.asarray(
        [
            (1.0 + float(np.sum(knockoff >= threshold)))
            / max(1.0, float(np.sum(real >= threshold)))
            for threshold in grid
        ]
    )


def select_stabl_min(
    real_scores: ArrayLike,
    knockoff_scores: ArrayLike,
    threshold_grid: ArrayLike,
) -> SelectionResult:
    real = _max_scores(real_scores)
    grid = np.asarray(threshold_grid, dtype=float)
    curve = fdp_plus_curve(real_scores, knockoff_scores, grid)
    index = int(np.argmin(curve))
    estimated = float(curve[index])
    if estimated > 1.0:
        return SelectionResult(
            np.zeros(real.shape[0], dtype=bool), 1.0, estimated, "stabl_min"
        )
    threshold = float(grid[index])
    return SelectionResult(real >= threshold, threshold, estimated, "stabl_min")


def select_stabl_q(
    real_scores: ArrayLike,
    knockoff_scores: ArrayLike,
    threshold_grid: ArrayLike,
    *,
    q: float,
) -> SelectionResult:
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    real = _max_scores(real_scores)
    grid = np.asarray(threshold_grid, dtype=float)
    curve = fdp_plus_curve(real_scores, knockoff_scores, grid)
    candidates = np.flatnonzero(curve <= q)
    if candidates.size == 0:
        return SelectionResult(
            np.zeros(real.shape[0], dtype=bool), 1.0, float(curve.min()), "stabl_q"
        )
    index = int(candidates[0])
    threshold = float(grid[index])
    return SelectionResult(real >= threshold, threshold, float(curve[index]), "stabl_q")


def select_knockoff_plus_from_w(
    W: ArrayLike,
    *,
    q: float,
    rule_name: str,
) -> SelectionResult:
    """Apply the standard knockoff-plus threshold to antisymmetric W statistics."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    statistics = np.asarray(W, dtype=float).reshape(-1)
    if not np.isfinite(statistics).all():
        raise ValueError("Knockoff statistics must be finite")
    candidates = np.sort(np.unique(np.abs(statistics[statistics != 0])))
    threshold = np.inf
    estimated = np.inf
    for candidate in candidates:
        ratio = (1.0 + float(np.sum(statistics <= -candidate))) / max(
            1.0, float(np.sum(statistics >= candidate))
        )
        if ratio <= q:
            threshold = float(candidate)
            estimated = float(ratio)
            break
    selected = (
        np.zeros(statistics.shape[0], dtype=bool)
        if not np.isfinite(threshold)
        else statistics >= threshold
    )
    return SelectionResult(selected, float(threshold), float(estimated), rule_name)


def select_stabl_knockoff_plus(
    real_scores: ArrayLike,
    knockoff_scores: ArrayLike,
    *,
    q: float,
    rule_name: str = "stabl_knockoff_plus",
) -> SelectionResult:
    """Knockoff-plus applied to paired STABL maximum-frequency differences."""
    W = _max_scores(real_scores) - _max_scores(knockoff_scores)
    return select_knockoff_plus_from_w(W, q=q, rule_name=rule_name)


def select_lcd_knockoff_plus(
    lcd_statistics: ArrayLike,
    *,
    q: float,
) -> SelectionResult:
    return select_knockoff_plus_from_w(
        lcd_statistics, q=q, rule_name="lcd_knockoff_plus"
    )


def apply_selection_rule(
    rule: SelectionRule,
    real_scores: ArrayLike,
    knockoff_scores: ArrayLike,
    *,
    q: float,
    threshold_grid: ArrayLike,
    lcd_statistics: ArrayLike | None = None,
) -> SelectionResult:
    if rule == "stabl_min":
        return select_stabl_min(real_scores, knockoff_scores, threshold_grid)
    if rule == "stabl_q":
        return select_stabl_q(
            real_scores, knockoff_scores, threshold_grid, q=q
        )
    if rule in {"stabl_knockoff_plus", "knockoff_plus"}:
        return select_stabl_knockoff_plus(
            real_scores, knockoff_scores, q=q, rule_name=rule
        )
    if rule == "lcd_knockoff_plus":
        if lcd_statistics is None:
            raise ValueError("lcd_knockoff_plus requires fitted LCD statistics")
        return select_lcd_knockoff_plus(lcd_statistics, q=q)
    raise ValueError(f"Unsupported selection rule: {rule}")

def selection_metrics(
    selected: ArrayLike,
    support: ArrayLike,
    p: int,
    *,
    target_fdr: float,
) -> dict[str, float | int]:
    selected_array = np.asarray(selected, dtype=bool)
    support_array = np.asarray(support, dtype=int)
    if selected_array.shape != (p,):
        raise ValueError("Selection vector has an incompatible shape")
    truth = np.zeros(p, dtype=bool)
    truth[support_array] = True
    tp = int(np.sum(selected_array & truth))
    fp = int(np.sum(selected_array & ~truth))
    count = int(selected_array.sum())
    union = int(np.sum(selected_array | truth))
    fdp = float(fp / max(1, count))
    return {
        "n_selected": count,
        "true_positives": tp,
        "false_positives": fp,
        "fdp": fdp,
        "power": float(tp / max(1, int(truth.sum()))),
        "support_jaccard": float(tp / max(1, union)),
        "fdp_exceeds_target": int(fdp > target_fdr),
    }


def max_selection_scores(scores: ArrayLike) -> NDArray[np.float64]:
    """Return one maximum STABL selection-frequency score per feature."""
    values = np.asarray(scores, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return np.max(values, axis=1)
    raise ValueError("STABL scores must be one or two dimensional")


def ranking_metrics(
    real_scores: ArrayLike,
    support: ArrayLike,
    p: int,
) -> dict[str, float]:
    """Threshold-independent support-ranking metrics for generator studies."""
    values = max_selection_scores(real_scores)
    if values.shape != (p,):
        raise ValueError("Real STABL scores have an incompatible feature dimension")
    truth = np.zeros(p, dtype=int)
    truth[np.asarray(support, dtype=int)] = 1
    order = np.argsort(-values, kind="mergesort")
    output: dict[str, float] = {
        "average_precision": float(average_precision_score(truth, values)),
        "ranking_auroc": float(roc_auc_score(truth, values)),
        "mean_signal_score": float(np.mean(values[truth == 1])),
        "mean_null_score": float(np.mean(values[truth == 0])),
        "median_signal_rank": float(
            np.median(np.flatnonzero(truth[order] == 1) + 1)
        ),
    }
    for k in (5, 10, 15, 20):
        if k <= p:
            output[f"precision_at_{k}"] = float(np.mean(truth[order[:k]]))
    return output


def selected_indices_string(selected: ArrayLike) -> str:
    indices = np.flatnonzero(np.asarray(selected, dtype=bool))
    return ";".join(str(int(item)) for item in indices)


def parse_selected_indices(value: Any) -> set[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value)
    if not text:
        return set()
    return {int(item) for item in text.split(";") if item}


def mean_pairwise_jaccard(sets: Sequence[set[int]]) -> float:
    if len(sets) < 2:
        return np.nan
    values: list[float] = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            left = sets[left_index]
            right = sets[right_index]
            union = left | right
            values.append(1.0 if not union else len(left & right) / len(union))
    return float(np.mean(values))


def imputation_recovery_metrics(
    X_complete: pd.DataFrame,
    X_completed: pd.DataFrame,
    mask: NDArray[np.bool_],
    *,
    true_covariance: ArrayLike | None = None,
    precision_ridge: float = 1e-3,
    compute_precision: bool = True,
) -> dict[str, float]:
    truth = X_complete.to_numpy(dtype=float)
    estimate = X_completed.to_numpy(dtype=float)
    difference = estimate[mask] - truth[mask]
    metrics: dict[str, float] = {
        "masked_rmse": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
        "masked_mae": float(np.mean(np.abs(difference))) if difference.size else 0.0,
    }
    sample_cov_truth = np.cov(truth, rowvar=False)
    sample_cov_estimate = np.cov(estimate, rowvar=False)
    sample_corr_truth = np.corrcoef(truth, rowvar=False)
    sample_corr_estimate = np.corrcoef(estimate, rowvar=False)
    metrics["sample_cov_fro_relative"] = relative_frobenius(
        sample_cov_estimate, sample_cov_truth
    )
    metrics["sample_corr_fro_relative"] = relative_frobenius(
        sample_corr_estimate, sample_corr_truth
    )
    if true_covariance is not None:
        Sigma = np.asarray(true_covariance, dtype=float)
        metrics["true_cov_fro_relative"] = relative_frobenius(
            sample_cov_estimate, Sigma
        )
    if compute_precision:
        p = truth.shape[1]
        truth_precision = np.linalg.inv(sample_cov_truth + precision_ridge * np.eye(p))
        estimate_precision = np.linalg.inv(
            sample_cov_estimate + precision_ridge * np.eye(p)
        )
        metrics["ridge_precision_fro_relative"] = relative_frobenius(
            estimate_precision, truth_precision
        )
    return metrics


def pair_diagnostics(X: ArrayLike, X_tilde: ArrayLike) -> dict[str, float]:
    real = np.asarray(X, dtype=float)
    knockoff = np.asarray(X_tilde, dtype=float)
    if real.shape != knockoff.shape:
        raise ValueError("X and X_tilde must have equal shape")
    pair_corrs = []
    for j in range(real.shape[1]):
        if np.std(real[:, j]) <= 1e-12 or np.std(knockoff[:, j]) <= 1e-12:
            continue
        pair_corrs.append(float(np.corrcoef(real[:, j], knockoff[:, j])[0, 1]))
    pair_corr = float(np.nanmean(pair_corrs)) if pair_corrs else np.nan
    variances = np.var(real, axis=0, ddof=1)
    cov_pairs = np.asarray(
        [np.cov(real[:, j], knockoff[:, j], ddof=1)[0, 1] for j in range(real.shape[1])]
    )
    valid = variances > 1e-12
    s_relative = float(np.mean(1.0 - cov_pairs[valid] / variances[valid]))
    cov_real = np.cov(real, rowvar=False)
    cov_knockoff = np.cov(knockoff, rowvar=False)
    cross = np.cov(np.concatenate([real, knockoff], axis=1), rowvar=False)[
        : real.shape[1], real.shape[1] :
    ]
    knockoff_std = np.std(knockoff, axis=0, ddof=1)
    return {
        "pair_corr_mean": pair_corr,
        "s_relative_mean": s_relative,
        "cov_kk_fro_relative": relative_frobenius(cov_knockoff, cov_real),
        "cov_xk_offdiag_rmse": offdiag_rmse(cross, cov_real),
        "near_constant_knockoff_columns": int(np.sum(knockoff_std <= 1e-10)),
        "knockoff_abs_max": float(np.max(np.abs(knockoff))),
    }


def _paired_c2st_auc(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    seed: int,
) -> float:
    if left.shape != right.shape:
        raise ValueError("C2ST matrices must have equal shape")
    n = left.shape[0]
    if n < 4:
        return np.nan
    features = np.vstack([left, right])
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    groups = np.concatenate([np.arange(n), np.arange(n)])
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=10000,
            random_state=seed,
        ),
    )
    probabilities = cross_val_predict(
        classifier,
        features,
        labels,
        groups=groups,
        cv=GroupKFold(n_splits=min(5, n)),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    auc = float(roc_auc_score(labels, probabilities))
    return max(auc, 1.0 - auc)


def c2st_diagnostics(
    X: ArrayLike,
    X_tilde: ArrayLike,
    *,
    seed: int,
) -> dict[str, float]:
    """Marginal and paired-swap classifier two-sample diagnostics."""
    real = np.asarray(X, dtype=float)
    knockoff = np.asarray(X_tilde, dtype=float)
    marginal = _paired_c2st_auc(real, knockoff, seed=seed)
    original_pairs = np.column_stack([real, knockoff])
    swapped_pairs = np.column_stack([knockoff, real])
    swap = _paired_c2st_auc(original_pairs, swapped_pairs, seed=seed + 1)
    return {"marginal_c2st_auc": marginal, "swap_c2st_auc": swap}


def relative_frobenius(estimate: ArrayLike, reference: ArrayLike) -> float:
    est = np.asarray(estimate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    return float(np.linalg.norm(est - ref, ord="fro") / max(np.linalg.norm(ref, ord="fro"), 1e-12))


def offdiag_rmse(estimate: ArrayLike, reference: ArrayLike) -> float:
    est = np.asarray(estimate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    keep = ~np.eye(est.shape[0], dtype=bool)
    return float(np.sqrt(np.mean((est[keep] - ref[keep]) ** 2)))


def summarize_results(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    target_fdr: float,
) -> pd.DataFrame:
    valid = frame[frame.get("error", "").fillna("").eq("")].copy()
    if valid.empty:
        return pd.DataFrame()
    numeric = [
        "fdp",
        "power",
        "n_selected",
        "true_positives",
        "false_positives",
        "fdp_exceeds_target",
    ]
    grouped = valid.groupby(list(group_columns), dropna=False)
    rows: list[dict[str, Any]] = []
    for key, block in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key))
        row["n_success"] = int(len(block))
        row["empirical_fdr"] = float(block["fdp"].mean())
        row["fdp_sd"] = float(block["fdp"].std(ddof=1)) if len(block) > 1 else 0.0
        row["power_mean"] = float(block["power"].mean())
        row["power_sd"] = float(block["power"].std(ddof=1)) if len(block) > 1 else 0.0
        row["selected_mean"] = float(block["n_selected"].mean())
        row["fdp_exceedance_probability"] = float(block["fdp_exceeds_target"].mean())
        row["target_fdr"] = target_fdr
        for column in numeric:
            if column in block.columns:
                row[f"{column}_median"] = float(block[column].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_columns)).reset_index(drop=True)


def cap_features_by_variance(X: pd.DataFrame, p_value: int | None) -> pd.DataFrame:
    if p_value is None or X.shape[1] <= p_value:
        return X.copy()
    variance = X.var(axis=0, ddof=1).sort_values(ascending=False)
    selected = set(variance.index[:p_value])
    columns = [column for column in X.columns if column in selected]
    return X.loc[:, columns].copy()


def filter_constant_columns(
    X: pd.DataFrame,
    *,
    relative_tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, list[str]]:
    values = X.to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=0)
    ranges = np.ptp(values, axis=0)
    scales = np.maximum(1.0, np.max(np.abs(values), axis=0))
    keep = finite & (ranges > relative_tolerance * scales)
    dropped = X.columns[~keep].astype(str).tolist()
    filtered = X.loc[:, keep].copy()
    if filtered.shape[1] < 2:
        raise ValueError("Fewer than two usable features remain")
    return filtered, dropped


def complete_real_matrix(
    X: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:
    strategy = strategy.lower()
    if strategy == "drop-columns":
        completed = X.dropna(axis=1, how="any")
    elif strategy == "median":
        usable = X.dropna(axis=1, how="all")
        if usable.shape[1] < 2:
            raise ValueError("Fewer than two nonempty columns remain before median completion")
        completed = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(usable),
            index=usable.index,
            columns=usable.columns,
        )
    elif strategy == "error":
        if X.isna().any().any():
            raise ValueError("The real matrix contains missing values")
        completed = X.copy()
    else:
        raise ValueError("complete strategy must be drop-columns, median, or error")
    completed, _ = filter_constant_columns(completed)
    return completed


def select_one_row_per_group(
    X: pd.DataFrame,
    groups: pd.Series,
    mode: str,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series | None]:
    mode = mode.lower()
    if mode == "all":
        return X.copy(), groups.loc[X.index].copy()
    frame = pd.DataFrame({"group": groups.loc[X.index]}, index=X.index)
    if mode == "first":
        chosen = frame.groupby("group", sort=False).head(1).index
    elif mode == "random":
        rng = np.random.default_rng(seed)
        chosen_rows = []
        for _, block in frame.groupby("group", sort=False):
            chosen_rows.append(rng.choice(block.index.to_numpy()))
        chosen = pd.Index(chosen_rows)
    else:
        raise ValueError("subject mode must be first, random, or all")
    return X.loc[chosen].copy(), None


def load_real_dataset(
    *,
    dataset: str,
    data_path: str | Path | None,
    omic: str,
    x_csv: str | Path | None,
    groups_csv: str | Path | None,
    subject_mode: str,
    complete_strategy: str,
    seed: int,
) -> RealDataset:
    dataset = dataset.lower()
    metadata: dict[str, Any] = {
        "dataset": dataset,
        "omic": omic,
        "subject_mode": subject_mode,
        "complete_strategy": complete_strategy,
    }

    if dataset == "ool":
        if data_path is None:
            raise ValueError("--data-path is required for OOL")
        training = Path(data_path) / "Onset of Labor" / "Training"
        X = pd.read_csv(training / f"{omic}.csv", index_col=0).astype(float)
        groups = pd.read_csv(training / "ID.csv", index_col=0).iloc[:, 0]
        common = X.index.intersection(groups.index)
        X = X.loc[common]
        groups = groups.loc[common]
        metadata["n_rows_raw"] = int(X.shape[0])
        metadata["p_raw"] = int(X.shape[1])
        X = complete_real_matrix(X, complete_strategy)
        X, groups_after = select_one_row_per_group(
            X, groups, subject_mode, seed=seed
        )
        groups = groups_after
        label = f"OOL_{safe_label(omic)}_{safe_label(subject_mode)}"
    elif dataset == "ssi":
        if data_path is None:
            raise ValueError("--data-path is required for SSI")
        try:
            from stabl import data as stabl_data
        except ImportError as exc:
            raise RuntimeError("The local stabl package must be importable") from exc
        X_train, _, _, _, ids, _ = stabl_data.load_ssi(str(data_path))
        if omic not in X_train:
            raise ValueError(f"SSI omic {omic!r} was not found: {list(X_train)}")
        X = X_train[omic].astype(float)
        groups = None if ids is None else pd.Series(ids, index=X.index)
        metadata["n_rows_raw"] = int(X.shape[0])
        metadata["p_raw"] = int(X.shape[1])
        X = complete_real_matrix(X, complete_strategy)
        if groups is not None:
            X, groups = select_one_row_per_group(X, groups, subject_mode, seed=seed)
        label = f"SSI_{safe_label(omic)}"
    elif dataset == "dream":
        if data_path is None:
            raise ValueError("--data-path is required for DREAM")
        try:
            from stabl import data as stabl_data
        except ImportError as exc:
            raise RuntimeError("The local stabl package must be importable") from exc
        X_train, _, _, _, ids, _ = stabl_data.load_dream(str(data_path))
        if omic not in X_train:
            raise ValueError(f"DREAM omic {omic!r} was not found: {list(X_train)}")
        X = X_train[omic].astype(float)
        groups = None if ids is None else pd.Series(ids, index=X.index)
        metadata["n_rows_raw"] = int(X.shape[0])
        metadata["p_raw"] = int(X.shape[1])
        metadata["n_subjects_raw"] = (
            int(groups.nunique()) if groups is not None else int(X.shape[0])
        )
        X = complete_real_matrix(X, complete_strategy)
        if groups is not None:
            X, groups = select_one_row_per_group(X, groups, subject_mode, seed=seed)
        label = f"DREAM_{safe_label(omic)}_{safe_label(subject_mode)}"
    elif dataset == "csv":
        if x_csv is None:
            raise ValueError("--x-csv is required when --dataset csv")
        X = pd.read_csv(x_csv, index_col=0).astype(float)
        groups = None
        if groups_csv is not None:
            groups_frame = pd.read_csv(groups_csv, index_col=0)
            groups = groups_frame.iloc[:, 0]
            common = X.index.intersection(groups.index)
            X = X.loc[common]
            groups = groups.loc[common]
        metadata["n_rows_raw"] = int(X.shape[0])
        metadata["p_raw"] = int(X.shape[1])
        X = complete_real_matrix(X, complete_strategy)
        if groups is not None:
            X, groups = select_one_row_per_group(X, groups, subject_mode, seed=seed)
        label = f"CSV_{safe_label(Path(x_csv).stem)}"
    else:
        raise ValueError("dataset must be ool, ssi, dream, or csv")

    X, dropped = filter_constant_columns(X)
    metadata["n_rows_final"] = int(X.shape[0])
    metadata["p_final"] = int(X.shape[1])
    metadata["constant_columns_dropped"] = int(len(dropped))
    return RealDataset(X=X, groups=groups, dataset_label=label, metadata=metadata)


def load_generator_configs(
    *,
    config_json: str | Path | None,
    generators: Iterable[str],
    plsko_threshold_abs: float | None,
    plsko_threshold_q: float | None,
    plsko_ncomp: int | None,
    plsko_sparsity: float,
) -> tuple[GeneratorConfig, ...]:
    if config_json is not None:
        raw = json.loads(Path(config_json).read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("Generator config JSON must contain a nonempty list")
        configs = tuple(GeneratorConfig.from_mapping(item) for item in raw)
    else:
        built: list[GeneratorConfig] = []
        for generator in generators:
            if generator == "gaussian_equicorrelated":
                built.append(GeneratorConfig("equicorr", "gaussian_equicorrelated"))
            elif generator == "gaussian_equicorrelated_true_sigma":
                built.append(
                    GeneratorConfig(
                        "equicorr_true_sigma",
                        "gaussian_equicorrelated_true_sigma",
                    )
                )
            elif generator == "gaussian_mvr":
                built.append(GeneratorConfig("mvr", "gaussian_mvr"))
            elif generator == "gaussian_sdp":
                built.append(GeneratorConfig("sdp", "gaussian_sdp"))
            elif generator in {
                "gaussian_factor_equicorrelated",
                "gaussian_factor_mvr",
                "gaussian_factor_sdp",
            }:
                raise ValueError(
                    "Factor Gaussian generators require --generator-configs-json "
                    "so factor_rank is explicit"
                )
            elif generator == "official_plsko":
                built.append(
                    GeneratorConfig(
                        "plsko_cli",
                        "official_plsko",
                        threshold_abs=plsko_threshold_abs,
                        threshold_q=plsko_threshold_q,
                        ncomp=plsko_ncomp,
                        sparsity=plsko_sparsity,
                    )
                )
            else:
                raise ValueError(f"Unsupported generator: {generator}")
        configs = tuple(built)
    labels = [config.label for config in configs]
    if len(labels) != len(set(labels)):
        raise ValueError("Generator config labels must be unique")
    return configs


def generator_configs_to_json(configs: Sequence[GeneratorConfig]) -> list[dict[str, Any]]:
    return [asdict(config) for config in configs]
