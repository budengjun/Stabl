#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imputation_effect_on_stabl.py

Artificially mask the complete OOL 3-omic dataset (CyTOF + Proteomics +
Metabolomics), impute it with different strategies, and compare STABL results
across conditions to assess whether the imputation strategy materially affects
STABL. Multi-omic setup mirrors run_cv_OOL_CyPrMe (STABL is fit per omic, then
the features selected across omics are concatenated for the final model).

Differences vs the previous version, all aimed at making the run actually finish:

  1. Knockoffs are generated ONCE per (fold, omic) and reused across the three
     base learners via Stabl.fit(..., X_artificial=...). Knockoff generation is
     deterministic given random_state, so this is numerically identical to the
     old behaviour, just 3x cheaper.
  2. IterativeImputer-based imputers (iterative / rf / missforest) now use
     n_nearest_features and skip_complete. Without this, one fold of
     Metabolomics (3529 columns, every column missing under MCAR 0.2) costs
     ~36 h for BayesianRidge and ~55 h for RandomForest, per fold, per omic.
  3. Progress is logged for every fold with a running ETA, and per omic at
     DEBUG level. The old code only logged every n_splits // 10 folds, which
     meant hours of complete silence.
  4. BLAS threads are pinned to 1 so they do not fight with joblib workers.
  5. Results are checkpointed after every condition, and a run can resume from
     an interrupted state (--no-resume to disable).
  6. --smoke runs a tiny end-to-end configuration in a few minutes, which is
     what you should run first after any edit.
  7. Per-fold metrics and the raw per-fold predictions are saved, so conditions
     can be compared with paired tests offline (see paired_tests.py). All
     conditions share the same CV splits (random_state), so they are paired at
     the sample level.
  8. The cache directory name carries an MD5 fingerprint of every config field
     that affects results. Change n_splits or n_bootstraps and the old cache is
     simply not found, instead of being silently reused.

Usage
    python imputation_effect_on_stabl.py --smoke
    python imputation_effect_on_stabl.py --mechanism MCAR --missing-rate 0.2
    python imputation_effect_on_stabl.py --mechanism MNAR --missing-rate 0.3 \
        --n-splits 25 --n-bootstraps 100 --imputers simple knn iterative

Outputs: metrics_long.csv, feature_recovery.csv, diagnostics.csv,
fold_metrics.csv, predictions.csv.gz, imputation_effect_figure.png, plus a
_cache_<fingerprint>/ directory used for resume.
"""

# Thread pinning must happen before numpy / sklearn import their BLAS backend.
# Stabl uses joblib with n_jobs=-1; if BLAS is also multi-threaded, the workers
# oversubscribe every core and the run appears to hang while burning 100% CPU.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import time
import pickle
import hashlib
import argparse
import logging
import warnings
from dataclasses import dataclass, replace
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, Lasso, ElasticNet
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone

# Julia is only needed for non-Gaussian knockoff margins (feat_type != None).
# Gaussian MX knockoffs go through knockpy and do not need it. Kept guarded.
try:
    from julia.api import Julia

    _JL = Julia(compiled_modules=False)
except Exception:  # noqa: BLE001
    _JL = None

from stabl.stabl import Stabl
from stabl.adaptive import ALasso
from stabl.preprocessing import LowInfoFilter

warnings.filterwarnings("ignore")

_NOTIFIER = None  # set in main(), used by notify()


# ====================================================================
# 0. Configuration
# ====================================================================
@dataclass
class Config:
    data_path: str = "../Sample Data"
    out_dir: str = "./Results OOL CyPrMe imputer compare"

    dataset: str = "Onset of Labor"
    omics: tuple = ("CyTOF", "Proteomics", "Metabolomics")
    task_type: str = "regression"

    # Masking
    mechanism: str = "MCAR"          # MCAR / MAR / MNAR
    missing_rate: float = 0.20
    n_mask_repeats: int = 3

    imputers: tuple = ("simple", "knn", "iterative", "rf", "missforest")

    # Fold-safe imputation. True would leak test information via a global imputer.
    global_impute: bool = False

    # Optional unsupervised feature pre-screen, applied to the complete data
    # before masking and identically to every condition. None keeps all features.
    # Metabolomics (3529 columns) drives the O(p^3) knockoff cost; capping it at
    # ~1500 roughly halves total runtime. Variance-based, so it never touches y.
    max_features_per_omic: int = None

    # STABL
    artificial_type: str = "knockoff"   # second-order Model-X (Gaussian, equicorrelated)
    n_bootstraps: int = 100
    sample_fraction: float = 0.5
    fdr_low: float = 0.1
    fdr_high: float = 1.0

    # Outer CV. 100 splits x 300 bootstraps is a final-run config, not something
    # you want to pay 16 times over (1 Real + 5 imputers x 3 repeats).
    n_splits: int = 25
    test_size: float = 0.2
    random_state: int = 42

    base_learners: tuple = ("Lasso", "ALasso", "ElasticNet")
    seed: int = 42

    # Parallelism handed to Stabl's bootstrap loop.
    n_jobs: int = -1

    # IterativeImputer cost controls. n_nearest_features is the single most
    # important knob: it caps the predictor count per imputed column.
    iter_n_nearest: int = 30
    iter_max_iter: int = 3
    rf_n_estimators: int = 20
    rf_max_depth: int = 8
    rf_max_iter: int = 2

    resume: bool = True

    # The out_dir the user passed, before " - MCAR rate 0p2" gets appended.
    # The Real condition's cache lives next to it, shared across mechanisms.
    base_out_dir: str = None


# ====================================================================
# 1. Logging + notifications
# ====================================================================
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def notify(cfg, title, message, priority="default", tags=""):
    """Log always; push to ntfy when the run notifier is installed."""
    logging.info(f"[notify] {title} | {message}")
    if _NOTIFIER is not None:
        try:
            _NOTIFIER.send(title=title, message=message,
                           priority=priority, tags=tags)
        except Exception:  # noqa: BLE001
            pass


def _fmt_dur(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ====================================================================
# 2. Data loading
# ====================================================================
def load_complete_omics(cfg: Config):
    """
    Load the OOL training omics as {omic: DataFrame}. Each omic is complete-cased
    (NaN columns dropped) so that Real is a true ground truth. All omics share the
    same sample index, aligned to y and groups.
    """
    from os.path import join

    base = join(cfg.data_path, cfg.dataset, "Training")

    y = pd.read_csv(join(base, "DOS.csv"), index_col=0).iloc[:, 0]
    ids = pd.read_csv(join(base, "ID.csv"), index_col=0).iloc[:, 0]

    raw = {omic: pd.read_csv(join(base, f"{omic}.csv"), index_col=0)
           for omic in cfg.omics}

    common = y.index
    for omic, X in raw.items():
        common = common.intersection(X.index)
    y, ids = y.loc[common], ids.loc[common]

    data_dict = {}
    for omic, X in raw.items():
        X = X.loc[common]
        n_before = X.shape[1]
        X = X.dropna(axis=1, how="any")
        dropped = n_before - X.shape[1]
        assert not X.isna().any().any(), f"{omic} still contains NaN"
        X = X.astype(float)

        if cfg.max_features_per_omic and X.shape[1] > cfg.max_features_per_omic:
            keep = (X.var(axis=0)
                     .sort_values(ascending=False)
                     .index[:cfg.max_features_per_omic])
            X = X[sorted(keep, key=list(X.columns).index)]
            logging.info(f"[data] {omic}: pre-screened to top "
                         f"{cfg.max_features_per_omic} features by variance")

        data_dict[omic] = X
        logging.info(f"[data] {omic}: {X.shape[0]} samples x {X.shape[1]} features "
                     f"(dropped {dropped} NaN columns)")

    logging.info(f"[data] {len(common)} common samples; {ids.nunique()} groups; "
                 f"total {sum(v.shape[1] for v in data_dict.values())} features")
    return data_dict, y.astype(float), ids


# ====================================================================
# 3. Masking
# ====================================================================
def make_mask(X, y, mechanism, rate, rng):
    n, p = X.shape
    mask = np.zeros((n, p), dtype=bool)

    if mechanism == "MCAR":
        mask = rng.rand(n, p) < rate

    elif mechanism == "MAR":
        Xv = X.values
        for j in range(p):
            driver = rng.randint(p)
            while driver == j and p > 1:
                driver = rng.randint(p)
            r = pd.Series(Xv[:, driver]).rank(pct=True).values
            prob = np.clip(rate * 2.0 * r, 0, 1)
            mask[:, j] = rng.rand(n) < prob

    elif mechanism == "MNAR":
        Xv = X.values
        y_rank = y.rank(pct=True).values
        for j in range(p):
            col_rank = pd.Series(Xv[:, j]).rank(pct=True).values
            score = 0.5 * col_rank + 0.5 * y_rank
            prob = np.clip(rate * 2.0 * score, 0, 1)
            mask[:, j] = rng.rand(n) < prob
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    # Keep at least max(5, 20%) observed values per column.
    min_keep = max(5, int(0.2 * n))
    for j in range(p):
        obs = np.where(~mask[:, j])[0]
        if len(obs) < min_keep:
            miss_idx = np.where(mask[:, j])[0]
            give_back = rng.choice(miss_idx, size=min_keep - len(obs), replace=False)
            mask[give_back, j] = False
    return mask


def apply_mask(X, mask):
    arr = X.to_numpy(dtype=float, copy=True)
    arr[mask] = np.nan
    return pd.DataFrame(arr, index=X.index, columns=X.columns)


def mask_data_dict(data_dict, y, mechanism, rate, rng):
    out = {}
    total_realized, total_cells = 0, 0
    for omic, X in data_dict.items():
        m = make_mask(X, y, mechanism, rate, rng)
        out[omic] = apply_mask(X, m)
        total_realized += m.sum()
        total_cells += m.size
    logging.info(f"[mask] mechanism={mechanism} target={rate:.2f} "
                 f"overall realized={total_realized / total_cells:.3f}")
    return out


# ====================================================================
# 4. Imputer registry
# ====================================================================
def _kw_keep_empty():
    """keep_empty_features exists from sklearn 1.2. Without it, a column that is
    all-NaN inside one training fold gets dropped, so the feature set silently
    changes between folds."""
    try:
        SimpleImputer(keep_empty_features=True)
        return {"keep_empty_features": True}
    except TypeError:
        return {}


def build_imputer_factory(name, cfg: Config):
    name = name.lower()
    seed = cfg.seed
    ke = _kw_keep_empty()

    if name == "simple":
        return lambda: SimpleImputer(strategy="median", **ke)

    if name == "knn":
        return lambda: KNNImputer(n_neighbors=5, weights="distance", **ke)

    if name == "iterative":
        return lambda: IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=cfg.iter_max_iter,
            n_nearest_features=cfg.iter_n_nearest,
            skip_complete=True,
            sample_posterior=False,
            random_state=seed,
            **ke,
        )

    if name == "rf":
        return lambda: IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                n_jobs=1,          # never nest joblib inside joblib
                random_state=seed,
            ),
            max_iter=cfg.rf_max_iter,
            n_nearest_features=cfg.iter_n_nearest,
            skip_complete=True,
            random_state=seed,
            **ke,
        )

    if name == "missforest":
        try:
            import miceforest  # noqa: F401

            return lambda: _MiceForestWrapper(seed=seed)
        except Exception:  # noqa: BLE001
            logging.warning("[imputer] miceforest not detected; missforest falls back "
                            "to rf (IterativeImputer + RF). For real missForest: "
                            "pip install miceforest")
            return build_imputer_factory("rf", cfg)

    raise ValueError(f"Unknown imputer: {name}")


class _MiceForestWrapper:
    def __init__(self, seed=42, iterations=2):
        self.seed = seed
        self.iterations = iterations
        self.kernel_ = None
        self.columns_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        import miceforest as mf

        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        # miceforest stringifies variable names internally for LightGBM, then
        # looks them up back in the DataFrame. Integer column names (what
        # VarianceThreshold hands us) blow up with KeyError: '40'.
        self.columns_ = [f"v{i}" for i in range(self.n_features_in_)]
        Xdf = pd.DataFrame(X, columns=self.columns_)

        self.kernel_ = mf.ImputationKernel(
            Xdf, datasets=1, save_models=1, random_state=self.seed)  # 5.x: datasets
        self.kernel_.mice(self.iterations, verbose=False)
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(np.asarray(X, dtype=float), columns=self.columns_)
        completed = self.kernel_.impute_new_data(Xdf).complete_data(0)
        return np.nan_to_num(np.asarray(completed, dtype=float), nan=0.0)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


# ====================================================================
# 5. STABL components
# ====================================================================
def build_stabl_estimators(cfg: Config):
    fdr_range = np.arange(cfg.fdr_low, cfg.fdr_high, 0.01)
    lasso = Lasso(max_iter=int(1e6), random_state=cfg.random_state)
    alasso = ALasso(max_iter=int(1e6), random_state=cfg.random_state)
    en = ElasticNet(max_iter=int(1e6), random_state=cfg.random_state)

    base = Stabl(
        base_estimator=lasso,
        n_bootstraps=cfg.n_bootstraps,
        artificial_type=cfg.artificial_type,
        artificial_proportion=1.0,
        replace=False,
        fdr_threshold_range=fdr_range,
        sample_fraction=cfg.sample_fraction,
        random_state=cfg.random_state,
        lambda_grid={"alpha": np.logspace(0, 2, 10)},
        n_jobs=cfg.n_jobs,
        verbose=0,
    )

    est = {}
    if "Lasso" in cfg.base_learners:
        est["STABL Lasso"] = clone(base).set_params(
            base_estimator=lasso, lambda_grid={"alpha": np.logspace(0, 2, 10)})
    if "ALasso" in cfg.base_learners:
        est["STABL ALasso"] = clone(base).set_params(
            base_estimator=alasso, lambda_grid={"alpha": np.logspace(0, 2, 10)})
    if "ElasticNet" in cfg.base_learners:
        est["STABL ElasticNet"] = clone(base).set_params(
            base_estimator=en,
            lambda_grid=[{"alpha": np.logspace(0.5, 2, 5), "l1_ratio": [.9]}])
    return est


def build_preprocessor(cfg: Config, imputer_factory):
    steps = [
        ("variance", VarianceThreshold(0.01)),
        ("lif", LowInfoFilter(max_nan_fraction=1.0)),
        ("impute", imputer_factory() if imputer_factory is not None
         else SimpleImputer(strategy="median", **_kw_keep_empty())),
        ("std", StandardScaler()),
    ]
    return Pipeline(steps)


def make_knockoffs(cfg: Config, Xtr, proto):
    """
    Generate the artificial (knockoff) block once for a given (fold, omic).

    Stabl._make_artificial_features is deterministic given random_state, so the
    three base learners were previously each generating the identical matrix.
    We generate it once and pass it in via Stabl.fit(X_artificial=...).
    """
    if cfg.artificial_type is None:
        return None
    s = clone(proto)
    s._make_artificial_features(
        X=Xtr,
        artificial_type=cfg.artificial_type,
        nb_noise=int(Xtr.shape[1] * 1.0),
        random_state=cfg.random_state,
    )
    return np.asarray(s.X_artificial_)


# ====================================================================
# 6. CV for one condition (multi-omic)
# ====================================================================
def run_condition(name, data_dict, y, groups, cfg: Config, imputer_factory=None):
    """
    Multi-omic CV mirroring multi_omic_stabl_cv: per fold, STABL is fit on each
    omic separately; features selected across omics are concatenated for the
    final LinearRegression. imputer_factory=None means the input is already
    complete (Real, or global pre-impute); otherwise it is fit per fold per omic.
    Selected feature names are namespaced as "{omic}::{feature}".
    """
    estimators = build_stabl_estimators(cfg)
    cv = GroupShuffleSplit(n_splits=cfg.n_splits, test_size=cfg.test_size,
                           random_state=cfg.random_state)

    preds = {m: pd.DataFrame(index=y.index, columns=[f"fold{k}" for k in range(cfg.n_splits)],
                             dtype=float) for m in estimators}
    nfeat = {m: [] for m in estimators}
    fold_feats = {m: [] for m in estimators}

    t_start = time.time()

    for k, (tr, te) in enumerate(cv.split(y, y, groups=groups)):
        t_fold = time.time()
        tr_idx, te_idx = y.index[tr], y.index[te]
        g_tr = groups.loc[tr_idx].values
        ytr = y.loc[tr_idx]

        std_frames = {}
        sel = {m: [] for m in estimators}   # list of (omic, feature)

        for omic, X in data_dict.items():
            t_omic = time.time()

            pre = build_preprocessor(cfg, imputer_factory)
            Xtr = pd.DataFrame(pre.fit_transform(X.loc[tr_idx]),
                               index=tr_idx, columns=pre.get_feature_names_out())
            Xte = pd.DataFrame(pre.transform(X.loc[te_idx]),
                               index=te_idx, columns=pre.get_feature_names_out())
            std_frames[omic] = (Xtr, Xte)
            t_pre = time.time() - t_omic

            # One knockoff block, shared by all base learners.
            t0 = time.time()
            proto = next(iter(estimators.values()))
            X_art = make_knockoffs(cfg, Xtr, proto)
            t_ko = time.time() - t0

            t0 = time.time()
            for m, stabl in estimators.items():
                s = clone(stabl)
                s.fit(Xtr, ytr, groups=g_tr, X_artificial=X_art)
                for f in s.get_feature_names_out():
                    sel[m].append((omic, f))
            t_fit = time.time() - t0

            logging.debug(
                f"    [{name}] fold {k + 1} | {omic} p={Xtr.shape[1]} | "
                f"pre {t_pre:.1f}s knockoff {t_ko:.1f}s stabl {t_fit:.1f}s")

        for m in estimators:
            nfeat[m].append(len(sel[m]))
            fold_feats[m].append([f"{omic}::{f}" for (omic, f) in sel[m]])
            if sel[m]:
                Xtr_sel = pd.concat(
                    [std_frames[o][0][f].rename(f"{o}::{f}") for (o, f) in sel[m]],
                    axis=1)
                Xte_sel = pd.concat(
                    [std_frames[o][1][f].rename(f"{o}::{f}") for (o, f) in sel[m]],
                    axis=1)
                lr = LinearRegression().fit(Xtr_sel, ytr)
                preds[m].loc[te_idx, f"fold{k}"] = lr.predict(Xte_sel)
            else:
                preds[m].loc[te_idx, f"fold{k}"] = float(np.mean(ytr))

        # Every fold, with an ETA. Silence is what made this look hung.
        dt = time.time() - t_fold
        done = k + 1
        eta = (time.time() - t_start) / done * (cfg.n_splits - done)
        nf = {m.replace("STABL ", ""): nfeat[m][-1] for m in estimators}
        logging.info(f"    [{name}] fold {done}/{cfg.n_splits} done in {_fmt_dur(dt)} "
                     f"| n_features {nf} | ETA {_fmt_dur(eta)}")

    out = {}
    for m in estimators:
        yhat = preds[m].median(axis=1)
        valid = yhat.notna()
        r2 = r2_score(y[valid], yhat[valid]) if valid.sum() > 1 else np.nan
        mae = mean_absolute_error(y[valid], yhat[valid]) if valid.sum() > 1 else np.nan

        # Per-fold metrics. NOTE: GroupShuffleSplit draws n_splits independent
        # test sets, so a sample appears in several of them and the folds are
        # NOT independent. Do not run an n=n_splits paired test on these. Use
        # the sample-level pairing in paired_tests.py instead, which is what
        # `preds` below is saved for.
        fm = []
        for k in range(cfg.n_splits):
            col = preds[m][f"fold{k}"]
            te = col.notna()
            if te.sum() > 1:
                fm.append(dict(
                    fold=k,
                    n_test=int(te.sum()),
                    r2=r2_score(y[te], col[te]),
                    mae=mean_absolute_error(y[te], col[te]),
                    n_features=nfeat[m][k],
                ))

        out[m] = {
            "r2": r2,
            "mae": mae,
            "n_features_mean": float(np.mean(nfeat[m])),
            "n_features_std": float(np.std(nfeat[m])),
            "jaccard_stability": jaccard_stability(fold_feats[m]),
            "floor_effect_flag": float(np.mean(nfeat[m])) < 1.0,
            "fold_feats": fold_feats[m],
            "fold_metrics": fm,
            "preds": preds[m],            # n_samples x n_splits, NaN off-fold
        }
    return out


def jaccard_stability(fold_feats):
    sets = [set(f) for f in fold_feats if len(f) > 0]
    if len(sets) < 2:
        return np.nan
    vals = []
    for a, b in combinations(sets, 2):
        u = a | b
        vals.append(len(a & b) / len(u) if u else np.nan)
    return float(np.nanmean(vals))


def feature_recovery(feats_cond, feats_real, min_fold_frac=0.5):
    def stable_set(fold_feats):
        c = Counter()
        for f in fold_feats:
            c.update(set(f))
        thr = max(1, int(min_fold_frac * len(fold_feats)))
        return set(k for k, v in c.items() if v >= thr)

    A, B = stable_set(feats_cond), stable_set(feats_real)
    if not A and not B:
        return dict(precision=np.nan, recall=np.nan, jaccard=np.nan,
                    n_cond=0, n_real=0)
    inter = len(A & B)
    return dict(
        precision=inter / len(A) if A else np.nan,
        recall=inter / len(B) if B else np.nan,
        jaccard=inter / len(A | B) if (A | B) else np.nan,
        n_cond=len(A), n_real=len(B))


# ====================================================================
# 7. Checkpointing / resume
# ====================================================================
# Every field here changes the numbers. n_jobs and out_dir deliberately do not
# appear: they affect speed and file layout, not results, and including them
# would force a full recompute every time the core count changes.
_FINGERPRINT_KEYS = (
    "omics", "mechanism", "missing_rate", "max_features_per_omic",
    "artificial_type", "n_bootstraps", "sample_fraction",
    "fdr_low", "fdr_high", "n_splits", "test_size", "random_state",
    "base_learners", "seed", "global_impute",
    "iter_n_nearest", "iter_max_iter",
    "rf_n_estimators", "rf_max_depth", "rf_max_iter",
)

# The Real condition runs on the complete data: no mask is ever applied, so it
# does not depend on the missingness mechanism, the missing rate, or any imputer
# setting. Its cache is therefore keyed on a narrower fingerprint and stored
# outside the mechanism-specific out_dir, so MCAR / MAR / MNAR all share one
# copy instead of recomputing an identical ~50 min result three times.
_REAL_IRRELEVANT_KEYS = (
    "mechanism", "missing_rate", "global_impute",
    "iter_n_nearest", "iter_max_iter",
    "rf_n_estimators", "rf_max_depth", "rf_max_iter",
)


def _fingerprint(cfg: Config, keys):
    blob = "|".join(f"{k}={getattr(cfg, k)!r}" for k in keys)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:8]


def cfg_fingerprint(cfg: Config):
    return _fingerprint(cfg, _FINGERPRINT_KEYS)


def real_fingerprint(cfg: Config):
    keys = tuple(k for k in _FINGERPRINT_KEYS if k not in _REAL_IRRELEVANT_KEYS)
    return _fingerprint(cfg, keys)


def _cache_dir(cfg: Config):
    """Cache directory is namespaced by a fingerprint of the result-affecting
    config. Without it, bumping n_splits from 25 to 100 and rerunning with
    resume=True would silently load the 25-split results and report them as
    100-split ones, with no warning anywhere."""
    d = os.path.join(cfg.out_dir, f"_cache_{cfg_fingerprint(cfg)}")
    os.makedirs(d, exist_ok=True)
    return d


def _real_cache_dir(cfg: Config):
    base = cfg.base_out_dir or cfg.out_dir
    d = os.path.join(f"{base} - shared", f"_cache_real_{real_fingerprint(cfg)}")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path(cfg: Config, key):
    safe = key.replace("/", "_").replace(" ", "_")
    if key == "Real":
        return os.path.join(_real_cache_dir(cfg), "Real.pkl")
    return os.path.join(_cache_dir(cfg), f"{safe}.pkl")


def load_cached(cfg: Config, key):
    p = _cache_path(cfg, key)
    if cfg.resume and os.path.exists(p):
        try:
            with open(p, "rb") as f:
                res = pickle.load(f)
            logging.info(f"[resume] loaded cached condition '{key}'")
            return res
        except Exception:  # noqa: BLE001
            logging.warning(f"[resume] cache for '{key}' is unreadable, recomputing")
    return None


def save_cached(cfg: Config, key, res):
    with open(_cache_path(cfg, key), "wb") as f:
        pickle.dump(res, f)


# ====================================================================
# 8. Orchestration
# ====================================================================
def run_experiment(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    payload = {k: (list(v) if isinstance(v, tuple) else v)
               for k, v in cfg.__dict__.items()}
    payload["cache_fingerprint"] = cfg_fingerprint(cfg)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(payload, f, indent=2)

    t0 = time.time()
    n_conditions = 1 + len(cfg.imputers) * cfg.n_mask_repeats
    fp = cfg_fingerprint(cfg)
    logging.info(f"[cache] fingerprint={fp} dir={_cache_dir(cfg)}")
    logging.info(f"[cache] real_fingerprint={real_fingerprint(cfg)} "
                f"dir={_real_cache_dir(cfg)}")
    notify(cfg, "STABL imputation run started",
           f"dataset={cfg.dataset} omics={list(cfg.omics)} "
           f"mechanism={cfg.mechanism} rate={cfg.missing_rate} "
           f"imputers={list(cfg.imputers)} fold_safe={not cfg.global_impute} "
           f"n_splits={cfg.n_splits} n_bootstraps={cfg.n_bootstraps} "
           f"conditions={n_conditions} fingerprint={fp}",
           tags="rocket")

    data_dict, y, groups = load_complete_omics(cfg)

    rows, recov_rows, fold_rows, pred_frames = [], [], [], []

    # Caches written before fold_metrics/preds existed simply lack those keys.
    # Degrade gracefully rather than invalidating them: a 15 h rerun is not
    # worth it, and the aggregate metrics in those caches are still correct.
    _MISSING = {"warned": False}

    def collect(cond, imp, rep, res):
        for m, r in res.items():
            rows.append(dict(condition=cond, imputer=imp, base_learner=m,
                             mask_repeat=rep,
                             **{k: v for k, v in r.items()
                                if k not in ("fold_feats", "fold_metrics", "preds")}))

            fm = r.get("fold_metrics")
            pr = r.get("preds")
            if fm is None or pr is None:
                if not _MISSING["warned"]:
                    logging.warning(
                        "[cache] some cached conditions predate per-fold logging; "
                        "fold_metrics.csv and predictions.csv.gz will cover only "
                        "the conditions that have it. Rerun with --no-resume to "
                        "regenerate everything.")
                    _MISSING["warned"] = True
                continue

            for row in fm:
                fold_rows.append(dict(condition=cond, imputer=imp,
                                      base_learner=m, mask_repeat=rep, **row))

            # Only the test rows of each fold are populated; the rest is NaN.
            # melt + dropna keeps the file to the ~n_splits * n_test rows that
            # actually carry a prediction. (stack()'s NaN handling varies by
            # pandas version, so do not rely on it.)
            df = (pr.rename_axis("sample")
                    .reset_index()
                    .melt(id_vars="sample", var_name="fold", value_name="yhat")
                    .dropna(subset=["yhat"]))
            df.insert(0, "mask_repeat", rep)
            df.insert(0, "base_learner", m)
            df.insert(0, "condition", cond)
            pred_frames.append(df)

    def flush():
        pd.DataFrame(rows).to_csv(
            os.path.join(cfg.out_dir, "metrics_long.csv"), index=False)
        if recov_rows:
            pd.DataFrame(recov_rows).to_csv(
                os.path.join(cfg.out_dir, "feature_recovery.csv"), index=False)
        if fold_rows:
            pd.DataFrame(fold_rows).to_csv(
                os.path.join(cfg.out_dir, "fold_metrics.csv"), index=False)
        if pred_frames:
            pd.concat(pred_frames, ignore_index=True).to_csv(
                os.path.join(cfg.out_dir, "predictions.csv.gz"),
                index=False, compression="gzip")

    # ---- Real ----
    logging.info("=== Condition 1/%d: Real (complete benchmark) ===" % n_conditions)
    real = load_cached(cfg, "Real")
    if real is None:
        real = run_condition("Real", data_dict, y, groups, cfg, imputer_factory=None)
        save_cached(cfg, "Real", real)
    notify(cfg, "STABL run progress", "Real condition completed",
           tags="white_check_mark")

    collect("Real", "Real", -1, real)

    # ---- Imputed conditions ----
    ci = 1
    for imp_name in cfg.imputers:
        factory = build_imputer_factory(imp_name, cfg)
        for rep in range(cfg.n_mask_repeats):
            ci += 1
            key = f"{imp_name}_rep{rep}"
            logging.info(f"=== Condition {ci}/{n_conditions}: {imp_name} | "
                         f"mask repeat {rep + 1}/{cfg.n_mask_repeats} ===")

            res = load_cached(cfg, key)
            if res is None:
                rng = np.random.RandomState(cfg.seed + 1000 * rep)
                masked = mask_data_dict(data_dict, y, cfg.mechanism,
                                        cfg.missing_rate, rng)

                if cfg.global_impute:
                    filled = {}
                    for omic, Xm in masked.items():
                        imp = factory()
                        filled[omic] = pd.DataFrame(imp.fit_transform(Xm.values),
                                                    index=Xm.index,
                                                    columns=Xm.columns)
                    res = run_condition(imp_name, filled, y, groups, cfg,
                                        imputer_factory=None)
                else:
                    res = run_condition(imp_name, masked, y, groups, cfg,
                                        imputer_factory=factory)
                save_cached(cfg, key, res)

            collect(imp_name, imp_name, rep, res)
            for m, r in res.items():
                rec = feature_recovery(r["fold_feats"], real[m]["fold_feats"])
                recov_rows.append(dict(imputer=imp_name, base_learner=m,
                                       mask_repeat=rep, **rec))

            # Checkpoint after every condition so a crash never costs more than one.
            flush()

            notify(cfg, "STABL run progress",
                   f"{imp_name} repeat {rep + 1}/{cfg.n_mask_repeats} completed "
                   f"({ci}/{n_conditions}); elapsed {_fmt_dur(time.time() - t0)}",
                   tags="white_check_mark")

    flush()
    df = pd.DataFrame(rows)
    df_rec = pd.DataFrame(recov_rows)

    diag = (df.groupby(["condition", "base_learner"])
              .agg(r2=("r2", "mean"), mae=("mae", "mean"),
                   n_features=("n_features_mean", "mean"),
                   jaccard_stability=("jaccard_stability", "mean"),
                   floor_effect=("floor_effect_flag", "mean"))
              .reset_index())
    diag.to_csv(os.path.join(cfg.out_dir, "diagnostics.csv"), index=False)
    logging.info("=== Diagnostics table (floor_effect marks no-signal conditions) ===\n"
                 + diag.to_string(index=False))

    make_figure(df, df_rec, cfg)

    dt = time.time() - t0
    notify(cfg, "STABL imputation run finished",
           f"Elapsed {_fmt_dur(dt)}; results in {cfg.out_dir}",
           priority="high", tags="tada")
    logging.info(f"Finished in {_fmt_dur(dt)}. Results saved to {cfg.out_dir}")
    return df, df_rec, diag


# ====================================================================
# 9. Plotting
# ====================================================================
def make_figure(df, df_rec, cfg: Config):
    conditions = ["Real"] + list(cfg.imputers)
    base_learners = [f"STABL {b}" for b in cfg.base_learners]
    colors = {"STABL Lasso": "#E8873A", "STABL ALasso": "#2E6FB7",
              "STABL ElasticNet": "#3F9B4F"}

    def agg(metric):
        out = {}
        for b in base_learners:
            means, stds = [], []
            for c in conditions:
                sub = df[(df.condition == c) & (df.base_learner == b)][metric]
                means.append(sub.mean())
                stds.append(sub.std())
            out[b] = (np.array(means, float), np.array(stds, float))
        return out

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    xpos = np.arange(len(conditions))
    panels = [
        (axes[0, 0], "r2", "Predictive Performance", "Predictive Accuracy ($R^2$)"),
        (axes[0, 1], "mae", "Error Analysis", "Error Magnitude (MAE)"),
        (axes[1, 0], "n_features_mean", "Stability of Features",
         "Feature Selection Stability\n(# Selected Features, all omics)"),
    ]
    for ax, metric, title, ylab in panels:
        data = agg(metric)
        for b in base_learners:
            means, stds = data[b]
            ax.errorbar(xpos, means, yerr=stds, marker="o", capsize=3,
                        label=b, color=colors.get(b), lw=2)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylab)
        ax.set_xticks(xpos)
        ax.set_xticklabels(conditions, rotation=30, ha="right")
        ax.axvline(0.5, color="grey", ls="--", lw=0.8)
        ax.grid(alpha=0.3)

    axD = axes[1, 1]
    imp_conditions = list(cfg.imputers)
    xposD = np.arange(len(imp_conditions))
    for b in base_learners:
        means, stds = [], []
        for c in imp_conditions:
            sub = df_rec[(df_rec.imputer == c) & (df_rec.base_learner == b)]["jaccard"]
            means.append(sub.mean())
            stds.append(sub.std())
        axD.errorbar(xposD, means, yerr=stds, marker="s", capsize=3,
                     label=b, color=colors.get(b), lw=2)
    axD.set_title("Feature Recovery vs Real", fontweight="bold")
    axD.set_ylabel("Jaccard(selected features, Real)")
    axD.set_xticks(xposD)
    axD.set_xticklabels(imp_conditions, rotation=30, ha="right")
    axD.set_ylim(0, 1)
    axD.grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(base_learners),
               frameon=False)
    fig.suptitle(f"Imputation effect on STABL | {cfg.dataset} 3-omic | "
                 f"{cfg.mechanism} rate={cfg.missing_rate} "
                 f"fold_safe={not cfg.global_impute}", y=1.02, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(cfg.out_dir, "imputation_effect_figure.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logging.info(f"[figure] saved -> {path}")


# ====================================================================
# 10. CLI
# ====================================================================
def parse_args():
    d = Config()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--mechanism", default=d.mechanism,
                   choices=["MCAR", "MAR", "MNAR"])
    p.add_argument("--missing-rate", type=float, default=d.missing_rate)
    p.add_argument("--n-splits", type=int, default=d.n_splits)
    p.add_argument("--n-bootstraps", type=int, default=d.n_bootstraps)
    p.add_argument("--mask-repeats", type=int, default=d.n_mask_repeats)
    p.add_argument("--imputers", nargs="+", default=list(d.imputers),
                   choices=["simple", "knn", "iterative", "rf", "missforest"])
    p.add_argument("--max-features", type=int, default=d.max_features_per_omic,
                   help="Cap features per omic by variance before masking. "
                        "Metabolomics has 3529 columns and drives the O(p^3) "
                        "knockoff cost; 1500 is a reasonable cap.")
    p.add_argument("--n-jobs", type=int, default=d.n_jobs)
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore cached conditions and recompute everything.")
    p.add_argument("--debug", action="store_true",
                   help="Log per-omic timing (preprocess / knockoff / stabl).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny end-to-end run. Use this first after any edit.")
    return p.parse_args()


def _log_name(cfg: Config):
    rate = str(cfg.missing_rate).replace(".", "p")
    return f"imputation_effect_on_stabl_CyPrMe_{cfg.mechanism}_rate_{rate}.log"


def _run_suffix(cfg: Config):
    rate = str(cfg.missing_rate).replace(".", "p")
    return f"{cfg.mechanism} rate {rate}"


def main():
    global _NOTIFIER

    a = parse_args()

    cfg = Config(
        mechanism=a.mechanism,
        missing_rate=a.missing_rate,
        n_splits=a.n_splits,
        n_bootstraps=a.n_bootstraps,
        n_mask_repeats=a.mask_repeats,
        imputers=tuple(a.imputers),
        max_features_per_omic=a.max_features,
        n_jobs=a.n_jobs,
        out_dir=a.out_dir,
        resume=not a.no_resume,
    )

    if a.smoke:
        cfg = replace(
            cfg,
            n_splits=2,
            n_bootstraps=20,
            n_mask_repeats=1,
            imputers=("simple", "knn"),
            max_features_per_omic=cfg.max_features_per_omic or 300,
            out_dir=cfg.out_dir + " - SMOKE",
            resume=False,
        )

    cfg.base_out_dir = cfg.out_dir          # 加后缀之前的原始路径
    cfg.out_dir = f"{cfg.out_dir} - {_run_suffix(cfg)}"

    from run_notifications import install_run_notifier

    _NOTIFIER = install_run_notifier(
        f"STABL imputation effect (CyPrMe, {cfg.mechanism}, rate={cfg.missing_rate})",
        log_name=_log_name(cfg),
    )

    setup_logging()
    if a.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info(f"[config] {cfg}")
    run_experiment(cfg)


if __name__ == "__main__":
    main()