#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imputation_effect_on_stabl.py

Artificially mask the complete OOL 3-omic dataset (CyTOF + Proteomics +
Metabolomics), impute it with different strategies, and compare STABL results
across conditions to assess whether the imputation strategy materially affects
STABL. Multi-omic setup mirrors run_cv_OOL_CyPrMe (STABL is fit per omic, then
the features selected across omics are concatenated for the final model).

For experimental design, metrics, and methodological trade-offs, see:
    Imputation_Effect_on_STABL_Proposal.docx

Fixed by decision (edit Config, not CLI): environment paths (following the
run_cv_SSI_imputer_compare_full convention), dataset = Onset of Labor, the three
omics, MX (Model-X) knockoff, and the full OOL compute config
(n_splits=100, n_bootstraps=300). The CLI exposes only the two experimental axes
actually varied run-to-run (mechanism, missing rate).

Usage
    python imputation_effect_on_stabl.py --mechanism MNAR --missing-rate 0.2

Runtime logging and ntfy start/finish/fail push are handled by run_notifications.
Outputs: metrics_long.csv, feature_recovery.csv, diagnostics.csv,
imputation_effect_figure.png.
"""


import os
import sys
import time
import argparse
import logging
import warnings
from dataclasses import dataclass, field
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

# Julia is initialized before importing stabl, matching
# run_cv_SSI_imputer_compare_full, where knockoff generation goes through the
# Julia/Bigsimr backend. Guarded so environments without Julia still import;
# knockoff will then fail at fit time if the backend truly needs it.
try:
    from julia.api import Julia
    _JL = Julia(compiled_modules=False)
except Exception as _jl_err:  # noqa: F841
    _JL = None

from stabl.stabl import Stabl
from stabl.adaptive import ALasso
from stabl.preprocessing import LowInfoFilter

warnings.filterwarnings("ignore")


# ====================================================================
# 0. Configuration
# ====================================================================
@dataclass
class Config:
    # Environment paths, matching run_cv_SSI_imputer_compare_full: the script
    # runs from a subdir (e.g. "Notebook examples"), so Sample Data is one level
    # up; stabl is installed/importable, so there is no repo-path concept;
    # outputs follow the "./Results ..." convention.
    data_path: str = "../Sample Data"
    out_dir: str = "./Results OOL CyPrMe imputer compare"

    # Dataset and omics. Fixed: Onset of Labor, three omics (like run_cv_OOL_CyPrMe).
    dataset: str = "Onset of Labor"
    omics: tuple = ("CyTOF", "Proteomics", "Metabolomics")
    task_type: str = "regression"

    # masking (CLI: mechanism, missing_rate)
    mechanism: str = "MCAR"          # MCAR / MAR / MNAR
    missing_rate: float = 0.20
    n_mask_repeats: int = 3

    # imputers to compare
    imputers: tuple = ("simple", "knn", "iterative", "rf", "missforest")

    # Fold-safe imputation (recommended). True would leak test info via a global imputer.
    global_impute: bool = False

    # STABL. Fixed to the full OOL config from run_cv_OOL_CyPrMe.
    # "knockoff" = second-order Model-X (Gaussian, equicorrelated) knockoff = MX knockoff.
    artificial_type: str = "knockoff"
    n_bootstraps: int = 300
    sample_fraction: float = 0.5
    fdr_low: float = 0.1
    fdr_high: float = 1.0

    # Outer CV. Fixed to OOL full: 100 splits, 20% test.
    n_splits: int = 100
    test_size: float = 0.2
    random_state: int = 42

    base_learners: tuple = ("Lasso", "ALasso", "ElasticNet")
    seed: int = 42


# ====================================================================
# 1. Logging + ntfy notifications
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


def notify(cfg: Config, title: str, message: str,
           priority: str = "default", tags: str = ""):
    """Log progress. ntfy start/finish/fail events are handled by run_notifications."""
    logging.info(f"[notify] {title} | {message}")


# ====================================================================
# 2. Data loading: complete 3-omic dict + y + groups
# ====================================================================
def load_complete_omics(cfg: Config):
    """
    Load the OOL training omics as a dict {omic: DataFrame}. Each omic is
    complete-cased (NaN columns dropped) so that Real is a true ground truth.
    All omics share the same sample index, aligned to y and groups.
    """
    from os.path import join
    base = join(cfg.data_path, cfg.dataset, "Training")

    y = pd.read_csv(join(base, "DOS.csv"), index_col=0).iloc[:, 0]
    ids = pd.read_csv(join(base, "ID.csv"), index_col=0).iloc[:, 0]

    raw = {omic: pd.read_csv(join(base, f"{omic}.csv"), index_col=0)
           for omic in cfg.omics}

    # Common sample index across all omics and y
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
        data_dict[omic] = X.astype(float)
        logging.info(f"[data] {omic}: {X.shape[0]} samples x {X.shape[1]} features "
                     f"(dropped {dropped} NaN columns)")

    logging.info(f"[data] {len(common)} common samples; {ids.nunique()} groups; "
                 f"total {sum(v.shape[1] for v in data_dict.values())} features")
    return data_dict, y.astype(float), ids


# ====================================================================
# 3. Masking: MCAR / MAR / MNAR (applied per omic)
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
    """Mask every omic independently with the same mechanism and rate."""
    out = {}
    total_target, total_realized, total_cells = 0, 0, 0
    for omic, X in data_dict.items():
        m = make_mask(X, y, mechanism, rate, rng)
        out[omic] = apply_mask(X, m)
        total_realized += m.sum()
        total_cells += m.size
    logging.info(f"[mask] mechanism={mechanism} target={rate:.2f} "
                 f"overall realized={total_realized/total_cells:.3f}")
    return out


# ====================================================================
# 4. Imputer registry
# ====================================================================
def build_imputer_factory(name, seed):
    name = name.lower()
    if name == "simple":
        return lambda: SimpleImputer(strategy="median")
    if name == "knn":
        return lambda: KNNImputer(n_neighbors=5, weights="distance")
    if name == "iterative":
        return lambda: IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                                        sample_posterior=False, random_state=seed)
    if name == "rf":
        return lambda: IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, n_jobs=-1,
                                            random_state=seed),
            max_iter=5, random_state=seed)
    if name == "missforest":
        try:
            import miceforest  # noqa
            return lambda: _MiceForestWrapper(seed=seed)
        except Exception:
            logging.warning("[imputer] miceforest was not detected; missforest falls back "
                            "to rf (IterativeImputer+RF). For real missForest, "
                            "pip install miceforest")
            return build_imputer_factory("rf", seed)
    raise ValueError(f"Unknown imputer: {name}")


class _MiceForestWrapper:
    def __init__(self, seed=42, iterations=3):
        self.seed = seed
        self.iterations = iterations
        self.kernel_ = None
        self.columns_ = None

    def fit(self, X, y=None):
        import miceforest as mf
        Xdf = pd.DataFrame(X).copy()
        self.columns_ = Xdf.columns
        self.kernel_ = mf.ImputationKernel(Xdf, num_datasets=1,
                                           random_state=self.seed)
        self.kernel_.mice(self.iterations)
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X, columns=self.columns_).copy()
        completed = self.kernel_.impute_new_data(Xdf).complete_data(0)
        return np.asarray(completed, dtype=float)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ====================================================================
# 5. STABL components
# ====================================================================
def build_stabl_estimators(cfg):
    fdr_range = np.arange(cfg.fdr_low, cfg.fdr_high, 0.01)
    lasso = Lasso(max_iter=int(1e6), random_state=cfg.random_state)
    alasso = ALasso(max_iter=int(1e6), random_state=cfg.random_state)
    en = ElasticNet(max_iter=int(1e6), random_state=cfg.random_state)

    base = Stabl(
        base_estimator=lasso, n_bootstraps=cfg.n_bootstraps,
        artificial_type=cfg.artificial_type, artificial_proportion=1.0,
        replace=False, fdr_threshold_range=fdr_range,
        sample_fraction=cfg.sample_fraction, random_state=cfg.random_state,
        lambda_grid={"alpha": np.logspace(0, 2, 10)}, verbose=0,
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


def build_preprocessor(cfg, imputer_factory):
    steps = [
        ("variance", VarianceThreshold(0.01)),
        ("lif", LowInfoFilter(max_nan_fraction=1.0)),
        ("impute", imputer_factory() if imputer_factory is not None
         else SimpleImputer(strategy="median")),
        ("std", StandardScaler()),
    ]
    return Pipeline(steps)


# ====================================================================
# 6. CV for one condition (multi-omic)
# ====================================================================
def run_condition(name, data_dict, y, groups, cfg, imputer_factory=None):
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

    preds = {m: pd.DataFrame(index=y.index) for m in estimators}
    nfeat = {m: [] for m in estimators}
    fold_feats = {m: [] for m in estimators}

    for k, (tr, te) in enumerate(cv.split(y, y, groups=groups)):
        tr_idx, te_idx = y.index[tr], y.index[te]
        g_tr = groups.loc[tr_idx].values
        ytr = y.loc[tr_idx]

        # Per-omic standardized train/test frames (original column names kept).
        std_frames = {}
        sel = {m: [] for m in estimators}   # list of (omic, feature)
        for omic, X in data_dict.items():
            pre = build_preprocessor(cfg, imputer_factory)
            Xtr = pd.DataFrame(pre.fit_transform(X.loc[tr_idx]),
                               index=tr_idx, columns=pre.get_feature_names_out())
            Xte = pd.DataFrame(pre.transform(X.loc[te_idx]),
                               index=te_idx, columns=pre.get_feature_names_out())
            std_frames[omic] = (Xtr, Xte)
            for m, stabl in estimators.items():
                s = clone(stabl)
                s.fit(Xtr, ytr, groups=g_tr)
                for f in s.get_feature_names_out():
                    sel[m].append((omic, f))

        # Final model on features concatenated across omics.
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

        if (k + 1) % max(1, cfg.n_splits // 10) == 0:
            logging.info(f"    [{name}] fold {k+1}/{cfg.n_splits} done")

    out = {}
    for m in estimators:
        yhat = preds[m].median(axis=1)
        valid = yhat.notna()
        r2 = r2_score(y[valid], yhat[valid]) if valid.sum() > 1 else np.nan
        mae = mean_absolute_error(y[valid], yhat[valid]) if valid.sum() > 1 else np.nan
        out[m] = {
            "r2": r2, "mae": mae,
            "n_features_mean": float(np.mean(nfeat[m])),
            "n_features_std": float(np.std(nfeat[m])),
            "jaccard_stability": jaccard_stability(fold_feats[m]),
            "floor_effect_flag": float(np.mean(nfeat[m])) < 1.0,
            "fold_feats": fold_feats[m],
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
# 7. Orchestration
# ====================================================================
def run_experiment(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    t0 = time.time()
    notify(cfg, "STABL imputation run started",
           f"dataset={cfg.dataset} omics={list(cfg.omics)} "
           f"mechanism={cfg.mechanism} rate={cfg.missing_rate} "
           f"imputers={list(cfg.imputers)} fold_safe={not cfg.global_impute}",
           tags="rocket")

    data_dict, y, groups = load_complete_omics(cfg)

    logging.info("=== Condition: Real (complete benchmark) ===")
    real = run_condition("Real", data_dict, y, groups, cfg, imputer_factory=None)
    notify(cfg, "STABL run progress", "Real condition completed", tags="white_check_mark")

    rows, recov_rows = [], []
    for m, r in real.items():
        rows.append(dict(condition="Real", imputer="Real", base_learner=m,
                         mask_repeat=-1,
                         **{k: v for k, v in r.items() if k != "fold_feats"}))

    for imp_name in cfg.imputers:
        factory = build_imputer_factory(imp_name, cfg.seed)
        for rep in range(cfg.n_mask_repeats):
            logging.info(f"=== Condition: {imp_name} | mask repeat "
                         f"{rep+1}/{cfg.n_mask_repeats} ===")
            rng = np.random.RandomState(cfg.seed + 1000 * rep)
            masked = mask_data_dict(data_dict, y, cfg.mechanism,
                                    cfg.missing_rate, rng)

            if cfg.global_impute:
                filled = {}
                for omic, Xm in masked.items():
                    imp = factory()
                    filled[omic] = pd.DataFrame(imp.fit_transform(Xm.values),
                                                index=Xm.index, columns=Xm.columns)
                res = run_condition(imp_name, filled, y, groups, cfg,
                                    imputer_factory=None)
            else:
                res = run_condition(imp_name, masked, y, groups, cfg,
                                    imputer_factory=factory)

            for m, r in res.items():
                rows.append(dict(condition=imp_name, imputer=imp_name,
                                 base_learner=m, mask_repeat=rep,
                                 **{k: v for k, v in r.items()
                                    if k != "fold_feats"}))
                rec = feature_recovery(r["fold_feats"], real[m]["fold_feats"])
                recov_rows.append(dict(imputer=imp_name, base_learner=m,
                                       mask_repeat=rep, **rec))
            notify(cfg, "STABL run progress",
                   f"{imp_name} repeat {rep+1}/{cfg.n_mask_repeats} completed",
                   tags="white_check_mark")

    df = pd.DataFrame(rows)
    df_rec = pd.DataFrame(recov_rows)
    df.to_csv(os.path.join(cfg.out_dir, "metrics_long.csv"), index=False)
    df_rec.to_csv(os.path.join(cfg.out_dir, "feature_recovery.csv"), index=False)

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
           f"Elapsed {dt/60:.1f} minutes; results in {cfg.out_dir}",
           priority="high", tags="tada")
    logging.info(f"Finished in {dt/60:.1f} minutes. Results saved to {cfg.out_dir}")
    return df, df_rec, diag


# ====================================================================
# 8. Plotting
# ====================================================================
def make_figure(df, df_rec, cfg):
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
# 9. CLI
# ====================================================================
def parse_args():
    # CLI = only the two experimental axes varied run-to-run. Paths, dataset,
    # omics, MX knockoff, n_splits=100, n_bootstraps=300, imputer list, mask
    # repeats and fold-safe are all fixed in Config.
    d = Config()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mechanism", default=d.mechanism, choices=["MCAR", "MAR", "MNAR"])
    p.add_argument("--missing-rate", type=float, default=d.missing_rate)
    return p.parse_args()


def _log_name(cfg: Config):
    rate = str(cfg.missing_rate).replace(".", "p")
    return f"imputation_effect_on_stabl_CyPrMe_{cfg.mechanism}_rate_{rate}.log"


def _run_suffix(cfg: Config):
    rate = str(cfg.missing_rate).replace(".", "p")
    return f"{cfg.mechanism} rate {rate}"


def main():
    a = parse_args()
    cfg = Config(mechanism=a.mechanism, missing_rate=a.missing_rate)

    cfg.out_dir = f"{cfg.out_dir} - {_run_suffix(cfg)}"

    from run_notifications import install_run_notifier

    install_run_notifier(
        f"STABL imputation effect (CyPrMe, {cfg.mechanism}, rate={cfg.missing_rate})",
        log_name=_log_name(cfg),
    )

    setup_logging()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
