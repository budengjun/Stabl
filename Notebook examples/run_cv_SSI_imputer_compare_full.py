"""
Full-scale comparison of SimpleImputer vs IterativeImputer for STABL on
Biobank SSI / CyTOF.csv.

This is run_cv_SSI.py at full CV rigor and full estimator suite, with two
additions layered on top:
    1. imputer selection: a CLI mode switch (simple | iterative), CyTOF-only
       block, missingness report, and imputation strategy passthrough;
    2. run notifier: install_run_notifier for logging / completion notice.

Usage:
    python run_cv_SSI_imputer_compare_full.py simple
    python run_cv_SSI_imputer_compare_full.py iterative

Compare the outputs after both runs:
    ./Results SSI - simple/Summary/Scores training CV.csv
    ./Results SSI - iterative/Summary/Scores training CV.csv

Primary metrics to inspect:
    - ROC AUC
    - N features
    - CVS (Jaccard stability, median [Q1, Q3])
    - Artificial/real SF mean and median (decoy calibration)

Note on runtime: this is the heaviest configuration (5x20 outer CV, 5x5 inner
CV, up to 500 bootstraps per STABL Lasso fit, four base learners plus their
STABL variants, two imputer modes). Expect a long wall-clock time.
"""

import sys


if len(sys.argv) != 2 or sys.argv[1] not in ("simple", "iterative"):
    print("Usage: python run_cv_SSI_imputer_compare_full.py [simple|iterative]")
    sys.exit(1)


imputer_mode = sys.argv[1]

from run_notifications import install_run_notifier

install_run_notifier(
    f"SSI CyTOF imputer compare full ({imputer_mode})",
    log_name=f"ssi_imputer_compare_full_{imputer_mode}.log",
)

from julia.api import Julia

import numpy as np
from stabl import data
from stabl.multi_omic_pipelines import multi_omic_stabl_cv
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from stabl.stabl import Stabl
from stabl.adaptive import ALogitLasso
from groupyr import LogisticSGL
from sklearn.base import clone

jl = Julia(compiled_modules=False)


random_seed = 1
# Full CV settings, identical to run_cv_SSI.py.
outter_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=random_seed)
chosen_inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

artificial_type = "knockoff"
X_train, X_valid, y_train, y_valid, ids, task_type = data.load_ssi(
    "../Sample Data/Biobank SSI"
)

# Keep only CyTOF because this experiment targets the omic block with real
# missing values. Proteomics has no within-table missing values here, so it
# would not exercise the imputer at all.
X_cytof = X_train["CyTOF"]
missing_by_row = X_cytof.isna().sum(axis=1)
missing_by_col = X_cytof.isna().sum(axis=0)
print("SSI/CyTOF missingness")
print(f"  samples: {X_cytof.shape[0]}")
print(f"  features: {X_cytof.shape[1]}")
print(f"  missing cells: {int(missing_by_row.sum())}")
print(f"  samples with missing values: {int((missing_by_row > 0).sum())}")
print(f"  features with missing values: {int((missing_by_col > 0).sum())}")

X_train = {"CyTOF": X_cytof}

# Lasso
lasso = LogisticRegression(
    penalty="l1",
    class_weight="balanced",
    max_iter=int(1e6),
    solver="liblinear",
    random_state=random_seed,
)
lasso_cv = GridSearchCV(
    lasso,
    param_grid={"C": np.logspace(-2, 2, 30)},
    scoring="roc_auc",
    cv=chosen_inner_cv,
    n_jobs=-1,
)

# ElasticNet
en = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    class_weight="balanced",
    max_iter=int(1e5),
    random_state=random_seed,
)
en_params = {"C": np.logspace(-2, 1, 10), "l1_ratio": [0.5, 0.7, 0.9]}
en_cv = GridSearchCV(
    en, param_grid=en_params, scoring="roc_auc", cv=chosen_inner_cv, n_jobs=-1
)

# ALasso
alasso = ALogitLasso(
    penalty="l1",
    solver="liblinear",
    max_iter=int(1e6),
    class_weight="balanced",
    random_state=random_seed,
)
alasso_cv = GridSearchCV(
    alasso,
    scoring="roc_auc",
    param_grid={"C": np.logspace(-2, 2, 30)},
    cv=chosen_inner_cv,
    n_jobs=-1,
)

# SGL
sgl = LogisticSGL(max_iter=int(1e4), l1_ratio=0.5)
sgl_cv = GridSearchCV(
    sgl,
    scoring="roc_auc",
    param_grid={"alpha": np.logspace(-2, 0, 10), "l1_ratio": [0.5, 0.7, 0.9]},
    cv=chosen_inner_cv,
    n_jobs=-1,
)

# Stabl
stabl = Stabl(
    lasso,
    n_bootstraps=500,
    artificial_type=artificial_type,
    artificial_proportion=1.0,
    replace=False,
    fdr_threshold_range=np.arange(0.1, 1, 0.01),
    sample_fraction=0.5,
    random_state=random_seed,
    lambda_grid={"C": np.linspace(0.01, 1, 10)},
    verbose=1,
)

stabl_alasso = clone(stabl).set_params(
    base_estimator=alasso, lambda_grid={"C": np.linspace(0.01, 10, 10)}, verbose=1
)

stabl_en = clone(stabl).set_params(
    base_estimator=en,
    n_bootstraps=100,
    lambda_grid=[
        {"C": np.logspace(-2, 0, 5), "l1_ratio": [0.5]},
        {"C": np.logspace(-2, 0, 5), "l1_ratio": [0.7]},
        {"C": np.logspace(-2, 0, 5), "l1_ratio": [0.9]},
    ],
    verbose=1,
)

stabl_sgl = clone(stabl).set_params(
    base_estimator=sgl,
    n_bootstraps=50,
    perc_corr_group_threshold=99,
    lambda_grid=[
        {"alpha": np.logspace(-2, 0, 5), "l1_ratio": [0.5]},
        {"alpha": np.logspace(-2, 0, 5), "l1_ratio": [0.7]},
        {"alpha": np.logspace(-2, 0, 5), "l1_ratio": [0.9]},
    ],
    verbose=1,
)


estimators = {
    "lasso": lasso_cv,
    "alasso": alasso_cv,
    "en": en_cv,
    "sgl": sgl_cv,
    "stabl_lasso": stabl,
    "stabl_alasso": stabl_alasso,
    "stabl_en": stabl_en,
    "stabl_sgl": stabl_sgl,
}

models = [
    "STABL Lasso",
    "Lasso",
    "STABL ALasso",
    "ALasso",
    "STABL ElasticNet",
    "ElasticNet",
    # "STABL SGL-90",
    # "STABL SGL-95",
    # "SGL-90",
    # "SGL-95",
]

print(f"Run FULL CV on SSI/CyTOF dataset, imputer = {imputer_mode}")
multi_omic_stabl_cv(
    X_train,
    y_train,
    outer_splitter=outter_cv,
    inner_splitter=chosen_inner_cv,
    estimators=estimators,
    task_type=task_type,
    save_path=f"./Results SSI - {imputer_mode}",
    outer_groups=ids,
    # Single omic block: early fusion would just re-run the same block, and
    # late fusion collapses to a trivial one-omic weighting, so n_iter_lf stays
    # at 1 (any larger value buys nothing with one block).
    early_fusion=False,
    models=models,
    late_fusion=True,
    n_iter_lf=1,
    sgl_corr_percentile=[90, 95],
    imputation_strategy=imputer_mode,
    random_state=random_seed,
)