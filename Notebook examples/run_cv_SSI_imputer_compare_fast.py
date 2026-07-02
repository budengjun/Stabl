"""
Fast comparison of SimpleImputer vs IterativeImputer for STABL on
Biobank SSI / CyTOF.csv.

Usage:
    python run_cv_SSI_imputer_compare_fast.py simple
    python run_cv_SSI_imputer_compare_fast.py iterative

Compare the outputs after both runs:
    ./Results SSI Fast - simple/Summary/Scores training CV.csv
    ./Results SSI Fast - iterative/Summary/Scores training CV.csv

Primary metrics to inspect:
    - ROC AUC
    - N features
    - CVS (Jaccard stability, median [Q1, Q3])
"""

import sys


if len(sys.argv) != 2 or sys.argv[1] not in ("simple", "iterative"):
    print("Usage: python run_cv_SSI_imputer_compare.py [simple|iterative]")
    sys.exit(1)


imputer_mode = sys.argv[1]

from run_notifications import install_run_notifier

install_run_notifier(
    f"SSI CyTOF imputer compare ({imputer_mode})",
    log_name=f"ssi_imputer_compare_{imputer_mode}.log",
)

from julia.api import Julia
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.base import clone

from stabl import data
from stabl.multi_omic_pipelines import multi_omic_stabl_cv
from stabl.stabl import Stabl

jl = Julia(compiled_modules=False)

random_seed = 1

# Fast/development settings.
outer_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=random_seed,
)
chosen_inner_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=42,
)

artificial_type = "knockoff"
X_train, X_valid, y_train, y_valid, ids, task_type = data.load_ssi(
    "../Sample Data/Biobank SSI"
)

# Keep only CyTOF because this experiment targets the omic block with real
# missing values. Proteomics has no within-table missing values here.
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

# Lasso baseline. It receives the same imputation strategy through the shared
# preprocessing pipeline inside multi_omic_stabl_cv.
lasso = LogisticRegression(
    penalty="l1",
    class_weight="balanced",
    max_iter=int(1e6),
    solver="liblinear",
    random_state=random_seed,
)
lasso_cv = GridSearchCV(
    lasso,
    param_grid={"C": np.logspace(-2, 2, 15)},
    scoring="roc_auc",
    cv=chosen_inner_cv,
    n_jobs=-1,
)

# ElasticNet baseline.
en = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    class_weight="balanced",
    max_iter=int(1e5),
    random_state=random_seed,
)
en_cv = GridSearchCV(
    en,
    param_grid={"C": np.logspace(-2, 1, 6), "l1_ratio": [0.5, 0.7, 0.9]},
    scoring="roc_auc",
    cv=chosen_inner_cv,
    n_jobs=-1,
)

stabl = Stabl(
    lasso,
    n_bootstraps=50,
    artificial_type=artificial_type,
    artificial_proportion=1.0,
    replace=False,
    fdr_threshold_range=np.arange(0.1, 1, 0.02),
    sample_fraction=0.5,
    random_state=random_seed,
    lambda_grid={"C": np.linspace(0.01, 1, 6)},
    verbose=1,
)

stabl_en = clone(stabl).set_params(
    base_estimator=en,
    n_bootstraps=25,
    lambda_grid=[
        {"C": np.logspace(-2, 0, 3), "l1_ratio": [0.5]},
        {"C": np.logspace(-2, 0, 3), "l1_ratio": [0.7]},
        {"C": np.logspace(-2, 0, 3), "l1_ratio": [0.9]},
    ],
    verbose=1,
)

estimators = {
    "lasso": lasso_cv,
    "en": en_cv,
    "stabl_lasso": stabl,
    "stabl_en": stabl_en,
}

models = [
    "STABL Lasso",
    "Lasso",
    "STABL ElasticNet",
    "ElasticNet",
]

print(f"Run FAST CV on SSI/CyTOF dataset, imputer = {imputer_mode}")
multi_omic_stabl_cv(
    X_train,
    y_train,
    outer_splitter=outer_cv,
    inner_splitter=chosen_inner_cv,
    estimators=estimators,
    task_type=task_type,
    save_path=f"./Results SSI Fast - {imputer_mode}",
    outer_groups=ids,
    early_fusion=False,
    models=models,
    late_fusion=True,
    n_iter_lf=1,
    imputation_strategy=imputer_mode,
    random_state=random_seed,
)
