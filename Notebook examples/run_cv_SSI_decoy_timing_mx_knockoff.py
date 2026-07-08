"""
Direction 1 test bed on Biobank SSI / CyTOF with real missing values.

This version runs only these three STABL models:

    STABL Lasso
    STABL ALasso
    STABL ElasticNet

The imputer is fixed to SimpleImputer median mode.
The artificial feature type is fixed to MX knockoffs.
The only command line variable is the timing of artificial feature injection.

Usage:

    python run_cv_SSI_decoy_timing_mx_knockoff.py post_impute
    python run_cv_SSI_decoy_timing_mx_knockoff.py pre_impute

Expected outputs are saved under:

    ./Results SSI Decoy - <timing>/Training CV/
"""

import sys


VALID_TIMINGS = ("post_impute", "pre_impute")
if len(sys.argv) != 2 or sys.argv[1] not in VALID_TIMINGS:
    print("Usage: python run_cv_SSI_decoy_timing_mx_knockoff.py [post_impute|pre_impute]")
    sys.exit(1)


decoy_timing = sys.argv[1]
imputer_mode = "simple"
artificial_type = "knockoff"


from run_notifications import install_run_notifier

install_run_notifier(
    f"SSI CyTOF decoy timing ({decoy_timing})",
    log_name=f"ssi_decoy_timing_{decoy_timing}.log",
)


import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from stabl import data
from stabl.adaptive import ALogitLasso
from stabl.multi_omic_pipelines import multi_omic_stabl_cv
from stabl.stabl import Stabl


random_seed = 1


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


X_train, X_valid, y_train, y_valid, ids, task_type = data.load_ssi(
    "../Sample Data/Biobank SSI"
)


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


# Base learner 1: Lasso logistic regression.
lasso = LogisticRegression(
    penalty="l1",
    class_weight="balanced",
    max_iter=int(1e6),
    solver="liblinear",
    random_state=random_seed,
)


# Base learner 2: Adaptive Lasso logistic regression.
alasso = ALogitLasso(
    penalty="l1",
    solver="liblinear",
    max_iter=int(1e6),
    class_weight="balanced",
    random_state=random_seed,
)


# Base learner 3: Elastic Net logistic regression.
en = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    class_weight="balanced",
    max_iter=int(1e5),
    random_state=random_seed,
)


# STABL Lasso.
stabl_lasso = Stabl(
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


# STABL ALasso.
stabl_alasso = clone(stabl_lasso).set_params(
    base_estimator=alasso,
    lambda_grid={"C": np.linspace(0.01, 10, 6)},
    verbose=1,
)


# STABL ElasticNet.
stabl_en = clone(stabl_lasso).set_params(
    base_estimator=en,
    lambda_grid=[
        {"C": np.logspace(-2, 0, 4), "l1_ratio": [0.5]},
        {"C": np.logspace(-2, 0, 4), "l1_ratio": [0.7]},
        {"C": np.logspace(-2, 0, 4), "l1_ratio": [0.9]},
    ],
    verbose=1,
)


estimators = {
    "stabl_lasso": stabl_lasso,
    "stabl_alasso": stabl_alasso,
    "stabl_en": stabl_en,
}


models = [
    "STABL Lasso",
    "STABL ALasso",
    "STABL ElasticNet",
]


print(
    "Run Direction 1 CV on SSI/CyTOF dataset, "
    f"imputer = {imputer_mode}, artificial type = {artificial_type}, "
    f"decoy timing = {decoy_timing}, "
    "models = STABL Lasso, STABL ALasso, STABL ElasticNet"
)

multi_omic_stabl_cv(
    X_train,
    y_train,
    outer_splitter=outer_cv,
    inner_splitter=chosen_inner_cv,
    estimators=estimators,
    task_type=task_type,
    save_path=f"./Results SSI Decoy - {decoy_timing}",
    outer_groups=ids,
    early_fusion=False,
    models=models,
    late_fusion=False,
    n_iter_lf=1,
    imputation_strategy=imputer_mode,
    artificial_injection_timing=decoy_timing,
    random_state=random_seed,
)
