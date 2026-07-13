"""
Direction 1 test bed on Biobank SSI / CyTOF with real missing values.

Runs STABL Lasso, STABL ALasso and STABL ElasticNet with MX knockoff decoys.
The main variable is artificial feature injection timing. For pre_impute, the
repaired knockoff branch can use stochastic posterior completion and optional
multiple-imputation pooling.

Examples:
    python run_cv_SSI_decoy_timing_mx_knockoff.py post_impute
    python run_cv_SSI_decoy_timing_mx_knockoff.py pre_impute --n-knockoff-imputations 1 --output-suffix fixed_M1
    python run_cv_SSI_decoy_timing_mx_knockoff.py pre_impute --n-knockoff-imputations 5 --output-suffix fixed_M5
    python run_cv_SSI_decoy_timing_mx_knockoff.py pre_impute --knockoff-impute-strategy median --knockoff-remask --output-suffix remask_median
"""

import argparse

VALID_TIMINGS = ("post_impute", "pre_impute")
VALID_KNOCKOFF_IMPUTERS = (
    "iterative_posterior",
    "simple",
    "median",
    "iterative",
    "knn",
)

parser = argparse.ArgumentParser()
parser.add_argument("timing", choices=VALID_TIMINGS)
parser.add_argument(
    "--knockoff-impute-strategy",
    choices=VALID_KNOCKOFF_IMPUTERS,
    default="iterative_posterior",
    help="Imputer used only by the repaired pre_impute knockoff branch.",
)
parser.add_argument(
    "--n-knockoff-imputations",
    type=int,
    default=1,
    help="Number of stochastic completions to pool in the repaired pre_impute knockoff branch.",
)
parser.add_argument(
    "--knockoff-remask",
    action="store_true",
    help=(
        "pre_impute knockoff branch only. Copy the paired real-feature missingness "
        "pattern onto the generated knockoffs, impute the knockoff block a second "
        "time, then re-standardize real and knockoff blocks jointly."
    ),
)
parser.add_argument(
    "--output-suffix",
    default="",
    help="Optional suffix appended to the output folder and log name.",
)
args = parser.parse_args()

if args.n_knockoff_imputations < 1:
    raise ValueError("--n-knockoff-imputations must be >= 1")


decoy_timing = args.timing
imputer_mode = "simple"
artificial_type = "knockoff"
suffix = f" - {args.output_suffix}" if args.output_suffix else ""
save_dir = f"./Results SSI Decoy - {decoy_timing}{suffix}"
log_suffix = f"_{args.output_suffix}" if args.output_suffix else ""


from run_notifications import install_run_notifier

install_run_notifier(
    f"SSI CyTOF decoy timing ({decoy_timing}{suffix})",
    log_name=f"ssi_decoy_timing_{decoy_timing}{log_suffix}.log",
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

lasso = LogisticRegression(
    penalty="l1",
    class_weight="balanced",
    max_iter=int(1e6),
    solver="liblinear",
    random_state=random_seed,
)

alasso = ALogitLasso(
    penalty="l1",
    solver="liblinear",
    max_iter=int(1e6),
    class_weight="balanced",
    random_state=random_seed,
)

en = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    class_weight="balanced",
    max_iter=int(1e5),
    random_state=random_seed,
)

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

stabl_alasso = clone(stabl_lasso).set_params(
    base_estimator=alasso,
    lambda_grid={"C": np.linspace(0.01, 10, 6)},
    verbose=1,
)

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
    f"knockoff imputer = {args.knockoff_impute_strategy}, "
    f"M = {args.n_knockoff_imputations}, "
    f"knockoff remask = {args.knockoff_remask}, "
    "models = STABL Lasso, STABL ALasso, STABL ElasticNet"
)

multi_omic_stabl_cv(
    X_train,
    y_train,
    outer_splitter=outer_cv,
    inner_splitter=chosen_inner_cv,
    estimators=estimators,
    task_type=task_type,
    save_path=save_dir,
    outer_groups=ids,
    early_fusion=False,
    models=models,
    late_fusion=False,
    n_iter_lf=1,
    imputation_strategy=imputer_mode,
    artificial_injection_timing=decoy_timing,
    knockoff_impute_strategy=args.knockoff_impute_strategy,
    n_knockoff_imputations=args.n_knockoff_imputations,
    knockoff_remask=args.knockoff_remask,
    random_state=random_seed,
)
