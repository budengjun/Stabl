import sys
import atexit
import urllib.request
import traceback
import signal
from pathlib import Path
from datetime import datetime
import socket

_NTFY_TOPIC = "yaolong-dream-cv"
_SCRIPT_DIR = Path(__file__).resolve().parent
_LOG_PATH = _SCRIPT_DIR / "dream_fast.log"
_START_TIME = datetime.now()
_HOSTNAME = socket.gethostname()

_FAILED = False
_EXIT_REASON = "completed"

_LOG_FILE = open(_LOG_PATH, "w", encoding="utf-8", buffering=1)


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


sys.stdout = _Tee(sys.__stdout__, _LOG_FILE)
sys.stderr = _Tee(sys.__stderr__, _LOG_FILE)


def _send_ntfy(title, message, priority="high", tags="bell"):
    url = f"https://ntfy.sh/{_NTFY_TOPIC}"

    req = urllib.request.Request(
        url,
        data=message.encode("utf-8"),
        method="POST",
        headers={
            "Title": title,
            "Priority": priority,
            "Tags": tags,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            response.read()
    except Exception as error:
        print(f"[ntfy failed] {error}", file=sys.__stderr__)


print(f"===== DREAM fast run started at {_START_TIME} on {_HOSTNAME} =====")
print(f"===== Log file: {_LOG_PATH} =====")


_send_ntfy(
    title="DREAM fast started",
    message=f"python run_cv_DREAM_fast.py started on {_HOSTNAME}. Output is being written to dream_fast.log.",
    priority="default",
    tags="rocket",
)


def _handle_exception(exc_type, exc_value, exc_traceback):
    global _FAILED, _EXIT_REASON
    _FAILED = True
    _EXIT_REASON = f"exception: {exc_type.__name__}"
    traceback.print_exception(exc_type, exc_value, exc_traceback)


sys.excepthook = _handle_exception


def _handle_signal(signum, frame):
    global _FAILED, _EXIT_REASON
    _FAILED = True
    _EXIT_REASON = f"terminated by signal {signum}"
    print(f"\n===== DREAM fast run interrupted by signal {signum} =====", flush=True)
    raise SystemExit(128 + signum)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _finish_notify():
    end_time = datetime.now()
    elapsed = end_time - _START_TIME

    print()
    print(f"===== DREAM fast run ended at {end_time} =====")
    print(f"===== Elapsed time: {elapsed} =====")
    print(f"===== Status: {'FAILED' if _FAILED else 'SUCCESS'} =====")
    print(f"===== Reason: {_EXIT_REASON} =====")
    print(f"===== Log file: {_LOG_PATH} =====", flush=True)

    if _FAILED:
        _send_ntfy(
            title="DREAM fast failed",
            message=f"run_cv_DREAM_fast.py did not finish successfully. Reason: {_EXIT_REASON}. Check dream_fast.log. Elapsed: {elapsed}",
            priority="urgent",
            tags="warning",
        )
    else:
        _send_ntfy(
            title="DREAM fast finished",
            message=f"run_cv_DREAM_fast.py has finished. Output file dream_fast.log is ready. Elapsed: {elapsed}",
            priority="high",
            tags="white_check_mark",
        )

    try:
        _LOG_FILE.close()
    except Exception:
        pass


atexit.register(_finish_notify)

"""
Faster development version of run_cv_DREAM.py.

Main changes vs original:
- Outer CV: 100 splits -> 50 splits
- Inner CV: 5 folds x 5 repeats -> 3 folds x 3 repeat
- GridSearch parameter grids are smaller
- STABL bootstraps reduced
- Late fusion iterations: 1000 -> 50
- n_jobs capped to avoid nested parallelism overload
- Results saved to ./Results Dream Fast so original outputs are not overwritten

Use this for smoke tests / development. For final experiments, gradually increase
N_OUTER_SPLITS, INNER_REPEATS, STABL_BOOTSTRAPS, and N_ITER_LF.
"""

from julia.api import Julia


import random

import numpy as np
from stabl import data
from stabl.multi_omic_pipelines import multi_omic_stabl_cv
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    GroupShuffleSplit,
    GridSearchCV,
)
from sklearn.linear_model import LogisticRegression
from stabl.stabl import Stabl
from stabl.adaptive import ALogitLasso
from groupyr import LogisticSGL
from sklearn.base import clone

# ---------------------------------------------------------------------
# Fast/development controls
# ---------------------------------------------------------------------

jl = Julia(compiled_modules=False)
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Original was: 5 splits x 5 repeats = 25 inner folds.
# This version uses 9 inner folds total.
INNER_SPLITS = 3
INNER_REPEATS = 3

# Original was 100 outer splits.
# This version uses 50 outer splits.
OUTER_SPLITS = 50

# Original late-fusion iterations were 1000.
N_ITER_LF = 50

# Original STABL bootstraps were 100 for lasso/alasso and 50 for en/sgl.
STABL_BOOTSTRAPS_MAIN = 50
STABL_BOOTSTRAPS_SECONDARY = 25

# Avoid nested parallelism overload. If your server has more cores, you can try 4 or -1.
N_JOBS = -1

chosen_inner_cv = RepeatedStratifiedKFold(
    n_splits=INNER_SPLITS,
    n_repeats=INNER_REPEATS,
    random_state=random_seed,
)

outter_group_cv = GroupShuffleSplit(
    n_splits=OUTER_SPLITS,
    test_size=0.2,
    random_state=random_seed,
)

artificial_type = "knockoff"

X_train, X_valid, y_train, y_valid, ids, task_type = data.load_dream(
    "../Sample Data/Dream"
)
y_train = y_train.astype(int)

for name, df in X_train.items():
    df.columns = df.columns.str.replace("/", "_", regex=True)
    X_train[name] = df

# ---------------------------------------------------------------------
# Base models + smaller parameter grids
# ---------------------------------------------------------------------

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
    param_grid={"C": np.logspace(-3, 0, 8)},  # original: 30 values
    scoring="roc_auc",
    cv=chosen_inner_cv,
    n_jobs=N_JOBS,
)

# ElasticNet
en = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    class_weight="balanced",
    max_iter=int(1e5),
    random_state=random_seed,
)
en_params = {
    "C": np.logspace(-2, 1, 6),  # original: 10 values
    "l1_ratio": [0.2, 0.5, 0.8],  # original equivalent: 3 values
}
en_cv = GridSearchCV(
    en,
    param_grid=en_params,
    scoring="roc_auc",
    cv=chosen_inner_cv,
    n_jobs=N_JOBS,
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
    param_grid={"C": np.logspace(-3, 0, 8)},  # original: 30 values
    cv=chosen_inner_cv,
    n_jobs=N_JOBS,
)

# SGL
# Kept available but not included in `models` by default below.
sgl = LogisticSGL(max_iter=int(1e3), l1_ratio=0.5)
sgl_cv = GridSearchCV(
    sgl,
    scoring="roc_auc",
    param_grid={
        "alpha": np.logspace(-3, 0, 6),
        "l1_ratio": [0.2, 0.5, 0.8],
    },
    cv=chosen_inner_cv,
    n_jobs=N_JOBS,
)

# ---------------------------------------------------------------------
# STABL models with reduced bootstrap/grid sizes
# ---------------------------------------------------------------------

stabl = Stabl(
    lasso,
    n_bootstraps=STABL_BOOTSTRAPS_MAIN,
    artificial_type=artificial_type,
    artificial_proportion=0.5,
    replace=False,
    fdr_threshold_range=np.arange(0.1, 1, 0.05),  # original step: 0.01
    sample_fraction=0.5,
    random_state=random_seed,
    lambda_grid={"C": np.linspace(0.004, 0.4, 5)},  # original: 10 values
    verbose=0,
)
stabl_rp = clone(stabl).set_params(artificial_type="random_permutation")

stabl_alasso = clone(stabl).set_params(
    base_estimator=alasso,
    lambda_grid={"C": np.linspace(0.004, 4, 5)},
    verbose=0,
)
stabl_alasso_rp = clone(stabl_alasso).set_params(artificial_type="random_permutation")

stabl_en = clone(stabl).set_params(
    base_estimator=en,
    n_bootstraps=STABL_BOOTSTRAPS_SECONDARY,
    lambda_grid=[
        {"C": np.logspace(-3, -2, 3), "l1_ratio": [0.2]},
        {"C": np.logspace(-3, -2, 3), "l1_ratio": [0.5]},
        {"C": np.logspace(-3, -2, 3), "l1_ratio": [0.8]},
    ],
    verbose=0,
)
stabl_en_rp = clone(stabl_en).set_params(artificial_type="random_permutation")

stabl_sgl = clone(stabl).set_params(
    base_estimator=sgl,
    n_bootstraps=STABL_BOOTSTRAPS_SECONDARY,
    lambda_grid=[
        {"alpha": np.logspace(-3, -2, 3), "l1_ratio": [0.2]},
        {"alpha": np.logspace(-3, -2, 3), "l1_ratio": [0.5]},
        {"alpha": np.logspace(-3, -2, 3), "l1_ratio": [0.8]},
    ],
    verbose=0,
)

estimators = {
    "lasso": lasso_cv,
    "alasso": alasso_cv,
    "en": en_cv,
    "sgl": sgl_cv,
    "stabl_lasso": stabl_rp,
    "stabl_alasso": stabl_alasso,
    "stabl_en": stabl_en,
    "stabl_sgl": stabl_sgl,
}

# Development default: fewer models.
models = [
    "Lasso",
    "ElasticNet",
    "STABL Lasso",
    "STABL ElasticNet",
    # "STABL ALasso", "ALasso",
    # "STABL SGL-90", "SGL-90",
    # "STABL SGL-95", "SGL-95",
]

print("Run FAST CV on dream dataset")
print(f"Outer splits: {OUTER_SPLITS}")
print(f"Inner CV: {INNER_SPLITS} folds x {INNER_REPEATS} repeat(s)")
print(
    f"STABL bootstraps: main={STABL_BOOTSTRAPS_MAIN}, secondary={STABL_BOOTSTRAPS_SECONDARY}"
)
print(f"Late fusion iterations: {N_ITER_LF}")
print(f"Random seed: {random_seed}")
print("Imputation strategy: iterative")
print(f"Models: {models}")

multi_omic_stabl_cv(
    X_train,
    y_train,
    outer_splitter=outter_group_cv,
    inner_splitter=chosen_inner_cv,
    estimators=estimators,
    task_type=task_type,
    save_path="./Results Dream Fast",
    outer_groups=ids,
    early_fusion=False,
    models=models,
    late_fusion=True,
    n_iter_lf=N_ITER_LF,
    imputation_strategy="iterative",
    random_state=random_seed,
)
