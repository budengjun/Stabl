#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paired_tests.py

Paired comparison of imputation conditions, using predictions.csv.gz.

Why pairing is possible: every condition uses the same GroupShuffleSplit with
the same random_state, so condition A and condition B produce a prediction for
the same sample from the same training folds. The conditions differ only in the
imputation applied. That makes the per-sample error difference a genuinely
paired quantity, and a paired analysis is far more powerful than comparing the
three mask-repeat means (n=3).

Why NOT a paired test over folds: GroupShuffleSplit draws n_splits independent
test sets, so a sample appears in several of them. The folds are not
independent, and an n=25 paired test on fold-level R2 would badly overstate the
degrees of freedom. The unit of analysis here is the sample, and the unit of
resampling is the subject (ID.csv), because samples from the same subject are
correlated.

Reported per comparison:
  mean_diff  : mean(|err_A| - |err_B|) across samples. Negative => A is better.
  ci_lo/hi   : 95% CI from a cluster bootstrap that resamples SUBJECTS, not
               samples. This is the number to trust.
  wilcoxon_p : signed-rank p over samples. Ignores the subject clustering, so
               it is anti-conservative. Reported for reference only.

Usage
    python paired_tests.py --out-dir "./Results OOL CyPrMe imputer compare - MCAR rate 0p2"
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DATA = "../Sample Data/Onset of Labor/Training"


def load(out_dir):
    path = os.path.join(out_dir, "predictions.csv.gz")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found.\n"
            "This file is only produced by runs that were NOT resumed from a "
            "pre-fingerprint cache. Rerun the experiment (or that condition) "
            "with the current script."
        )

    y = pd.read_csv(os.path.join(DATA, "DOS.csv"), index_col=0).iloc[:, 0]
    ids = pd.read_csv(os.path.join(DATA, "ID.csv"), index_col=0).iloc[:, 0]
    preds = pd.read_csv(path)

    # Final prediction per sample = median across the folds it was tested in,
    # matching how r2 is computed in the main script. Averaged over mask repeats.
    med = (preds.groupby(["condition", "base_learner", "mask_repeat", "sample"])["yhat"]
                .median()
                .reset_index())
    med = (med.groupby(["condition", "base_learner", "sample"])["yhat"]
              .mean()
              .reset_index())

    med["y"] = med["sample"].map(y)
    med["abs_err"] = (med["y"] - med["yhat"]).abs()
    med["subject"] = med["sample"].map(ids)
    return med


def paired(med, cond_a, cond_b, learner, n_boot=5000, seed=0):
    sub = med[med.base_learner == learner]
    a = sub[sub.condition == cond_a].set_index("sample")["abs_err"]
    b = sub[sub.condition == cond_b].set_index("sample")["abs_err"]
    common = a.index.intersection(b.index)
    if len(common) < 10:
        return None

    d = a.loc[common] - b.loc[common]        # >0 means A has larger error
    subj = sub.drop_duplicates("sample").set_index("sample")["subject"].loc[common]

    try:
        _, pval = wilcoxon(d.values)
    except ValueError:
        pval = np.nan

    # Cluster bootstrap: resample subjects with replacement, take all of each
    # resampled subject's samples. This respects the within-subject correlation
    # that a naive sample-level bootstrap would ignore.
    rng = np.random.RandomState(seed)
    groups = subj.unique()
    by_subject = {g: d.loc[subj.index[subj == g]].values for g in groups}

    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        boots[i] = np.concatenate([by_subject[g] for g in pick]).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return dict(
        base_learner=learner, cond_a=cond_a, cond_b=cond_b,
        mean_diff=d.mean(), ci_lo=lo, ci_hi=hi,
        significant=bool(lo > 0 or hi < 0),
        wilcoxon_p=pval,
        n_samples=len(common), n_subjects=len(groups),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    med = load(args.out_dir)
    conds = [c for c in med.condition.unique() if c != "Real"]
    learners = sorted(med.base_learner.unique())

    rows = []
    for lr in learners:
        for c in conds:                       # cost of imputing at all
            r = paired(med, c, "Real", lr, n_boot=args.n_boot)
            if r:
                rows.append(r)
        for i in range(len(conds)):           # imputer vs imputer
            for j in range(i + 1, len(conds)):
                r = paired(med, conds[i], conds[j], lr, n_boot=args.n_boot)
                if r:
                    rows.append(r)

    res = pd.DataFrame(rows)
    out = os.path.join(args.out_dir, "paired_tests.csv")
    res.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print()
    print("mean_diff < 0 means cond_a has SMALLER absolute error than cond_b.")
    print("Trust the CI, not wilcoxon_p: the CI accounts for subject clustering.")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()