"""
Diagnostic 1 for E7: is block-wise missingness in SSI CyTOF associated with the outcome y?

If it is, then median-imputing a re-masked knockoff column injects a constant block that
functions as a missingness indicator, which would explain why Artificial SF went UP in E7
instead of down.

Uses the same data.load_ssi loader as run_cv_SSI_decoy_timing_mx_knockoff.py, so the matrix
tested here is the exact matrix the experiment saw, NaNs intact, before any imputer.

Run from the same directory as the run script:
    python diag_missingness_vs_y.py
    python diag_missingness_vs_y.py --outdir ./Diagnostics

Outputs
-------
missingness_vs_y_per_feature.csv   per-feature missing rate, Fisher exact P, BH q
missingness_vs_y_per_pattern.csv   one test per unique block missingness pattern
missingness_vs_y_summary.txt       verdict
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from stabl import data

DATA_DIR = "../Sample Data/Biobank SSI"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--outdir", default="./Diagnostics")
    ap.add_argument("--alpha", type=float, default=0.05)
    return ap.parse_args()


def fisher_2x2(mask, y):
    a = int(((mask == 1) & (y == 1)).sum())
    b = int(((mask == 1) & (y == 0)).sum())
    c = int(((mask == 0) & (y == 1)).sum())
    d = int(((mask == 0) & (y == 0)).sum())
    odds, p = fisher_exact([[a, b], [c, d]])
    return a, b, c, d, odds, p


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X_train, X_valid, y_train, y_valid, ids, task_type = data.load_ssi(args.data_dir)
    X = X_train["CyTOF"]
    y = y_train.loc[X.index].astype(int).values

    M = X.isna()
    print("SSI/CyTOF missingness")
    print(f"  samples: {X.shape[0]}")
    print(f"  features: {X.shape[1]}")
    print(f"  missing cells: {int(M.values.sum())}")
    print(f"  samples with missing values: {int(M.any(axis=1).sum())}")
    print(f"  features with missing values: {int(M.any(axis=0).sum())}")
    print(f"  y prevalence: {y.mean():.3f}")
    print()

    assert set(np.unique(y)) <= {0, 1}, "y must be binary"

    feat_missing = M.columns[M.any(axis=0)]

    # ---- per-feature ----
    rows = []
    for f in feat_missing:
        m = M[f].values.astype(int)
        a, b, c, d, odds, p = fisher_2x2(m, y)
        rows.append({
            "feature": f,
            "missing_rate": m.mean(),
            "n_missing": int(m.sum()),
            "missing_y1": a, "missing_y0": b,
            "observed_y1": c, "observed_y0": d,
            "odds_ratio": odds,
            "p_fisher": p,
        })
    res = pd.DataFrame(rows).set_index("feature")
    res["q_BH"] = multipletests(res["p_fisher"], method="fdr_bh")[1]
    res = res.sort_values("p_fisher")
    res.to_csv(outdir / "missingness_vs_y_per_feature.csv")

    # ---- per unique missingness pattern ----
    # Missingness is block-wise (stimulation conditions), so many features share one pattern.
    # Testing every feature separately inflates the count of significant hits, since those
    # tests are not independent. Collapsing to unique patterns is the level to judge on.
    Mi = M[feat_missing].astype(int)
    groups = {}
    for f in feat_missing:
        pat = tuple(Mi[f].to_numpy())
        groups.setdefault(pat, []).append(f)

    prow = []
    for pat, members in groups.items():
        m = np.array(pat)
        a, b, c, d, odds, p = fisher_2x2(m, y)
        prow.append({
            "representative_feature": members[0],
            "n_features_sharing_pattern": len(members),
            "n_missing_samples": int(m.sum()),
            "missing_y1": a, "missing_y0": b,
            "observed_y1": c, "observed_y0": d,
            "odds_ratio": odds,
            "p_fisher": p,
        })
    pat_res = pd.DataFrame(prow).sort_values("p_fisher").reset_index(drop=True)
    pat_res["q_BH"] = multipletests(pat_res["p_fisher"], method="fdr_bh")[1]
    pat_res.to_csv(outdir / "missingness_vs_y_per_pattern.csv", index=False)

    n_feat = len(res)
    n_pat = len(pat_res)
    exp_feat = args.alpha * n_feat
    exp_pat = args.alpha * n_pat

    pd.set_option("display.width", 220)
    lines = [
        "Diagnostic 1  missingness vs outcome (Fisher exact)",
        "",
        "feature level (tests are correlated, read with caution)",
        f"  features tested   {n_feat}",
        f"  p < {args.alpha}          {int((res['p_fisher'] < args.alpha).sum())}"
        f"   (null expectation {exp_feat:.1f})",
        f"  BH q < {args.alpha}       {int((res['q_BH'] < args.alpha).sum())}",
        "",
        "pattern level (this is the level to judge on)",
        f"  unique patterns   {n_pat}",
        f"  p < {args.alpha}          {int((pat_res['p_fisher'] < args.alpha).sum())}"
        f"   (null expectation {exp_pat:.1f})",
        f"  BH q < {args.alpha}       {int((pat_res['q_BH'] < args.alpha).sum())}",
        "",
        "top patterns by P value",
        pat_res.head(10).to_string(index=False),
        "",
        "Interpretation",
        "  any pattern surviving BH, or hits well above the null expectation:",
        "      missingness carries outcome information. The constant block written by median",
        "      imputation into a re-masked knockoff column acts as a missingness indicator,",
        "      which explains the E7 rise in Artificial SF.",
        "  hits at or below the null expectation:",
        "      that mechanism is not supported. The E7 shift would then come from the joint",
        "      StandardScaler being dragged by the constant blocks, which depresses Real SF",
        "      across the board. Diagnostic 2 separates the two by stratifying on missing rate.",
    ]
    txt = "\n".join(lines)
    (outdir / "missingness_vs_y_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
