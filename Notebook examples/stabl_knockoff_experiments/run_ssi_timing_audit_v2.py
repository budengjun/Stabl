#!/usr/bin/env python3
"""Paired knockoff timing audit for SSI CyTOF with real missing values.

The script repeats the current 5 by 3 outer cross validation structure and, for
one or more repeated knockoff draws within each training fold, stores paired
STABL score paths.  Those score files can later be analysed with
``derandomized_stabl_v2.py`` without rerunning the expensive STABL fits.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "8")

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from derandomized_stabl_v2 import save_score_draws
from experiment_utils import atomic_write_csv, construct_pair, fit_stabl_scores, write_json
from knockoff_exchangeability_v2 import marginal_classifier_test, second_moment_report, swap_classifier_test


VALID_GENERATORS = (
    "gaussian_equicorrelated",
    "gaussian_mvr",
    "official_plsko",
)
VALID_TIMINGS = (
    "post_median",
    "posterior_then_knockoff",
    "posterior_remask_negative_control",
)
VALID_MODELS = ("lasso", "alasso", "elasticnet")


def parse_list(value: str, allowed: tuple[str, ...], name: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(items) - set(allowed))
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid {name}: {invalid}. Allowed: {allowed}")
    return items


def load_ssi(data_path: str):
    try:
        from stabl import data
    except ImportError as exc:
        raise RuntimeError("The stabl package must be importable") from exc
    X_train, _, y_train, _, ids, task_type = data.load_ssi(data_path)
    X = X_train["CyTOF"].astype(float)
    y = pd.Series(y_train, index=X.index)
    # The bundled SSI loader returns ids=None.  SSI has one row per patient, so
    # no grouped split is required.  Preserve groups only if a future loader
    # supplies real repeated-measures identifiers.
    groups = None if ids is None else pd.Series(ids, index=X.index)
    print(
        f"[data] SSI CyTOF: {X.shape[0]} samples x {X.shape[1]} features, "
        f"{int(X.isna().sum().sum())} missing cells",
        flush=True,
    )
    return X, y, groups, task_type


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../Sample Data/Biobank SSI")
    parser.add_argument("--out-dir", default="./SSI_knockoff_timing_audit_v2")
    parser.add_argument(
        "--generators", default="gaussian_equicorrelated,gaussian_mvr"
    )
    parser.add_argument("--timings", default=",".join(VALID_TIMINGS))
    parser.add_argument("--models", default="lasso,alasso,elasticnet")
    parser.add_argument("--n-knockoff-draws", type=int, default=1)
    parser.add_argument("--n-bootstraps", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--ctst-repeats", type=int, default=3)
    parser.add_argument("--ctst-permutations", type=int, default=100)
    parser.add_argument(
        "--ctst-permutation-mode", choices=("fast", "refit", "none"), default="fast"
    )
    parser.add_argument("--random-state", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plsko-threshold-abs", type=float, default=None)
    parser.add_argument("--plsko-threshold-q", type=float, default=0.8)
    parser.add_argument("--plsko-ncomp", type=int, default=None)
    parser.add_argument("--plsko-sparsity", type=float, default=1.0)
    args = parser.parse_args()

    generators = parse_list(args.generators, VALID_GENERATORS, "generators")
    timings = parse_list(args.timings, VALID_TIMINGS, "timings")
    models = parse_list(args.models, VALID_MODELS, "models")
    if args.n_knockoff_draws < 1:
        raise ValueError("n_knockoff_draws must be at least 1")

    out_dir = Path(args.out_dir).expanduser().resolve()
    score_dir = out_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "audit_rows.csv"

    existing = pd.read_csv(result_path) if args.resume and result_path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    completed = set()
    if not existing.empty:
        key_columns = ["fold", "generator", "timing", "model", "draw"]
        valid = existing[existing["error"].fillna("").eq("")]
        completed = set(map(tuple, valid[key_columns].itertuples(index=False, name=None)))

    X, y, groups, task_type = load_ssi(args.data_path)
    write_json(vars(args), out_dir / "config.json")

    outer = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=args.random_state
    )

    for fold, (train_pos, _) in enumerate(outer.split(X, y)):
        train_index = X.index[train_pos]
        X_missing = X.loc[train_index]
        y_train = y.loc[train_index]
        groups_train = None if groups is None else groups.loc[train_index].to_numpy()
        feature_names = X_missing.columns.astype(str).tolist()

        for generator in generators:
            for timing in timings:
                for model_name in models:
                    real_draws: list[np.ndarray] = []
                    knockoff_draws: list[np.ndarray] = []
                    score_path = score_dir / (
                        f"fold_{fold:02d}__{generator}__{timing}__{model_name}.npz"
                    )

                    for draw in range(args.n_knockoff_draws):
                        key = (fold, generator, timing, model_name, draw)
                        if args.resume and key in completed:
                            print(f"[skip] {key}", flush=True)
                            continue

                        completion_seed = args.random_state + fold * 10000 + draw * 131 + 11
                        knockoff_seed = args.random_state + fold * 10000 + draw * 131 + 71
                        record: dict[str, object] = {
                            "fold": fold,
                            "generator": generator,
                            "timing": timing,
                            "model": model_name,
                            "draw": draw,
                            "n_train": len(train_index),
                            "p": X_missing.shape[1],
                            "missing_rate": float(X_missing.isna().to_numpy().mean()),
                            "completion_seed": completion_seed,
                            "knockoff_seed": knockoff_seed,
                            "error": "",
                        }
                        print(
                            f"[start] fold={fold} generator={generator} timing={timing} "
                            f"model={model_name} draw={draw}",
                            flush=True,
                        )

                        try:
                            pair = construct_pair(
                                X_missing,
                                generator,
                                timing,
                                completion_seed=completion_seed,
                                knockoff_seed=knockoff_seed,
                                plsko_threshold_abs=args.plsko_threshold_abs,
                                plsko_threshold_q=args.plsko_threshold_q,
                                plsko_ncomp=args.plsko_ncomp,
                                plsko_sparsity=args.plsko_sparsity,
                            )
                            record["generation_seconds"] = pair.generation_seconds
                            marginal = marginal_classifier_test(
                                pair.X,
                                pair.X_tilde,
                                groups=groups_train,
                                n_repeats=args.ctst_repeats,
                                n_permutations=args.ctst_permutations,
                                permutation_mode=args.ctst_permutation_mode,
                                random_state=knockoff_seed + 1,
                            )
                            record.update(marginal.to_dict())
                            ctst = swap_classifier_test(
                                pair.X,
                                pair.X_tilde,
                                groups=groups_train,
                                n_repeats=args.ctst_repeats,
                                n_permutations=args.ctst_permutations,
                                permutation_mode=args.ctst_permutation_mode,
                                random_state=knockoff_seed,
                            )
                            record.update(ctst.to_dict())
                            record.update(
                                second_moment_report(
                                    pair.X, pair.X_tilde, Sigma_used=pair.Sigma_used
                                )
                            )

                            real_score, knockoff_score, model = fit_stabl_scores(
                                pair.X,
                                pair.X_tilde,
                                y_train.to_numpy(),
                                task="classification",
                                model_name=model_name,
                                groups=groups_train,
                                n_bootstraps=args.n_bootstraps,
                                n_jobs=args.n_jobs,
                                random_state=knockoff_seed,
                            )
                            real_max = real_score.max(axis=1)
                            knockoff_max = knockoff_score.max(axis=1)
                            W = real_max - knockoff_max
                            record.update(
                                {
                                    "theta": float(model.fdr_min_threshold_),
                                    "min_fdp_plus": float(model.min_fdr_),
                                    "n_selected_classic": int(
                                        np.sum(real_max > model.fdr_min_threshold_)
                                    ),
                                    "real_sf_mean": float(real_max.mean()),
                                    "knockoff_sf_mean": float(knockoff_max.mean()),
                                    "sf_mean_ratio": float(
                                        knockoff_max.mean() / max(real_max.mean(), 1e-12)
                                    ),
                                    "W_positive_fraction": float(np.mean(W > 0)),
                                    "W_mean": float(W.mean()),
                                }
                            )
                            real_draws.append(real_score)
                            knockoff_draws.append(knockoff_score)
                            print(
                                f"[done] theta={record['theta']:.3f} "
                                f"nsel={record['n_selected_classic']} "
                                f"paircorr={record['pair_corr_mean']:.3f} "
                                f"srel={record['s_relative_mean']:.3f}",
                                flush=True,
                            )
                        except Exception as exc:
                            record["error"] = repr(exc)
                            record["traceback"] = traceback.format_exc(limit=8)
                            print(f"[failed] {exc!r}", flush=True)

                        rows.append(record)
                        atomic_write_csv(pd.DataFrame(rows), result_path)

                    if real_draws:
                        save_score_draws(
                            score_path,
                            real_draws,
                            knockoff_draws,
                            feature_names,
                            fold=fold,
                            generator=generator,
                            timing=timing,
                            model=model_name,
                            dataset="SSI_CyTOF",
                        )
                        print(f"[scores] wrote {score_path}", flush=True)

    frame = pd.DataFrame(rows)
    atomic_write_csv(frame, result_path)
    valid = frame[frame["error"].fillna("").eq("")]
    metric_candidates = [
        "marginal_auc_oriented_mean",
        "marginal_permutation_pvalue",
        "swap_auc_oriented_mean",
        "permutation_pvalue",
        "pair_corr_mean",
        "s_relative_mean",
        "cov_kk_fro_relative",
        "theta",
        "min_fdp_plus",
        "n_selected_classic",
        "sf_mean_ratio",
        "W_positive_fraction",
    ]
    metrics = [column for column in metric_candidates if column in valid.columns]
    if not valid.empty and metrics:
        summary = (
            valid.groupby(["generator", "timing", "model"])[metrics]
            .agg(["mean", "std", "median"])
            .reset_index()
        )
        summary.to_csv(out_dir / "summary.csv", index=False)
        print("\n=== Summary ===")
        print(summary.to_string(index=False))
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
