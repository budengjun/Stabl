#!/usr/bin/env python3
"""Half synthetic benchmark with known support for STABL timing experiments.

The covariate matrix is synthetic block correlated Gaussian data.  Outcomes are
generated from a sparse conditional model, so the true support is known.  The
script then injects MCAR, MAR, or MNAR missingness and compares completion
strategies and knockoff generators using empirical FDP and power.

Repeated knockoff copies are generated on the same synthetic dataset.  This
makes it valid to compare single draw STABL, score averaged STABL, and the
proposed e value aggregation implementation as distinct analysis procedures.
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
from scipy.special import expit

from derandomized_stabl_v2 import (
    derandomized_stabl_select,
    save_score_draws,
    stabl_fdp_plus_select,
)
from experiment_utils import (
    apply_mask,
    atomic_write_csv,
    construct_pair,
    fit_stabl_scores,
    make_missing_mask,
    write_json,
)
from knockoff_exchangeability_v2 import marginal_classifier_test, second_moment_report, swap_classifier_test


VALID_GENERATORS = ("gaussian_equicorrelated", "gaussian_mvr", "official_plsko")
VALID_TIMINGS = (
    "post_median",
    "posterior_then_knockoff",
    "posterior_remask_negative_control",
)


def parse_list(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    items = tuple(x.strip() for x in value.split(",") if x.strip())
    invalid = sorted(set(items) - set(allowed))
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}. Allowed: {allowed}")
    if not items:
        raise ValueError(f"At least one {label} is required")
    return items


def block_covariance(p: int, block_size: int, rho: float) -> np.ndarray:
    if not 0 <= rho < 1:
        raise ValueError("rho must lie in [0, 1)")
    Sigma = np.eye(p)
    for start in range(0, p, block_size):
        stop = min(p, start + block_size)
        Sigma[start:stop, start:stop] = rho
        np.fill_diagonal(Sigma[start:stop, start:stop], 1.0)
    return Sigma


def simulate_dataset(
    n: int,
    p: int,
    n_signal: int,
    block_size: int,
    rho: float,
    task: str,
    signal_strength: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    Sigma = block_covariance(p, block_size, rho)
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
    X = rng.standard_normal((n, p)) @ L.T
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    support = np.sort(rng.choice(p, size=n_signal, replace=False))
    beta = np.zeros(p)
    beta[support] = signal_strength * rng.choice([-1.0, 1.0], size=n_signal)
    raw = X @ beta
    raw = raw / max(raw.std(), 1e-12)

    if task == "classification":
        probability = expit(raw)
        y = rng.binomial(1, probability)
        # Avoid a degenerate simulated class split.
        if len(np.unique(y)) < 2:
            y = (raw > np.median(raw)).astype(int)
    elif task == "regression":
        y = raw + rng.normal(scale=1.0, size=n)
    else:
        raise ValueError(task)

    columns = [f"x{j:04d}" for j in range(p)]
    index = [f"sample_{i:04d}" for i in range(n)]
    return (
        pd.DataFrame(X, index=index, columns=columns),
        pd.Series(y, index=index, name="y"),
        support,
        beta,
        Sigma,
    )


def selection_metrics(selection: np.ndarray, support: np.ndarray, p: int):
    truth = np.zeros(p, dtype=bool)
    truth[support] = True
    selection = np.asarray(selection, dtype=bool)
    tp = int(np.sum(selection & truth))
    fp = int(np.sum(selection & ~truth))
    n_selected = int(selection.sum())
    return {
        "n_selected": n_selected,
        "true_positives": tp,
        "false_positives": fp,
        "fdp": fp / max(1, n_selected),
        "power": tp / max(1, int(truth.sum())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="./half_synthetic_benchmark")
    parser.add_argument("--n-replicates", type=int, default=20)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--p", type=int, default=300)
    parser.add_argument("--n-signal", type=int, default=15)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--rho", type=float, default=0.60)
    parser.add_argument("--task", choices=("classification", "regression"), default="classification")
    parser.add_argument("--signal-strength", type=float, default=1.0)
    parser.add_argument("--mechanism", choices=("MCAR", "MAR", "MNAR"), default="MAR")
    parser.add_argument("--missing-rate", type=float, default=0.20)
    parser.add_argument("--generators", default="gaussian_equicorrelated,gaussian_mvr")
    parser.add_argument("--timings", default=",".join(VALID_TIMINGS))
    parser.add_argument("--n-knockoff-draws", type=int, default=10)
    parser.add_argument("--n-bootstraps", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--alpha-ebh", type=float, default=0.10)
    parser.add_argument("--alpha-kn", type=float, default=None)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--ctst-repeats", type=int, default=3)
    parser.add_argument("--ctst-permutations", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--plsko-threshold-abs", type=float, default=None)
    parser.add_argument("--plsko-threshold-q", type=float, default=0.8)
    parser.add_argument("--plsko-ncomp", type=int, default=None)
    parser.add_argument("--plsko-sparsity", type=float, default=1.0)
    args = parser.parse_args()

    if args.n_signal <= 0 or args.n_signal >= args.p:
        raise ValueError("n_signal must lie between 1 and p minus 1")
    if args.n_knockoff_draws < 1:
        raise ValueError("n_knockoff_draws must be positive")
    generators = parse_list(args.generators, VALID_GENERATORS, "generators")
    timings = parse_list(args.timings, VALID_TIMINGS, "timings")

    out_dir = Path(args.out_dir).expanduser().resolve()
    score_dir = out_dir / "scores"
    selection_dir = out_dir / "selections"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(exist_ok=True)
    selection_dir.mkdir(exist_ok=True)
    write_json({**vars(args), "generators_resolved": generators, "timings_resolved": timings}, out_dir / "config.json")

    draw_rows: list[dict] = []
    aggregate_rows: list[dict] = []

    for replicate in range(args.n_replicates):
        dataset_seed = args.random_state + replicate * 100000
        X, y, support, beta, Sigma_true = simulate_dataset(
            args.n,
            args.p,
            args.n_signal,
            args.block_size,
            args.rho,
            args.task,
            args.signal_strength,
            dataset_seed,
        )
        mask_rng = np.random.RandomState(dataset_seed + 19)
        mask = make_missing_mask(X, y, args.mechanism, args.missing_rate, mask_rng)
        X_missing = apply_mask(X, mask)
        feature_names = X.columns.tolist()
        np.savez_compressed(
            out_dir / f"truth_rep_{replicate:03d}.npz",
            support=support,
            beta=beta,
            Sigma_true=Sigma_true,
            feature_names=np.asarray(feature_names),
        )

        for generator in generators:
            for timing in timings:
                real_draws: list[np.ndarray] = []
                knockoff_draws: list[np.ndarray] = []
                cell_failed = False

                for draw in range(args.n_knockoff_draws):
                    completion_seed = dataset_seed + draw * 173 + 31
                    knockoff_seed = dataset_seed + draw * 173 + 97
                    record = {
                        "replicate": replicate,
                        "generator": generator,
                        "timing": timing,
                        "draw": draw,
                        "n": args.n,
                        "p": args.p,
                        "n_signal": args.n_signal,
                        "mechanism": args.mechanism,
                        "target_missing_rate": args.missing_rate,
                        "realized_missing_rate": float(mask.mean()),
                        "task": args.task,
                        "error": "",
                    }
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
                        real_score, knockoff_score, model = fit_stabl_scores(
                            pair.X,
                            pair.X_tilde,
                            y.to_numpy(),
                            task=args.task,
                            model_name="lasso",
                            groups=None,
                            n_bootstraps=args.n_bootstraps,
                            n_jobs=args.n_jobs,
                            random_state=knockoff_seed,
                        )
                        real_draws.append(real_score)
                        knockoff_draws.append(knockoff_score)
                        real_max = real_score.max(axis=1)
                        knockoff_max = knockoff_score.max(axis=1)
                        selection = real_max > float(model.fdr_min_threshold_)
                        record.update(
                            {
                                "generation_seconds": pair.generation_seconds,
                                "theta": float(model.fdr_min_threshold_),
                                "min_fdp_plus": float(model.min_fdr_),
                                **{f"classic_{key}": value for key, value in selection_metrics(selection, support, args.p).items()},
                                "W_positive_fraction": float(np.mean(real_max - knockoff_max > 0)),
                            }
                        )
                        if args.diagnostics:
                            record.update(second_moment_report(pair.X, pair.X_tilde, Sigma_used=pair.Sigma_used))
                            marginal = marginal_classifier_test(
                                pair.X,
                                pair.X_tilde,
                                n_repeats=args.ctst_repeats,
                                n_permutations=args.ctst_permutations,
                                permutation_mode="fast",
                                random_state=knockoff_seed + 1,
                            )
                            record.update(marginal.to_dict())
                            ctst = swap_classifier_test(
                                pair.X,
                                pair.X_tilde,
                                n_repeats=args.ctst_repeats,
                                n_permutations=args.ctst_permutations,
                                permutation_mode="fast",
                                random_state=knockoff_seed,
                            )
                            record.update(ctst.to_dict())
                        print(
                            f"[draw] rep={replicate} gen={generator} timing={timing} "
                            f"draw={draw} fdp={record['classic_fdp']:.3f} "
                            f"power={record['classic_power']:.3f}",
                            flush=True,
                        )
                    except Exception as exc:
                        record["error"] = repr(exc)
                        record["traceback"] = traceback.format_exc(limit=8)
                        cell_failed = True
                        print(f"[failed] rep={replicate} {generator} {timing} draw={draw}: {exc!r}", flush=True)
                    draw_rows.append(record)
                    atomic_write_csv(pd.DataFrame(draw_rows), out_dir / "draw_results.csv")

                if cell_failed or len(real_draws) != args.n_knockoff_draws:
                    aggregate_rows.append(
                        {
                            "replicate": replicate,
                            "generator": generator,
                            "timing": timing,
                            "method": "cell_incomplete",
                            "error": "One or more draws failed",
                        }
                    )
                    continue

                score_path = score_dir / f"rep_{replicate:03d}__{generator}__{timing}.npz"
                save_score_draws(
                    score_path,
                    real_draws,
                    knockoff_draws,
                    feature_names,
                    replicate=replicate,
                    generator=generator,
                    timing=timing,
                    dataset="block_gaussian_half_synthetic",
                )

                real_flat = np.vstack([x.max(axis=1) for x in real_draws])
                knockoff_flat = np.vstack([x.max(axis=1) for x in knockoff_draws])

                first_sel, first_theta, first_q, _ = stabl_fdp_plus_select(real_draws[0], knockoff_draws[0])
                mean_sel, mean_theta, mean_q, _ = stabl_fdp_plus_select(real_flat.mean(axis=0), knockoff_flat.mean(axis=0))
                derand_table, derand_summary = derandomized_stabl_select(
                    real_draws,
                    knockoff_draws,
                    alpha_ebh=args.alpha_ebh,
                    alpha_kn=args.alpha_kn,
                    feature_names=feature_names,
                )
                derand_sel = derand_table.set_index("feature").loc[feature_names, "selected_ebh"].to_numpy(dtype=bool)

                method_specs = [
                    ("single_draw_fdp_plus", first_sel, first_theta, first_q),
                    ("mean_score_fdp_plus", mean_sel, mean_theta, mean_q),
                    ("derandomized_e_bh", derand_sel, np.nan, np.nan),
                ]
                for method, selection, theta, q_plus in method_specs:
                    aggregate_rows.append(
                        {
                            "replicate": replicate,
                            "generator": generator,
                            "timing": timing,
                            "method": method,
                            "theta": theta,
                            "q_plus": q_plus,
                            "n_draws": args.n_knockoff_draws,
                            "error": "",
                            **selection_metrics(selection, support, args.p),
                        }
                    )
                derand_table.to_csv(
                    selection_dir / f"rep_{replicate:03d}__{generator}__{timing}__derandomized.csv",
                    index=False,
                )
                atomic_write_csv(pd.DataFrame(aggregate_rows), out_dir / "aggregate_results.csv")

    draw_frame = pd.DataFrame(draw_rows)
    aggregate_frame = pd.DataFrame(aggregate_rows)
    atomic_write_csv(draw_frame, out_dir / "draw_results.csv")
    atomic_write_csv(aggregate_frame, out_dir / "aggregate_results.csv")

    valid = aggregate_frame[aggregate_frame["error"].fillna("").eq("")]
    if not valid.empty:
        summary = (
            valid.groupby(["generator", "timing", "method"])[
                ["fdp", "power", "n_selected", "true_positives", "false_positives"]
            ]
            .agg(["mean", "std", "median"])
            .reset_index()
        )
        summary.to_csv(out_dir / "summary.csv", index=False)
        print("\n=== Half synthetic summary ===")
        print(summary.to_string(index=False))
    print(f"\nWrote benchmark to {out_dir}")


if __name__ == "__main__":
    main()
