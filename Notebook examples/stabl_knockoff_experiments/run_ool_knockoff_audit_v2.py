#!/usr/bin/env python3
"""Generator and timing audit on the Onset of Labor three omic dataset.

This script is designed to answer three separate questions without mixing them:

1. Does the knockoff generator produce obvious exchangeability violations or
   nearly identical copies of the original variables?
2. Does posterior completion behave differently from deterministic median
   completion?
3. Does the known remasking and second imputation pattern create a detectable
   negative control failure?

The default generator set contains only knockpy Gaussian generators.  The
``official_plsko`` option calls the authors' R package through the provided
bridge.  The unvalidated Python PLSKO port is intentionally excluded.
"""

from __future__ import annotations

# Pin numerical libraries before importing NumPy.  Override these variables in
# the shell if a different allocation is desired.
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
from sklearn.model_selection import GroupShuffleSplit

from derandomized_stabl_v2 import load_score_draws, save_score_draws
from experiment_utils import (
    apply_mask,
    atomic_write_csv,
    cap_features_by_variance,
    construct_pair,
    fit_stabl_scores,
    make_missing_mask,
    median_complete,
    posterior_complete,
    write_json,
)
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
VALID_OMICS = ("CyTOF", "Proteomics", "Metabolomics")


def filter_fold_features(
    X_missing: pd.DataFrame,
    *,
    relative_tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop fold-specific constant or unusable columns once for all cells.

    The filter is computed from a deterministic median completion of the
    training fold, before looping over timing, generator, or draw.  Therefore
    every method in the same fold and omic sees exactly the same feature set.
    """
    if X_missing.empty:
        raise ValueError("Cannot filter an empty feature matrix")

    medians = X_missing.median(axis=0, skipna=True)
    completed_reference = X_missing.fillna(medians)
    values = completed_reference.to_numpy(dtype=float)

    finite_columns = np.isfinite(values).all(axis=0)
    column_range = np.ptp(values, axis=0)
    column_scale = np.maximum(1.0, np.max(np.abs(values), axis=0))
    nonconstant = column_range > relative_tolerance * column_scale
    keep = finite_columns & nonconstant

    dropped = X_missing.columns[~keep].astype(str).tolist()
    filtered = X_missing.loc[:, keep].copy()
    if filtered.shape[1] == 0:
        raise ValueError("All training-fold features were constant or unusable")
    return filtered, dropped


def parse_csv_list(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(items) - set(allowed))
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid {label}: {invalid}. Allowed values: {allowed}")
    if not items:
        raise argparse.ArgumentTypeError(f"At least one {label} is required")
    return items


def load_ool(data_path: Path, omics: tuple[str, ...], max_features: int | None):
    training = data_path / "Onset of Labor" / "Training"
    y = pd.read_csv(training / "DOS.csv", index_col=0).iloc[:, 0].astype(float)
    groups = pd.read_csv(training / "ID.csv", index_col=0).iloc[:, 0]
    raw = {
        omic: pd.read_csv(training / f"{omic}.csv", index_col=0).astype(float)
        for omic in omics
    }

    common = y.index.intersection(groups.index)
    for X in raw.values():
        common = common.intersection(X.index)
    y = y.loc[common]
    groups = groups.loc[common]

    data: dict[str, pd.DataFrame] = {}
    for omic, X in raw.items():
        X = X.loc[common]
        before = X.shape[1]
        complete = X.dropna(axis=1, how="any")
        incomplete_dropped = before - complete.shape[1]
        X = cap_features_by_variance(complete, max_features)
        variance_capped = complete.shape[1] - X.shape[1]
        data[omic] = X
        print(
            f"[data] {omic}: {X.shape[0]} samples x {X.shape[1]} features "
            f"after dropping {incomplete_dropped} incomplete columns and "
            f"capping {variance_capped} additional columns by variance",
            flush=True,
        )
    print(f"[data] {len(common)} samples, {groups.nunique()} subjects", flush=True)
    return data, y, groups


def completed_keys(frame: pd.DataFrame) -> set[tuple]:
    required = ["fold", "omic", "generator", "timing", "draw"]
    if frame.empty or not set(required).issubset(frame.columns):
        return set()
    ok = frame.get("error", pd.Series("", index=frame.index)).fillna("").eq("")
    return set(map(tuple, frame.loc[ok, required].itertuples(index=False, name=None)))


KEY_COLUMNS = ("fold", "omic", "generator", "timing", "draw")


def _record_key(record: dict[str, object]) -> tuple:
    return tuple(record[column] for column in KEY_COLUMNS)


def upsert_record(rows: list[dict[str, object]], record: dict[str, object]) -> None:
    """Replace any previous row for the same experiment unit, then append."""
    key = _record_key(record)
    rows[:] = [row for row in rows if _record_key(row) != key]
    rows.append(record)


def atomic_save_score_draws(
    path: Path,
    real_scores: list[np.ndarray],
    knockoff_scores: list[np.ndarray],
    feature_names: list[str],
    **metadata: object,
) -> None:
    """Write an NPZ checkpoint atomically so interruption cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        save_score_draws(
            temporary,
            real_scores,
            knockoff_scores,
            feature_names,
            **metadata,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def valid_single_draw_checkpoint(path: Path, feature_names: list[str]) -> bool:
    """Return True only for a readable, shape-consistent one-draw checkpoint."""
    if not path.exists():
        return False
    try:
        real, knockoff, names, _ = load_score_draws(path)
    except Exception:
        return False
    if len(real) != 1 or len(knockoff) != 1 or names != feature_names:
        return False
    real_array = np.asarray(real[0])
    knockoff_array = np.asarray(knockoff[0])
    return (
        real_array.shape == knockoff_array.shape
        and real_array.ndim >= 1
        and real_array.shape[0] == len(feature_names)
        and np.isfinite(real_array).all()
        and np.isfinite(knockoff_array).all()
    )


def combine_draw_checkpoints(
    checkpoint_paths: list[Path],
    combined_path: Path,
    feature_names: list[str],
    **metadata: object,
) -> bool:
    """Combine ordered one-draw checkpoints into the file used for analysis."""
    if not all(valid_single_draw_checkpoint(path, feature_names) for path in checkpoint_paths):
        if combined_path.exists():
            combined_path.unlink()
        return False

    real_draws: list[np.ndarray] = []
    knockoff_draws: list[np.ndarray] = []
    for path in checkpoint_paths:
        real, knockoff, names, _ = load_score_draws(path)
        if names != feature_names:
            raise ValueError(f"Feature-name mismatch in {path}")
        real_draws.append(np.asarray(real[0]))
        knockoff_draws.append(np.asarray(knockoff[0]))

    atomic_save_score_draws(
        combined_path,
        real_draws,
        knockoff_draws,
        feature_names,
        **metadata,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../Sample Data")
    parser.add_argument("--out-dir", default="./OOL_knockoff_audit_v2")
    parser.add_argument("--omics", default=",".join(VALID_OMICS))
    parser.add_argument(
        "--generators",
        default="gaussian_equicorrelated,gaussian_mvr",
        help="Comma separated generator names. official_plsko requires R and the PLSKO package.",
    )
    parser.add_argument("--timings", default=",".join(VALID_TIMINGS))
    parser.add_argument("--mechanism", choices=("MCAR", "MAR", "MNAR"), default="MAR")
    parser.add_argument("--missing-rate", type=float, default=0.20)
    parser.add_argument("--max-features", type=int, default=1500)
    parser.add_argument(
        "--fold-variance-relative-tolerance",
        type=float,
        default=1e-12,
        help=(
            "Relative range tolerance used to remove constant or numerically "
            "degenerate training-fold features before all method comparisons."
        ),
    )
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--n-knockoff-draws", type=int, default=1)
    parser.add_argument("--fit-stabl", action="store_true")
    parser.add_argument("--n-bootstraps", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--ctst-repeats", type=int, default=3)
    parser.add_argument("--ctst-permutations", type=int, default=100)
    parser.add_argument(
        "--ctst-permutation-mode", choices=("fast", "refit", "none"), default="fast"
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plsko-threshold-abs", type=float, default=None)
    parser.add_argument("--plsko-threshold-q", type=float, default=0.8)
    parser.add_argument("--plsko-ncomp", type=int, default=None)
    parser.add_argument("--plsko-sparsity", type=float, default=1.0)
    args = parser.parse_args()

    omics = parse_csv_list(args.omics, VALID_OMICS, "omics")
    generators = parse_csv_list(args.generators, VALID_GENERATORS, "generators")
    timings = parse_csv_list(args.timings, VALID_TIMINGS, "timings")
    if args.n_knockoff_draws < 1:
        raise ValueError("n_knockoff_draws must be at least 1")

    out_dir = Path(args.out_dir).expanduser().resolve()
    score_dir = out_dir / "scores"
    score_checkpoint_dir = out_dir / "score_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    score_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "audit_rows.csv"

    existing = pd.read_csv(result_path) if args.resume and result_path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    done = completed_keys(existing)

    data, y, groups = load_ool(Path(args.data_path), omics, args.max_features)

    # Generate one mask per omic on the complete dataset, then reuse it in every
    # fold, timing, and generator.  This mirrors the Direction 2 experiment and
    # removes missingness draw as an avoidable confound.
    masked_data: dict[str, pd.DataFrame] = {}
    mask_rates: dict[str, float] = {}
    for omic_index, (omic, X) in enumerate(data.items()):
        rng = np.random.RandomState(args.random_state + 1000 * omic_index)
        mask = make_missing_mask(X, y, args.mechanism, args.missing_rate, rng)
        masked_data[omic] = apply_mask(X, mask)
        mask_rates[omic] = float(mask.mean())
        print(f"[mask] {omic}: realized missing rate {mask.mean():.3f}", flush=True)

    config = vars(args).copy()
    config.update(
        {
            "omics_resolved": omics,
            "generators_resolved": generators,
            "timings_resolved": timings,
            "realized_missing_rates": mask_rates,
        }
    )
    write_json(config, out_dir / "config.json")

    splitter = GroupShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    for fold, (train_pos, _) in enumerate(splitter.split(y, y, groups=groups)):
        train_index = y.index[train_pos]
        y_train = y.loc[train_index]
        groups_train = groups.loc[train_index].to_numpy()

        for omic_index, (omic, X_masked_full) in enumerate(masked_data.items()):
            X_missing_unfiltered = X_masked_full.loc[train_index]
            p_before_filter = X_missing_unfiltered.shape[1]
            X_missing, dropped_features = filter_fold_features(
                X_missing_unfiltered,
                relative_tolerance=args.fold_variance_relative_tolerance,
            )
            feature_names = X_missing.columns.astype(str).tolist()
            p_after_filter = X_missing.shape[1]
            fold_missing_rate = float(X_missing.isna().to_numpy().mean())

            if dropped_features:
                print(
                    f"[filter] fold={fold} omic={omic}: dropped "
                    f"{len(dropped_features)} constant/unusable features; "
                    f"retained {p_after_filter}/{p_before_filter}",
                    flush=True,
                )

            # The completion is fixed within a fold and omic.  Derandomized
            # draws vary only the knockoff construction, not the imputed X.
            completion_seed = (
                args.random_state + fold * 10000 + omic_index * 1000 + 7
            )
            completion_cache: dict[str, pd.DataFrame] = {}
            if "post_median" in timings:
                completion_cache["post_median"] = median_complete(X_missing)
            if any(timing.startswith("posterior") for timing in timings):
                posterior = posterior_complete(X_missing, completion_seed)
                completion_cache["posterior"] = posterior

            for generator in generators:
                for timing in timings:
                    score_path = score_dir / (
                        f"fold_{fold:02d}__{omic}__{generator}__{timing}.npz"
                    )
                    checkpoint_paths = [
                        score_checkpoint_dir
                        / (
                            f"fold_{fold:02d}__{omic}__{generator}__{timing}"
                            f"__draw_{draw:03d}.npz"
                        )
                        for draw in range(args.n_knockoff_draws)
                    ]

                    for draw in range(args.n_knockoff_draws):
                        key = (fold, omic, generator, timing, draw)
                        draw_score_path = checkpoint_paths[draw]
                        audit_complete = key in done
                        score_complete = (
                            not args.fit_stabl
                            or valid_single_draw_checkpoint(draw_score_path, feature_names)
                        )
                        if args.resume and audit_complete and score_complete:
                            print(f"[skip] {key}", flush=True)
                            continue
                        if args.resume and audit_complete and not score_complete:
                            print(
                                f"[repair] {key}: audit row exists but score checkpoint "
                                "is missing or invalid; rerunning this draw",
                                flush=True,
                            )

                        knockoff_seed = args.random_state + fold * 10000 + draw * 101 + 53
                        record: dict[str, object] = {
                            "fold": fold,
                            "omic": omic,
                            "generator": generator,
                            "timing": timing,
                            "draw": draw,
                            "n_train": len(train_index),
                            "p": X_missing.shape[1],
                            "p_before_fold_filter": p_before_filter,
                            "p_dropped_fold_filter": len(dropped_features),
                            "fold_realized_missing_rate": fold_missing_rate,
                            "mechanism": args.mechanism,
                            "target_missing_rate": args.missing_rate,
                            "realized_missing_rate": mask_rates[omic],
                            "completion_seed": completion_seed,
                            "knockoff_seed": knockoff_seed,
                            "error": "",
                        }
                        print(
                            f"[start] fold={fold} omic={omic} generator={generator} "
                            f"timing={timing} draw={draw}",
                            flush=True,
                        )

                        try:
                            completed_override = (
                                completion_cache["post_median"]
                                if timing == "post_median"
                                else completion_cache["posterior"]
                            )
                            pair = construct_pair(
                                X_missing,
                                generator,
                                timing,
                                completion_seed=completion_seed,
                                knockoff_seed=knockoff_seed,
                                completed_override=completed_override,
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

                            if args.fit_stabl:
                                real_score, knockoff_score, model = fit_stabl_scores(
                                    pair.X,
                                    pair.X_tilde,
                                    y_train.to_numpy(),
                                    task="regression",
                                    groups=groups_train,
                                    n_bootstraps=args.n_bootstraps,
                                    n_jobs=args.n_jobs,
                                    random_state=knockoff_seed,
                                )
                                real_max = real_score.max(axis=1)
                                knockoff_max = knockoff_score.max(axis=1)
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
                                        "W_positive_fraction": float(
                                            np.mean(real_max - knockoff_max > 0)
                                        ),
                                    }
                                )
                                # Save this draw before marking the audit row successful.
                                # If the process is killed during the write, os.replace keeps
                                # the previous final checkpoint intact or leaves no final file.
                                atomic_save_score_draws(
                                    draw_score_path,
                                    [real_score],
                                    [knockoff_score],
                                    feature_names,
                                    fold=fold,
                                    omic=omic,
                                    generator=generator,
                                    timing=timing,
                                    draw=draw,
                                    mechanism=args.mechanism,
                                    missing_rate=args.missing_rate,
                                    p_before_fold_filter=p_before_filter,
                                    p_dropped_fold_filter=len(dropped_features),
                                    completion_seed=completion_seed,
                                    knockoff_seed=knockoff_seed,
                                )
                                record["score_checkpoint"] = str(draw_score_path)

                            print(
                                f"[done] paircorr={record['pair_corr_mean']:.3f} "
                                f"srel={record['s_relative_mean']:.3f} "
                                f"swap={record['swap_auc_oriented_mean']:.3f} "
                                f"seconds={record['generation_seconds']:.1f}",
                                flush=True,
                            )
                        except Exception as exc:
                            record["error"] = repr(exc)
                            record["traceback"] = traceback.format_exc(limit=8)
                            print(f"[failed] {exc!r}", flush=True)

                        upsert_record(rows, record)
                        atomic_write_csv(pd.DataFrame(rows), result_path)

                    if args.fit_stabl:
                        combined = combine_draw_checkpoints(
                            checkpoint_paths,
                            score_path,
                            feature_names,
                            fold=fold,
                            omic=omic,
                            generator=generator,
                            timing=timing,
                            mechanism=args.mechanism,
                            missing_rate=args.missing_rate,
                            p_before_fold_filter=p_before_filter,
                            p_dropped_fold_filter=len(dropped_features),
                            completion_seed=completion_seed,
                            n_draws=args.n_knockoff_draws,
                        )
                        if combined:
                            print(
                                f"[scores] wrote complete {args.n_knockoff_draws}-draw "
                                f"file {score_path}",
                                flush=True,
                            )
                        else:
                            missing = [
                                str(path)
                                for path in checkpoint_paths
                                if not valid_single_draw_checkpoint(path, feature_names)
                            ]
                            print(
                                f"[scores] incomplete combination; {len(missing)} draw "
                                "checkpoint(s) still missing or invalid",
                                flush=True,
                            )

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
        "cov_xk_offdiag_rmse",
        "generation_seconds",
        "theta",
        "min_fdp_plus",
        "n_selected_classic",
        "sf_mean_ratio",
    ]
    metrics = [column for column in metric_candidates if column in valid.columns]
    if not valid.empty and metrics:
        summary = (
            valid.groupby(["omic", "generator", "timing"])[metrics]
            .agg(["mean", "std", "median"])
            .reset_index()
        )
        summary.to_csv(out_dir / "summary.csv", index=False)
        print("\n=== Summary ===")
        print(summary.to_string(index=False))
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
