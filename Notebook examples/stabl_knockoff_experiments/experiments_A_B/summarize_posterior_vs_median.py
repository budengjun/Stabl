#!/usr/bin/env python3
"""Summarize the posterior-versus-median original-STABL experiment.

Primary analysis:
  * original STABL FDP+ argmin rule (stabl_min)
  * realistic sample-Sigma Gaussian equicorrelated knockoffs
  * paired contrasts versus median within replicate/mechanism/draw

Secondary analysis:
  * distribution-recovery diagnostics
  * known-Sigma oracle checks where theoretically aligned
  * agreement with the oracle-complete STABL selection set
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

COMPLETION_LABELS = {
    "oracle_complete": "Oracle complete",
    "median": "Median",
    "exact_gaussian_posterior": "Exact posterior",
    "bayesianridge_posterior": "BayesianRidge posterior",
}

PRIMARY_COMPLETIONS = (
    "median",
    "exact_gaussian_posterior",
    "bayesianridge_posterior",
)


def _safe_wilcoxon(diff: pd.Series) -> float:
    values = pd.to_numeric(diff, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    try:
        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def _paired_rows(
    frame: pd.DataFrame,
    *,
    index_columns: list[str],
    condition_column: str,
    baseline: str,
    comparators: Iterable[str],
    metrics: Iterable[str],
    source: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_values, group in frame.groupby(
        [c for c in index_columns if c not in {"replicate", "completion", "draw"}],
        dropna=False,
    ):
        grouping_columns = [
            c for c in index_columns if c not in {"replicate", "completion", "draw"}
        ]
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_meta = dict(zip(grouping_columns, group_values))
        pair_index = [c for c in ("replicate", "draw") if c in group.columns]
        base = group[group[condition_column] == baseline].set_index(pair_index)
        if base.empty:
            continue
        for comparator in comparators:
            comp = group[group[condition_column] == comparator].set_index(pair_index)
            common = base.index.intersection(comp.index)
            if len(common) == 0:
                continue
            for metric in metrics:
                if metric not in base.columns or metric not in comp.columns:
                    continue
                b = pd.to_numeric(base.loc[common, metric], errors="coerce")
                c = pd.to_numeric(comp.loc[common, metric], errors="coerce")
                diff = c - b
                valid = pd.concat([b.rename("baseline"), c.rename("comparator")], axis=1).dropna()
                if valid.empty:
                    continue
                diff_valid = valid["comparator"] - valid["baseline"]
                rows.append(
                    {
                        "source": source,
                        **group_meta,
                        "baseline": baseline,
                        "comparator": comparator,
                        "metric": metric,
                        "n_pairs": int(len(valid)),
                        "baseline_mean": float(valid["baseline"].mean()),
                        "comparator_mean": float(valid["comparator"].mean()),
                        "mean_difference_comparator_minus_median": float(diff_valid.mean()),
                        "median_difference_comparator_minus_median": float(diff_valid.median()),
                        "wilcoxon_p": _safe_wilcoxon(diff_valid),
                        "comparator_better_fraction": float(
                            np.mean(_metric_improvement(metric, diff_valid.to_numpy(dtype=float)))
                        ),
                    }
                )
    return rows


def _metric_improvement(metric: str, diff: np.ndarray) -> np.ndarray:
    lower_better = {
        "masked_rmse",
        "masked_mae",
        "sample_cov_fro_relative",
        "sample_corr_fro_relative",
        "true_cov_fro_relative",
        "cov_kk_fro_relative",
        "cov_xk_offdiag_rmse",
        "pair_corr_mean",
        "estimated_fdp",
        "fdp",
        "false_positives",
        "oracle_selected_count_abs_difference",
    }
    higher_better = {
        "s_relative_mean",
        "power",
        "true_positives",
        "oracle_selection_jaccard",
    }
    if metric in lower_better:
        return diff < 0
    if metric in higher_better:
        return diff > 0
    return np.full(diff.shape, False)


def _parse_score_name(path: Path) -> dict[str, object] | None:
    pattern = re.compile(
        r"rep_(?P<rep>\d+)__(?P<mechanism>[^_]+)__(?P<completion>.+?)__"
        r"(?P<generator>.+?)__draw_(?P<draw>\d+)\.npz$"
    )
    match = pattern.match(path.name)
    if not match:
        return None
    return {
        "replicate": int(match.group("rep")),
        "mechanism": match.group("mechanism"),
        "completion": match.group("completion"),
        "generator_label": match.group("generator"),
        "draw": int(match.group("draw")),
    }


def _selection_masks(out_dir: Path, selection: pd.DataFrame) -> pd.DataFrame:
    primary = selection[
        selection["selection_rule"].eq("stabl_min")
        & selection.get("error", pd.Series("", index=selection.index)).fillna("").eq("")
    ].copy()
    thresholds = {
        (
            int(row.replicate),
            str(row.mechanism),
            str(row.completion),
            str(row.generator_label),
            int(row.draw),
        ): float(row.threshold)
        for row in primary.itertuples()
    }
    records: list[dict[str, object]] = []
    for score_path in sorted((out_dir / "scores").glob("*.npz")):
        meta = _parse_score_name(score_path)
        if meta is None:
            continue
        key = (
            meta["replicate"],
            meta["mechanism"],
            meta["completion"],
            meta["generator_label"],
            meta["draw"],
        )
        threshold = thresholds.get(key)
        if threshold is None or not np.isfinite(threshold):
            continue
        with np.load(score_path, allow_pickle=False) as payload:
            real_scores = np.asarray(payload["real_scores"], dtype=float)
        if real_scores.ndim == 3 and real_scores.shape[0] == 1:
            real_scores = real_scores[0]
        max_scores = np.max(real_scores, axis=-1)
        mask = max_scores >= threshold
        records.append({**meta, "selected_mask": mask, "n_selected_recomputed": int(mask.sum())})
    return pd.DataFrame(records)


def _oracle_agreement(mask_frame: pd.DataFrame) -> pd.DataFrame:
    if mask_frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    key_cols = ["replicate", "mechanism", "generator_label", "draw"]
    for key, group in mask_frame.groupby(key_cols, dropna=False):
        oracle_rows = group[group["completion"].eq("oracle_complete")]
        if oracle_rows.empty:
            continue
        oracle = np.asarray(oracle_rows.iloc[0]["selected_mask"], dtype=bool)
        for row in group.itertuples():
            selected = np.asarray(row.selected_mask, dtype=bool)
            union = int(np.logical_or(oracle, selected).sum())
            intersection = int(np.logical_and(oracle, selected).sum())
            jaccard = 1.0 if union == 0 else intersection / union
            rows.append(
                {
                    **dict(zip(key_cols, key if isinstance(key, tuple) else (key,))),
                    "completion": row.completion,
                    "oracle_selection_jaccard": float(jaccard),
                    "oracle_selected_count": int(oracle.sum()),
                    "completion_selected_count": int(selected.sum()),
                    "oracle_selected_count_abs_difference": int(abs(selected.sum() - oracle.sum())),
                    "oracle_intersection": intersection,
                    "oracle_union": union,
                }
            )
    return pd.DataFrame(rows)


def _save_figures(
    out_dir: Path,
    recovery: pd.DataFrame,
    diagnostics: pd.DataFrame,
    primary: pd.DataFrame,
    oracle_agreement: pd.DataFrame,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    noto_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if noto_path.exists():
        font_manager.fontManager.addfont(str(noto_path))
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    order = [
        "oracle_complete",
        "median",
        "exact_gaussian_posterior",
        "bayesianridge_posterior",
    ]

    grouped = (
        recovery.groupby(["completion", "mechanism"], as_index=False)[
            "true_cov_fro_relative"
        ].mean()
    )
    pivot = grouped.pivot(index="completion", columns="mechanism", values="true_cov_fro_relative").reindex(order)
    ax = pivot.rename(index=COMPLETION_LABELS).plot(kind="bar", figsize=(8.3, 4.8))
    ax.set_ylabel("True covariance relative error")
    ax.set_xlabel("")
    ax.set_title("Completion recovery under MCAR and MAR")
    ax.legend(title="Missingness")
    ax.figure.tight_layout()
    ax.figure.savefig(fig_dir / "01_completion_true_covariance.png", dpi=220)
    plt.close(ax.figure)

    d = diagnostics[diagnostics["generator_label"].eq("equicorr")]
    diag_group = (
        d.groupby(["completion", "mechanism"], as_index=False)[
            ["cov_kk_fro_relative", "cov_xk_offdiag_rmse"]
        ].mean()
    )
    for metric, filename, ylabel in [
        ("cov_kk_fro_relative", "02_knockoff_covariance_match.png", "Cov(X_tilde) relative error"),
        ("cov_xk_offdiag_rmse", "03_cross_covariance_match.png", "Cross-covariance off-diagonal RMSE"),
    ]:
        pivot = diag_group.pivot(index="completion", columns="mechanism", values=metric).reindex(order)
        ax = pivot.rename(index=COMPLETION_LABELS).plot(kind="bar", figsize=(8.3, 4.8))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_title("Sample-Sigma knockoff diagnostic")
        ax.legend(title="Missingness")
        ax.figure.tight_layout()
        ax.figure.savefig(fig_dir / filename, dpi=220)
        plt.close(ax.figure)

    primary_group = (
        primary.groupby(["completion", "mechanism"], as_index=False)[["fdp", "power"]].mean()
    )
    for metric, filename, ylabel in [
        ("fdp", "04_stabl_min_fdp.png", "Mean realized FDP"),
        ("power", "05_stabl_min_power.png", "Mean power"),
    ]:
        pivot = primary_group.pivot(index="completion", columns="mechanism", values=metric).reindex(order)
        ax = pivot.rename(index=COMPLETION_LABELS).plot(kind="bar", figsize=(8.3, 4.8))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_title("Original STABL (stabl_min), sample-Sigma generator")
        ax.legend(title="Missingness")
        ax.figure.tight_layout()
        ax.figure.savefig(fig_dir / filename, dpi=220)
        plt.close(ax.figure)

    if not oracle_agreement.empty:
        a = oracle_agreement[oracle_agreement["generator_label"].eq("equicorr")]
        grouped = (
            a.groupby(["completion", "mechanism"], as_index=False)[
                "oracle_selection_jaccard"
            ].mean()
        )
        pivot = grouped.pivot(index="completion", columns="mechanism", values="oracle_selection_jaccard").reindex(order)
        ax = pivot.rename(index=COMPLETION_LABELS).plot(kind="bar", figsize=(8.3, 4.8))
        ax.set_ylabel("Mean Jaccard with oracle-complete selection")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.set_title("Selection agreement with complete-data STABL")
        ax.legend(title="Missingness")
        ax.figure.tight_layout()
        ax.figure.savefig(fig_dir / "06_oracle_selection_agreement.png", dpi=220)
        plt.close(ax.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    selection = pd.read_csv(out_dir / "selection_results.csv")
    recovery = pd.read_csv(out_dir / "imputation_recovery.csv")
    diagnostics = pd.read_csv(out_dir / "draw_diagnostics.csv")

    valid_selection = selection[
        selection.get("error", pd.Series("", index=selection.index)).fillna("").eq("")
    ].copy()
    primary = valid_selection[
        valid_selection["selection_rule"].eq("stabl_min")
        & valid_selection["generator_label"].eq("equicorr")
    ].copy()

    primary_summary = (
        primary.groupby(["mechanism", "completion"], as_index=False)
        .agg(
            n_success=("fdp", "size"),
            estimated_fdp_mean=("estimated_fdp", "mean"),
            selected_mean=("n_selected", "mean"),
            true_positives_mean=("true_positives", "mean"),
            false_positives_mean=("false_positives", "mean"),
            empirical_fdr=("fdp", "mean"),
            fdp_sd=("fdp", "std"),
            power_mean=("power", "mean"),
            power_sd=("power", "std"),
        )
        .sort_values(["mechanism", "completion"])
    )
    primary_summary.to_csv(out_dir / "primary_stabl_min_summary.csv", index=False)

    contrast_rows: list[dict[str, object]] = []
    contrast_rows.extend(
        _paired_rows(
            recovery,
            index_columns=["mechanism", "replicate", "completion"],
            condition_column="completion",
            baseline="median",
            comparators=("exact_gaussian_posterior", "bayesianridge_posterior"),
            metrics=(
                "masked_rmse",
                "masked_mae",
                "sample_cov_fro_relative",
                "sample_corr_fro_relative",
                "true_cov_fro_relative",
            ),
            source="completion_recovery",
        )
    )
    contrast_rows.extend(
        _paired_rows(
            diagnostics[diagnostics["generator_label"].eq("equicorr")],
            index_columns=["mechanism", "generator_label", "replicate", "draw", "completion"],
            condition_column="completion",
            baseline="median",
            comparators=("exact_gaussian_posterior", "bayesianridge_posterior"),
            metrics=(
                "pair_corr_mean",
                "s_relative_mean",
                "cov_kk_fro_relative",
                "cov_xk_offdiag_rmse",
            ),
            source="knockoff_diagnostics_sample_sigma",
        )
    )
    contrast_rows.extend(
        _paired_rows(
            primary,
            index_columns=["mechanism", "generator_label", "replicate", "draw", "completion"],
            condition_column="completion",
            baseline="median",
            comparators=("exact_gaussian_posterior", "bayesianridge_posterior"),
            metrics=(
                "estimated_fdp",
                "n_selected",
                "true_positives",
                "false_positives",
                "fdp",
                "power",
            ),
            source="original_stabl_sample_sigma",
        )
    )

    masks = _selection_masks(out_dir, selection)
    agreement = _oracle_agreement(masks)
    if not agreement.empty:
        agreement.drop(columns=["selected_mask"], errors="ignore").to_csv(
            out_dir / "oracle_selection_agreement.csv", index=False
        )
        agreement_summary = (
            agreement.groupby(["mechanism", "completion", "generator_label"], as_index=False)
            .agg(
                n_pairs=("oracle_selection_jaccard", "size"),
                oracle_selection_jaccard_mean=("oracle_selection_jaccard", "mean"),
                oracle_selection_jaccard_sd=("oracle_selection_jaccard", "std"),
                selected_count_abs_difference_mean=(
                    "oracle_selected_count_abs_difference",
                    "mean",
                ),
            )
        )
        agreement_summary.to_csv(out_dir / "oracle_selection_agreement_summary.csv", index=False)
        contrast_rows.extend(
            _paired_rows(
                agreement[agreement["generator_label"].eq("equicorr")],
                index_columns=["mechanism", "generator_label", "replicate", "draw", "completion"],
                condition_column="completion",
                baseline="median",
                comparators=("exact_gaussian_posterior", "bayesianridge_posterior"),
                metrics=(
                    "oracle_selection_jaccard",
                    "oracle_selected_count_abs_difference",
                ),
                source="oracle_selection_agreement",
            )
        )

    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(out_dir / "paired_completion_contrasts.csv", index=False)

    if not args.no_figures:
        _save_figures(out_dir, recovery, diagnostics, primary, agreement)

    print("Primary analysis: original STABL stabl_min with sample-Sigma equicorr")
    print(primary_summary.to_string(index=False))
    print(f"\nWrote: {out_dir / 'paired_completion_contrasts.csv'}")
    if not agreement.empty:
        print(f"Wrote: {out_dir / 'oracle_selection_agreement_summary.csv'}")
    if not args.no_figures:
        print(f"Figures: {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
