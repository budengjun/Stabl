#!/usr/bin/env python3
"""Summarize V6.1 validity-aware pooling and compare it with V6 naive pooling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from posterior_pooling_common import decode_indices

COMPLETION_LABELS = {
    "oracle_complete": "Oracle complete",
    "median": "Median",
    "exact_gaussian_posterior": "Exact posterior",
    "bayesianridge_posterior": "BayesianRidge posterior",
}
RULE_LABELS = {
    "mean_path_naive": "Naive mean path",
    "mean_real_component_artificial_count": "Validity-aware count",
    "component_selection_vote": "Component vote",
}


def safe_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.allclose(values, 0):
        return 1.0
    try:
        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, B: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(B, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_rows(
    frame: pd.DataFrame,
    *,
    baseline_filter: dict[str, object],
    comparator_filter: dict[str, object],
    label: str,
    metrics: Iterable[str],
    group_columns: Iterable[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_values, group in frame.groupby(list(group_columns), dropna=False):
        meta = dict(zip(group_columns, group_values if isinstance(group_values, tuple) else (group_values,)))
        baseline = group.copy()
        comparator = group.copy()
        for key, value in baseline_filter.items():
            baseline = baseline[baseline[key].eq(value)]
        for key, value in comparator_filter.items():
            comparator = comparator[comparator[key].eq(value)]
        baseline = baseline.set_index("replicate")
        comparator = comparator.set_index("replicate")
        common = baseline.index.intersection(comparator.index)
        if common.empty:
            continue
        for metric in metrics:
            if metric not in baseline or metric not in comparator:
                continue
            b = pd.to_numeric(baseline.loc[common, metric], errors="coerce")
            c = pd.to_numeric(comparator.loc[common, metric], errors="coerce")
            valid = pd.concat([b.rename("baseline"), c.rename("comparator")], axis=1).dropna()
            if valid.empty:
                continue
            diff = (valid.comparator - valid.baseline).to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean_ci(diff, seed=20260720 + len(rows) * 31)
            rows.append({
                **meta,
                "contrast": label,
                "metric": metric,
                "n_pairs": int(len(diff)),
                "baseline_mean": float(valid.baseline.mean()),
                "comparator_mean": float(valid.comparator.mean()),
                "mean_difference": float(diff.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wilcoxon_p": safe_wilcoxon(diff),
            })
    return rows


def oracle_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = [
        "replicate", "mechanism", "generator_label", "pool_size",
        "pooling_aggregator", "pooling_rule", "vote_fraction",
    ]
    for values, group in frame.groupby(keys, dropna=False):
        oracle = group[group.completion.eq("oracle_complete")]
        if oracle.empty:
            continue
        oracle_row = oracle.iloc[0]
        p = int(oracle_row.p)
        oracle_mask = decode_indices(oracle_row.selected_indices, p)
        for row in group.itertuples():
            selected = decode_indices(row.selected_indices, p)
            union = int(np.logical_or(oracle_mask, selected).sum())
            intersection = int(np.logical_and(oracle_mask, selected).sum())
            rows.append({
                **dict(zip(keys, values)),
                "completion": row.completion,
                "oracle_selection_jaccard": 1.0 if union == 0 else intersection / union,
                "oracle_selected_count_abs_difference": int(
                    abs(selected.sum() - oracle_mask.sum())
                ),
            })
    return pd.DataFrame(rows)


def save_figures(out_dir: Path, frame: pd.DataFrame, agreement: pd.DataFrame) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    primary = frame[
        frame.generator_label.eq("equicorr")
        & frame.pooling_aggregator.eq("mean")
        & frame.pool_size.eq(frame.pool_size.max())
        & ~frame.pooling_rule.eq("component_selection_vote")
    ].copy()
    primary["rule_label"] = primary.pooling_rule.map(RULE_LABELS)
    primary["completion_label"] = primary.completion.map(COMPLETION_LABELS)

    for mechanism in sorted(primary.mechanism.unique()):
        subset = primary[primary.mechanism.eq(mechanism)]
        grouped = subset.groupby(["completion_label", "rule_label"], as_index=False).agg(
            realized_fdp=("fdp", "mean"), estimated_fdp=("estimated_fdp", "mean")
        )
        labels = [f"{row.completion_label}\n{row.rule_label}" for row in grouped.itertuples()]
        x = np.arange(len(grouped))
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        width = 0.38
        ax.bar(x - width / 2, grouped.realized_fdp, width, label="Realized FDP")
        ax.bar(x + width / 2, grouped.estimated_fdp, width, label="Estimated FDP+")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(0, max(1.0, float(grouped.realized_fdp.max()) * 1.15))
        ax.set_ylabel("FDP")
        ax.set_title(f"{mechanism}: calibration at M={int(frame.pool_size.max())}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"01_{mechanism}_realized_vs_estimated_fdp.png", dpi=220)
        plt.close(fig)

        for metric, ylabel, filename in [
            ("power", "Mean power", f"02_{mechanism}_power_by_rule.png"),
            ("n_selected", "Mean selected features", f"03_{mechanism}_selected_by_rule.png"),
        ]:
            grouped_metric = subset.groupby(["completion_label", "rule_label"], as_index=False)[metric].mean()
            pivot = grouped_metric.pivot(index="completion_label", columns="rule_label", values=metric)
            ax = pivot.plot(kind="bar", figsize=(8.8, 4.8))
            ax.set_ylabel(ylabel)
            ax.set_title(f"{mechanism}: {ylabel} at M={int(frame.pool_size.max())}")
            ax.set_xlabel("")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / filename, dpi=220)
            plt.close()

    if not agreement.empty:
        max_M = int(agreement.pool_size.max())
        subset = agreement[
            agreement.pool_size.eq(max_M)
            & agreement.generator_label.eq("equicorr")
            & agreement.pooling_aggregator.eq("mean")
        ].copy()
        subset["rule_label"] = subset.pooling_rule.map(RULE_LABELS)
        subset["completion_label"] = subset.completion.map(COMPLETION_LABELS)
        for mechanism in sorted(subset.mechanism.unique()):
            a = subset[subset.mechanism.eq(mechanism)]
            grouped = a.groupby(["completion_label", "rule_label"], as_index=False)[
                "oracle_selection_jaccard"
            ].mean()
            pivot = grouped.pivot(index="completion_label", columns="rule_label", values="oracle_selection_jaccard")
            ax = pivot.plot(kind="bar", figsize=(8.8, 4.8))
            ax.set_ylim(0, 1)
            ax.set_ylabel("Mean Jaccard with oracle-complete selection")
            ax.set_title(f"{mechanism}: oracle agreement at M={max_M}")
            ax.set_xlabel("")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / f"04_{mechanism}_oracle_jaccard_by_rule.png", dpi=220)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    path = out_dir / "validity_aware_selection_results.csv"
    frame = pd.read_csv(path)
    valid = frame[frame.get("error", pd.Series("", index=frame.index)).fillna("").eq("")].copy()
    valid["calibration_gap"] = valid.fdp - valid.estimated_fdp

    summary = (
        valid.groupby(
            ["mechanism", "completion", "generator_label", "pool_size",
             "pooling_aggregator", "pooling_rule", "vote_fraction"],
            as_index=False,
        )
        .agg(
            n_success=("fdp", "size"),
            empirical_fdr=("fdp", "mean"),
            fdp_sd=("fdp", "std"),
            power_mean=("power", "mean"),
            power_sd=("power", "std"),
            selected_mean=("n_selected", "mean"),
            true_positives_mean=("true_positives", "mean"),
            false_positives_mean=("false_positives", "mean"),
            estimated_fdp_mean=("estimated_fdp", "mean"),
            calibration_gap_mean=("calibration_gap", "mean"),
            mean_component_estimated_fdp=("mean_component_estimated_fdp", "mean"),
            component_selection_jaccard_mean=("component_selection_jaccard_mean", "mean"),
            artificial_dilution_mean=("artificial_dilution_at_threshold", "mean"),
        )
        .sort_values(["mechanism", "pool_size", "pooling_rule", "completion"])
    )
    summary.to_csv(out_dir / "validity_aware_summary.csv", index=False)

    contrast_rows: list[dict[str, object]] = []
    nonvote = valid[~valid.pooling_rule.eq("component_selection_vote")]
    contrast_rows.extend(paired_rows(
        nonvote,
        baseline_filter={"pooling_rule": "mean_path_naive"},
        comparator_filter={"pooling_rule": "mean_real_component_artificial_count"},
        label="validity_aware_minus_naive_at_fixed_completion_M",
        metrics=("estimated_fdp", "fdp", "power", "n_selected", "false_positives"),
        group_columns=("mechanism", "generator_label", "pool_size", "pooling_aggregator", "completion"),
    ))
    for rule in ["mean_path_naive", "mean_real_component_artificial_count"]:
        contrast_rows.extend(paired_rows(
            nonvote[nonvote.pooling_rule.eq(rule)],
            baseline_filter={"completion": "median"},
            comparator_filter={"completion": "exact_gaussian_posterior"},
            label=f"exact_posterior_minus_median__{rule}",
            metrics=("estimated_fdp", "fdp", "power", "n_selected", "false_positives"),
            group_columns=("mechanism", "generator_label", "pool_size", "pooling_aggregator"),
        ))
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(out_dir / "validity_aware_paired_contrasts.csv", index=False)

    agreement = oracle_agreement(valid)
    agreement.to_csv(out_dir / "validity_aware_oracle_agreement.csv", index=False)
    if not agreement.empty:
        agreement_summary = (
            agreement.groupby(
                ["mechanism", "completion", "generator_label", "pool_size",
                 "pooling_aggregator", "pooling_rule", "vote_fraction"],
                as_index=False,
            )
            .agg(
                n_success=("oracle_selection_jaccard", "size"),
                oracle_selection_jaccard_mean=("oracle_selection_jaccard", "mean"),
                oracle_selection_jaccard_sd=("oracle_selection_jaccard", "std"),
                selected_count_abs_difference_mean=("oracle_selected_count_abs_difference", "mean"),
            )
        )
        agreement_summary.to_csv(
            out_dir / "validity_aware_oracle_agreement_summary.csv", index=False
        )

    if not args.no_figures:
        save_figures(out_dir, valid, agreement)

    display = summary[
        summary.generator_label.eq("equicorr")
        & summary.pooling_aggregator.eq("mean")
        & summary.pool_size.eq(summary.pool_size.max())
    ][[
        "mechanism", "completion", "pooling_rule", "vote_fraction",
        "empirical_fdr", "estimated_fdp_mean", "calibration_gap_mean",
        "power_mean", "selected_mean",
    ]]
    print("\nV6.1 validity-aware pooling summary at maximum M")
    print(display.to_string(index=False))
    print(f"\nSaved V6.1 summaries to {out_dir}")


if __name__ == "__main__":
    main()
