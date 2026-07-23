#!/usr/bin/env python3
"""Summarize Experiment A V6 nested posterior-pooling results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
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
PRIMARY_COMPLETIONS = (
    "median",
    "exact_gaussian_posterior",
    "bayesianridge_posterior",
)


def safe_wilcoxon(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or np.allclose(finite, 0.0):
        return 1.0
    try:
        return float(wilcoxon(finite, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, B: int = 10000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(B, array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_contrast_rows(
    frame: pd.DataFrame,
    *,
    baseline_completion: str = "median",
    comparators: Iterable[str] = (
        "exact_gaussian_posterior",
        "bayesianridge_posterior",
    ),
    metrics: Iterable[str] = (
        "estimated_fdp",
        "n_selected",
        "true_positives",
        "false_positives",
        "fdp",
        "power",
    ),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_columns = [
        "mechanism", "generator_label", "pool_size", "pooling_aggregator",
    ]
    for group_values, group in frame.groupby(group_columns, dropna=False):
        meta = dict(zip(group_columns, group_values))
        base = group[group.completion.eq(baseline_completion)].set_index("replicate")
        if base.empty:
            continue
        for comparator in comparators:
            comp = group[group.completion.eq(comparator)].set_index("replicate")
            common = base.index.intersection(comp.index)
            for metric in metrics:
                if len(common) == 0 or metric not in base or metric not in comp:
                    continue
                b = pd.to_numeric(base.loc[common, metric], errors="coerce")
                c = pd.to_numeric(comp.loc[common, metric], errors="coerce")
                valid = pd.concat([b.rename("baseline"), c.rename("comparator")], axis=1).dropna()
                if valid.empty:
                    continue
                diff = (valid.comparator - valid.baseline).to_numpy(dtype=float)
                ci_low, ci_high = bootstrap_mean_ci(
                    diff,
                    seed=20260717 + int(meta["pool_size"]) * 101 + len(rows),
                )
                rows.append({
                    **meta,
                    "contrast_type": "completion_at_fixed_M",
                    "baseline": baseline_completion,
                    "comparator": comparator,
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


def within_completion_pooling_rows(
    frame: pd.DataFrame,
    *,
    baseline_M: int = 1,
    metrics: Iterable[str] = (
        "estimated_fdp",
        "n_selected",
        "true_positives",
        "false_positives",
        "fdp",
        "power",
        "component_selection_jaccard_mean",
    ),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_columns = [
        "mechanism", "generator_label", "pooling_aggregator", "completion",
    ]
    for group_values, group in frame.groupby(group_columns, dropna=False):
        meta = dict(zip(group_columns, group_values))
        base = group[group.pool_size.eq(baseline_M)].set_index("replicate")
        if base.empty:
            continue
        for target_M in sorted(int(v) for v in group.pool_size.unique() if int(v) != baseline_M):
            comp = group[group.pool_size.eq(target_M)].set_index("replicate")
            common = base.index.intersection(comp.index)
            for metric in metrics:
                if len(common) == 0 or metric not in base or metric not in comp:
                    continue
                b = pd.to_numeric(base.loc[common, metric], errors="coerce")
                c = pd.to_numeric(comp.loc[common, metric], errors="coerce")
                valid = pd.concat([b.rename("M1"), c.rename("target")], axis=1).dropna()
                if valid.empty:
                    continue
                diff = (valid.target - valid.M1).to_numpy(dtype=float)
                ci_low, ci_high = bootstrap_mean_ci(
                    diff,
                    seed=20260718 + target_M * 103 + len(rows),
                )
                rows.append({
                    **meta,
                    "contrast_type": "within_completion_pooling_gain",
                    "baseline_pool_size": baseline_M,
                    "target_pool_size": target_M,
                    "metric": metric,
                    "n_pairs": int(len(diff)),
                    "baseline_mean": float(valid.M1.mean()),
                    "target_mean": float(valid.target.mean()),
                    "mean_difference": float(diff.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wilcoxon_p": safe_wilcoxon(diff),
                })
    return rows


def difference_in_differences_rows(
    frame: pd.DataFrame,
    *,
    baseline_completion: str = "median",
    posterior_completions: Iterable[str] = (
        "exact_gaussian_posterior",
        "bayesianridge_posterior",
    ),
    baseline_M: int = 1,
    metrics: Iterable[str] = ("fdp", "power", "n_selected", "false_positives"),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_columns = ["mechanism", "generator_label", "pooling_aggregator"]
    for group_values, group in frame.groupby(group_columns, dropna=False):
        meta = dict(zip(group_columns, group_values))
        target_Ms = sorted(int(v) for v in group.pool_size.unique() if int(v) != baseline_M)
        for posterior in posterior_completions:
            for target_M in target_Ms:
                subsets = {}
                for completion, M, label in [
                    (posterior, target_M, "post_M"),
                    (posterior, baseline_M, "post_1"),
                    (baseline_completion, target_M, "med_M"),
                    (baseline_completion, baseline_M, "med_1"),
                ]:
                    subsets[label] = group[
                        group.completion.eq(completion) & group.pool_size.eq(M)
                    ].set_index("replicate")
                common = set.intersection(*(set(item.index) for item in subsets.values()))
                if not common:
                    continue
                common_index = sorted(common)
                for metric in metrics:
                    if any(metric not in item for item in subsets.values()):
                        continue
                    values = {
                        label: pd.to_numeric(item.loc[common_index, metric], errors="coerce")
                        for label, item in subsets.items()
                    }
                    valid = pd.DataFrame(values).dropna()
                    if valid.empty:
                        continue
                    did = (
                        valid.post_M - valid.post_1
                        - (valid.med_M - valid.med_1)
                    ).to_numpy(dtype=float)
                    ci_low, ci_high = bootstrap_mean_ci(
                        did,
                        seed=20260719 + target_M * 107 + len(rows),
                    )
                    rows.append({
                        **meta,
                        "contrast_type": "posterior_specific_pooling_gain_DID",
                        "posterior_completion": posterior,
                        "baseline_completion": baseline_completion,
                        "baseline_pool_size": baseline_M,
                        "target_pool_size": target_M,
                        "metric": metric,
                        "n_pairs": int(len(did)),
                        "mean_difference_in_differences": float(did.mean()),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "wilcoxon_p": safe_wilcoxon(did),
                    })
    return rows


def oracle_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = [
        "replicate", "mechanism", "generator_label", "pool_size", "pooling_aggregator",
    ]
    for key_values, group in frame.groupby(keys, dropna=False):
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
                **dict(zip(keys, key_values)),
                "completion": row.completion,
                "oracle_selection_jaccard": 1.0 if union == 0 else intersection / union,
                "oracle_selected_count": int(oracle_mask.sum()),
                "completion_selected_count": int(selected.sum()),
                "oracle_selected_count_abs_difference": int(abs(selected.sum() - oracle_mask.sum())),
                "oracle_intersection": intersection,
                "oracle_union": union,
            })
    return pd.DataFrame(rows)


def save_figures(out_dir: Path, primary: pd.DataFrame, agreement: pd.DataFrame) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    noto_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if noto_path.exists():
        font_manager.fontManager.addfont(str(noto_path))
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    plot_data = primary[
        primary.generator_label.eq("equicorr")
        & primary.pooling_aggregator.eq("mean")
    ].copy()
    completion_order = [
        "oracle_complete", "median", "exact_gaussian_posterior", "bayesianridge_posterior"
    ]

    for mechanism in sorted(plot_data.mechanism.unique()):
        subset = plot_data[plot_data.mechanism.eq(mechanism)]
        for metric, ylabel, filename in [
            ("fdp", "Mean realized FDP", f"01_{mechanism}_fdp_vs_M.png"),
            ("power", "Mean power", f"02_{mechanism}_power_vs_M.png"),
            ("n_selected", "Mean selected features", f"03_{mechanism}_selected_vs_M.png"),
            (
                "component_selection_jaccard_mean",
                "Mean component-set pairwise Jaccard",
                f"04_{mechanism}_component_stability_vs_M.png",
            ),
        ]:
            grouped = subset.groupby(["completion", "pool_size"], as_index=False)[metric].mean()
            fig, ax = plt.subplots(figsize=(7.8, 4.8))
            for completion in completion_order:
                line = grouped[grouped.completion.eq(completion)].sort_values("pool_size")
                if line.empty:
                    continue
                ax.plot(
                    line.pool_size,
                    line[metric],
                    marker="o",
                    label=COMPLETION_LABELS[completion],
                )
            ax.set_xlabel("Pool size M")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{mechanism}: {ylabel} versus pooled component count")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / filename, dpi=220)
            plt.close(fig)

    if not agreement.empty:
        a = agreement[
            agreement.generator_label.eq("equicorr")
            & agreement.pooling_aggregator.eq("mean")
        ]
        for mechanism in sorted(a.mechanism.unique()):
            subset = a[a.mechanism.eq(mechanism)]
            grouped = subset.groupby(["completion", "pool_size"], as_index=False)[
                "oracle_selection_jaccard"
            ].mean()
            fig, ax = plt.subplots(figsize=(7.8, 4.8))
            for completion in completion_order[1:]:
                line = grouped[grouped.completion.eq(completion)].sort_values("pool_size")
                if line.empty:
                    continue
                ax.plot(
                    line.pool_size,
                    line.oracle_selection_jaccard,
                    marker="o",
                    label=COMPLETION_LABELS[completion],
                )
            ax.set_xlabel("Pool size M")
            ax.set_ylabel("Mean Jaccard with oracle-complete selection")
            ax.set_ylim(0, 1)
            ax.set_title(f"{mechanism}: oracle-selection agreement versus M")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"05_{mechanism}_oracle_jaccard_vs_M.png", dpi=220)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    pooled = pd.read_csv(out_dir / "pooled_selection_results.csv")
    valid = pooled[pooled.get("error", pd.Series("", index=pooled.index)).fillna("").eq("")].copy()
    primary = valid[
        valid.selection_rule.eq("stabl_min")
        & valid.generator_label.eq("equicorr")
        & valid.pooling_aggregator.eq("mean")
    ].copy()

    summary = (
        valid.groupby(
            [
                "mechanism", "completion", "generator_label",
                "pool_size", "pooling_aggregator", "selection_rule",
            ],
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
            component_selection_jaccard_mean=("component_selection_jaccard_mean", "mean"),
            real_component_score_sd_mean=("real_component_max_score_sd_mean", "mean"),
            artificial_component_score_sd_mean=("artificial_component_max_score_sd_mean", "mean"),
        )
        .sort_values(["mechanism", "completion", "generator_label", "pool_size"])
    )
    summary.to_csv(out_dir / "pooling_summary.csv", index=False)

    contrast_rows: list[dict[str, object]] = []
    contrast_rows.extend(paired_contrast_rows(primary))
    contrast_rows.extend(within_completion_pooling_rows(primary))
    contrast_rows.extend(difference_in_differences_rows(primary))
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(out_dir / "pooling_paired_contrasts.csv", index=False)

    agreement = oracle_agreement(valid)
    agreement.to_csv(out_dir / "pooling_oracle_selection_agreement.csv", index=False)
    if not agreement.empty:
        agreement_summary = (
            agreement.groupby(
                [
                    "mechanism", "completion", "generator_label",
                    "pool_size", "pooling_aggregator",
                ],
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
            out_dir / "pooling_oracle_selection_agreement_summary.csv", index=False
        )

    if not args.no_figures:
        save_figures(out_dir, primary, agreement)

    print("\nPrimary V6 pooled STABL summary")
    columns = [
        "mechanism", "completion", "pool_size", "n_success",
        "empirical_fdr", "power_mean", "selected_mean",
        "component_selection_jaccard_mean",
    ]
    print(summary[
        summary.generator_label.eq("equicorr")
        & summary.pooling_aggregator.eq("mean")
    ][columns].to_string(index=False))
    print(f"\nSaved V6 summaries to {out_dir}")


if __name__ == "__main__":
    main()
