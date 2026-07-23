#!/usr/bin/env python3
"""Summarize V9 real-X completion replacement results.

Posterior completion draws are repeated ordinary STABL runs.  Inference is
performed at the simulation-replicate level: metrics are first averaged across
completion draws, then paired against the median baseline within replicate.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from research_common import atomic_write_csv


PRIMARY_COMPLETIONS = (
    "oracle_complete",
    "median",
    "bayesianridge_mean",
    "bayesianridge_posterior",
)


def parse_selected(value: Any) -> set[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {int(item) for item in text.split(";") if item != ""}


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def mean_pairwise_jaccard(sets: Iterable[set[int]]) -> float:
    items = list(sets)
    if len(items) < 2:
        return 1.0
    values = [jaccard(a, b) for a, b in itertools.combinations(items, 2)]
    return float(np.mean(values)) if values else 1.0


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_boot: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_test(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    try:
        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def valid_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "error" in frame.columns:
        frame = frame[frame["error"].fillna("").eq("")].copy()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    selection = valid_frame(out_dir / "selection_results.csv")
    recovery = valid_frame(out_dir / "imputation_recovery.csv")
    diagnostics = valid_frame(out_dir / "draw_diagnostics.csv")
    if selection.empty:
        raise RuntimeError("No successful selection rows were found")

    group = [
        "dataset",
        "p_setting",
        "actual_p",
        "mechanism",
        "completion",
        "generator_label",
    ]
    metric_columns = [
        "fdp",
        "power",
        "n_selected",
        "true_positives",
        "false_positives",
        "estimated_fdp",
        "threshold",
    ]
    summary = (
        selection.groupby(group, dropna=False)[metric_columns]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if str(part) != "").rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    counts = selection.groupby(group, dropna=False).size().rename("n_draw_rows").reset_index()
    summary = summary.merge(counts, on=group, how="left")
    atomic_write_csv(summary, out_dir / "completion_replacement_summary.csv")

    # Average completion draws within each simulation replicate before inference.
    replicate_group = [
        "dataset",
        "p_setting",
        "actual_p",
        "replicate",
        "mechanism",
        "completion",
        "generator_label",
    ]
    replicate_metrics = (
        selection.groupby(replicate_group, dropna=False)[metric_columns]
        .mean()
        .reset_index()
    )
    atomic_write_csv(replicate_metrics, out_dir / "replicate_level_metrics.csv")

    contrast_rows: list[dict[str, Any]] = []
    comparison_keys = ["dataset", "p_setting", "actual_p", "mechanism", "generator_label"]
    for keys, block in replicate_metrics.groupby(comparison_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base_meta = dict(zip(comparison_keys, keys))
        comparisons = [
            ("median", "bayesianridge_mean"),
            ("median", "bayesianridge_posterior"),
            ("median", "oracle_complete"),
            ("bayesianridge_mean", "bayesianridge_posterior"),
        ]
        for reference, comparator in comparisons:
            ref = block[block["completion"] == reference].set_index("replicate")
            comp = block[block["completion"] == comparator].set_index("replicate")
            common = ref.index.intersection(comp.index)
            if common.empty:
                continue
            for metric in metric_columns:
                difference = comp.loc[common, metric].to_numpy() - ref.loc[common, metric].to_numpy()
                ci_low, ci_high = bootstrap_mean_ci(
                    difference,
                    seed=args.bootstrap_seed + len(contrast_rows),
                )
                contrast_rows.append(
                    {
                        **base_meta,
                        "reference": reference,
                        "comparator": comparator,
                        "metric": metric,
                        "n_replicates": int(len(common)),
                        "reference_mean": float(ref.loc[common, metric].mean()),
                        "comparator_mean": float(comp.loc[common, metric].mean()),
                        "mean_difference": float(difference.mean()),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "paired_wilcoxon_p": paired_test(difference),
                        "comparator_better_fraction": float(
                            np.mean(
                                difference < 0
                                if metric in {"fdp", "false_positives", "estimated_fdp"}
                                else difference > 0
                            )
                        ),
                    }
                )
    contrasts = pd.DataFrame(contrast_rows)
    atomic_write_csv(contrasts, out_dir / "paired_completion_contrasts.csv")

    # Oracle selection agreement uses the oracle row with the same replicate and draw.
    agreement_rows: list[dict[str, Any]] = []
    oracle_keys = [
        "dataset",
        "p_setting",
        "actual_p",
        "replicate",
        "mechanism",
        "completion_draw",
        "generator_label",
    ]
    indexed = selection.set_index(oracle_keys + ["completion"], drop=False)
    for _, row in selection.iterrows():
        if row["completion"] == "oracle_complete":
            continue
        key = tuple(row[col] for col in oracle_keys) + ("oracle_complete",)
        if key not in indexed.index:
            continue
        oracle = indexed.loc[key]
        if isinstance(oracle, pd.DataFrame):
            oracle = oracle.iloc[0]
        selected = parse_selected(row.get("selected_indices"))
        oracle_selected = parse_selected(oracle.get("selected_indices"))
        agreement_rows.append(
            {
                **{col: row[col] for col in oracle_keys},
                "completion": row["completion"],
                "oracle_selection_jaccard": jaccard(selected, oracle_selected),
                "oracle_selected_count_abs_difference": abs(len(selected) - len(oracle_selected)),
            }
        )
    agreement = pd.DataFrame(agreement_rows)
    atomic_write_csv(agreement, out_dir / "oracle_selection_agreement.csv")
    if not agreement.empty:
        agreement_summary = (
            agreement.groupby(
                ["dataset", "p_setting", "actual_p", "mechanism", "completion", "generator_label"],
                dropna=False,
            )[
                ["oracle_selection_jaccard", "oracle_selected_count_abs_difference"]
            ]
            .agg(["mean", "std", "median"])
            .reset_index()
        )
        agreement_summary.columns = [
            "_".join(str(part) for part in col if str(part) != "").rstrip("_")
            if isinstance(col, tuple)
            else str(col)
            for col in agreement_summary.columns
        ]
        atomic_write_csv(
            agreement_summary, out_dir / "oracle_selection_agreement_summary.csv"
        )

    # Within-replicate selection stability across repeated ordinary STABL runs.
    stability_rows: list[dict[str, Any]] = []
    stability_group = [
        "dataset",
        "p_setting",
        "actual_p",
        "replicate",
        "mechanism",
        "completion",
        "generator_label",
    ]
    for keys, block in selection.groupby(stability_group, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        stability_rows.append(
            {
                **dict(zip(stability_group, keys)),
                "n_completion_draws": int(block["completion_draw"].nunique()),
                "selection_pairwise_jaccard": mean_pairwise_jaccard(
                    parse_selected(value) for value in block["selected_indices"]
                ),
                "selected_count_sd": float(block["n_selected"].std(ddof=1))
                if len(block) > 1
                else 0.0,
                "fdp_sd": float(block["fdp"].std(ddof=1)) if len(block) > 1 else 0.0,
                "power_sd": float(block["power"].std(ddof=1)) if len(block) > 1 else 0.0,
            }
        )
    stability = pd.DataFrame(stability_rows)
    atomic_write_csv(stability, out_dir / "selection_stability.csv")
    stability_summary = (
        stability.groupby(
            ["dataset", "p_setting", "actual_p", "mechanism", "completion", "generator_label"],
            dropna=False,
        )[
            ["selection_pairwise_jaccard", "selected_count_sd", "fdp_sd", "power_sd"]
        ]
        .mean()
        .reset_index()
    )
    atomic_write_csv(stability_summary, out_dir / "selection_stability_summary.csv")

    recovery_summary = (
        recovery.groupby(
            ["dataset", "p_setting", "actual_p", "mechanism", "completion"],
            dropna=False,
        )
        .agg(
            masked_rmse_mean=("masked_rmse", "mean"),
            masked_mae_mean=("masked_mae", "mean"),
            sample_cov_error_mean=("sample_cov_fro_relative", "mean"),
            sample_corr_error_mean=("sample_corr_fro_relative", "mean"),
        )
        .reset_index()
    )
    atomic_write_csv(recovery_summary, out_dir / "imputation_recovery_summary.csv")

    diagnostic_summary = (
        diagnostics.groupby(
            ["dataset", "p_setting", "actual_p", "mechanism", "completion", "generator_label"],
            dropna=False,
        )
        .agg(
            pair_corr_mean=("pair_corr_mean", "mean"),
            s_relative_mean=("s_relative_mean", "mean"),
            cov_kk_error_mean=("cov_kk_fro_relative", "mean"),
            cov_xk_error_mean=("cov_xk_offdiag_rmse", "mean"),
        )
        .reset_index()
    )
    atomic_write_csv(diagnostic_summary, out_dir / "knockoff_diagnostics_summary.csv")

    # Compact figures, one chart per file.
    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    display = {
        "oracle_complete": "Oracle complete",
        "median": "Median",
        "bayesianridge_mean": "BayesianRidge mean",
        "bayesianridge_posterior": "BayesianRidge posterior",
    }
    order = [item for item in PRIMARY_COMPLETIONS if item in selection["completion"].unique()]

    for metric, ylabel, filename in [
        ("fdp", "Mean realized FDP", "01_fdp.png"),
        ("power", "Mean power", "02_power.png"),
    ]:
        plot_data = (
            selection.groupby(["mechanism", "completion"], dropna=False)[metric]
            .mean()
            .unstack("completion")
            .reindex(columns=order)
        )
        ax = plot_data.rename(columns=display).plot(kind="bar", figsize=(7.2, 4.6))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Missingness mechanism")
        ax.set_title(f"V9 real-X semisynthetic {metric}")
        ax.legend(title="Completion")
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=220, bbox_inches="tight")
        plt.close()

    if not agreement.empty:
        plot_data = (
            agreement.groupby(["mechanism", "completion"], dropna=False)[
                "oracle_selection_jaccard"
            ]
            .mean()
            .unstack("completion")
            .reindex(columns=[c for c in order if c != "oracle_complete"])
        )
        ax = plot_data.rename(columns=display).plot(kind="bar", figsize=(7.2, 4.6))
        ax.set_ylabel("Mean Jaccard with complete-data STABL")
        ax.set_xlabel("Missingness mechanism")
        ax.set_title("V9 oracle-selection agreement")
        ax.legend(title="Completion")
        plt.tight_layout()
        plt.savefig(figures / "03_oracle_jaccard.png", dpi=220, bbox_inches="tight")
        plt.close()

    plot_data = (
        stability.groupby(["mechanism", "completion"], dropna=False)[
            "selection_pairwise_jaccard"
        ]
        .mean()
        .unstack("completion")
        .reindex(columns=order)
    )
    ax = plot_data.rename(columns=display).plot(kind="bar", figsize=(7.2, 4.6))
    ax.set_ylabel("Mean pairwise Jaccard across repeated runs")
    ax.set_xlabel("Missingness mechanism")
    ax.set_title("V9 selection stability")
    ax.legend(title="Completion")
    plt.tight_layout()
    plt.savefig(figures / "04_selection_stability.png", dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Summary written to {out_dir}")
    print("Primary files:")
    for name in (
        "completion_replacement_summary.csv",
        "replicate_level_metrics.csv",
        "paired_completion_contrasts.csv",
        "oracle_selection_agreement_summary.csv",
        "selection_stability_summary.csv",
        "imputation_recovery_summary.csv",
        "knockoff_diagnostics_summary.csv",
    ):
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
