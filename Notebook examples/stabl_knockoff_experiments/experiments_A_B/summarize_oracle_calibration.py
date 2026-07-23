#!/usr/bin/env python3
"""Summarize V8 real-X oracle difficulty calibration cells."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_common import atomic_write_csv


def parse_selected(value: Any) -> set[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value).strip()
    return {int(item) for item in text.split(";") if item} if text else set()


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return 1.0 if not union else float(len(a & b) / len(union))


def mean_pairwise_jaccard(values: Iterable[Any]) -> float:
    sets = [parse_selected(value) for value in values]
    if len(sets) < 2:
        return 1.0
    scores = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(scores)) if scores else 1.0


def distance_to_interval(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-power-low", type=float, default=0.30)
    parser.add_argument("--target-power-high", type=float, default=0.60)
    parser.add_argument("--soft-fdp-ceiling", type=float, default=0.65)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    rows: list[pd.DataFrame] = []
    for selection_path in sorted(out_dir.glob("p_*__strength_*/selection_results.csv")):
        cell_dir = selection_path.parent
        config = json.loads((cell_dir / "config.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(selection_path)
        if "error" in frame.columns:
            frame = frame[frame["error"].fillna("").eq("")].copy()
        frame["calibration_signal_strength"] = float(config["signal_strength"])
        frame["calibration_cell"] = cell_dir.name
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No calibration selection files found under {out_dir}")

    selection = pd.concat(rows, ignore_index=True)
    atomic_write_csv(selection, out_dir / "oracle_calibration_all_draws.csv")

    replicate_group = [
        "dataset",
        "p_setting",
        "actual_p",
        "calibration_signal_strength",
        "replicate",
        "generator_label",
    ]
    metrics = [
        "fdp",
        "power",
        "n_selected",
        "true_positives",
        "false_positives",
        "estimated_fdp",
        "threshold",
        "fdp_exceeds_target",
    ]
    replicate_metrics = (
        selection.groupby(replicate_group, dropna=False)[metrics].mean().reset_index()
    )
    atomic_write_csv(
        replicate_metrics, out_dir / "oracle_calibration_replicate_metrics.csv"
    )

    summary_group = [
        "dataset",
        "p_setting",
        "actual_p",
        "calibration_signal_strength",
        "generator_label",
    ]
    summary = (
        replicate_metrics.groupby(summary_group, dropna=False)
        .agg(
            empirical_fdr=("fdp", "mean"),
            empirical_fdr_sd=("fdp", "std"),
            power_mean=("power", "mean"),
            power_sd=("power", "std"),
            selected_mean=("n_selected", "mean"),
            true_positives_mean=("true_positives", "mean"),
            false_positives_mean=("false_positives", "mean"),
            estimated_fdp_mean=("estimated_fdp", "mean"),
            fdp_exceedance_probability=("fdp_exceeds_target", "mean"),
            n_replicates=("replicate", "nunique"),
        )
        .reset_index()
    )

    stability_rows: list[dict[str, Any]] = []
    stability_group = [
        "dataset",
        "p_setting",
        "actual_p",
        "calibration_signal_strength",
        "replicate",
        "generator_label",
    ]
    for keys, block in selection.groupby(stability_group, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        stability_rows.append(
            {
                **dict(zip(stability_group, keys)),
                "n_draws": int(block["completion_draw"].nunique()),
                "selection_pairwise_jaccard": mean_pairwise_jaccard(
                    block["selected_indices"]
                ),
                "selected_count_sd": float(block["n_selected"].std(ddof=1))
                if len(block) > 1
                else 0.0,
            }
        )
    stability = pd.DataFrame(stability_rows)
    atomic_write_csv(stability, out_dir / "oracle_calibration_stability.csv")
    stability_summary = (
        stability.groupby(summary_group, dropna=False)
        .agg(
            selection_stability_mean=("selection_pairwise_jaccard", "mean"),
            selected_count_sd_mean=("selected_count_sd", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(stability_summary, on=summary_group, how="left")

    summary["power_interval_penalty"] = summary["power_mean"].map(
        lambda value: distance_to_interval(
            float(value), args.target_power_low, args.target_power_high
        )
    )
    summary["fdp_soft_penalty"] = np.maximum(
        0.0, summary["empirical_fdr"] - args.soft_fdp_ceiling
    )
    summary["floor_penalty"] = (summary["selected_mean"] < 3).astype(float) * 0.25
    summary["calibration_rank_score"] = (
        summary["power_interval_penalty"]
        + summary["fdp_soft_penalty"]
        + summary["floor_penalty"]
    )
    summary = summary.sort_values(
        ["calibration_rank_score", "empirical_fdr", "power_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary["meets_power_window"] = summary["power_mean"].between(
        args.target_power_low, args.target_power_high
    )
    summary["meets_soft_fdp_ceiling"] = summary["empirical_fdr"] <= args.soft_fdp_ceiling
    summary["recommended_for_replacement_pilot"] = False
    if not summary.empty:
        summary.loc[0, "recommended_for_replacement_pilot"] = True
    atomic_write_csv(summary, out_dir / "oracle_calibration_summary.csv")
    atomic_write_csv(
        summary.head(1), out_dir / "oracle_calibration_recommendation.csv"
    )

    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    for metric, ylabel, filename in [
        ("power_mean", "Mean oracle STABL power", "01_oracle_power.png"),
        ("empirical_fdr", "Mean oracle STABL realized FDP", "02_oracle_fdp.png"),
        (
            "selection_stability_mean",
            "Mean pairwise Jaccard across downstream draws",
            "03_oracle_stability.png",
        ),
    ]:
        plot = summary.pivot_table(
            index="calibration_signal_strength",
            columns="actual_p",
            values=metric,
            aggfunc="first",
        ).sort_index()
        ax = plot.plot(kind="bar", figsize=(7.2, 4.6))
        ax.set_xlabel("Signal strength")
        ax.set_ylabel(ylabel)
        ax.set_title("V8 SSI real-X oracle difficulty calibration")
        ax.legend(title="p")
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=220, bbox_inches="tight")
        plt.close()

    print("\nOracle calibration summary")
    print(
        summary[
            [
                "rank",
                "actual_p",
                "calibration_signal_strength",
                "power_mean",
                "empirical_fdr",
                "selected_mean",
                "selection_stability_mean",
                "recommended_for_replacement_pilot",
            ]
        ].to_string(index=False)
    )
    if not summary.empty:
        best = summary.iloc[0]
        print(
            "\nRecommended next pilot setting: "
            f"p={int(best['actual_p'])}, "
            f"signal_strength={best['calibration_signal_strength']:g}. "
            "This is a calibration ranking, not proof that the setting is easy or FDR-controlled."
        )


if __name__ == "__main__":
    main()
