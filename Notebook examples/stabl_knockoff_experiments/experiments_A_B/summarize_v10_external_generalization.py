#!/usr/bin/env python3
"""Summarize V10 external omic generalization blocks.

All inferential summaries use the simulation replicate as the unit.  Repeated
ordinary STABL runs are averaged within replicate before paired BR-mean minus
Median contrasts are computed.  The 10-replicate default is a directional
pilot and is never labeled confirmatory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_common import atomic_write_csv
from v10_common import block_output_dirs, favorable_direction, is_favorable
from v91_common import (
    dataframe_to_markdown,
    holm_adjust,
    paired_wilcoxon_p,
    percentile_bootstrap_mean_ci,
)


def parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 1 for item in parsed):
        raise ValueError("k values must be positive")
    return parsed


def valid_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "error" in frame.columns:
        frame = frame[frame["error"].fillna("").eq("")].copy()
    return frame


def build_contrast_rows(
    frame: pd.DataFrame,
    *,
    metrics: list[str],
    metadata_columns: list[str],
    bootstrap_seed: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, block in frame.groupby(metadata_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(metadata_columns, keys))
        median = block[block["completion"].eq("median")].set_index("replicate")
        br = block[block["completion"].eq("bayesianridge_mean")].set_index("replicate")
        common = median.index.intersection(br.index)
        if common.empty:
            continue
        for metric in metrics:
            if metric not in median.columns or metric not in br.columns:
                continue
            difference = (
                br.loc[common, metric].to_numpy(dtype=float)
                - median.loc[common, metric].to_numpy(dtype=float)
            )
            ci_low, ci_high = percentile_bootstrap_mean_ci(
                difference,
                seed=bootstrap_seed + len(rows),
                n_bootstrap=n_bootstrap,
            )
            direction = favorable_direction(metric)
            rows.append(
                {
                    **meta,
                    "metric": metric,
                    "contrast": "bayesianridge_mean minus median",
                    "favorable_direction": direction,
                    "n_pairs": int(len(common)),
                    "median_mean": float(median.loc[common, metric].mean()),
                    "bayesianridge_mean": float(br.loc[common, metric].mean()),
                    "mean_difference": float(np.mean(difference)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "paired_wilcoxon_p": paired_wilcoxon_p(difference),
                    "fraction_replicates_favorable": float(
                        np.mean(difference < 0)
                        if direction == "lower"
                        else np.mean(difference > 0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--k-values", default="5,10,15,20")
    parser.add_argument("--pilot-power-loss-guard", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    prereg_path = out_dir / "v10_preregistration.json"
    if not prereg_path.exists():
        raise FileNotFoundError(prereg_path)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    k_values = parse_int_list(args.k_values)
    block_dirs = block_output_dirs(out_dir)
    if not block_dirs:
        raise RuntimeError(f"No completed V10 block outputs found under {out_dir}")

    selection_parts: list[pd.DataFrame] = []
    agreement_parts: list[pd.DataFrame] = []
    ranking_parts: list[pd.DataFrame] = []
    recovery_parts: list[pd.DataFrame] = []
    diagnostics_parts: list[pd.DataFrame] = []
    integrity_rows: list[dict[str, Any]] = []

    expected_reps = int(prereg["n_replicates"])
    expected_draws = int(prereg["n_completion_draws"])
    expected_mechanisms = len(prereg["missingness_mechanisms"])
    expected_selection_rows = expected_reps * expected_draws * expected_mechanisms * 3

    for block_dir in block_dirs:
        label = block_dir.name
        selection = valid_csv(block_dir / "selection_results.csv")
        selection["block_label"] = label
        selection_parts.append(selection)

        selection_keys = [
            "replicate",
            "mechanism",
            "completion",
            "completion_draw",
            "generator_label",
        ]
        duplicate_selection = int(selection.duplicated(selection_keys).sum())
        integrity_rows.append(
            {
                "block_label": label,
                "table": "selection_results",
                "expected_rows": expected_selection_rows,
                "successful_rows": int(len(selection)),
                "duplicate_keys": duplicate_selection,
                "complete": bool(
                    len(selection) == expected_selection_rows
                    and duplicate_selection == 0
                ),
            }
        )

        agreement_path = block_dir / "oracle_selection_agreement.csv"
        if agreement_path.exists():
            agreement = pd.read_csv(agreement_path)
            agreement["block_label"] = label
            agreement_parts.append(agreement)

        ranking_path = (
            block_dir
            / "v10_threshold_independent_ranking"
            / "ranking_replicate_metrics.csv"
        )
        if ranking_path.exists():
            ranking = pd.read_csv(ranking_path)
            ranking["block_label"] = label
            ranking_parts.append(ranking)
            ranking_integrity = pd.read_csv(
                ranking_path.parent / "ranking_integrity_audit.csv"
            )
            integrity_rows.append(
                {
                    "block_label": label,
                    "table": "threshold_independent_scores",
                    "expected_rows": expected_selection_rows,
                    "successful_rows": int(ranking_integrity["n_score_files"].sum()),
                    "duplicate_keys": 0,
                    "complete": bool(ranking_integrity["complete_pairing"].all()),
                }
            )

        recovery = valid_csv(block_dir / "imputation_recovery.csv")
        recovery["block_label"] = label
        recovery_parts.append(recovery)
        diagnostics = valid_csv(block_dir / "draw_diagnostics.csv")
        diagnostics["block_label"] = label
        diagnostics_parts.append(diagnostics)

    integrity = pd.DataFrame(integrity_rows)
    atomic_write_csv(integrity, out_dir / "v10_integrity_audit.csv")

    selection_all = pd.concat(selection_parts, ignore_index=True)
    atomic_write_csv(selection_all, out_dir / "v10_selection_all_draws.csv")
    selection_metrics = [
        "fdp",
        "power",
        "n_selected",
        "true_positives",
        "false_positives",
        "estimated_fdp",
        "threshold",
    ]
    replicate_selection = (
        selection_all.groupby(
            [
                "block_label",
                "dataset",
                "actual_p",
                "replicate",
                "mechanism",
                "completion",
                "generator_label",
            ],
            as_index=False,
        )[selection_metrics]
        .mean()
    )

    if agreement_parts:
        agreement_all = pd.concat(agreement_parts, ignore_index=True)
        atomic_write_csv(agreement_all, out_dir / "v10_oracle_agreement_all_draws.csv")
        replicate_agreement = (
            agreement_all.groupby(
                [
                    "block_label",
                    "dataset",
                    "actual_p",
                    "replicate",
                    "mechanism",
                    "completion",
                    "generator_label",
                ],
                as_index=False,
            )[
                [
                    "oracle_selection_jaccard",
                    "oracle_selected_count_abs_difference",
                ]
            ]
            .mean()
        )
        replicate_selection = replicate_selection.merge(
            replicate_agreement,
            on=[
                "block_label",
                "dataset",
                "actual_p",
                "replicate",
                "mechanism",
                "completion",
                "generator_label",
            ],
            how="left",
        )
    atomic_write_csv(
        replicate_selection, out_dir / "v10_replicate_selection_metrics.csv"
    )

    primary_selection_metrics = ["fdp", "power", "oracle_selection_jaccard"]
    selection_contrasts = build_contrast_rows(
        replicate_selection,
        metrics=primary_selection_metrics,
        metadata_columns=["block_label", "dataset", "actual_p", "mechanism", "generator_label"],
        bootstrap_seed=args.bootstrap_seed,
        n_bootstrap=args.n_bootstrap,
    )
    atomic_write_csv(selection_contrasts, out_dir / "v10_primary_selection_contrasts.csv")

    ranking_contrasts = pd.DataFrame()
    ranking_all = pd.DataFrame()
    ranking_metrics_primary = [
        *[f"precision_at_{k}" for k in k_values],
        "average_precision",
        "median_signal_rank",
    ]
    ranking_metrics_reference = [
        "support_ranking_auroc",
        "mean_signal_score",
        "mean_null_score",
    ]
    if ranking_parts:
        ranking_all = pd.concat(ranking_parts, ignore_index=True)
        atomic_write_csv(ranking_all, out_dir / "v10_ranking_replicate_metrics.csv")
        ranking_contrasts = build_contrast_rows(
            ranking_all,
            metrics=ranking_metrics_primary + ranking_metrics_reference,
            metadata_columns=["block_label", "mechanism", "generator_label"],
            bootstrap_seed=args.bootstrap_seed + 100_000,
            n_bootstrap=args.n_bootstrap,
        )
        adjusted_parts = []
        for _, block in ranking_contrasts.groupby(
            ["block_label", "mechanism"], sort=False
        ):
            block = block.copy()
            primary_mask = block["metric"].isin(ranking_metrics_primary)
            block["holm_p_primary_within_block_mechanism"] = np.nan
            block.loc[
                primary_mask, "holm_p_primary_within_block_mechanism"
            ] = holm_adjust(block.loc[primary_mask, "paired_wilcoxon_p"].to_numpy())
            adjusted_parts.append(block)
        ranking_contrasts = pd.concat(adjusted_parts, ignore_index=True)
        atomic_write_csv(ranking_contrasts, out_dir / "v10_ranking_contrasts.csv")

    recovery_all = pd.concat(recovery_parts, ignore_index=True)
    diagnostics_all = pd.concat(diagnostics_parts, ignore_index=True)
    atomic_write_csv(recovery_all, out_dir / "v10_imputation_recovery_all.csv")
    atomic_write_csv(diagnostics_all, out_dir / "v10_knockoff_diagnostics_all.csv")

    recovery_summary = (
        recovery_all.groupby(["block_label", "mechanism", "completion"], as_index=False)
        .agg(
            masked_rmse=("masked_rmse", "mean"),
            masked_mae=("masked_mae", "mean"),
            sample_cov_error=("sample_cov_fro_relative", "mean"),
            sample_corr_error=("sample_corr_fro_relative", "mean"),
        )
    )
    diagnostic_summary = (
        diagnostics_all.groupby(
            ["block_label", "mechanism", "completion", "generator_label"],
            as_index=False,
        )
        .agg(
            pair_corr=("pair_corr_mean", "mean"),
            s_relative=("s_relative_mean", "mean"),
            cov_kk_error=("cov_kk_fro_relative", "mean"),
            cross_cov_rmse=("cov_xk_offdiag_rmse", "mean"),
        )
    )
    atomic_write_csv(recovery_summary, out_dir / "v10_imputation_recovery_summary.csv")
    atomic_write_csv(diagnostic_summary, out_dir / "v10_knockoff_diagnostics_summary.csv")

    # Pilot progression gate, evaluated separately for each block and mechanism.
    gate_rows: list[dict[str, Any]] = []
    for block_label in sorted(selection_contrasts["block_label"].unique()):
        for mechanism in prereg["missingness_mechanisms"]:
            selection_block = selection_contrasts[
                selection_contrasts["block_label"].eq(block_label)
                & selection_contrasts["mechanism"].eq(mechanism)
            ]
            ranking_block = ranking_contrasts[
                ranking_contrasts.get("block_label", pd.Series(dtype=str)).eq(block_label)
                & ranking_contrasts.get("mechanism", pd.Series(dtype=str)).eq(mechanism)
            ] if not ranking_contrasts.empty else pd.DataFrame()

            def delta(frame: pd.DataFrame, metric: str) -> float:
                row = frame[frame["metric"].eq(metric)]
                return float(row.iloc[0]["mean_difference"]) if len(row) == 1 else float("nan")

            delta_fdp = delta(selection_block, "fdp")
            delta_power = delta(selection_block, "power")
            delta_jaccard = delta(selection_block, "oracle_selection_jaccard")
            precision_deltas = [
                delta(ranking_block, f"precision_at_{k}") for k in k_values
            ] if not ranking_block.empty else []
            criteria = {
                "fdp_point_favorable": bool(np.isfinite(delta_fdp) and delta_fdp < 0),
                "oracle_jaccard_point_favorable": bool(
                    np.isfinite(delta_jaccard) and delta_jaccard > 0
                ),
                "all_precision_at_k_point_favorable": bool(
                    precision_deltas and all(value > 0 for value in precision_deltas)
                ),
                "no_large_power_loss": bool(
                    np.isfinite(delta_power)
                    and delta_power >= -args.pilot_power_loss_guard
                ),
            }
            pass_gate = bool(all(criteria.values()))
            gate_rows.append(
                {
                    "block_label": block_label,
                    "mechanism": mechanism,
                    "delta_fdp": delta_fdp,
                    "delta_power": delta_power,
                    "delta_oracle_jaccard": delta_jaccard,
                    **{f"delta_precision_at_{k}": value for k, value in zip(k_values, precision_deltas)},
                    **criteria,
                    "pilot_progression_gate": pass_gate,
                    "recommended_action": (
                        "advance_to_50_replicates"
                        if pass_gate
                        else "retain_as_directional_or_redesign_before_expansion"
                    ),
                    "interpretation": "pilot_only_not_confirmatory",
                }
            )
    gate = pd.DataFrame(gate_rows)
    atomic_write_csv(gate, out_dir / "v10_pilot_progression_gate.csv")

    consistency_rows: list[dict[str, Any]] = []
    combined_contrasts = pd.concat(
        [selection_contrasts, ranking_contrasts], ignore_index=True, sort=False
    ) if not ranking_contrasts.empty else selection_contrasts.copy()
    for (mechanism, metric), block in combined_contrasts.groupby(
        ["mechanism", "metric"], dropna=False
    ):
        favorable = [
            is_favorable(metric, float(value)) for value in block["mean_difference"]
        ]
        consistency_rows.append(
            {
                "mechanism": mechanism,
                "metric": metric,
                "n_blocks": int(len(block)),
                "n_blocks_favorable": int(sum(favorable)),
                "all_blocks_favorable": bool(all(favorable)),
                "equal_block_mean_difference": float(block["mean_difference"].mean()),
                "scientific_role": "cross_block_directional_replication",
            }
        )
    consistency = pd.DataFrame(consistency_rows)
    atomic_write_csv(consistency, out_dir / "v10_cross_block_consistency.csv")

    if not integrity["complete"].all():
        overall = "incomplete_outputs"
    elif gate["pilot_progression_gate"].all():
        overall = "advance_all_block_mechanism_cells_to_50_replicates"
    elif gate["pilot_progression_gate"].any():
        overall = "block_or_mechanism_specific_support"
    else:
        overall = "pilot_does_not_support_immediate_expansion"
    decision = pd.DataFrame(
        [
            {
                "overall_pilot_decision": overall,
                "n_gate_cells": int(len(gate)),
                "n_gate_cells_passed": int(gate["pilot_progression_gate"].sum()),
                "confirmatory_claim_allowed": False,
                "next_step": (
                    "Run 50 fresh replicates only for cells passing the preregistered pilot gate"
                ),
            }
        ]
    )
    atomic_write_csv(decision, out_dir / "v10_pilot_decision.csv")

    figures = out_dir / "figures_v10"
    figures.mkdir(exist_ok=True)
    for metric in ["fdp", "power", "oracle_selection_jaccard"]:
        block = selection_contrasts[selection_contrasts["metric"].eq(metric)].copy()
        if block.empty:
            continue
        block["label"] = block["block_label"] + " | " + block["mechanism"]
        block = block.sort_values(["block_label", "mechanism"])
        fig, ax = plt.subplots(figsize=(8.0, max(3.4, 0.55 * len(block) + 1.5)))
        y = np.arange(len(block))
        ax.errorbar(
            block["mean_difference"],
            y,
            xerr=np.vstack(
                [
                    block["mean_difference"] - block["bootstrap_ci_low"],
                    block["bootstrap_ci_high"] - block["mean_difference"],
                ]
            ),
            fmt="o",
            capsize=3,
        )
        ax.axvline(0.0, linewidth=1)
        ax.set_yticks(y, block["label"])
        ax.set_xlabel(f"BayesianRidge mean minus Median: {metric}")
        ax.set_title(f"V10 external generalization pilot: {metric}")
        fig.tight_layout()
        fig.savefig(figures / f"contrast_{metric}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    report = [
        "# V10 External Omic Generalization Pilot",
        "",
        "This is a directional pilot. The simulation replicate is the inference unit; repeated STABL runs are averaged within replicate.",
        "",
        "## Integrity",
        "",
        dataframe_to_markdown(integrity, index=False),
        "",
        "## Primary selection contrasts",
        "",
        dataframe_to_markdown(selection_contrasts, index=False),
        "",
        "## Threshold-independent ranking contrasts",
        "",
        dataframe_to_markdown(ranking_contrasts, index=False) if not ranking_contrasts.empty else "Scores were not saved; ranking analysis unavailable.",
        "",
        "## Pilot progression gate",
        "",
        dataframe_to_markdown(gate, index=False),
        "",
        "## Cross-block directional consistency",
        "",
        dataframe_to_markdown(consistency, index=False),
        "",
        "## Overall pilot decision",
        "",
        dataframe_to_markdown(decision, index=False),
    ]
    (out_dir / "v10_external_generalization_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(f"[V10 summary] integrity complete={bool(integrity['complete'].all())}")
    print(f"[V10 summary] overall pilot decision={overall}")
    print(f"[V10 summary] outputs written to {out_dir}")


if __name__ == "__main__":
    main()
