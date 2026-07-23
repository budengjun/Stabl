#!/usr/bin/env python3
"""Summarize V11 missingness-rate and block-missingness stress results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_common import atomic_write_csv
from v11_common import favorable_direction, paired_contrast, rate_label, valid_rows
from v91_common import dataframe_to_markdown, holm_adjust


PRIMARY_METRICS = (
    "fdp",
    "power",
    "oracle_selection_jaccard",
    "average_precision",
    "support_ranking_auroc",
)
SUPPORTIVE_METRICS = (
    "precision_at_5",
    "precision_at_10",
    "precision_at_15",
    "precision_at_20",
    "median_signal_rank",
    "mean_signal_score",
    "mean_null_score",
    "n_selected",
    "true_positives",
    "false_positives",
    "estimated_fdp",
    "masked_rmse",
    "masked_mae",
    "sample_cov_fro_relative",
    "sample_corr_fro_relative",
    "pair_corr_mean",
    "s_relative_mean",
    "pair_corr_abs_error_to_oracle",
    "s_relative_abs_error_to_oracle",
    "cov_kk_fro_relative",
    "cov_xk_offdiag_rmse",
)


def aggregate_source(
    frame: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    available = [metric for metric in metrics if metric in frame.columns]
    keys = ["replicate", "mechanism", "completion"]
    return frame.groupby(keys, as_index=False)[available].mean()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Summarize V11 missingness stress results.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=60260730)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--power-loss-guard", type=float, default=0.05)
    parser.add_argument("--fdp-worsening-guard", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    prereg = json.loads((out_dir / "v11_preregistration.json").read_text(encoding="utf-8"))
    blocks = prereg["blocks"]
    rates = tuple(float(item) for item in prereg["missing_rates"])
    mechanisms = tuple(str(item) for item in prereg["missingness_mechanisms"])
    n_replicates = int(prereg["n_replicates"])
    n_draws = int(prereg["n_completion_draws"])
    analysis_status = str(prereg["analysis_status"])

    audit_rows: list[dict[str, Any]] = []
    all_replicates: list[pd.DataFrame] = []

    for block in blocks:
        block_label = str(block["label"])
        for rate in rates:
            cell_dir = out_dir / block_label / rate_label(rate)
            selection_raw = valid_rows(cell_dir / "selection_results.csv")
            replicate = pd.read_csv(cell_dir / "replicate_level_metrics.csv")
            oracle_raw = pd.read_csv(cell_dir / "oracle_selection_agreement.csv")
            recovery_raw = valid_rows(cell_dir / "imputation_recovery.csv")
            diagnostic_raw = valid_rows(cell_dir / "draw_diagnostics.csv")
            ranking_path = cell_dir / "v11_threshold_independent_ranking" / "ranking_replicate_metrics.csv"
            if not ranking_path.exists():
                raise FileNotFoundError(f"Missing V11 ranking output: {ranking_path}")
            ranking = pd.read_csv(ranking_path)

            expected_selection = n_replicates * len(mechanisms) * 3 * n_draws
            expected_replicate = n_replicates * len(mechanisms) * 3
            expected_oracle = n_replicates * len(mechanisms) * 2 * n_draws
            table_specs = [
                (
                    "selection_results",
                    selection_raw,
                    expected_selection,
                    ["replicate", "mechanism", "completion", "completion_draw", "generator_label"],
                ),
                (
                    "replicate_level_metrics",
                    replicate,
                    expected_replicate,
                    ["replicate", "mechanism", "completion", "generator_label"],
                ),
                (
                    "oracle_selection_agreement",
                    oracle_raw,
                    expected_oracle,
                    ["replicate", "mechanism", "completion", "completion_draw", "generator_label"],
                ),
                (
                    "ranking_replicate_metrics",
                    ranking,
                    expected_replicate,
                    ["replicate", "mechanism", "completion", "generator_label"],
                ),
                (
                    "imputation_recovery",
                    recovery_raw,
                    expected_selection,
                    ["replicate", "mechanism", "completion", "completion_draw"],
                ),
                (
                    "draw_diagnostics",
                    diagnostic_raw,
                    expected_selection,
                    ["replicate", "mechanism", "completion", "completion_draw", "generator_label"],
                ),
            ]
            for table_name, frame, expected, key in table_specs:
                duplicates = int(frame.duplicated(key).sum())
                audit_rows.append(
                    {
                        "block_label": block_label,
                        "missing_rate": rate,
                        "table": table_name,
                        "expected_rows": expected,
                        "successful_rows": len(frame),
                        "duplicate_keys": duplicates,
                        "complete": bool(len(frame) == expected and duplicates == 0),
                    }
                )

            oracle_rep = aggregate_source(
                oracle_raw,
                ["oracle_selection_jaccard", "oracle_selected_count_abs_difference"],
            )
            recovery_rep = aggregate_source(
                recovery_raw,
                [
                    "missing_rate_actual",
                    "missing_rate_row_sd",
                    "missing_rate_column_sd",
                    "max_row_missing_fraction",
                    "max_column_missing_fraction",
                    "n_fully_observed_rows",
                    "n_fully_observed_columns",
                    "masked_rmse",
                    "masked_mae",
                    "sample_cov_fro_relative",
                    "sample_corr_fro_relative",
                ],
            )
            diagnostics_rep = aggregate_source(
                diagnostic_raw,
                [
                    "pair_corr_mean",
                    "s_relative_mean",
                    "cov_kk_fro_relative",
                    "cov_xk_offdiag_rmse",
                ],
            )
            ranking_columns = [
                "precision_at_5",
                "precision_at_10",
                "precision_at_15",
                "precision_at_20",
                "average_precision",
                "median_signal_rank",
                "support_ranking_auroc",
                "mean_signal_score",
                "mean_null_score",
            ]
            ranking_rep = ranking[["replicate", "mechanism", "completion", *ranking_columns]].copy()

            merged = replicate.merge(
                oracle_rep,
                on=["replicate", "mechanism", "completion"],
                how="left",
            ).merge(
                ranking_rep,
                on=["replicate", "mechanism", "completion"],
                how="left",
            ).merge(
                recovery_rep,
                on=["replicate", "mechanism", "completion"],
                how="left",
            ).merge(
                diagnostics_rep,
                on=["replicate", "mechanism", "completion"],
                how="left",
            )
            merged["block_label"] = block_label
            merged["missing_rate"] = rate
            merged["analysis_status"] = analysis_status

            for geometry_metric, derived_name in (
                ("pair_corr_mean", "pair_corr_abs_error_to_oracle"),
                ("s_relative_mean", "s_relative_abs_error_to_oracle"),
            ):
                oracle_geometry = (
                    merged[merged["completion"].eq("oracle_complete")]
                    .set_index(["replicate", "mechanism"])[geometry_metric]
                )
                merged[derived_name] = [
                    abs(float(value) - float(oracle_geometry.loc[(int(rep), str(mechanism))]))
                    for rep, mechanism, value in zip(
                        merged["replicate"], merged["mechanism"], merged[geometry_metric]
                    )
                ]
            all_replicates.append(merged)

    audit = pd.DataFrame(audit_rows)
    atomic_write_csv(audit, out_dir / "v11_integrity_audit.csv")
    replicate_metrics = pd.concat(all_replicates, ignore_index=True)
    atomic_write_csv(replicate_metrics, out_dir / "v11_replicate_metrics.csv")

    all_metrics = [metric for metric in (*PRIMARY_METRICS, *SUPPORTIVE_METRICS) if metric in replicate_metrics.columns]
    contrast_rows: list[dict[str, Any]] = []
    seed_offset = 0
    for (block_label, missing_rate, mechanism), cell in replicate_metrics.groupby(
        ["block_label", "missing_rate", "mechanism"], sort=True
    ):
        for metric in all_metrics:
            row = paired_contrast(
                cell,
                index_columns=["replicate"],
                metric=metric,
                seed=args.bootstrap_seed + seed_offset,
                n_bootstrap=args.n_bootstrap,
            )
            row.update(
                {
                    "block_label": block_label,
                    "missing_rate": float(missing_rate),
                    "mechanism": mechanism,
                    "analysis_status": analysis_status,
                }
            )
            contrast_rows.append(row)
            seed_offset += 1
    contrasts = pd.DataFrame(contrast_rows)

    contrasts["holm_p_primary_within_block_mechanism"] = np.nan
    primary_family = {"oracle_selection_jaccard", "average_precision"}
    for (block_label, mechanism), indices in contrasts[
        contrasts["metric"].isin(primary_family)
    ].groupby(["block_label", "mechanism"]).groups.items():
        values = contrasts.loc[indices, "paired_wilcoxon_p"].to_numpy(dtype=float)
        contrasts.loc[indices, "holm_p_primary_within_block_mechanism"] = holm_adjust(values)
    atomic_write_csv(contrasts, out_dir / "v11_paired_contrasts.csv")

    trajectory_metrics = [metric for metric in all_metrics if metric not in {"pair_corr_abs_error_to_oracle", "s_relative_abs_error_to_oracle"}]
    trajectory = (
        replicate_metrics.groupby(
            ["block_label", "missing_rate", "mechanism", "completion"], as_index=False
        )[trajectory_metrics]
        .mean()
    )
    atomic_write_csv(trajectory, out_dir / "v11_rate_trajectory_summary.csv")

    gate_rows: list[dict[str, Any]] = []
    for (block_label, missing_rate, mechanism), cell in contrasts.groupby(
        ["block_label", "missing_rate", "mechanism"], sort=True
    ):
        indexed = cell.set_index("metric")
        required = {"fdp", "power", "oracle_selection_jaccard", "average_precision"}
        if not required.issubset(indexed.index):
            raise RuntimeError(f"Missing gate metrics for {(block_label, missing_rate, mechanism)}")
        fdp = indexed.loc["fdp"]
        power = indexed.loc["power"]
        oracle = indexed.loc["oracle_selection_jaccard"]
        ap = indexed.loc["average_precision"]
        precision_rows = indexed.loc[[name for name in indexed.index if str(name).startswith("precision_at_")]]
        all_precision_favorable = bool(
            not precision_rows.empty and (precision_rows["mean_difference"] > 0).all()
        )
        directional = bool(
            oracle["mean_difference"] > 0
            and ap["mean_difference"] > 0
            and power["mean_difference"] >= -args.power_loss_guard
            and fdp["mean_difference"] <= args.fdp_worsening_guard
        )
        strong = bool(
            oracle["bootstrap_ci_low"] > 0
            and ap["bootstrap_ci_low"] > 0
            and power["bootstrap_ci_low"] >= -args.power_loss_guard
            and fdp["bootstrap_ci_high"] <= args.fdp_worsening_guard
        )
        adverse = bool(
            oracle["mean_difference"] < 0
            and ap["mean_difference"] < 0
            and (
                power["mean_difference"] < -args.power_loss_guard
                or fdp["mean_difference"] > args.fdp_worsening_guard
            )
        )
        classification = (
            "strong_support"
            if strong
            else "directional_support"
            if directional
            else "adverse"
            if adverse
            else "mixed_or_boundary"
        )
        gate_rows.append(
            {
                "block_label": block_label,
                "missing_rate": float(missing_rate),
                "mechanism": mechanism,
                "analysis_status": analysis_status,
                "delta_fdp": float(fdp["mean_difference"]),
                "delta_power": float(power["mean_difference"]),
                "delta_oracle_jaccard": float(oracle["mean_difference"]),
                "delta_average_precision": float(ap["mean_difference"]),
                "all_precision_at_k_point_estimates_favorable": all_precision_favorable,
                "directional_gate_passed": directional,
                "strong_ci_gate_passed": strong,
                "cell_classification": classification,
                "scientific_decision_allowed": analysis_status == "stress",
            }
        )
    gate = pd.DataFrame(gate_rows)
    atomic_write_csv(gate, out_dir / "v11_cell_progression_gate.csv")

    boundary_rows: list[dict[str, Any]] = []
    for (block_label, mechanism), cell in gate.groupby(["block_label", "mechanism"], sort=True):
        directional_rates = cell.loc[cell["directional_gate_passed"], "missing_rate"]
        strong_rates = cell.loc[cell["strong_ci_gate_passed"], "missing_rate"]
        boundary_rows.append(
            {
                "block_label": block_label,
                "mechanism": mechanism,
                "highest_rate_directionally_supported": float(directional_rates.max()) if not directional_rates.empty else np.nan,
                "highest_rate_strongly_supported": float(strong_rates.max()) if not strong_rates.empty else np.nan,
                "n_directional_cells": int(cell["directional_gate_passed"].sum()),
                "n_strong_cells": int(cell["strong_ci_gate_passed"].sum()),
                "n_mixed_or_adverse_cells": int((~cell["directional_gate_passed"]).sum()),
            }
        )
    boundary = pd.DataFrame(boundary_rows)
    atomic_write_csv(boundary, out_dir / "v11_support_boundary_summary.csv")

    auc_rows: list[dict[str, Any]] = []
    for (block_label, mechanism, completion), cell in trajectory.groupby(
        ["block_label", "mechanism", "completion"], sort=True
    ):
        cell = cell.sort_values("missing_rate")
        x = cell["missing_rate"].to_numpy(dtype=float)
        for metric in PRIMARY_METRICS:
            if metric not in cell.columns or len(cell) < 2:
                continue
            auc_rows.append(
                {
                    "block_label": block_label,
                    "mechanism": mechanism,
                    "completion": completion,
                    "metric": metric,
                    "rate_min": float(x.min()),
                    "rate_max": float(x.max()),
                    "auc_over_missing_rate": float(np.trapz(cell[metric].to_numpy(dtype=float), x)),
                }
            )
    auc_columns = [
        "block_label",
        "mechanism",
        "completion",
        "metric",
        "rate_min",
        "rate_max",
        "auc_over_missing_rate",
    ]
    auc = pd.DataFrame(auc_rows, columns=auc_columns)
    atomic_write_csv(auc, out_dir / "v11_auc_over_missing_rates.csv")
    auc_contrast_rows: list[dict[str, Any]] = []
    # AUC across missingness rates is undefined for a one-rate smoke run.
    # In that case, write schema-valid empty outputs instead of grouping a
    # columnless DataFrame. The cell-level smoke results remain fully usable.
    if not auc.empty:
        for (block_label, mechanism, metric), cell in auc.groupby(
            ["block_label", "mechanism", "metric"], sort=True
        ):
            values = cell.set_index("completion")["auc_over_missing_rate"]
            if not {"median", "bayesianridge_mean"}.issubset(values.index):
                continue
            difference = float(values["bayesianridge_mean"] - values["median"])
            direction = favorable_direction(metric)
            auc_contrast_rows.append(
                {
                    "block_label": block_label,
                    "mechanism": mechanism,
                    "metric": metric,
                    "median_auc": float(values["median"]),
                    "bayesianridge_mean_auc": float(values["bayesianridge_mean"]),
                    "auc_difference": difference,
                    "favorable_direction": direction,
                    "auc_difference_favorable": bool(
                        difference < 0 if direction == "lower" else difference > 0
                    ),
                }
            )
    auc_contrast_columns = [
        "block_label",
        "mechanism",
        "metric",
        "median_auc",
        "bayesianridge_mean_auc",
        "auc_difference",
        "favorable_direction",
        "auc_difference_favorable",
    ]
    atomic_write_csv(
        pd.DataFrame(auc_contrast_rows, columns=auc_contrast_columns),
        out_dir / "v11_auc_contrasts.csv",
    )

    slope_rows: list[dict[str, Any]] = []
    slope_metrics = ["fdp", "power", "oracle_selection_jaccard", "average_precision", "support_ranking_auroc"]
    for (block_label, mechanism, completion, replicate), cell in replicate_metrics.groupby(
        ["block_label", "mechanism", "completion", "replicate"], sort=True
    ):
        cell = cell.sort_values("missing_rate")
        x = cell["missing_rate"].to_numpy(dtype=float)
        if len(np.unique(x)) < 2:
            continue
        for metric in slope_metrics:
            slope_rows.append(
                {
                    "block_label": block_label,
                    "mechanism": mechanism,
                    "completion": completion,
                    "replicate": int(replicate),
                    "metric": metric,
                    "slope_per_unit_missing_rate": float(np.polyfit(x, cell[metric].to_numpy(dtype=float), 1)[0]),
                }
            )
    slope_columns = [
        "block_label",
        "mechanism",
        "completion",
        "replicate",
        "metric",
        "slope_per_unit_missing_rate",
    ]
    slopes = pd.DataFrame(slope_rows, columns=slope_columns)
    atomic_write_csv(slopes, out_dir / "v11_replicate_rate_slopes.csv")
    slope_contrast_rows: list[dict[str, Any]] = []
    # Rate slopes likewise require at least two distinct rates.
    if not slopes.empty:
        for (block_label, mechanism, metric), cell in slopes.groupby(
            ["block_label", "mechanism", "metric"], sort=True
        ):
            renamed = cell.rename(columns={"slope_per_unit_missing_rate": metric})
            row = paired_contrast(
                renamed,
                index_columns=["replicate"],
                metric=metric,
                seed=args.bootstrap_seed + seed_offset,
                n_bootstrap=args.n_bootstrap,
            )
            row.update({"block_label": block_label, "mechanism": mechanism})
            slope_contrast_rows.append(row)
            seed_offset += 1
    slope_contrast_columns = [
        "metric",
        "reference",
        "comparator",
        "favorable_direction",
        "n_pairs",
        "reference_mean",
        "comparator_mean",
        "mean_difference",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "paired_wilcoxon_p",
        "fraction_pairs_favorable",
        "block_label",
        "mechanism",
    ]
    atomic_write_csv(
        pd.DataFrame(slope_contrast_rows, columns=slope_contrast_columns),
        out_dir / "v11_rate_slope_contrasts.csv",
    )

    decision = pd.DataFrame(
        [
            {
                "analysis_status": analysis_status,
                "integrity_complete": bool(audit["complete"].all()),
                "n_cells": int(len(gate)),
                "n_strong_support_cells": int(gate["strong_ci_gate_passed"].sum()),
                "n_directional_support_cells": int(gate["directional_gate_passed"].sum()),
                "n_mixed_or_adverse_cells": int((~gate["directional_gate_passed"]).sum()),
                "overall_decision": (
                    "smoke_only_no_scientific_decision"
                    if analysis_status == "smoke"
                    else "pilot_boundary_map_only"
                    if analysis_status == "pilot"
                    else "stress_map_completed_no_universal_superiority_claim"
                ),
            }
        ]
    )
    atomic_write_csv(decision, out_dir / "v11_mapping_decision.csv")

    figures = out_dir / "figures_v11"
    figures.mkdir(exist_ok=True)
    for (block_label, mechanism), cell in trajectory.groupby(["block_label", "mechanism"], sort=True):
        for metric in PRIMARY_METRICS:
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            for completion, series in cell.groupby("completion", sort=False):
                series = series.sort_values("missing_rate")
                ax.plot(series["missing_rate"], series[metric], marker="o", label=completion)
            ax.set_xlabel("Missingness rate")
            ax.set_ylabel(metric)
            ax.set_title(f"V11 {block_label} {mechanism}: {metric}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / f"{block_label}__{mechanism}__{metric}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

    report_lines = [
        "# V11 Missingness Stress Map",
        "",
        f"Analysis status: **{analysis_status}**",
        "",
        f"Overall decision: **{decision.iloc[0]['overall_decision']}**",
        "",
        "## Integrity",
        "",
        dataframe_to_markdown(audit, index=False),
        "",
        "## Cell-level support map",
        "",
        dataframe_to_markdown(gate, index=False),
        "",
        "## Estimated support boundaries",
        "",
        dataframe_to_markdown(boundary, index=False),
        "",
        "Positive BR-minus-Median differences favor BR mean for power, Oracle Jaccard, "
        "average precision, and ranking AUROC. Negative differences favor BR mean for FDP. "
        "Pilot status is directional screening only. Stress status maps the tested cells and "
        "does not justify a universal-superiority claim.",
    ]
    (out_dir / "v11_missingness_stress_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[V11 summary] integrity complete={bool(audit['complete'].all())}")
    print(f"[V11 summary] overall decision={decision.iloc[0]['overall_decision']}")
    print(f"[V11 summary] outputs written to {out_dir}")


if __name__ == "__main__":
    main()
