#!/usr/bin/env python3
"""Summarize V9.1 Stage 1 convergence and over-smoothing sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_common import atomic_write_csv
from v91_common import dataframe_to_markdown, paired_wilcoxon_p, percentile_bootstrap_mean_ci


STAGE1A_METRICS: tuple[tuple[str, str], ...] = (
    ("masked_rmse", "decrease"),
    ("sample_cov_fro_relative", "decrease"),
    ("sample_corr_fro_relative", "decrease"),
    ("masked_imputed_variance_ratio_median", "decrease"),
    ("completed_variance_ratio_median", "decrease"),
    ("pair_corr_mean", "increase"),
    ("s_relative_mean", "decrease"),
    ("cov_kk_fro_relative", "unspecified"),
    ("cov_xk_offdiag_rmse", "unspecified"),
    ("corr_condition_number_effective", "increase"),
    ("corr_condition_number_ridge_1e3", "increase"),
    ("corr_min_eigenvalue", "decrease"),
    ("corr_min_positive_eigenvalue", "decrease"),
)

STAGE1B_METRICS = (
    "fdp",
    "power",
    "oracle_selection_jaccard",
    "n_selected",
    "true_positives",
    "false_positives",
)


def valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    error = frame["error"] if "error" in frame.columns else pd.Series("", index=frame.index)
    return frame[error.fillna("").eq("")].copy()


def monotonic_fraction(wide: pd.DataFrame, direction: str, tolerance: float = 1e-12) -> float:
    values = wide.to_numpy(dtype=float)
    if values.shape[1] < 2:
        return float("nan")
    if direction == "increase":
        passed = np.all(np.diff(values, axis=1) >= -tolerance, axis=1)
    elif direction == "decrease":
        passed = np.all(np.diff(values, axis=1) <= tolerance, axis=1)
    else:
        return float("nan")
    return float(np.mean(passed))


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Summarize V9.1 max_iter sensitivity.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    max_iters = [int(value) for value in config["max_iters"]]
    n_replicates = int(config["n_replicates"])
    n_draws = int(config["n_downstream_draws"])
    run_downstream = bool(config["run_downstream"])

    completion = valid_rows(pd.read_csv(out_dir / "completion_diagnostics.csv"))
    knockoff = valid_rows(pd.read_csv(out_dir / "knockoff_diagnostics.csv"))
    selection_path = out_dir / "selection_results.csv"
    selection = valid_rows(pd.read_csv(selection_path)) if selection_path.exists() else pd.DataFrame()

    audit_rows = [
        {
            "table": "completion_diagnostics",
            "expected_rows": n_replicates * len(max_iters),
            "successful_rows": len(completion),
            "duplicate_keys": int(completion.duplicated(["replicate", "max_iter"]).sum()),
        },
        {
            "table": "knockoff_diagnostics",
            "expected_rows": n_replicates * (len(max_iters) + 1) * n_draws,
            "successful_rows": len(knockoff),
            "duplicate_keys": int(
                knockoff.duplicated(["replicate", "variant", "max_iter", "downstream_draw"]).sum()
            ),
        },
    ]
    if run_downstream:
        audit_rows.append(
            {
                "table": "selection_results",
                "expected_rows": n_replicates * (len(max_iters) + 1) * n_draws,
                "successful_rows": len(selection),
                "duplicate_keys": int(
                    selection.duplicated(["replicate", "variant", "max_iter", "downstream_draw"]).sum()
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit["complete"] = (
        (audit["expected_rows"] == audit["successful_rows"])
        & audit["duplicate_keys"].eq(0)
    )
    atomic_write_csv(audit, out_dir / "v91_integrity_audit.csv")

    br_knockoff = knockoff[knockoff["variant"].eq("bayesianridge_mean")].copy()
    diagnostic_columns = [
        "pair_corr_mean",
        "s_relative_mean",
        "cov_kk_fro_relative",
        "cov_xk_offdiag_rmse",
    ]
    br_knockoff_rep = (
        br_knockoff.groupby(["replicate", "max_iter"], as_index=False)[diagnostic_columns]
        .mean()
    )
    completion_columns = [
        "masked_rmse",
        "masked_mae",
        "sample_cov_fro_relative",
        "sample_corr_fro_relative",
        "masked_imputed_variance_ratio_median",
        "masked_imputed_variance_ratio_mean",
        "completed_variance_ratio_median",
        "completed_variance_ratio_mean",
        "corr_condition_number_effective",
        "corr_condition_number_ridge_1e3",
        "corr_min_eigenvalue",
        "corr_min_positive_eigenvalue",
        "corr_max_eigenvalue",
        "n_iter",
        "convergence_warning_count",
        "converged_without_warning",
    ]
    stage1a = completion[["replicate", "max_iter", *completion_columns]].merge(
        br_knockoff_rep,
        on=["replicate", "max_iter"],
        how="inner",
        validate="one_to_one",
    )
    atomic_write_csv(stage1a, out_dir / "stage1a_replicate_metrics.csv")

    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for metric, expected_direction in STAGE1A_METRICS:
        if metric not in stage1a.columns:
            continue
        for max_iter, block in stage1a.groupby("max_iter"):
            values = block[metric].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "metric": metric,
                    "expected_direction_with_more_iterations": expected_direction,
                    "max_iter": int(max_iter),
                    "n_replicates": int(np.isfinite(values).sum()),
                    "mean": float(np.nanmean(values)),
                    "sd": float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else 0.0,
                    "median": float(np.nanmedian(values)),
                }
            )
        wide = stage1a.pivot(index="replicate", columns="max_iter", values=metric).sort_index(axis=1)
        if set(max_iters).issubset(wide.columns):
            wide = wide[max_iters].dropna()
            monotonic = monotonic_fraction(wide, expected_direction)
            for start, stop in ((max_iters[0], value) for value in max_iters[1:]):
                differences = wide[stop].to_numpy() - wide[start].to_numpy()
                ci_low, ci_high = percentile_bootstrap_mean_ci(
                    differences,
                    seed=args.bootstrap_seed + len(contrast_rows),
                    n_bootstrap=args.n_bootstrap,
                )
                favorable = np.nan
                if expected_direction == "increase":
                    favorable = float(np.mean(differences > 0))
                elif expected_direction == "decrease":
                    favorable = float(np.mean(differences < 0))
                contrast_rows.append(
                    {
                        "metric": metric,
                        "expected_direction_with_more_iterations": expected_direction,
                        "from_max_iter": start,
                        "to_max_iter": stop,
                        "n_pairs": int(len(differences)),
                        "mean_difference_to_minus_from": float(np.mean(differences)),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "paired_wilcoxon_p": paired_wilcoxon_p(differences),
                        "fraction_pairs_in_expected_direction": favorable,
                        "fraction_replicates_monotonic_across_all_budgets": monotonic,
                    }
                )
    stage1a_summary = pd.DataFrame(summary_rows)
    stage1a_contrasts = pd.DataFrame(contrast_rows)
    atomic_write_csv(stage1a_summary, out_dir / "stage1a_summary.csv")
    atomic_write_csv(stage1a_contrasts, out_dir / "stage1a_directional_contrasts.csv")

    stage1b_rep = pd.DataFrame()
    stage1b_summary = pd.DataFrame()
    if run_downstream and not selection.empty:
        oracle = (
            selection[selection["variant"].eq("oracle_complete")]
            .groupby("replicate", as_index=False)[list(STAGE1B_METRICS)]
            .mean()
        )
        br = (
            selection[selection["variant"].eq("bayesianridge_mean")]
            .groupby(["replicate", "max_iter"], as_index=False)[list(STAGE1B_METRICS)]
            .mean()
        )
        stage1b_rep = br.copy()
        atomic_write_csv(stage1b_rep, out_dir / "stage1b_replicate_metrics.csv")
        rows: list[dict[str, Any]] = []
        baseline = stage1b_rep[stage1b_rep["max_iter"].eq(max_iters[0])].set_index("replicate")
        for metric in STAGE1B_METRICS:
            for max_iter, block in stage1b_rep.groupby("max_iter"):
                block_indexed = block.set_index("replicate")
                values = block_indexed[metric]
                row: dict[str, Any] = {
                    "metric": metric,
                    "max_iter": int(max_iter),
                    "n_replicates": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0,
                    "analysis_status": "exploratory_descriptive_only",
                }
                if int(max_iter) == max_iters[0]:
                    row["difference_vs_iter_10"] = 0.0
                    row["direction_vs_iter_10"] = "reference"
                else:
                    common = baseline.index.intersection(block_indexed.index)
                    difference = (
                        block_indexed.loc[common, metric].to_numpy()
                        - baseline.loc[common, metric].to_numpy()
                    )
                    mean_difference = float(np.mean(difference))
                    row["difference_vs_iter_10"] = mean_difference
                    row["direction_vs_iter_10"] = (
                        "increase" if mean_difference > 0 else "decrease" if mean_difference < 0 else "no_change"
                    )
                rows.append(row)
            oracle_values = oracle[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "metric": metric,
                    "max_iter": 0,
                    "n_replicates": int(np.isfinite(oracle_values).sum()),
                    "mean": float(np.nanmean(oracle_values)),
                    "sd": float(np.nanstd(oracle_values, ddof=1)) if np.isfinite(oracle_values).sum() > 1 else 0.0,
                    "difference_vs_iter_10": float("nan"),
                    "direction_vs_iter_10": "oracle_reference",
                    "analysis_status": "exploratory_descriptive_only",
                }
            )
        stage1b_summary = pd.DataFrame(rows)
        atomic_write_csv(stage1b_summary, out_dir / "stage1b_exploratory_summary.csv")

    figures = out_dir / "figures_v91"
    figures.mkdir(exist_ok=True)
    plot_metrics = [
        "masked_rmse",
        "masked_imputed_variance_ratio_median",
        "pair_corr_mean",
        "s_relative_mean",
        "corr_condition_number_ridge_1e3",
    ]
    for metric in plot_metrics:
        if metric not in stage1a.columns:
            continue
        plot = stage1a.groupby("max_iter", as_index=False)[metric].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.errorbar(plot["max_iter"], plot["mean"], yerr=plot["std"], marker="o", capsize=4)
        ax.set_xlabel("IterativeImputer max_iter")
        ax.set_ylabel(metric)
        ax.set_title(f"V9.1 Stage 1A: {metric}")
        fig.tight_layout()
        fig.savefig(figures / f"stage1a_{metric}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if not stage1b_rep.empty:
        for metric in ("fdp", "power", "oracle_selection_jaccard"):
            plot = stage1b_rep.groupby("max_iter", as_index=False)[metric].agg(["mean", "std"]).reset_index()
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            ax.errorbar(plot["max_iter"], plot["mean"], yerr=plot["std"], marker="o", capsize=4)
            ax.set_xlabel("IterativeImputer max_iter")
            ax.set_ylabel(metric)
            ax.set_title(f"V9.1 Stage 1B exploratory: {metric}")
            fig.tight_layout()
            fig.savefig(figures / f"stage1b_{metric}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

    warning_summary = (
        completion.groupby("max_iter", as_index=False)[["n_iter", "convergence_warning_count", "converged_without_warning"]]
        .mean()
    )
    lines = [
        "# V9.1 Stage 1 Convergence Sensitivity",
        "",
        "Stage 1A is a prospective directional mechanism test. Stage 1B is exploratory and descriptive only at the default 10 replicates.",
        "",
        "## Integrity",
        "",
        dataframe_to_markdown(audit, index=False),
        "",
        "## IterativeImputer convergence",
        "",
        dataframe_to_markdown(warning_summary, index=False),
        "",
        "## Stage 1A directional contrasts",
        "",
        dataframe_to_markdown(stage1a_contrasts, index=False) if not stage1a_contrasts.empty else "No Stage 1A contrasts available.",
    ]
    if not stage1b_summary.empty:
        lines.extend(
            [
                "",
                "## Stage 1B exploratory downstream summary",
                "",
                "No significance claim is permitted from this table under the V9.1 preregistration.",
                "",
                dataframe_to_markdown(stage1b_summary, index=False),
            ]
        )
    (out_dir / "v91_stage1_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[V9.1 summary] integrity complete={bool(audit['complete'].all())}")
    print(f"[V9.1 summary] outputs written to {out_dir}")


if __name__ == "__main__":
    main()
