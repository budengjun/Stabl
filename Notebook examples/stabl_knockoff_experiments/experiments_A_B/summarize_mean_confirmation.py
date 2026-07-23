#!/usr/bin/env python3
"""Confirmatory inference for V9 BayesianRidge mean vs median.

All downstream metrics are averaged across repeated ordinary STABL runs within
simulation replicate before paired inference.  The three primary endpoints are
realized FDP, power, and Jaccard agreement with complete-data STABL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from research_common import atomic_write_csv


PRIMARY_ENDPOINTS = (
    ("fdp", "lower"),
    ("power", "higher"),
    ("oracle_selection_jaccard", "higher"),
)


def valid_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "error" in frame.columns:
        frame = frame[frame["error"].fillna("").eq("")].copy()
    return frame


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    index = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[index].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    try:
        return float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return np.nan


def holm_adjust(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    result = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return pd.Series(result, index=p_values.index)
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        result[idx] = running
    return pd.Series(result, index=p_values.index)


def paired_contrast(
    frame: pd.DataFrame,
    *,
    metric: str,
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    pivot = frame.pivot_table(index="replicate", columns="completion", values=metric, aggfunc="mean")
    needed = ["median", "bayesianridge_mean"]
    pivot = pivot.dropna(subset=needed)
    diff = pivot["bayesianridge_mean"].to_numpy() - pivot["median"].to_numpy()
    low, high = bootstrap_mean_ci(diff, seed=seed, n_boot=n_boot)
    return {
        "metric": metric,
        "n_replicates": int(len(pivot)),
        "median_mean": float(pivot["median"].mean()),
        "bayesianridge_mean_mean": float(pivot["bayesianridge_mean"].mean()),
        "mean_difference": float(diff.mean()),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "paired_wilcoxon_p": paired_p(diff),
        "br_mean_better_fraction": float(
            np.mean(diff < 0 if metric in {"fdp", "false_positives", "n_selected"} else diff > 0)
        ),
    }


def integrity_audit(
    selection: pd.DataFrame,
    recovery: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    completions = tuple(config.get("completion_methods_resolved", []))
    mechanisms = tuple(config.get("mechanisms_resolved", []))
    n_reps = int(config.get("n_replicates", selection["replicate"].nunique()))
    n_draws = int(config.get("n_completion_draws", selection["completion_draw"].nunique()))
    n_generators = max(1, selection["generator_label"].nunique())
    expected_selection = n_reps * len(mechanisms) * len(completions) * n_draws * n_generators
    expected_recovery = n_reps * len(mechanisms) * len(completions) * n_draws
    selection_key = [
        "p_setting", "actual_p", "replicate", "mechanism", "completion",
        "completion_draw", "generator_label",
    ]
    recovery_key = [
        "p_setting", "actual_p", "replicate", "mechanism", "completion", "completion_draw",
    ]
    diagnostic_key = selection_key
    rows = [
        {
            "table": "selection_results",
            "expected_rows": expected_selection,
            "successful_rows": int(len(selection)),
            "duplicate_keys": int(selection.duplicated(selection_key).sum()),
            "complete": bool(len(selection) == expected_selection and not selection.duplicated(selection_key).any()),
        },
        {
            "table": "draw_diagnostics",
            "expected_rows": expected_selection,
            "successful_rows": int(len(diagnostics)),
            "duplicate_keys": int(diagnostics.duplicated(diagnostic_key).sum()),
            "complete": bool(len(diagnostics) == expected_selection and not diagnostics.duplicated(diagnostic_key).any()),
        },
        {
            "table": "imputation_recovery",
            "expected_rows": expected_recovery,
            "successful_rows": int(len(recovery)),
            "duplicate_keys": int(recovery.duplicated(recovery_key).sum()),
            "complete": bool(len(recovery) == expected_recovery and not recovery.duplicated(recovery_key).any()),
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--fdp-noninferiority-margin", type=float, default=0.0)
    parser.add_argument("--power-noninferiority-margin", type=float, default=0.0)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    selection = valid_frame(out_dir / "selection_results.csv")
    recovery = valid_frame(out_dir / "imputation_recovery.csv")
    diagnostics = valid_frame(out_dir / "draw_diagnostics.csv")

    required = {"oracle_complete", "median", "bayesianridge_mean"}
    found = set(selection["completion"].unique())
    if found != required:
        raise RuntimeError(f"V9 requires exactly {sorted(required)}, found {sorted(found)}")

    audit = integrity_audit(selection, recovery, diagnostics, config)
    atomic_write_csv(audit, out_dir / "confirmatory_integrity_audit.csv")
    if not audit["complete"].all() and not args.allow_incomplete:
        raise RuntimeError(
            "Confirmatory result tables are incomplete. Inspect "
            "confirmatory_integrity_audit.csv or use --allow-incomplete for diagnosis only."
        )

    replicate_metrics = (
        selection.groupby(
            ["dataset", "p_setting", "actual_p", "replicate", "mechanism", "completion", "generator_label"],
            dropna=False,
        )[["fdp", "power", "n_selected", "true_positives", "false_positives", "estimated_fdp", "threshold"]]
        .mean()
        .reset_index()
    )

    agreement = valid_frame(out_dir / "oracle_selection_agreement.csv")
    agreement_replicate = (
        agreement.groupby(
            ["dataset", "p_setting", "actual_p", "replicate", "mechanism", "completion", "generator_label"],
            dropna=False,
        )[["oracle_selection_jaccard", "oracle_selected_count_abs_difference"]]
        .mean()
        .reset_index()
    )
    combined = replicate_metrics.merge(
        agreement_replicate,
        on=["dataset", "p_setting", "actual_p", "replicate", "mechanism", "completion", "generator_label"],
        how="left",
    )
    atomic_write_csv(combined, out_dir / "confirmatory_replicate_metrics.csv")

    primary_rows: list[dict[str, Any]] = []
    grouping = ["dataset", "p_setting", "actual_p", "mechanism", "generator_label"]
    counter = 0
    for keys, block in combined.groupby(grouping, dropna=False):
        meta = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        for metric, direction in PRIMARY_ENDPOINTS:
            row = paired_contrast(
                block,
                metric=metric,
                seed=args.bootstrap_seed + counter,
                n_boot=args.n_bootstrap,
            )
            row.update(meta)
            row["favorable_direction"] = direction
            primary_rows.append(row)
            counter += 1
    primary = pd.DataFrame(primary_rows)
    primary["holm_adjusted_p"] = holm_adjust(primary["paired_wilcoxon_p"])
    atomic_write_csv(primary, out_dir / "confirmatory_primary_contrasts.csv")

    # Secondary selection, stability, completion, and knockoff mechanism contrasts.
    secondary_rows: list[dict[str, Any]] = []
    for keys, block in combined.groupby(grouping, dropna=False):
        meta = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        for metric in (
            "n_selected", "true_positives", "false_positives", "estimated_fdp",
            "threshold", "oracle_selected_count_abs_difference",
        ):
            row = paired_contrast(
                block,
                metric=metric,
                seed=args.bootstrap_seed + 1000 + len(secondary_rows),
                n_boot=args.n_bootstrap,
            )
            row.update(meta)
            secondary_rows.append(row)

    stability = valid_frame(out_dir / "selection_stability.csv")
    for keys, block in stability.groupby(grouping, dropna=False):
        meta = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        for metric in ("selection_pairwise_jaccard", "selected_count_sd", "fdp_sd", "power_sd"):
            row = paired_contrast(
                block,
                metric=metric,
                seed=args.bootstrap_seed + 2000 + len(secondary_rows),
                n_boot=args.n_bootstrap,
            )
            row.update(meta)
            secondary_rows.append(row)

    recovery_rep = (
        recovery.groupby(
            ["dataset", "p_setting", "actual_p", "replicate", "mechanism", "completion"],
            dropna=False,
        )[["masked_rmse", "masked_mae", "sample_cov_fro_relative", "sample_corr_fro_relative"]]
        .mean()
        .reset_index()
    )
    for keys, block in recovery_rep.groupby(["dataset", "p_setting", "actual_p", "mechanism"], dropna=False):
        meta = dict(zip(["dataset", "p_setting", "actual_p", "mechanism"], keys if isinstance(keys, tuple) else (keys,)))
        meta["generator_label"] = "not_applicable"
        for metric in ("masked_rmse", "masked_mae", "sample_cov_fro_relative", "sample_corr_fro_relative"):
            row = paired_contrast(
                block,
                metric=metric,
                seed=args.bootstrap_seed + 3000 + len(secondary_rows),
                n_boot=args.n_bootstrap,
            )
            row.update(meta)
            secondary_rows.append(row)

    diagnostic_rep = (
        diagnostics.groupby(
            ["dataset", "p_setting", "actual_p", "replicate", "mechanism", "completion", "generator_label"],
            dropna=False,
        )[["pair_corr_mean", "s_relative_mean", "cov_kk_fro_relative", "cov_xk_offdiag_rmse"]]
        .mean()
        .reset_index()
    )
    for keys, block in diagnostic_rep.groupby(grouping, dropna=False):
        meta = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        for metric in ("pair_corr_mean", "s_relative_mean", "cov_kk_fro_relative", "cov_xk_offdiag_rmse"):
            row = paired_contrast(
                block,
                metric=metric,
                seed=args.bootstrap_seed + 4000 + len(secondary_rows),
                n_boot=args.n_bootstrap,
            )
            row.update(meta)
            secondary_rows.append(row)
    secondary = pd.DataFrame(secondary_rows)
    atomic_write_csv(secondary, out_dir / "confirmatory_secondary_contrasts.csv")

    decision_rows: list[dict[str, Any]] = []
    for keys, block in primary.groupby(grouping, dropna=False):
        meta = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        by_metric = block.set_index("metric")
        fdp = by_metric.loc["fdp"]
        power = by_metric.loc["power"]
        jaccard = by_metric.loc["oracle_selection_jaccard"]
        fdp_nonworse = bool(fdp.bootstrap_ci_high <= args.fdp_noninferiority_margin)
        power_nonworse = bool(power.bootstrap_ci_low >= -args.power_noninferiority_margin)
        stable_improvement = bool(
            (fdp.bootstrap_ci_high < 0)
            or (power.bootstrap_ci_low > 0)
            or (jaccard.bootstrap_ci_low > 0)
        )
        pointwise_favorable = bool(
            fdp.mean_difference <= args.fdp_noninferiority_margin
            and power.mean_difference >= -args.power_noninferiority_margin
            and (
                fdp.mean_difference < 0
                or power.mean_difference > 0
                or jaccard.mean_difference > 0
            )
        )
        if fdp_nonworse and power_nonworse and stable_improvement:
            status = "supported"
        elif pointwise_favorable:
            status = "directionally_promising_not_confirmed"
        else:
            status = "not_supported"
        decision_rows.append(
            {
                **meta,
                "fdp_nonworse": fdp_nonworse,
                "power_nonworse": power_nonworse,
                "stable_improvement": stable_improvement,
                "pointwise_favorable": pointwise_favorable,
                "mechanism_decision": status,
                "delta_fdp": fdp.mean_difference,
                "delta_fdp_ci_low": fdp.bootstrap_ci_low,
                "delta_fdp_ci_high": fdp.bootstrap_ci_high,
                "delta_power": power.mean_difference,
                "delta_power_ci_low": power.bootstrap_ci_low,
                "delta_power_ci_high": power.bootstrap_ci_high,
                "delta_oracle_jaccard": jaccard.mean_difference,
                "delta_oracle_jaccard_ci_low": jaccard.bootstrap_ci_low,
                "delta_oracle_jaccard_ci_high": jaccard.bootstrap_ci_high,
            }
        )
    decisions = pd.DataFrame(decision_rows)
    statuses = set(decisions["mechanism_decision"])
    if statuses == {"supported"}:
        overall = "confirmed_across_mcar_and_mar"
    elif "supported" in statuses and "not_supported" not in statuses:
        overall = "mechanism_specific_support"
    elif statuses <= {"supported", "directionally_promising_not_confirmed"}:
        overall = "promising_but_not_fully_confirmed"
    else:
        overall = "not_confirmed"
    decisions["overall_decision"] = overall
    atomic_write_csv(decisions, out_dir / "confirmatory_decision.csv")

    figures = out_dir / "figures_confirmatory"
    figures.mkdir(exist_ok=True)
    for metric, direction in PRIMARY_ENDPOINTS:
        plot = primary[primary["metric"] == metric].copy()
        plot = plot.sort_values(["mechanism", "generator_label"])
        labels = [f"{m} | {g}" for m, g in zip(plot["mechanism"], plot["generator_label"])]
        x = plot["mean_difference"].to_numpy()
        low = plot["bootstrap_ci_low"].to_numpy()
        high = plot["bootstrap_ci_high"].to_numpy()
        y = np.arange(len(plot))
        fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.8 * len(plot) + 1.8)))
        ax.errorbar(x, y, xerr=np.vstack([x - low, high - x]), fmt="o", capsize=4)
        ax.axvline(0.0, linewidth=1)
        ax.set_yticks(y, labels)
        ax.set_xlabel(f"BayesianRidge mean minus Median: {metric}")
        ax.set_title(f"V9 paired 95% bootstrap CI, favorable direction: {direction}")
        fig.tight_layout()
        fig.savefig(figures / f"primary_{metric}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    report_lines = [
        "# V9 BayesianRidge Mean Confirmation",
        "",
        f"Overall decision: **{overall}**",
        "",
        "Inference uses simulation replicates as the independent unit after averaging repeated ordinary STABL runs.",
        "",
        "## Mechanism decisions",
        "",
    ]
    for _, row in decisions.iterrows():
        report_lines.extend(
            [
                f"### {row['mechanism']}",
                "",
                f"Decision: **{row['mechanism_decision']}**",
                "",
                f"Delta FDP: {row['delta_fdp']:.4f} [{row['delta_fdp_ci_low']:.4f}, {row['delta_fdp_ci_high']:.4f}]",
                "",
                f"Delta power: {row['delta_power']:.4f} [{row['delta_power_ci_low']:.4f}, {row['delta_power_ci_high']:.4f}]",
                "",
                f"Delta Oracle Jaccard: {row['delta_oracle_jaccard']:.4f} [{row['delta_oracle_jaccard_ci_low']:.4f}, {row['delta_oracle_jaccard_ci_high']:.4f}]",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Files",
            "",
            "* confirmatory_integrity_audit.csv",
            "* confirmatory_primary_contrasts.csv",
            "* confirmatory_secondary_contrasts.csv",
            "* confirmatory_decision.csv",
            "* confirmatory_replicate_metrics.csv",
        ]
    )
    (out_dir / "confirmatory_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Confirmatory summary written to {out_dir}")
    print(f"Overall decision: {overall}")


if __name__ == "__main__":
    main()
