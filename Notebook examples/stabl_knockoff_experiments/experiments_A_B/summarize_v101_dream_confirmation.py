#!/usr/bin/env python3
"""Confirmatory summary for V10.1B DREAM BR mean vs Median."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_common import atomic_write_csv
from v91_common import (
    dataframe_to_markdown,
    holm_adjust,
    paired_wilcoxon_p,
    percentile_bootstrap_mean_ci,
)


def valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    error = frame["error"] if "error" in frame.columns else pd.Series("", index=frame.index)
    return frame[error.fillna("").eq("")].copy()


def paired_contrast(
    frame: pd.DataFrame,
    *,
    mechanism: str,
    metric: str,
    method_column: str,
    reference: str = "median",
    comparator: str = "bayesianridge_mean",
    seed: int,
    n_bootstrap: int,
) -> tuple[dict[str, Any], np.ndarray]:
    subset = frame[frame["mechanism"].eq(mechanism)].copy()
    wide = subset.pivot(index="replicate", columns=method_column, values=metric)
    required = {reference, comparator}
    if not required.issubset(wide.columns):
        raise ValueError(f"Missing paired methods for {mechanism} {metric}: {sorted(required - set(wide.columns))}")
    wide = wide[[reference, comparator]].dropna()
    differences = wide[comparator].to_numpy(dtype=float) - wide[reference].to_numpy(dtype=float)
    ci_low, ci_high = percentile_bootstrap_mean_ci(
        differences,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    row = {
        "mechanism": mechanism,
        "metric": metric,
        "reference": reference,
        "comparator": comparator,
        "n_replicates": int(len(wide)),
        "reference_mean": float(wide[reference].mean()),
        "comparator_mean": float(wide[comparator].mean()),
        "mean_difference": float(np.mean(differences)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "paired_wilcoxon_p": paired_wilcoxon_p(differences),
        "comparator_better_fraction": float(
            np.mean(differences < 0) if metric in {"fdp", "median_signal_rank", "mean_null_score"} else np.mean(differences > 0)
        ),
    }
    return row, differences


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Summarize V10.1B DREAM confirmation.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=50260730)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--power-noninferiority-margin", type=float, default=0.03)
    parser.add_argument("--fdp-nonworsening-margin", type=float, default=0.03)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    prereg = json.loads((out_dir / "v101_preregistration.json").read_text(encoding="utf-8"))
    analysis_status = str(prereg.get("analysis_status", "unknown"))
    n_replicates = int(prereg["n_replicates"])
    n_draws = int(prereg["n_completion_draws"])

    selection_raw = valid_rows(pd.read_csv(out_dir / "selection_results.csv"))
    replicate = pd.read_csv(out_dir / "replicate_level_metrics.csv")
    oracle_raw = pd.read_csv(out_dir / "oracle_selection_agreement.csv")
    ranking_dir = out_dir / "v101_threshold_independent_ranking"
    ranking_rep = pd.read_csv(ranking_dir / "ranking_replicate_metrics.csv")

    oracle_rep = (
        oracle_raw.groupby(["replicate", "mechanism", "completion"], as_index=False)[
            ["oracle_selection_jaccard", "oracle_selected_count_abs_difference"]
        ].mean()
    )

    audit = pd.DataFrame(
        [
            {
                "table": "selection_results",
                "expected_rows": n_replicates * 2 * 3 * n_draws,
                "successful_rows": len(selection_raw),
                "duplicate_keys": int(selection_raw.duplicated(["replicate", "mechanism", "completion", "completion_draw", "generator_label"]).sum()),
            },
            {
                "table": "replicate_level_metrics",
                "expected_rows": n_replicates * 2 * 3,
                "successful_rows": len(replicate),
                "duplicate_keys": int(replicate.duplicated(["replicate", "mechanism", "completion", "generator_label"]).sum()),
            },
            {
                "table": "oracle_selection_agreement",
                "expected_rows": n_replicates * 2 * 2 * n_draws,
                "successful_rows": len(oracle_raw),
                "duplicate_keys": int(oracle_raw.duplicated(["replicate", "mechanism", "completion", "completion_draw", "generator_label"]).sum()),
            },
            {
                "table": "ranking_replicate_metrics",
                "expected_rows": n_replicates * 2 * 3,
                "successful_rows": len(ranking_rep),
                "duplicate_keys": int(ranking_rep.duplicated(["replicate", "mechanism", "completion", "generator_label"]).sum()),
            },
        ]
    )
    audit["complete"] = (
        audit["expected_rows"].eq(audit["successful_rows"])
        & audit["duplicate_keys"].eq(0)
    )
    atomic_write_csv(audit, out_dir / "v101_integrity_audit.csv")

    rows: list[dict[str, Any]] = []
    seed_offset = 0
    metric_sources = [
        (replicate, "completion", "fdp"),
        (replicate, "completion", "power"),
        (oracle_rep, "completion", "oracle_selection_jaccard"),
        (ranking_rep, "completion", "average_precision"),
        (ranking_rep, "completion", "support_ranking_auroc"),
        (ranking_rep, "completion", "precision_at_5"),
        (ranking_rep, "completion", "precision_at_10"),
        (ranking_rep, "completion", "precision_at_15"),
        (ranking_rep, "completion", "precision_at_20"),
        (ranking_rep, "completion", "mean_null_score"),
        (ranking_rep, "completion", "mean_signal_score"),
    ]
    for mechanism in ("MCAR", "MAR"):
        for source, method_column, metric in metric_sources:
            row, _ = paired_contrast(
                source,
                mechanism=mechanism,
                metric=metric,
                method_column=method_column,
                seed=args.bootstrap_seed + seed_offset,
                n_bootstrap=args.n_bootstrap,
            )
            rows.append(row)
            seed_offset += 1
    contrasts = pd.DataFrame(rows)

    for mechanism in ("MCAR", "MAR"):
        mask = contrasts["mechanism"].eq(mechanism) & contrasts["metric"].isin(
            ["oracle_selection_jaccard", "average_precision"]
        )
        contrasts.loc[mask, "holm_p_within_mechanism_primary_superiority"] = holm_adjust(
            contrasts.loc[mask, "paired_wilcoxon_p"].to_numpy(dtype=float)
        )
    atomic_write_csv(contrasts, out_dir / "v101_primary_and_supportive_contrasts.csv")

    decision_rows: list[dict[str, Any]] = []
    for mechanism in ("MCAR", "MAR"):
        block = contrasts[contrasts["mechanism"].eq(mechanism)].set_index("metric")
        oracle = block.loc["oracle_selection_jaccard"]
        average_precision = block.loc["average_precision"]
        power = block.loc["power"]
        fdp = block.loc["fdp"]

        oracle_superiority = bool(oracle["bootstrap_ci_low"] > 0)
        ranking_superiority = bool(average_precision["bootstrap_ci_low"] > 0)
        power_noninferiority = bool(
            power["bootstrap_ci_low"] >= -args.power_noninferiority_margin
        )
        fdp_nonworsening = bool(
            fdp["bootstrap_ci_high"] <= args.fdp_nonworsening_margin
        )
        supported = bool(
            oracle_superiority
            and ranking_superiority
            and power_noninferiority
            and fdp_nonworsening
        )
        decision_rows.append(
            {
                "mechanism": mechanism,
                "analysis_status": analysis_status,
                "oracle_jaccard_superiority": oracle_superiority,
                "average_precision_superiority": ranking_superiority,
                "power_noninferiority_margin": args.power_noninferiority_margin,
                "power_noninferiority": power_noninferiority,
                "fdp_nonworsening_margin": args.fdp_nonworsening_margin,
                "fdp_nonworsening": fdp_nonworsening,
                "mechanism_supported": supported if analysis_status == "confirmatory" else False,
                "interpretation": (
                    "confirmatory_support" if supported and analysis_status == "confirmatory"
                    else "smoke_only_no_scientific_decision" if analysis_status == "smoke"
                    else "confirmatory_gate_not_fully_met"
                ),
            }
        )
    decision = pd.DataFrame(decision_rows)
    if analysis_status == "smoke":
        overall = "smoke_only_no_scientific_decision"
    else:
        count = int(decision["mechanism_supported"].sum())
        overall = (
            "confirmed_across_mcar_and_mar" if count == 2
            else "mechanism_specific_confirmation" if count == 1
            else "not_confirmed"
        )
    decision["overall_decision"] = overall
    atomic_write_csv(decision, out_dir / "v101_confirmatory_decision.csv")

    primary_display = contrasts[
        contrasts["metric"].isin(
            ["fdp", "power", "oracle_selection_jaccard", "average_precision", "support_ranking_auroc"]
        )
    ].copy()
    lines = [
        "# V10.1B DREAM confirmation",
        "",
        f"Analysis status: **{analysis_status}**",
        "",
        f"Overall decision: **{overall}**",
        "",
        "## Integrity",
        "",
        dataframe_to_markdown(audit, index=False),
        "",
        "## Primary and key supportive contrasts",
        "",
        dataframe_to_markdown(primary_display, index=False),
        "",
        "## Confirmatory gate",
        "",
        dataframe_to_markdown(decision, index=False),
        "",
        "The primary comparison is BayesianRidge conditional mean minus Median. "
        "Positive values favor BR mean for Oracle Jaccard and average precision. "
        "For FDP, lower values are favorable. The power and FDP margins were fixed "
        "before the fresh confirmation run.",
    ]
    (out_dir / "v101_dream_confirmation_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[V10.1B summary] integrity complete={bool(audit['complete'].all())}")
    print(f"[V10.1B summary] overall decision={overall}")
    print(f"[V10.1B summary] outputs written to {out_dir}")


if __name__ == "__main__":
    main()
