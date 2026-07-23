#!/usr/bin/env python3
"""V9.1 Stage 2: threshold-independent ranking analysis of V9 score files.

The script does not rerun STABL. It reads each saved V9 score path, defines the
feature score as the maximum real-feature selection frequency across the
regularization grid, and evaluates fixed-size and global support ranking.

Primary fixed-size metrics are tie-aware Precision@k for k=5,10,15,20.
TP@k and Recall@k are saved as deterministic derivatives, not treated as
independent evidence. Global metrics are average precision and median signal
rank. AUROC is retained as a reference diagnostic only.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_common import atomic_write_csv, parse_csv_items
from v91_common import (
    dataframe_to_markdown,
    holm_adjust,
    max_feature_scores,
    paired_wilcoxon_p,
    percentile_bootstrap_mean_ci,
    ranking_metrics,
)


SCORE_PATTERN = re.compile(
    r"^p_(?P<p>\d+)__rep_(?P<replicate>\d+)__(?P<mechanism>MCAR|MAR|MNAR|BLOCK)__"
    r"(?P<completion>.+?)__cdraw_(?P<draw>\d+)__(?P<generator>.+)\.npz$"
)


def parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 1 for item in parsed):
        raise ValueError("k values must be positive integers")
    return tuple(parsed)


def find_score_directory(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    if (path / "scores").is_dir():
        return path, path / "scores"
    candidates = [candidate for candidate in path.rglob("scores") if candidate.is_dir()]
    if len(candidates) == 1:
        return candidates[0].parent, candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No scores directory found under {path}")
    raise RuntimeError(f"Multiple scores directories found under {path}; pass the exact V9 output directory")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="V9.1 threshold-independent ranking reanalysis.",
    )
    parser.add_argument("--v9-out-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--k-values", default="5,10,15,20")
    parser.add_argument("--mechanisms", default="MCAR,MAR")
    parser.add_argument("--completions", default="oracle_complete,median,bayesianridge_mean")
    parser.add_argument("--generator", default="equicorr")
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    args = parser.parse_args()

    k_values = parse_int_list(args.k_values)
    mechanisms = tuple(item.upper() for item in parse_csv_items(args.mechanisms))
    completions = parse_csv_items(args.completions)
    if "median" not in completions or "bayesianridge_mean" not in completions:
        raise ValueError("Primary ranking analysis requires median and bayesianridge_mean")

    v9_root, score_dir = find_score_directory(Path(args.v9_out_dir))
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else v9_root / "v91_threshold_independent_ranking"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for score_path in sorted(score_dir.glob("*.npz")):
        if score_path.name.startswith("._"):
            continue
        match = SCORE_PATTERN.match(score_path.name)
        if not match:
            skipped.append({"file": score_path.name, "reason": "filename_pattern_mismatch"})
            continue
        meta = match.groupdict()
        mechanism = meta["mechanism"]
        completion = meta["completion"]
        generator = meta["generator"]
        if mechanism not in mechanisms or completion not in completions or generator != args.generator:
            continue
        try:
            with np.load(score_path, allow_pickle=True) as stored:
                feature_names = np.asarray(stored["feature_names"])
                p = int(feature_names.size)
                support = np.asarray(stored["support"], dtype=int)
                feature_score = max_feature_scores(stored["real_scores"], p)
                metrics = ranking_metrics(feature_score, support, k_values=k_values)
                row: dict[str, Any] = {
                    "score_file": score_path.name,
                    "p": p,
                    "replicate": int(meta["replicate"]),
                    "mechanism": mechanism,
                    "completion": completion,
                    "completion_draw": int(meta["draw"]),
                    "generator_label": generator,
                    "n_signal": int(support.size),
                    "random_average_precision_baseline": float(support.size / p),
                    **metrics,
                }
                rows.append(row)
        except Exception as exc:
            skipped.append({"file": score_path.name, "reason": repr(exc)})

    run_metrics = pd.DataFrame(rows)
    if run_metrics.empty:
        raise RuntimeError("No matching V9 score files were successfully read")
    atomic_write_csv(run_metrics, out_dir / "ranking_run_metrics.csv")
    if skipped:
        atomic_write_csv(pd.DataFrame(skipped), out_dir / "ranking_skipped_files.csv")

    metric_columns = [
        *[f"precision_at_{k}" for k in k_values],
        *[f"tp_at_{k}" for k in k_values],
        *[f"recall_at_{k}" for k in k_values],
        "average_precision",
        "median_signal_rank",
        "support_ranking_auroc",
        "mean_signal_score",
        "mean_null_score",
        "random_average_precision_baseline",
    ]
    replicate_metrics = (
        run_metrics.groupby(
            ["p", "replicate", "mechanism", "completion", "generator_label"],
            as_index=False,
        )[metric_columns]
        .mean()
    )
    atomic_write_csv(replicate_metrics, out_dir / "ranking_replicate_metrics.csv")

    draw_counts = (
        run_metrics.groupby(["mechanism", "completion", "replicate"], as_index=False)
        .agg(n_draws=("completion_draw", "nunique"))
    )
    expected_counts = (
        run_metrics.groupby(["mechanism", "completion"], as_index=False)
        .agg(
            n_score_files=("score_file", "size"),
            n_replicates=("replicate", "nunique"),
        )
        .merge(
            draw_counts.groupby(["mechanism", "completion"], as_index=False)
            .agg(min_draws_per_replicate=("n_draws", "min"), max_draws_per_replicate=("n_draws", "max")),
            on=["mechanism", "completion"],
            how="left",
        )
    )
    expected_counts["complete_pairing"] = (
        expected_counts["n_replicates"].eq(run_metrics["replicate"].nunique())
        & expected_counts["min_draws_per_replicate"].eq(run_metrics["completion_draw"].nunique())
        & expected_counts["max_draws_per_replicate"].eq(run_metrics["completion_draw"].nunique())
    )
    atomic_write_csv(expected_counts, out_dir / "ranking_integrity_audit.csv")

    primary_metrics = [
        *[f"precision_at_{k}" for k in k_values],
        "average_precision",
        "median_signal_rank",
    ]
    reference_metrics = [
        "support_ranking_auroc",
        "mean_signal_score",
        "mean_null_score",
    ]

    def build_contrasts(metrics: list[str], seed_offset: int) -> pd.DataFrame:
        contrast_rows: list[dict[str, Any]] = []
        for mechanism in mechanisms:
            block = replicate_metrics[replicate_metrics["mechanism"].eq(mechanism)]
            median = block[block["completion"].eq("median")].set_index("replicate")
            br = block[block["completion"].eq("bayesianridge_mean")].set_index("replicate")
            common = median.index.intersection(br.index)
            if common.empty:
                continue
            for metric in metrics:
                differences = br.loc[common, metric].to_numpy(dtype=float) - median.loc[common, metric].to_numpy(dtype=float)
                ci_low, ci_high = percentile_bootstrap_mean_ci(
                    differences,
                    seed=args.bootstrap_seed + seed_offset + len(contrast_rows),
                    n_bootstrap=args.n_bootstrap,
                )
                favorable_direction = "lower" if metric == "median_signal_rank" or metric == "mean_null_score" else "higher"
                favorable_fraction = (
                    float(np.mean(differences < 0))
                    if favorable_direction == "lower"
                    else float(np.mean(differences > 0))
                )
                contrast_rows.append(
                    {
                        "mechanism": mechanism,
                        "metric": metric,
                        "contrast": "bayesianridge_mean minus median",
                        "favorable_direction": favorable_direction,
                        "n_pairs": int(len(common)),
                        "median_mean": float(median.loc[common, metric].mean()),
                        "bayesianridge_mean": float(br.loc[common, metric].mean()),
                        "mean_difference": float(np.mean(differences)),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "paired_wilcoxon_p": paired_wilcoxon_p(differences),
                        "fraction_replicates_favorable": favorable_fraction,
                    }
                )
        result = pd.DataFrame(contrast_rows)
        if not result.empty:
            adjusted_blocks = []
            for mechanism, block in result.groupby("mechanism", sort=False):
                block = block.copy()
                block["holm_p_within_mechanism"] = holm_adjust(block["paired_wilcoxon_p"].to_numpy())
                adjusted_blocks.append(block)
            result = pd.concat(adjusted_blocks, ignore_index=True)
        return result

    primary = build_contrasts(primary_metrics, 0)
    reference = build_contrasts(reference_metrics, 10_000)
    atomic_write_csv(primary, out_dir / "ranking_primary_contrasts.csv")
    atomic_write_csv(reference, out_dir / "ranking_reference_contrasts.csv")

    completion_summary = (
        replicate_metrics.groupby(["mechanism", "completion"], as_index=False)[
            [*primary_metrics, "support_ranking_auroc", "random_average_precision_baseline"]
        ]
        .mean()
    )
    atomic_write_csv(completion_summary, out_dir / "ranking_completion_summary.csv")

    consistency_rows = []
    for mechanism in mechanisms:
        block = primary[
            primary["mechanism"].eq(mechanism)
            & primary["metric"].str.startswith("precision_at_")
        ]
        if block.empty:
            continue
        favorable = np.where(
            block["favorable_direction"].eq("higher"),
            block["mean_difference"] > 0,
            block["mean_difference"] < 0,
        )
        ci_favorable = np.where(
            block["favorable_direction"].eq("higher"),
            block["bootstrap_ci_low"] > 0,
            block["bootstrap_ci_high"] < 0,
        )
        consistency_rows.append(
            {
                "mechanism": mechanism,
                "n_k_values": int(len(block)),
                "all_precision_at_k_point_estimates_favorable": bool(np.all(favorable)),
                "n_precision_at_k_ci_excluding_zero_favorably": int(np.sum(ci_favorable)),
                "analysis_scope": "threshold_independent_support_ranking",
            }
        )
    consistency = pd.DataFrame(consistency_rows)
    atomic_write_csv(consistency, out_dir / "ranking_consistency_summary.csv")

    figures = out_dir / "figures_ranking"
    figures.mkdir(exist_ok=True)
    for mechanism in mechanisms:
        block = completion_summary[completion_summary["mechanism"].eq(mechanism)]
        if block.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for _, row in block.iterrows():
            values = [row[f"precision_at_{k}"] for k in k_values]
            ax.plot(k_values, values, marker="o", label=row["completion"])
        ax.axhline(float(run_metrics["n_signal"].iloc[0] / run_metrics["p"].iloc[0]), linewidth=1, linestyle=":")
        ax.set_xlabel("Fixed selection size k")
        ax.set_ylabel("Tie-aware Precision@k")
        ax.set_title(f"V9.1 threshold-independent ranking: {mechanism}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"precision_at_k_{mechanism}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    report_lines = [
        "# V9.1 Threshold-Independent Ranking",
        "",
        f"Source V9 output: `{v9_root}`",
        "",
        "Feature score is the maximum real-feature STABL selection frequency across the regularization grid.",
        "Precision@k is tie-aware and uses expected precision under uniform random tie-breaking when k cuts through a tied score block.",
        "",
        "## Integrity",
        "",
        dataframe_to_markdown(expected_counts, index=False),
        "",
        "## Primary paired contrasts",
        "",
        dataframe_to_markdown(primary, index=False),
        "",
        "## Consistency across fixed selection sizes",
        "",
        dataframe_to_markdown(consistency, index=False),
        "",
        "## Reference diagnostics",
        "",
        dataframe_to_markdown(reference, index=False),
    ]
    (out_dir / "v91_ranking_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[V9.1 ranking] read {len(run_metrics)} score files")
    print(f"[V9.1 ranking] outputs written to {out_dir}")


if __name__ == "__main__":
    main()
