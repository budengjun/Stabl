#!/usr/bin/env python3
"""Reanalyze existing V6 component score banks with V6.1 pooling rules.

This script does not refit STABL or regenerate knockoffs.  It is the preferred
first step after the V6 pilot because all 400 component score paths can be
reused directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from posterior_pooling_common import (
    encode_indices,
    max_score_dispersion,
    mean_pairwise_jaccard,
    parse_aggregators,
    parse_pool_sizes,
    pool_score_paths,
)
from research_common import (
    apply_selection_rule,
    atomic_savez_compressed,
    atomic_write_csv,
    selection_metrics,
)
from validity_aware_pooling import (
    curve_value_at_selected_threshold,
    parse_pooling_rules,
    parse_vote_fractions,
    select_component_vote,
    select_mean_real_component_artificial_count,
)

def parse_score_name(path: Path) -> tuple[int, str, str, str, int] | None:
    parts = path.stem.split("__")
    if len(parts) != 5 or not parts[0].startswith("rep_") or not parts[4].startswith("draw_"):
        return None
    return (
        int(parts[0].removeprefix("rep_")),
        parts[1],
        parts[2],
        parts[3],
        int(parts[4].removeprefix("draw_")),
    )



def upsert(rows: list[dict[str, Any]], row: dict[str, Any], keys: tuple[str, ...]) -> None:
    target = tuple(row.get(key) for key in keys)
    rows[:] = [old for old in rows if tuple(old.get(key) for key in keys) != target]
    rows.append(row)


def load_component(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        real = np.asarray(payload["real_scores"], dtype=float)
        artificial = np.asarray(payload["knockoff_scores"], dtype=float)
        support = np.asarray(payload["support"], dtype=int)
        names = np.asarray(payload["feature_names"]).astype(str)
    if real.ndim == 3 and real.shape[0] == 1:
        real = real[0]
    if artificial.ndim == 3 and artificial.shape[0] == 1:
        artificial = artificial[0]
    if real.ndim != 2 or artificial.shape != real.shape:
        raise RuntimeError(f"Invalid component score shapes in {path}")
    if not np.isfinite(real).all() or not np.isfinite(artificial).all():
        raise RuntimeError(f"Nonfinite component scores in {path}")
    return real, artificial, support, names


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pool-sizes", default=None)
    parser.add_argument("--pooling-aggregators", default="mean")
    parser.add_argument(
        "--pooling-rules",
        default=(
            "mean_path_naive,mean_real_component_artificial_count,"
            "component_selection_vote"
        ),
    )
    parser.add_argument("--vote-fractions", default="0.5,0.8")
    parser.add_argument("--target-fdr", type=float, default=None)
    parser.add_argument("--threshold-min", type=float, default=None)
    parser.add_argument("--threshold-max", type=float, default=None)
    parser.add_argument("--threshold-step", type=float, default=None)
    parser.add_argument("--save-pooled-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_dir).expanduser().resolve()
    score_dir = source / "component_scores"
    if not score_dir.is_dir():
        raise FileNotFoundError(f"Missing component score directory: {score_dir}")
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    source_pool_sizes = tuple(int(v) for v in source_config.get("pool_sizes_resolved", []))
    pool_sizes = parse_pool_sizes(args.pool_sizes) if args.pool_sizes else source_pool_sizes
    if not pool_sizes:
        raise ValueError("No pool sizes available")
    max_pool_size = max(pool_sizes)
    if max_pool_size > int(source_config.get("max_pool_size", 0)):
        raise ValueError("Requested pool size exceeds the existing V6 component bank")
    aggregators = parse_aggregators(args.pooling_aggregators)
    pooling_rules = parse_pooling_rules(args.pooling_rules)
    vote_fractions = parse_vote_fractions(args.vote_fractions)
    target_fdr = float(
        source_config.get("target_fdr", 0.10) if args.target_fdr is None else args.target_fdr
    )
    threshold_min = float(
        source_config.get("threshold_min", 0.10) if args.threshold_min is None else args.threshold_min
    )
    threshold_max = float(
        source_config.get("threshold_max", 0.99) if args.threshold_max is None else args.threshold_max
    )
    threshold_step = float(
        source_config.get("threshold_step", 0.01) if args.threshold_step is None else args.threshold_step
    )
    threshold_grid = np.arange(
        threshold_min, threshold_max + threshold_step / 2.0, threshold_step
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f"Output directory is not empty: {out_dir}")
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled_score_dir = out_dir / "pooled_scores"
    pooled_score_dir.mkdir(exist_ok=True)

    reanalysis_config = {
        "experiment": "V6.1_validity_aware_reanalysis",
        "version": "6.1",
        "source_dir": str(source),
        "source_config": source_config,
        "pool_sizes_resolved": pool_sizes,
        "pooling_aggregators_resolved": aggregators,
        "pooling_rules_resolved": pooling_rules,
        "vote_fractions_resolved": vote_fractions,
        "target_fdr": target_fdr,
        "threshold_grid": threshold_grid.tolist(),
        "primary_pooling_rule": "mean_real_component_artificial_count",
    }
    (out_dir / "config.json").write_text(
        json.dumps(reanalysis_config, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    diagnostics_path = source / "component_draw_diagnostics.csv"
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    missing_lookup: dict[tuple[int, str, str, str], float] = {}
    if not diagnostics.empty and "realized_missing_rate" in diagnostics:
        for keys, group in diagnostics.groupby(
            ["replicate", "mechanism", "completion", "generator_label"], dropna=False
        ):
            missing_lookup[tuple(keys)] = float(group.realized_missing_rate.mean())

    groups: dict[tuple[int, str, str, str], list[tuple[int, Path]]] = {}
    for path in sorted(score_dir.glob("*.npz")):
        parsed = parse_score_name(path)
        if parsed is None:
            continue
        replicate, mechanism, completion, generator_label, draw = parsed
        key = (replicate, mechanism, completion, generator_label)
        groups.setdefault(key, []).append((draw, path))
    if not groups:
        raise FileNotFoundError(f"No component score files found in {score_dir}")

    rows: list[dict[str, Any]] = []
    row_keys = (
        "replicate", "mechanism", "completion", "generator_label",
        "pool_size", "pooling_aggregator", "pooling_rule", "vote_fraction",
    )

    for key, entries in sorted(groups.items()):
        replicate, mechanism, completion, generator_label = key
        entries = sorted(entries)
        draws = [draw for draw, _ in entries]
        if draws[:max_pool_size] != list(range(max_pool_size)):
            raise RuntimeError(f"Missing nested component draws for {key}: found {draws}")
        real_components: list[np.ndarray] = []
        artificial_components: list[np.ndarray] = []
        support_ref: np.ndarray | None = None
        names_ref: np.ndarray | None = None
        for draw, path in entries[:max_pool_size]:
            real, artificial, support, names = load_component(path)
            if support_ref is None:
                support_ref = support
                names_ref = names
            elif not np.array_equal(support_ref, support) or not np.array_equal(names_ref, names):
                raise RuntimeError(f"Support or feature order mismatch within {key}")
            real_components.append(real)
            artificial_components.append(artificial)
        assert support_ref is not None and names_ref is not None
        p = int(real_components[0].shape[0])
        component_results = [
            apply_selection_rule(
                "stabl_min", real_components[index], artificial_components[index],
                q=target_fdr, threshold_grid=threshold_grid,
            )
            for index in range(max_pool_size)
        ]
        component_masks = [item.selected for item in component_results]
        component_estimated_fdps = np.asarray(
            [item.estimated_fdp for item in component_results], dtype=float
        )
        component_selected_counts = np.asarray(
            [item.selected.sum() for item in component_results], dtype=float
        )

        for pool_size in pool_sizes:
            for aggregator in aggregators:
                pooled_real = pool_score_paths(
                    real_components, pool_size=pool_size, aggregator=aggregator
                )
                pooled_artificial = pool_score_paths(
                    artificial_components, pool_size=pool_size, aggregator=aggregator
                )
                rule_results: list[tuple[str, float, Any, dict[str, float], dict[str, np.ndarray]]] = []
                if "mean_path_naive" in pooling_rules:
                    result = apply_selection_rule(
                        "stabl_min", pooled_real, pooled_artificial,
                        q=target_fdr, threshold_grid=threshold_grid,
                    )
                    rule_results.append(("mean_path_naive", -1.0, result, {}, {}))
                if "mean_real_component_artificial_count" in pooling_rules:
                    validity = select_mean_real_component_artificial_count(
                        pooled_real,
                        artificial_components,
                        pool_size=pool_size,
                        threshold_grid=threshold_grid,
                        naive_pooled_artificial_scores=pooled_artificial,
                    )
                    rule_results.append((
                        "mean_real_component_artificial_count",
                        -1.0,
                        validity.result,
                        curve_value_at_selected_threshold(validity),
                        {
                            "fdp_curve": validity.fdp_curve,
                            "real_count_curve": validity.real_count_curve,
                            "artificial_count_curve": validity.artificial_count_curve,
                            "artificial_count_sd_curve": validity.artificial_count_sd_curve,
                            "naive_artificial_count_curve": validity.naive_artificial_count_curve,
                        },
                    ))
                if "component_selection_vote" in pooling_rules:
                    for vote_fraction in vote_fractions:
                        result = select_component_vote(
                            component_masks,
                            pool_size=pool_size,
                            vote_fraction=vote_fraction,
                        )
                        rule_results.append((
                            "component_selection_vote", float(vote_fraction), result, {}, {}
                        ))

                for pooling_rule, vote_fraction, result, extra, curve_payload in rule_results:
                    row = {
                        "replicate": replicate,
                        "mechanism": mechanism,
                        "completion": completion,
                        "generator_label": generator_label,
                        "pool_size": pool_size,
                        "pooling_aggregator": aggregator,
                        "pooling_rule": pooling_rule,
                        "vote_fraction": vote_fraction,
                        "selection_rule": result.rule,
                        "threshold": result.threshold,
                        "estimated_fdp": result.estimated_fdp,
                        "mean_component_estimated_fdp": float(
                            component_estimated_fdps[:pool_size].mean()
                        ),
                        "mean_component_selected_count": float(
                            component_selected_counts[:pool_size].mean()
                        ),
                        "selected_indices": encode_indices(result.selected),
                        "n": int(source_config.get("n", 0)),
                        "p": p,
                        "n_signal": int(len(support_ref)),
                        "target_fdr": target_fdr,
                        "component_selection_jaccard_mean": mean_pairwise_jaccard(
                            component_masks[:pool_size]
                        ),
                        "realized_missing_rate": missing_lookup.get(key, float("nan")),
                        "error": "",
                        **extra,
                        **selection_metrics(
                            result.selected, support_ref, p, target_fdr=target_fdr
                        ),
                    }
                    real_disp = max_score_dispersion(real_components, pool_size=pool_size)
                    art_disp = max_score_dispersion(artificial_components, pool_size=pool_size)
                    row.update({f"real_{k}": v for k, v in real_disp.items()})
                    row.update({f"artificial_{k}": v for k, v in art_disp.items()})
                    upsert(rows, row, row_keys)
                    if args.save_pooled_scores:
                        vote_label = (
                            f"__vote_{vote_fraction:g}"
                            if pooling_rule == "component_selection_vote" else ""
                        )
                        atomic_savez_compressed(
                            pooled_score_dir / (
                                f"rep_{replicate:04d}__{mechanism}__{completion}__"
                                f"{generator_label}__M_{pool_size:03d}__{aggregator}__"
                                f"{pooling_rule}{vote_label}.npz"
                            ),
                            real_scores=pooled_real,
                            knockoff_scores=pooled_artificial,
                            selected=result.selected,
                            support=support_ref,
                            feature_names=names_ref,
                            threshold_grid=threshold_grid,
                            threshold=np.asarray(result.threshold),
                            estimated_fdp=np.asarray(result.estimated_fdp),
                            pool_size=np.asarray(pool_size),
                            pooling_rule=np.asarray(pooling_rule),
                            vote_fraction=np.asarray(vote_fraction),
                            **curve_payload,
                        )
        print(f"[reanalysis] {key} completed", flush=True)

    frame = pd.DataFrame(rows).sort_values(list(row_keys)).reset_index(drop=True)
    atomic_write_csv(frame, out_dir / "validity_aware_selection_results.csv")
    summary = (
        frame.groupby(
            ["mechanism", "completion", "generator_label", "pool_size",
             "pooling_aggregator", "pooling_rule", "vote_fraction"],
            as_index=False,
        )
        .agg(
            n_success=("fdp", "size"),
            empirical_fdr=("fdp", "mean"),
            power_mean=("power", "mean"),
            selected_mean=("n_selected", "mean"),
            true_positives_mean=("true_positives", "mean"),
            false_positives_mean=("false_positives", "mean"),
            estimated_fdp_mean=("estimated_fdp", "mean"),
            mean_component_estimated_fdp=("mean_component_estimated_fdp", "mean"),
            component_jaccard_mean=("component_selection_jaccard_mean", "mean"),
        )
    )
    summary["calibration_gap"] = summary.empirical_fdr - summary.estimated_fdp_mean
    atomic_write_csv(summary, out_dir / "validity_aware_summary.csv")
    print(f"\nV6.1 reanalysis completed. Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
