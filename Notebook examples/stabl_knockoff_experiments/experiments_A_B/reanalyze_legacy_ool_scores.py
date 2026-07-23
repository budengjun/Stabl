#!/usr/bin/env python3
"""Reapply STABL selection rules to the existing OOL score cache.

This script deliberately does not claim empirical FDR or power because the old
OOL clinical outcome has no known support.  It reuses the expensive fitted
selection-frequency paths to compare thresholds and selected-set sizes without
regenerating knockoffs or refitting STABL.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_common import apply_selection_rule, atomic_write_csv


COMBINED_PATTERN = re.compile(
    r"fold_(?P<fold>\d+)__(?P<omic>.+?)__(?P<generator>.+?)__(?P<timing>.+)\.npz$"
)
CHECKPOINT_PATTERN = re.compile(
    r"fold_(?P<fold>\d+)__(?P<omic>.+?)__(?P<generator>.+?)__"
    r"(?P<timing>.+?)__draw_(?P<draw>\d+)\.npz$"
)


def normalize_draws(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError(f"Expected scores with shape (draw, p, lambda), got {values.shape}")
    return values


def parse_name(path: Path) -> dict[str, Any]:
    match = CHECKPOINT_PATTERN.match(path.name)
    if match:
        info = match.groupdict()
        return {
            "fold": int(info["fold"]),
            "omic": info["omic"],
            "generator": info["generator"],
            "timing": info["timing"],
            "draw_from_name": int(info["draw"]),
        }
    match = COMBINED_PATTERN.match(path.name)
    if match:
        info = match.groupdict()
        return {
            "fold": int(info["fold"]),
            "omic": info["omic"],
            "generator": info["generator"],
            "timing": info["timing"],
            "draw_from_name": None,
        }
    raise ValueError(f"Unrecognized legacy score filename: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--score-dir",
        required=True,
        help="Old OOL scores/ or score_checkpoints/ directory",
    )
    parser.add_argument("--out-csv", default="legacy_ool_threshold_reanalysis.csv")
    parser.add_argument("--target-fdr", type=float, default=0.10)
    parser.add_argument(
        "--selection-rules", default="stabl_min,stabl_q,knockoff_plus"
    )
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--include-selected-features",
        action="store_true",
        help="Store selected feature names as a JSON string in the CSV",
    )
    args = parser.parse_args()

    if not 0.0 < args.target_fdr < 1.0:
        raise ValueError("target_fdr must lie in (0, 1)")
    score_dir = Path(args.score_dir).expanduser().resolve()
    files = sorted(score_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ score files found in {score_dir}")

    rules = tuple(item.strip() for item in args.selection_rules.split(",") if item.strip())
    threshold_grid = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2.0,
        args.threshold_step,
    )
    rows: list[dict[str, Any]] = []

    for index, path in enumerate(files, start=1):
        info = parse_name(path)
        with np.load(path, allow_pickle=True) as loaded:
            real_draws = normalize_draws(loaded["real_scores"])
            knockoff_draws = normalize_draws(loaded["knockoff_scores"])
            if real_draws.shape != knockoff_draws.shape:
                raise ValueError(f"Score shape mismatch in {path}")
            feature_names = (
                np.asarray(loaded["feature_names"]).astype(str)
                if "feature_names" in loaded.files
                else np.asarray([f"x{j}" for j in range(real_draws.shape[1])])
            )

        for local_draw in range(real_draws.shape[0]):
            draw = (
                info["draw_from_name"]
                if info["draw_from_name"] is not None
                else local_draw
            )
            for rule in rules:
                result = apply_selection_rule(
                    rule,  # type: ignore[arg-type]
                    real_draws[local_draw],
                    knockoff_draws[local_draw],
                    q=args.target_fdr,
                    threshold_grid=threshold_grid,
                )
                row: dict[str, Any] = {
                    "fold": info["fold"],
                    "omic": info["omic"],
                    "generator": info["generator"],
                    "timing": info["timing"],
                    "draw": int(draw),
                    "selection_rule": rule,
                    "target_fdr": args.target_fdr,
                    "threshold": result.threshold,
                    "estimated_fdp": result.estimated_fdp,
                    "n_selected": int(result.selected.sum()),
                    "source_file": str(path),
                }
                if args.include_selected_features:
                    row["selected_features_json"] = json.dumps(
                        feature_names[result.selected].tolist(), ensure_ascii=False
                    )
                rows.append(row)
        if index % 20 == 0 or index == len(files):
            print(f"[processed] {index}/{len(files)} files", flush=True)

    frame = pd.DataFrame(rows).sort_values(
        ["fold", "omic", "generator", "timing", "draw", "selection_rule"]
    )
    out_path = Path(args.out_csv).expanduser().resolve()
    atomic_write_csv(frame, out_path)

    summary = (
        frame.groupby(["omic", "generator", "timing", "selection_rule"], dropna=False)
        .agg(
            n_runs=("n_selected", "size"),
            selected_mean=("n_selected", "mean"),
            selected_median=("n_selected", "median"),
            zero_selection_rate=("n_selected", lambda x: float(np.mean(np.asarray(x) == 0))),
            estimated_fdp_mean=("estimated_fdp", "mean"),
        )
        .reset_index()
    )
    summary_path = out_path.with_name(f"{out_path.stem}_summary.csv")
    atomic_write_csv(summary, summary_path)
    print(f"[done] rows={len(frame)}", flush=True)
    print(f"[output] {out_path}", flush=True)
    print(f"[output] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
