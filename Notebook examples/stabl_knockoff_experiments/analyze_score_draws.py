#!/usr/bin/env python3
"""Analyze repeated paired STABL score draws saved by the audit runners.

Each input NPZ must contain repeated knockoff copies generated on the same
training dataset.  The script compares three procedures:

1. STABL FDP+ applied to the first draw
2. STABL FDP+ applied after averaging real and knockoff scores
3. Derandomised knockoffs using averaged e values and e BH

Outer cross validation folds are analysed separately.  They are never pooled as
if they were repeated knockoff copies.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from derandomized_stabl_v2 import (
    compare_score_aggregators,
    derandomized_stabl_select,
    load_score_draws,
    stabl_fdp_plus_select,
)


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def analyze_one(path: Path, alpha_ebh: float, alpha_kn: float | None):
    real, knockoff, names, metadata = load_score_draws(path)
    if len(real) != len(knockoff) or len(real) == 0:
        raise ValueError(f"{path} does not contain matched score draws")

    feature_table, method_table = compare_score_aggregators(
        real,
        knockoff,
        alpha_ebh=alpha_ebh,
        alpha_kn=alpha_kn,
        feature_names=names,
    )
    derand_table, derand_summary = derandomized_stabl_select(
        real,
        knockoff,
        alpha_ebh=alpha_ebh,
        alpha_kn=alpha_kn,
        feature_names=names,
    )

    real_flat = np.vstack([np.max(x, axis=1) if np.asarray(x).ndim == 2 else x for x in real])
    knock_flat = np.vstack(
        [np.max(x, axis=1) if np.asarray(x).ndim == 2 else x for x in knockoff]
    )
    mean_sel, theta_mean, q_mean, curve = stabl_fdp_plus_select(
        real_flat.mean(axis=0), knock_flat.mean(axis=0)
    )

    # Draw level diagnostics reveal whether the artificial block changes more
    # strongly across knockoff copies than the real block.
    draw_summary = pd.DataFrame(
        {
            "draw": np.arange(len(real_flat)),
            "real_mean": real_flat.mean(axis=1),
            "knockoff_mean": knock_flat.mean(axis=1),
            "real_sd_across_features": real_flat.std(axis=1),
            "knockoff_sd_across_features": knock_flat.std(axis=1),
            "W_positive_fraction": np.mean(real_flat - knock_flat > 0, axis=1),
        }
    )

    # Variability across repeated knockoff copies for each feature.
    variability = pd.DataFrame(
        {
            "feature": names,
            "real_between_draw_sd": real_flat.std(axis=0),
            "knockoff_between_draw_sd": knock_flat.std(axis=0),
            "mean_real_score": real_flat.mean(axis=0),
            "mean_knockoff_score": knock_flat.mean(axis=0),
            "mean_W": (real_flat - knock_flat).mean(axis=0),
            "selected_mean_fdp_plus": mean_sel,
        }
    )
    variability["knockoff_to_real_draw_sd_ratio"] = (
        variability["knockoff_between_draw_sd"]
        / variability["real_between_draw_sd"].clip(lower=1e-12)
    )

    row = {
        "file": str(path),
        "n_draws": len(real),
        "p": len(names),
        "alpha_ebh": alpha_ebh,
        "alpha_kn": derand_summary.alpha_kn,
        "mean_score_theta": theta_mean,
        "mean_score_q_plus": q_mean,
        "mean_score_n_selected": int(mean_sel.sum()),
        **{f"meta_{key}": _jsonable(value) for key, value in metadata.items()},
    }
    for _, method_row in method_table.iterrows():
        method = str(method_row["method"])
        for key, value in method_row.items():
            if key != "method":
                row[f"{method}_{key}"] = _jsonable(value)
    row.update({f"derand_{key}": _jsonable(value) for key, value in asdict(derand_summary).items()})

    return row, feature_table, derand_table, method_table, draw_summary, variability, curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="NPZ files or directories containing NPZ files")
    parser.add_argument("--out-dir", default="./score_draw_analysis")
    parser.add_argument("--alpha-ebh", type=float, default=0.10)
    parser.add_argument("--alpha-kn", type=float, default=None)
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.npz")))
        elif path.suffix == ".npz":
            paths.append(path)
        else:
            raise ValueError(f"Input is neither an NPZ file nor a directory: {path}")
    paths = sorted(dict.fromkeys(path.resolve() for path in paths))
    if not paths:
        raise ValueError("No score NPZ files found")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    details = out_dir / "details"
    details.mkdir(exist_ok=True)

    summary_rows = []
    failures = []
    for index, path in enumerate(paths):
        stem = f"{index:04d}__{path.stem}"
        try:
            (
                row,
                feature_table,
                derand_table,
                method_table,
                draw_summary,
                variability,
                curve,
            ) = analyze_one(path, args.alpha_ebh, args.alpha_kn)
            summary_rows.append(row)
            feature_table.to_csv(details / f"{stem}__feature_comparison.csv", index=False)
            derand_table.to_csv(details / f"{stem}__derandomized_features.csv", index=False)
            method_table.to_csv(details / f"{stem}__method_summary.csv", index=False)
            draw_summary.to_csv(details / f"{stem}__draw_summary.csv", index=False)
            variability.to_csv(details / f"{stem}__draw_variability.csv", index=False)
            curve.to_csv(details / f"{stem}__mean_score_fdp_curve.csv", index=False)
            print(f"[done] {path}", flush=True)
        except Exception as exc:
            failures.append({"file": str(path), "error": repr(exc)})
            print(f"[failed] {path}: {exc!r}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "aggregator_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    if not summary.empty:
        print("\n=== Aggregator summary ===")
        cols = [
            column
            for column in (
                "file",
                "n_draws",
                "p",
                "single_draw_fdp_plus_n_selected",
                "mean_score_fdp_plus_n_selected",
                "derandomised_e_bh_n_selected",
                "mean_score_theta",
                "derand_finite_threshold_fraction",
            )
            if column in summary.columns
        ]
        print(summary[cols].to_string(index=False))
    print(f"\nWrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
