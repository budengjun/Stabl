#!/usr/bin/env python3
"""Validate and summarize a V4 Experiment A or B output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_common import atomic_write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    selection_path = out_dir / "selection_results.csv"
    diagnostic_path = out_dir / "draw_diagnostics.csv"
    config_path = out_dir / "config.json"
    if not selection_path.exists() or not diagnostic_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Expected config.json, selection_results.csv, and draw_diagnostics.csv "
            f"under {out_dir}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection = pd.read_csv(selection_path)
    diagnostics = pd.read_csv(diagnostic_path)

    selection_errors = int(selection.get("error", pd.Series("", index=selection.index)).fillna("").ne("").sum())
    diagnostic_errors = int(diagnostics.get("error", pd.Series("", index=diagnostics.index)).fillna("").ne("").sum())

    print(f"experiment={config.get('experiment')} version={config.get('version')}")
    print(f"selection_rows={len(selection)} selection_errors={selection_errors}")
    print(f"diagnostic_rows={len(diagnostics)} diagnostic_errors={diagnostic_errors}")

    valid = selection[selection.get("error", pd.Series("", index=selection.index)).fillna("").eq("")].copy()
    if valid.empty:
        raise RuntimeError("No successful selection rows were found")

    if config.get("experiment") == "A_oracle_completion":
        group_columns = [
            "mechanism",
            "completion",
            "generator_label",
            "selection_rule",
        ]
    else:
        group_columns = [
            "dataset",
            "p_setting",
            "actual_p",
            "generator_label",
            "selection_rule",
        ]

    summary = (
        valid.groupby(group_columns, dropna=False)
        .agg(
            n_success=("fdp", "size"),
            nonempty_rate=("n_selected", lambda x: float(np.mean(np.asarray(x) > 0))),
            empirical_fdr=("fdp", "mean"),
            power_mean=("power", "mean"),
            selected_mean=("n_selected", "mean"),
            fdp_exceedance_probability=("fdp_exceeds_target", "mean"),
            estimated_fdp_median=("estimated_fdp", "median"),
        )
        .reset_index()
    )
    destination = out_dir / "v4_core_summary.csv"
    atomic_write_csv(summary, destination)
    print(f"wrote={destination}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
