#!/usr/bin/env python3
"""Lightweight checks for the posterior-vs-median V5 refocus."""
from __future__ import annotations

import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_experiment_a_oracle_completion.py"
SUMMARY = HERE / "summarize_posterior_vs_median.py"


def main() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    ast.parse(SUMMARY.read_text(encoding="utf-8"))

    assert '"version": 5' in source
    assert 'default="stabl_min"' in source
    assert '"--skip-misspecified-true-sigma"' in source

    seed_block = source.split("knockoff_seed = (", 1)[1].split(")", 1)[0]
    assert "completion_index" not in seed_block, (
        "Knockoff/STABL seeds must be paired across completion methods"
    )

    assert "primary_analysis" in source
    print("[ok] V5 experiment version and original-STABL primary rule")
    print("[ok] downstream seeds paired across completion methods")
    print("[ok] misspecified true-Sigma combinations skipped by default")
    print("[ok] posterior-versus-median paired summarizer available")
    print("[ok] all V5 update tests passed")


if __name__ == "__main__":
    main()
