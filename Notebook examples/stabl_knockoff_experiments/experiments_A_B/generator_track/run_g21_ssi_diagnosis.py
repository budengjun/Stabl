#!/usr/bin/env python3
"""Convenience Python launcher for the SSI G2.1 diagnosis.

Run this file from experiments_A_B. It performs the complete saved-score path
analysis and expands C2ST to every stored G2 knockoff draw.
"""

from __future__ import annotations

from pathlib import Path

from run_g21_saved_score_diagnosis import run


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    g2_dir = root / "generator_track_results" / "G2_ssi_pilot"
    out_dir = g2_dir / "G21_saved_score_diagnosis"
    run(
        [
            "--g2-dir",
            str(g2_dir),
            "--out-dir",
            str(out_dir),
            "--comparisons",
            (
                "plsko_tuned_ssi:equicorr,"
                "plsko_tuned_ssi:mvr,"
                "plsko_default_diagnostic:equicorr,"
                "plsko_tuned_ssi:plsko_default_diagnostic"
            ),
            "--expanded-c2st",
            "--save-full-threshold-path",
            "--make-figures",
            "--bootstrap-samples",
            "20000",
            "--random-state",
            "20260723",
        ]
    )


if __name__ == "__main__":
    main()
