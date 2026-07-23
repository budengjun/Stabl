#!/usr/bin/env python3
"""Convenience launcher for the default SSI plus Gaussian G2.2A audit."""

from __future__ import annotations

from pathlib import Path

from run_g22a_validity_calibration import build_parser, run


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = build_parser()
    args = parser.parse_args(
        [
            "--ssi-data-path",
            "/data/yhu94/Stabl/Sample Data/Biobank SSI",
            "--ssi-omic",
            "CyTOF",
            "--tuned-config-json",
            str(
                root
                / "generator_track_results"
                / "G1_ssi_pilot"
                / "frozen_plsko_generator.json"
            ),
            "--out-dir",
            str(root / "generator_track_results" / "G22A_validity_calibration"),
            "--p",
            "100",
            "--synthetic-replicates",
            "5",
            "--draws",
            "2",
            "--ssi-draws",
            "10",
            "--c2st-repeats",
            "3",
            "--n-single-features",
            "3",
            "--block-size",
            "25",
            "--local-swap-every",
            "2",
            "--overwrite",
        ]
    )
    out_dir = run(args)
    print(f"G2.2A SSI validity audit completed: {out_dir}")


if __name__ == "__main__":
    main()
