#!/usr/bin/env python3
"""Convenience launcher for default SSI G2.2B screening."""

from __future__ import annotations

from pathlib import Path

from run_g22b_stabl_aware_screening import build_parser, run


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = build_parser()
    args = parser.parse_args(
        [
            "--ssi-data-path",
            "/data/yhu94/Stabl/Sample Data/Biobank SSI",
            "--ssi-omic",
            "CyTOF",
            "--plsko-grid-json",
            str(Path(__file__).with_name("configs") / "g22_plsko_screen_grid.json"),
            "--c2st-gate-json",
            str(
                root
                / "generator_track_results"
                / "G22A_validity_calibration"
                / "g22a_recommended_c2st_gate.json"
            ),
            "--out-dir",
            str(root / "generator_track_results" / "G22B_stabl_aware_screening"),
            "--p",
            "100",
            "--validity-draws",
            "5",
            "--screen-replicates",
            "8",
            "--screen-draws",
            "2",
            "--n-bootstraps",
            "50",
            "--n-jobs",
            "8",
            "--overwrite",
        ]
    )
    out_dir = run(args)
    print(f"G2.2B SSI screening completed: {out_dir}")


if __name__ == "__main__":
    main()
