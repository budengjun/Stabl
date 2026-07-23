#!/usr/bin/env python3
"""Merge a frozen tuned PLSKO config with Gaussian Generator Track baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-plsko", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-default-plsko", action="store_true")
    args = parser.parse_args()

    frozen = json.loads(Path(args.frozen_plsko).read_text(encoding="utf-8"))
    if not isinstance(frozen, list) or len(frozen) != 1:
        raise ValueError("Frozen PLSKO JSON must contain exactly one configuration")
    configs = [
        {"label": "equicorr", "generator": "gaussian_equicorrelated"},
        {"label": "mvr", "generator": "gaussian_mvr"},
        frozen[0],
    ]
    if args.include_default_plsko:
        configs.append(
            {
                "label": "plsko_default_diagnostic",
                "generator": "official_plsko",
                "threshold_abs": None,
                "threshold_q": 0.8,
                "ncomp": None,
                "sparsity": 1.0,
            }
        )
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(configs, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
