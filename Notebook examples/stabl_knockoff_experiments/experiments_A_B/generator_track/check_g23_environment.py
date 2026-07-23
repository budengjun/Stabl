#!/usr/bin/env python3
"""Check the server environment for the G2.3 Gaussian-family experiment."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from research_common import GeneratorConfig, generate_knockoff


def main() -> int:
    failed = False
    print(f"Python: {sys.executable}")
    for name in ("numpy", "pandas", "scipy", "sklearn", "knockpy", "stabl"):
        try:
            module = importlib.import_module(name)
            print(f"[ok] {name}: {getattr(module, '__version__', 'available')}")
        except Exception as exc:  # noqa: BLE001
            failed = True
            print(f"[missing] {name}: {exc!r}")

    if failed:
        return 1

    rng = np.random.default_rng(20260727)
    X = pd.DataFrame(rng.normal(size=(80, 20)))
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    configs = [
        GeneratorConfig("equicorr", "gaussian_equicorrelated"),
        GeneratorConfig("mvr", "gaussian_mvr"),
        GeneratorConfig("sdp", "gaussian_sdp"),
        GeneratorConfig(
            "factor_r5_mvr",
            "gaussian_factor_mvr",
            factor_rank=5,
        ),
    ]
    for index, config in enumerate(configs):
        try:
            pair = generate_knockoff(X, config, seed=101 + index)
            print(
                f"[ok] {config.label}: shape={pair.X_tilde.shape}, "
                f"finite={np.isfinite(pair.X_tilde).all()}, "
                f"seconds={pair.generation_seconds:.3f}"
            )
        except Exception as exc:  # noqa: BLE001
            failed = True
            print(f"[failed] {config.label}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
