#!/usr/bin/env python3
"""Check the Python and optional R dependencies for Experiments A and B."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "knockpy", "stabl"]


def main() -> int:
    failed = False
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "available")
            print(f"[ok] {name}: {version}")
        except Exception as exc:
            failed = True
            print(f"[missing] {name}: {exc!r}")

    try:
        from stabl.stabl import Stabl  # noqa: F401

        print("[ok] stabl.stabl.Stabl")
    except Exception as exc:
        failed = True
        print(f"[missing] stabl.stabl.Stabl: {exc!r}")

    bridge = Path(__file__).with_name("official_plsko_bridge.R")
    tuning_bridge = Path(__file__).with_name("generator_track") / "official_plsko_tuning_bridge.R"
    if shutil.which("Rscript") is None:
        print("[optional missing] Rscript. Gaussian experiments still work.")
    else:
        result = subprocess.run(
            [
                "Rscript",
                "-e",
                "quit(status=ifelse(requireNamespace('PLSKO', quietly=TRUE), 0, 2))",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and bridge.exists():
            print("[ok] official R PLSKO generation bridge")
            if tuning_bridge.exists():
                print("[ok] official R PLSKO tuning bridge")
            else:
                print(f"[missing] PLSKO tuning bridge: {tuning_bridge}")
                failed = True
        else:
            print("[optional missing] official R PLSKO. Gaussian experiments still work.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
