#!/usr/bin/env python3
"""Check Python, STABL, knockpy, and optional official PLSKO dependencies."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from official_plsko_bridge import check_official_plsko_installation


def main() -> int:
    modules = ["numpy", "pandas", "scipy", "sklearn", "knockpy", "stabl"]
    failed = False
    print(f"Python: {sys.executable}")
    print(f"Working directory: {Path.cwd()}")
    for name in modules:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"[ok] {name}: {version}")
        except Exception as exc:
            failed = True
            print(f"[missing] {name}: {exc!r}")

    ok, message = check_official_plsko_installation()
    print(f"[{'ok' if ok else 'optional missing'}] official PLSKO: {message}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
