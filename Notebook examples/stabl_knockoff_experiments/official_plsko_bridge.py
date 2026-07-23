"""Python wrapper around the authors' official PLSKO R package.

The initial Python port supplied with this project is intentionally not used in
this wrapper.  PLSKO includes several implementation details and tuning rules,
so scientific comparisons should use the authors' package unless an independent
Python port has been validated by simulation against that package.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


@dataclass
class OfficialPLSKOSampler:
    X: ArrayLike
    threshold_abs: float | None = None
    threshold_q: float | None = 0.8
    ncomp: int | None = None
    sparsity: float = 1.0
    random_state: int = 0
    rscript: str = "Rscript"
    bridge_script: str | Path | None = None
    timeout_seconds: int = 7200

    def _bridge_path(self) -> Path:
        if self.bridge_script is not None:
            return Path(self.bridge_script).expanduser().resolve()
        return Path(__file__).with_name("official_plsko_bridge.R")

    def _validate_environment(self) -> None:
        if shutil.which(self.rscript) is None:
            raise RuntimeError(
                f"Could not find {self.rscript!r}. Install R or provide the full Rscript path."
            )
        bridge = self._bridge_path()
        if not bridge.exists():
            raise FileNotFoundError(f"PLSKO bridge script not found: {bridge}")

    def sample_knockoffs(self) -> NDArray[np.float64]:
        self._validate_environment()
        if isinstance(self.X, pd.DataFrame):
            X_df = self.X.copy()
        else:
            X_arr = np.asarray(self.X, dtype=float)
            if X_arr.ndim != 2:
                raise ValueError("X must be a two dimensional matrix")
            X_df = pd.DataFrame(
                X_arr,
                index=[f"sample_{i}" for i in range(X_arr.shape[0])],
                columns=[f"feature_{j}" for j in range(X_arr.shape[1])],
            )
        if not np.isfinite(X_df.to_numpy(dtype=float)).all():
            raise ValueError("OfficialPLSKOSampler currently requires complete finite X")
        if not 0.0 < self.sparsity <= 1.0:
            raise ValueError("sparsity must lie in (0, 1]")

        with tempfile.TemporaryDirectory(prefix="stabl_plsko_") as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "X.csv"
            output_csv = tmp_path / "X_knockoff.csv"
            X_df.to_csv(input_csv)

            command = [
                self.rscript,
                str(self._bridge_path()),
                "--input",
                str(input_csv),
                "--output",
                str(output_csv),
                "--seed",
                str(int(self.random_state)),
                "--sparsity",
                str(float(self.sparsity)),
            ]
            if self.threshold_abs is not None:
                command += ["--threshold-abs", str(float(self.threshold_abs))]
            if self.threshold_q is not None:
                command += ["--threshold-q", str(float(self.threshold_q))]
            if self.ncomp is not None:
                command += ["--ncomp", str(int(self.ncomp))]

            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Official PLSKO R process failed.\n"
                    f"Command: {' '.join(command)}\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            if not output_csv.exists():
                raise RuntimeError("PLSKO completed but did not create the output CSV")

            Xk_df = pd.read_csv(output_csv, index_col=0)
            Xk_df = Xk_df.loc[X_df.index, X_df.columns]
            Xk = Xk_df.to_numpy(dtype=float)
            if Xk.shape != X_df.shape:
                raise RuntimeError(f"PLSKO returned {Xk.shape}, expected {X_df.shape}")
            if not np.isfinite(Xk).all():
                raise RuntimeError("PLSKO returned NaN or infinite values")
            self.X_artificial_ = Xk
            return Xk


def check_official_plsko_installation(
    rscript: str = "Rscript", bridge_script: str | Path | None = None
) -> tuple[bool, str]:
    bridge = (
        Path(bridge_script).expanduser().resolve()
        if bridge_script is not None
        else Path(__file__).with_name("official_plsko_bridge.R")
    )
    if shutil.which(rscript) is None:
        return False, f"Could not find {rscript}"
    command = [
        rscript,
        "-e",
        "quit(status=ifelse(requireNamespace('PLSKO', quietly=TRUE), 0, 2))",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode == 2:
        return False, "R is available, but package PLSKO is not installed"
    if result.returncode != 0:
        return False, result.stderr.strip() or "Unknown R error"
    if not bridge.exists():
        return False, f"Bridge script is missing: {bridge}"
    return True, "Official PLSKO R package and bridge are available"
