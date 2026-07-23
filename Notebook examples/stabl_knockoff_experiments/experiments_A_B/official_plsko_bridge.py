"""Python wrapper for the authors' official PLSKO R package."""

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

    def sample_knockoffs(self) -> NDArray[np.float64]:
        if shutil.which(self.rscript) is None:
            raise RuntimeError(f"Could not find {self.rscript!r}")
        bridge = self._bridge_path()
        if not bridge.exists():
            raise FileNotFoundError(f"PLSKO bridge not found: {bridge}")
        if isinstance(self.X, pd.DataFrame):
            X_df = self.X.copy()
        else:
            values = np.asarray(self.X, dtype=float)
            if values.ndim != 2:
                raise ValueError("X must be two dimensional")
            X_df = pd.DataFrame(values)
        if not np.isfinite(X_df.to_numpy(dtype=float)).all():
            raise ValueError("PLSKO requires complete finite X")
        if not 0.0 < self.sparsity <= 1.0:
            raise ValueError("sparsity must lie in (0, 1]")

        with tempfile.TemporaryDirectory(prefix="stabl_plsko_") as temporary:
            directory = Path(temporary)
            input_path = directory / "X.csv"
            output_path = directory / "X_knockoff.csv"
            X_df.to_csv(input_path)
            command = [
                self.rscript,
                str(bridge),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--seed",
                str(int(self.random_state)),
                "--sparsity",
                str(float(self.sparsity)),
            ]
            if self.threshold_abs is not None:
                command.extend(["--threshold-abs", str(float(self.threshold_abs))])
            if self.threshold_q is not None:
                command.extend(["--threshold-q", str(float(self.threshold_q))])
            if self.ncomp is not None:
                command.extend(["--ncomp", str(int(self.ncomp))])
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Official PLSKO failed.\n"
                    f"Command: {' '.join(command)}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            if not output_path.exists():
                raise RuntimeError("PLSKO did not create its output file")
            knockoff = pd.read_csv(output_path, index_col=0)
            knockoff = knockoff.loc[X_df.index, X_df.columns]
            values = knockoff.to_numpy(dtype=float)
            if values.shape != X_df.shape or not np.isfinite(values).all():
                raise RuntimeError("PLSKO returned an invalid matrix")
            return values
