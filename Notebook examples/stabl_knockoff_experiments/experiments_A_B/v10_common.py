"""Shared helpers for V10 external omic generalization.

V10 evaluates whether the reliability advantage of BayesianRidge conditional-
mean completion generalizes beyond SSI CyTOF.  The default pilot contains one
same-cohort/different-modality block (SSI Proteomics) and one different-cohort
block (DREAM Phylotype).  Calibration and evaluation use disjoint random seeds.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from research_common import safe_label


@dataclass(frozen=True)
class ExternalBlock:
    label: str
    dataset: str
    data_path: str
    omic: str
    subject_mode: str = "first"
    reference_complete_strategy: str = "drop-columns"
    p: int = 100
    n_signal: int = 15
    task: str = "classification"
    signal_strength: float | None = None
    signal_strengths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

    @staticmethod
    def from_mapping(mapping: dict[str, Any]) -> "ExternalBlock":
        required = {"label", "dataset", "data_path", "omic"}
        missing = required - set(mapping)
        if missing:
            raise ValueError(f"Block config is missing fields: {sorted(missing)}")
        dataset = str(mapping["dataset"]).lower()
        if dataset not in {"ssi", "dream", "ool", "csv"}:
            raise ValueError(f"Unsupported V10 dataset: {dataset}")
        subject_mode = str(mapping.get("subject_mode", "first")).lower()
        if subject_mode not in {"first", "random", "all"}:
            raise ValueError("subject_mode must be first, random, or all")
        complete_strategy = str(
            mapping.get("reference_complete_strategy", "drop-columns")
        ).lower()
        if complete_strategy not in {"drop-columns", "error"}:
            raise ValueError(
                "reference_complete_strategy must be drop-columns or error"
            )
        task = str(mapping.get("task", "classification")).lower()
        if task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        p = int(mapping.get("p", 100))
        n_signal = int(mapping.get("n_signal", 15))
        if p < 2 or not 1 <= n_signal < p:
            raise ValueError("Require p >= 2 and 1 <= n_signal < p")
        strength = mapping.get("signal_strength")
        strength = None if strength in {None, ""} else float(strength)
        if strength is not None and strength <= 0:
            raise ValueError("signal_strength must be positive")
        candidates_raw = mapping.get("signal_strengths", [0.5, 1.0, 2.0, 4.0])
        candidates = tuple(float(item) for item in candidates_raw)
        if not candidates or any(item <= 0 for item in candidates):
            raise ValueError("signal_strengths must contain positive values")
        return ExternalBlock(
            label=safe_label(str(mapping["label"])),
            dataset=dataset,
            data_path=os.path.expandvars(os.path.expanduser(str(mapping["data_path"]))),
            omic=str(mapping["omic"]),
            subject_mode=subject_mode,
            reference_complete_strategy=complete_strategy,
            p=p,
            n_signal=n_signal,
            task=task,
            signal_strength=strength,
            signal_strengths=candidates,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "dataset": self.dataset,
            "data_path": self.data_path,
            "omic": self.omic,
            "subject_mode": self.subject_mode,
            "reference_complete_strategy": self.reference_complete_strategy,
            "p": self.p,
            "n_signal": self.n_signal,
            "task": self.task,
            "signal_strength": self.signal_strength,
            "signal_strengths": list(self.signal_strengths),
        }


def load_blocks(path: str | Path) -> tuple[ExternalBlock, ...]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("V10 blocks JSON must contain a nonempty list")
    blocks = tuple(ExternalBlock.from_mapping(item) for item in raw)
    labels = [block.label for block in blocks]
    if len(labels) != len(set(labels)):
        raise ValueError("Every V10 block label must be unique")
    return blocks


def parse_csv_items(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("Expected at least one comma-separated item")
    return items


def strength_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def read_calibration_recommendations(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        csv_path = path / "v10_calibration_recommendations.csv"
    else:
        csv_path = path
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path)
    required = {"block_label", "actual_p", "calibration_signal_strength"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Calibration recommendation is missing {sorted(missing)}")
    return {
        str(row.block_label): row._asdict()
        for row in frame.itertuples(index=False)
    }


def favorable_direction(metric: str) -> str:
    lower = {
        "fdp",
        "false_positives",
        "estimated_fdp",
        "oracle_selected_count_abs_difference",
        "median_signal_rank",
        "mean_null_score",
        "masked_rmse",
        "masked_mae",
        "sample_cov_fro_relative",
        "sample_corr_fro_relative",
        "cov_kk_fro_relative",
        "cov_xk_offdiag_rmse",
    }
    return "lower" if metric in lower else "higher"


def is_favorable(metric: str, difference: float) -> bool:
    return difference < 0 if favorable_direction(metric) == "lower" else difference > 0


def block_output_dirs(root: str | Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    candidates = []
    for child in sorted(root.iterdir() if root.exists() else []):
        if child.is_dir() and (child / "selection_results.csv").exists():
            candidates.append(child)
    return candidates


def validate_mechanisms(items: Iterable[str]) -> tuple[str, ...]:
    resolved = tuple(str(item).upper() for item in items)
    invalid = sorted(set(resolved) - {"MCAR", "MAR", "MNAR", "BLOCK"})
    if invalid:
        raise ValueError(f"Unsupported missingness mechanisms: {invalid}")
    return resolved
