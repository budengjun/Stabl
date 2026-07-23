#!/usr/bin/env python3
"""Move a completed G2.3 run into the established project layout.

The original G2.3 command resolved relative result paths from experiments_A_B,
which placed results and logs under the code directory. This utility moves the
run to experiment_B, moves pair caches to pair_cache, moves the log to logs,
and rewrites stored score paths so the result directory remains portable.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Iterable


def _write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _rewrite_score_paths(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "score_file" not in fieldnames:
        return 0

    changed = 0
    for row in rows:
        raw = str(row.get("score_file", "")).strip()
        if not raw:
            continue
        portable = str(Path("stage2_scores") / Path(raw).name)
        if raw != portable:
            row["score_file"] = portable
            changed += 1

    temp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(csv_path)
    return changed


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root. Defaults to the directory containing experiments_A_B.",
    )
    parser.add_argument("--run-name", default="G23_ssi_smoke")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration. Without this flag the script prints the plan only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inferred_root = Path(__file__).resolve().parents[2]
    root = Path(args.project_root).expanduser().resolve() if args.project_root else inferred_root
    run_name = args.run_name

    source = _first_existing(
        [
            root / "experiments_A_B" / "generator_track_results" / run_name,
            root / "stabl_knockoff_experiments" / "experiments_A_B" / "generator_track_results" / run_name,
        ]
    )
    destination = root / "experiment_B" / run_name
    cache_destination = root / "pair_cache" / run_name
    log_source = _first_existing(
        [
            root / "experiments_A_B" / "logs" / f"{run_name.lower()}.log",
            root / "experiments_A_B" / "logs" / "g23_ssi_smoke.log",
            root / "stabl_knockoff_experiments" / "experiments_A_B" / "logs" / "g23_ssi_smoke.log",
        ]
    )
    log_destination = root / "logs" / "g23_ssi_smoke.log"

    print(f"project_root={root}")
    print(f"source={source}")
    print(f"destination={destination}")
    print(f"pair_cache_destination={cache_destination}")
    print(f"log_source={log_source}")
    print(f"log_destination={log_destination}")

    if not args.apply:
        print("Dry run only. Re-run with --apply after reviewing the paths above.")
        return

    if source is None:
        if destination.exists():
            print(f"Result directory is already in the corrected location: {destination}")
            result_dir = destination
        else:
            raise FileNotFoundError("Could not find the original G2.3 result directory")
    else:
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        result_dir = destination
        print(f"Moved results to {result_dir}")

    cache_destination.mkdir(parents=True, exist_ok=True)
    for old_name, new_name in (
        ("stage1_pair_cache", "stage1"),
        ("stage2_pair_cache", "stage2"),
    ):
        old_cache = result_dir / old_name
        new_cache = cache_destination / new_name
        if old_cache.exists():
            if new_cache.exists():
                raise FileExistsError(f"Cache destination already exists: {new_cache}")
            shutil.move(str(old_cache), str(new_cache))
            print(f"Moved {old_name} to {new_cache}")

    changed = _rewrite_score_paths(result_dir / "g23_stage2_stabl_runs.csv")
    print(f"Rewrote {changed} score_file paths as portable relative paths")

    config_path = result_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["out_dir"] = str(result_dir)
        config["pair_cache_dir"] = str(cache_destination)
        config["project_root"] = str(root)
        config["layout_version"] = 2
        _write_json(config_path, config)

    status_path = result_dir / "g23_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["out_dir"] = str(result_dir)
        status["pair_cache_dir"] = str(cache_destination)
        status["layout_version"] = 2
        _write_json(status_path, status)

    if log_source is not None:
        log_destination.parent.mkdir(parents=True, exist_ok=True)
        if log_destination.exists() and log_destination.resolve() != log_source.resolve():
            raise FileExistsError(f"Log destination already exists: {log_destination}")
        if not log_destination.exists():
            shutil.move(str(log_source), str(log_destination))
            print(f"Moved log to {log_destination}")

    print("G2.3 layout repair completed.")


if __name__ == "__main__":
    main()
