#!/usr/bin/env python3
"""Reproduce the source index failure in STABL's p greater than 3000 branch.

This audit tracks column identities only.  It does not generate knockoffs and is
therefore fast even for large p.  A valid paired construction should produce
exactly one knockoff source for every original feature, with source j at
artificial position j after any recorded reordering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def simulate_current_branch(p: int, chunk_size: int, seed: int):
    rng = np.random.default_rng(seed)
    n_chunks = p // chunk_size + 1
    source_blocks = []
    for _ in range(n_chunks):
        source_blocks.append(rng.choice(p, size=chunk_size, replace=False))
    generated_sources = np.concatenate(source_blocks)

    first_downsample = rng.choice(len(generated_sources), size=p, replace=False)
    sources_after_first = generated_sources[first_downsample]

    second_indices = rng.choice(p, size=p, replace=False)
    final_sources = sources_after_first[second_indices]

    counts = np.bincount(final_sources, minlength=p)
    return {
        "p": p,
        "chunk_size": chunk_size,
        "n_chunks": n_chunks,
        "generated_columns_before_downsample": int(len(generated_sources)),
        "correct_pair_fraction_at_final_position": float(np.mean(final_sources == np.arange(p))),
        "n_sources_absent": int(np.sum(counts == 0)),
        "n_sources_once": int(np.sum(counts == 1)),
        "n_sources_repeated": int(np.sum(counts > 1)),
        "total_duplicate_excess": int(np.sum(np.clip(counts - 1, 0, None))),
        "max_source_multiplicity": int(counts.max()),
    }, final_sources, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=3529)
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="./chunking_bug_audit")
    args = parser.parse_args()
    if args.p <= args.chunk_size:
        raise ValueError("Choose p greater than chunk_size to exercise the branch")

    summary, final_sources, counts = simulate_current_branch(
        args.p, args.chunk_size, args.seed
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "artificial_position": np.arange(args.p),
            "original_source": final_sources,
            "is_correct_pair": final_sources == np.arange(args.p),
        }
    ).to_csv(out_dir / "final_pairing_map.csv", index=False)
    pd.DataFrame(
        {
            "original_feature": np.arange(args.p),
            "number_of_knockoff_copies": counts,
        }
    ).to_csv(out_dir / "source_multiplicity.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote audit to {out_dir}")


if __name__ == "__main__":
    main()
