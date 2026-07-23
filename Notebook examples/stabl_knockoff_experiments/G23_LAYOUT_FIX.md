# G2.3 layout correction

This patch corrects four layout defects in the first G2.3 package:

1. The original archive included a full `stabl_knockoff_experiments/` wrapper, which created a nested project when extracted inside the existing project root.
2. Relative `--out-dir` values were resolved from the shell current working directory, placing results below `experiments_A_B/`.
3. Pair caches were placed inside the result directory instead of the project level `pair_cache/` directory.
4. Absolute score paths were stored in CSV output, making completed results nonportable after moving or zipping.

Install the patch into the existing project root and use `repair_g23_layout.py` to migrate the completed smoke result without rerunning it.
