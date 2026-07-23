# G2.3 Gaussian family generator screening

## Correct project layout

The project root is:

```text
/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments
```

G2.3 now follows the existing layout:

```text
stabl_knockoff_experiments/
├── experiments_A_B/
│   └── generator_track/
├── experiment_B/
│   ├── G23_ssi_smoke/
│   ├── G23_ssi_pilot/
│   └── G23_ssi_confirmation/
├── pair_cache/
│   ├── G23_ssi_smoke/
│   ├── G23_ssi_pilot/
│   └── G23_ssi_confirmation/
└── logs/
```

Relative paths are resolved from the project root by the Python program. They no longer depend on the shell current working directory.

## Install this patch

Place the patch ZIP anywhere, then run:

```bash
EXP_ROOT="/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments"
unzip -o G23_layout_fix.zip -d "$EXP_ROOT"
```

The ZIP begins with `experiments_A_B/`. It does not contain another `stabl_knockoff_experiments/` wrapper directory.

## Repair the completed smoke run

First preview the migration:

```bash
EXP_ROOT="/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments"
cd "$EXP_ROOT"

python -u experiments_A_B/generator_track/repair_g23_layout.py \
  --project-root "$EXP_ROOT" \
  --run-name G23_ssi_smoke
```

After confirming the printed paths, apply it:

```bash
python -u experiments_A_B/generator_track/repair_g23_layout.py \
  --project-root "$EXP_ROOT" \
  --run-name G23_ssi_smoke \
  --apply
```

This moves:

```text
experiments_A_B/generator_track_results/G23_ssi_smoke
```

to:

```text
experiment_B/G23_ssi_smoke
```

It also moves pair caches to `pair_cache/G23_ssi_smoke`, moves the log to `logs/g23_ssi_smoke.log`, and rewrites `score_file` entries as portable relative paths.

## Environment

```bash
EXP_ROOT="/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments"
cd "$EXP_ROOT"
. ~/.profile
conda activate stabl
```

## Self test and environment check

```bash
python -u experiments_A_B/generator_track/run_g23_self_test.py
python -u experiments_A_B/generator_track/check_g23_environment.py
```

## Smoke command

The completed smoke does not need to be rerun solely because of the old directory placement. For a fresh smoke, use:

```bash
mkdir -p "$EXP_ROOT/logs"
set -o pipefail

python -u experiments_A_B/generator_track/run_g23_gaussian_family_screening.py \
  --ssi-data-path "/data/yhu94/Stabl/Sample Data/Biobank SSI" \
  --ssi-omic CyTOF \
  --generator-grid-json experiments_A_B/generator_track/configs/g23_gaussian_family_grid.json \
  --c2st-gate-json experiments_A_B/generator_track/configs/g23_c2st_gate.json \
  --out-dir experiment_B/G23_ssi_smoke \
  --pair-cache-dir pair_cache/G23_ssi_smoke \
  --p 50 \
  --validity-draws 2 \
  --c2st-repeats 2 \
  --c2st-folds 4 \
  --max-stage2-factor-candidates 2 \
  --screen-replicates 2 \
  --screen-draws 2 \
  --n-signal 10 \
  --signal-strength 4 \
  --n-bootstraps 20 \
  --n-jobs 8 \
  --bootstrap-samples 1000 \
  --overwrite \
  2>&1 | tee "$EXP_ROOT/logs/g23_ssi_smoke.log"
```

## Pilot command

```bash
mkdir -p "$EXP_ROOT/logs"
set -o pipefail

python -u experiments_A_B/generator_track/run_g23_gaussian_family_screening.py \
  --ssi-data-path "/data/yhu94/Stabl/Sample Data/Biobank SSI" \
  --ssi-omic CyTOF \
  --generator-grid-json experiments_A_B/generator_track/configs/g23_gaussian_family_grid.json \
  --c2st-gate-json experiments_A_B/generator_track/configs/g23_c2st_gate.json \
  --out-dir experiment_B/G23_ssi_pilot \
  --pair-cache-dir pair_cache/G23_ssi_pilot \
  --p 100 \
  --validity-draws 10 \
  --c2st-repeats 3 \
  --c2st-folds 5 \
  --max-stage2-factor-candidates 3 \
  --screen-replicates 10 \
  --screen-draws 3 \
  --n-signal 15 \
  --signal-strength 4 \
  --n-bootstraps 100 \
  --n-jobs 8 \
  --target-fdr 0.10 \
  --bootstrap-samples 10000 \
  --overwrite \
  2>&1 | tee "$EXP_ROOT/logs/g23_ssi_pilot.log"
```

Use `--resume` instead of `--overwrite` after an interrupted run.

## Path behavior fixed in version 2

1. Result paths are resolved against the project root.
2. Pair caches are stored under the project level `pair_cache/` directory.
3. Score paths written to CSV are relative to the result folder, so moving or zipping a completed result does not break them.
4. Output and cache locations do not alter the scientific configuration fingerprint.
5. The patch ZIP merges into the existing project and does not create a nested project directory.
