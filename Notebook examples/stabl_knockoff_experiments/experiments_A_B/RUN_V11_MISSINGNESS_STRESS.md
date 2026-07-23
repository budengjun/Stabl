# V11 run commands

## Environment

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"
mkdir -p experiment_B pair_cache logs
```

Create the block configuration:

```bash
cp experiments_A_B/v11_stress_blocks.example.json v11_stress_blocks.json
```

Run tests:

```bash
python -u experiments_A_B/test_v11_updates.py
```

## Smoke

Smoke uses a stage-specific seed offset and cannot be interpreted scientifically.

```bash
python -u experiments_A_B/run_v11_missingness_stress.py \
  --blocks-json v11_stress_blocks.json \
  --out-dir experiment_B/v11_missingness_stress_smoke \
  --pair-cache-dir pair_cache/experiment_B_v11_smoke \
  --analysis-status smoke \
  --missing-rates 0.2 \
  --mechanisms MCAR,MAR,BLOCK \
  --n-replicates 1 \
  --n-completion-draws 2 \
  --n-bootstraps 20 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/v11_missingness_stress_smoke.log
```

Expected selection rows: `2 blocks × 1 rate × 3 mechanisms × 3 completions × 1 replicate × 2 runs = 36`.

## Pilot

```bash
nice -n 10 python -u experiments_A_B/run_v11_missingness_stress.py \
  --blocks-json v11_stress_blocks.json \
  --out-dir experiment_B/v11_missingness_stress_pilot \
  --pair-cache-dir pair_cache/experiment_B_v11_pilot \
  --analysis-status pilot \
  --missing-rates 0.1,0.2,0.3,0.4 \
  --mechanisms MCAR,MAR,BLOCK \
  --n-replicates 10 \
  --n-completion-draws 3 \
  --n-bootstraps 50 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/v11_missingness_stress_pilot.log
```

Expected ordinary STABL fits: `2 × 4 × 3 × 3 × 10 × 3 = 2160`.

Resume after interruption by replacing `--overwrite` with `--resume` and using `tee -a`.

## Full stress map

Run only after reviewing the pilot.

```bash
nice -n 10 python -u experiments_A_B/run_v11_missingness_stress.py \
  --blocks-json v11_stress_blocks.json \
  --out-dir experiment_B/v11_missingness_stress \
  --pair-cache-dir pair_cache/experiment_B_v11_stress \
  --analysis-status stress \
  --missing-rates 0.1,0.2,0.3,0.4 \
  --mechanisms MCAR,MAR,BLOCK \
  --n-replicates 20 \
  --n-completion-draws 5 \
  --n-bootstraps 100 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/v11_missingness_stress.log
```

Expected ordinary STABL fits: `2 × 4 × 3 × 3 × 20 × 5 = 7200`.

## Integrity check

```bash
column -s, -t < experiment_B/v11_missingness_stress_pilot/v11_integrity_audit.csv | less -S
```

Every row must have `complete=True`.

## Cache cleanup

Delete caches only after integrity is complete and results are backed up:

```bash
rm -rf pair_cache/experiment_B_v11_smoke
rm -rf pair_cache/experiment_B_v11_pilot
```

The command also supports `--delete-pair-cache-after-success`.
