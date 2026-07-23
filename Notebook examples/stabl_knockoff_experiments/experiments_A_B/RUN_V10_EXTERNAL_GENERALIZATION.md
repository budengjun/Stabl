# Run V10 external omic generalization

## Environment

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"
```

Copy the example block file and edit paths only when your local directories
differ:

```bash
cp experiments_A_B/v10_external_blocks.example.json v10_external_blocks.json
```

## Test

```bash
python -u experiments_A_B/test_v10_updates.py
```

## Stage 0 smoke

```bash
python -u experiments_A_B/run_v10_oracle_calibration.py \
  --blocks-json v10_external_blocks.json \
  --out-dir experiment_B/v10_oracle_calibration_smoke \
  --pair-cache-dir pair_cache/experiment_B_v10_calibration_smoke \
  --n-replicates 2 \
  --n-completion-draws 1 \
  --n-bootstraps 20 \
  --n-jobs 8 \
  --overwrite
```

The smoke uses two replicates only for loader and engineering validation; use the formal command below for recommendations.

## Stage 0 formal calibration

```bash
nice -n 10 python -u experiments_A_B/run_v10_oracle_calibration.py \
  --blocks-json v10_external_blocks.json \
  --out-dir experiment_B/v10_oracle_calibration \
  --pair-cache-dir pair_cache/experiment_B_v10_calibration \
  --n-replicates 10 \
  --n-completion-draws 3 \
  --n-bootstraps 50 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/v10_oracle_calibration.log
```

## Stage 1 smoke

```bash
python -u experiments_A_B/run_v10_external_generalization.py \
  --blocks-json v10_external_blocks.json \
  --calibration-dir experiment_B/v10_oracle_calibration \
  --out-dir experiment_B/v10_external_generalization_smoke \
  --pair-cache-dir pair_cache/experiment_B_v10_smoke \
  --n-replicates 2 \
  --n-completion-draws 2 \
  --n-bootstraps 20 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/v10_external_generalization_smoke.log
```

## Stage 1 formal 10-replicate pilot

```bash
nice -n 10 python -u experiments_A_B/run_v10_external_generalization.py \
  --blocks-json v10_external_blocks.json \
  --calibration-dir experiment_B/v10_oracle_calibration \
  --out-dir experiment_B/v10_external_generalization_pilot \
  --pair-cache-dir pair_cache/experiment_B_v10_pilot \
  --n-replicates 10 \
  --n-completion-draws 5 \
  --n-bootstraps 100 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/v10_external_generalization_pilot.log
```

For resume, keep every scientific argument unchanged and replace `--overwrite`
with `--resume`.

## Main outputs

- `v10_integrity_audit.csv`
- `v10_primary_selection_contrasts.csv`
- `v10_ranking_contrasts.csv`
- `v10_pilot_progression_gate.csv`
- `v10_cross_block_consistency.csv`
- `v10_pilot_decision.csv`
- `v10_external_generalization_report.md`

## Pair cache cleanup

After all outputs are complete and archived:

```bash
rm -rf pair_cache/experiment_B_v10_calibration
rm -rf pair_cache/experiment_B_v10_pilot
```

Or add `--delete-pair-cache-after-success` to a run. Do not use automatic cache
deletion when you may need to resume the same run.
