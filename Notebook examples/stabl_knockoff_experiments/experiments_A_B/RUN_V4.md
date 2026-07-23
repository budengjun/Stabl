# Install and run V4

The experiment root on `fenn10` is:

```bash
/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments
```

## 1. Install V4

Upload `experiments_A_B_v4_paper_aligned.zip` to `/data/yhu94/Stabl/`, then run:

```bash
cd /data/yhu94/Stabl

export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"

mv "$EXP_ROOT/experiments_A_B" \
   "$EXP_ROOT/experiments_A_B_before_v4_$(date +%Y%m%d_%H%M%S)"

unzip -q experiments_A_B_v4_paper_aligned.zip -d "$EXP_ROOT"

export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"
```

## 2. Run lightweight V4 tests

```bash
python -u experiments_A_B/test_v4_updates.py
```

Expected final line:

```text
[ok] all V4 update tests passed
```

## 3. V4 integration smoke test

```bash
mkdir -p experiment_A/smoke_v4 pair_cache/experiment_A logs

python -u experiments_A_B/run_experiment_a_oracle_completion.py \
  --out-dir experiment_A/smoke_v4 \
  --pair-cache-dir pair_cache/experiment_A \
  --n-replicates 1 \
  --n 150 \
  --p 200 \
  --n-signal 25 \
  --block-size 25 \
  --rho 0.60 \
  --task classification \
  --signal-strength 2.0 \
  --mechanisms MCAR \
  --missing-rate 0.20 \
  --completion-methods oracle_complete,exact_gaussian_posterior \
  --generators gaussian_equicorrelated,gaussian_equicorrelated_true_sigma \
  --n-knockoff-draws 1 \
  --n-bootstraps 20 \
  --stabl-grid-size 30 \
  --lcd-cv-folds 3 \
  --lcd-grid-size 10 \
  --selection-rules stabl_min,stabl_q,stabl_knockoff_plus,lcd_knockoff_plus \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --lcd-n-jobs 1 \
  --skip-precision-recovery \
  --fail-fast \
  --overwrite \
  2>&1 | tee logs/experiment_A_smoke_v4.log
```

Expected output sizes:

```text
selection_results.csv: 16 rows
draw_diagnostics.csv: 4 rows
imputation_recovery.csv: 2 rows
errors: 0
```

## 4. Paper-aligned V4 confirmation

This is the next scientific run. It holds the V3 outcome design fixed and changes only the paper alignment and comparator:

- 30 STABL C values
- 100 bootstraps
- LCD knockoff-plus
- ideal completion branches only

Start tmux:

```bash
tmux new -s stabl_A_v4_confirm
```

Inside tmux:

```bash
cd "$EXP_ROOT"
mkdir -p experiment_A/confirmation_v4 pair_cache/experiment_A logs

nice -n 10 python -u experiments_A_B/run_experiment_a_oracle_completion.py \
  --out-dir experiment_A/confirmation_v4 \
  --pair-cache-dir pair_cache/experiment_A \
  --n-replicates 5 \
  --n 150 \
  --p 200 \
  --n-signal 25 \
  --block-size 25 \
  --rho 0.60 \
  --task classification \
  --signal-strength 2.0 \
  --mechanisms MCAR \
  --missing-rate 0.20 \
  --completion-methods oracle_complete,exact_gaussian_posterior \
  --generators gaussian_equicorrelated,gaussian_equicorrelated_true_sigma \
  --n-knockoff-draws 1 \
  --n-bootstraps 100 \
  --stabl-grid-size 30 \
  --stabl-c-min 0.01 \
  --stabl-c-max 1.0 \
  --lcd-cv-folds 5 \
  --lcd-grid-size 30 \
  --lcd-c-min 0.001 \
  --lcd-c-max 10.0 \
  --selection-rules stabl_min,stabl_q,stabl_knockoff_plus,lcd_knockoff_plus \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --lcd-n-jobs 1 \
  --skip-precision-recovery \
  --fail-fast \
  --overwrite \
  2>&1 | tee logs/experiment_A_confirmation_v4.log
```

Detach with `Ctrl+B`, then `D`.

Expected output sizes:

```text
selection_results.csv: 80 rows
draw_diagnostics.csv: 20 rows
imputation_recovery.csv: 10 rows
errors: 0
```

Summarize:

```bash
python -u experiments_A_B/summarize_v4_results.py \
  --out-dir experiment_A/confirmation_v4 \
  | tee logs/experiment_A_confirmation_v4_summary.log
```

## 5. Decision after confirmation

Proceed to the full run only after checking:

1. LCD produces at least some non-empty selections in the oracle/true-Sigma branch.
2. All error counts are zero.
3. Exact posterior continues to track oracle completion.
4. The contrast between sample-Sigma and true-Sigma generators is stable.

If every target-q rule remains empty, run a stronger-signal calibration before the full experiment:

```text
n_signal = 15
signal_strength = 4.0
```

Keep `q = 0.10`; do not raise q merely to force discoveries.

## 6. Full V4 Experiment A

This command starts 100 replicates from the beginning so it can be resumed safely without changing the configuration.

```bash
tmux new -s stabl_A_v4_main
```

Inside tmux:

```bash
cd "$EXP_ROOT"
mkdir -p experiment_A/main_v4 pair_cache/experiment_A logs

nice -n 10 python -u experiments_A_B/run_experiment_a_oracle_completion.py \
  --out-dir experiment_A/main_v4 \
  --pair-cache-dir pair_cache/experiment_A \
  --n-replicates 100 \
  --n 150 \
  --p 200 \
  --n-signal 25 \
  --block-size 25 \
  --rho 0.60 \
  --task classification \
  --signal-strength 2.0 \
  --mechanisms MCAR,MAR \
  --missing-rate 0.20 \
  --completion-methods oracle_complete,median,exact_gaussian_posterior,bayesianridge_posterior \
  --generators gaussian_equicorrelated,gaussian_equicorrelated_true_sigma \
  --n-knockoff-draws 1 \
  --n-bootstraps 100 \
  --stabl-grid-size 30 \
  --stabl-c-min 0.01 \
  --stabl-c-max 1.0 \
  --lcd-cv-folds 5 \
  --lcd-grid-size 30 \
  --lcd-c-min 0.001 \
  --lcd-c-max 10.0 \
  --selection-rules stabl_min,stabl_q,stabl_knockoff_plus,lcd_knockoff_plus \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --lcd-n-jobs 1 \
  --skip-precision-recovery \
  --fail-fast \
  2>&1 | tee logs/experiment_A_main_v4.log
```

Expected completed sizes:

```text
selection_results.csv: 6400 rows
draw_diagnostics.csv: 1600 rows
imputation_recovery.csv: 800 rows
```

## 7. Resume an interrupted run

Use the exact same scientific arguments and output directory, remove `--overwrite`, and add `--resume`:

```bash
...same command... \
  --resume \
  2>&1 | tee -a logs/experiment_A_main_v4.log
```

Do not use `--refresh-pair-cache` unless pair regeneration is intentional.
