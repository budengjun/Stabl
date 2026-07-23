# V5 run guide: posterior-sampling completion versus median imputation

All commands assume the project root is:

```bash
/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments
```

The primary analysis uses the original STABL `stabl_min` rule. The `target_fdr` field is retained only as an evaluation reference for realized FDP exceedance; it is not a hard selection constraint for `stabl_min`.

## 1. Install V5

Upload `experiments_A_B_v5_posterior_vs_median.zip` to `/data/yhu94/Stabl`, then run:

```bash
cd /data/yhu94/Stabl

export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"

mv \
  "$EXP_ROOT/experiments_A_B" \
  "$EXP_ROOT/experiments_A_B_before_v5_$(date +%Y%m%d_%H%M%S)"

unzip -q \
  experiments_A_B_v5_posterior_vs_median.zip \
  -d "$EXP_ROOT"

export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"
```

## 2. Static V5 checks

```bash
python -u experiments_A_B/test_v5_updates.py
```

Expected output ends with:

```text
[ok] all V5 update tests passed
```

## 3. Integration smoke

```bash
mkdir -p \
  experiment_A/posterior_vs_median_smoke_v5 \
  pair_cache/experiment_A_v5 \
  logs

python -u experiments_A_B/run_experiment_a_oracle_completion.py \
  --out-dir experiment_A/posterior_vs_median_smoke_v5 \
  --pair-cache-dir pair_cache/experiment_A_v5 \
  --n-replicates 1 \
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
  --selection-rules stabl_min \
  --n-knockoff-draws 1 \
  --n-bootstraps 20 \
  --stabl-grid-size 30 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --skip-misspecified-true-sigma \
  --skip-precision-recovery \
  --fail-fast \
  --overwrite \
  2>&1 | tee logs/posterior_vs_median_smoke_v5.log
```

Expected rows:

```text
selection_results.csv      12
 draw_diagnostics.csv      12
 imputation_recovery.csv    8
```

The 12 STABL cells are, per missingness mechanism, four sample-Sigma completion arms plus two theory-aligned true-Sigma arms.

Generate the paired summary:

```bash
python -u experiments_A_B/summarize_posterior_vs_median.py \
  --out-dir experiment_A/posterior_vs_median_smoke_v5
```

## 4. Paired 20-replicate pilot

```bash
tmux new -s stabl_posterior_pilot_v5
```

Inside tmux:

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"

mkdir -p experiment_A/posterior_vs_median_pilot_v5 pair_cache/experiment_A_v5 logs

nice -n 10 python -u experiments_A_B/run_experiment_a_oracle_completion.py \
  --out-dir experiment_A/posterior_vs_median_pilot_v5 \
  --pair-cache-dir pair_cache/experiment_A_v5 \
  --n-replicates 20 \
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
  --selection-rules stabl_min \
  --n-knockoff-draws 1 \
  --n-bootstraps 100 \
  --stabl-grid-size 30 \
  --stabl-c-min 0.01 \
  --stabl-c-max 1.0 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --skip-misspecified-true-sigma \
  --skip-precision-recovery \
  --fail-fast \
  --overwrite \
  2>&1 | tee logs/posterior_vs_median_pilot_v5.log
```

Detach with `Ctrl+B`, then `D`.

Expected rows:

```text
selection_results.csv      240
 draw_diagnostics.csv      240
 imputation_recovery.csv   160
```

After completion:

```bash
python -u experiments_A_B/summarize_posterior_vs_median.py \
  --out-dir experiment_A/posterior_vs_median_pilot_v5 \
  | tee logs/posterior_vs_median_pilot_v5_summary.log
```

Primary outputs:

```text
primary_stabl_min_summary.csv
paired_completion_contrasts.csv
oracle_selection_agreement.csv
oracle_selection_agreement_summary.csv
figures/
```

## 5. Main 100-replicate experiment

Run only after the smoke and 20-replicate pilot have zero errors and the paired summary is sensible.

```bash
tmux new -s stabl_posterior_main_v5
```

Use the pilot command with these replacements:

```text
--out-dir experiment_A/posterior_vs_median_main_v5
--n-replicates 100
```

Keep every other scientific parameter unchanged. Expected primary row counts are:

```text
selection_results.csv      1200
 draw_diagnostics.csv      1200
 imputation_recovery.csv    800
```

## 6. Resume after interruption

Use the identical command and output directory, remove `--overwrite`, add `--resume`, and append the log:

```text
--resume
2>&1 | tee -a logs/<same_log_name>.log
```

Do not use `--refresh-pair-cache` unless intentional regeneration is required.

## Primary interpretation

The main comparison is sample-Sigma equicorrelated knockoffs with `stabl_min`:

```text
median
vs exact_gaussian_posterior
vs bayesianridge_posterior
```

`oracle_complete` is the complete-data benchmark. Theory-aligned true-Sigma results are secondary oracle diagnostics. The report should judge posterior superiority in layers:

1. completion and covariance recovery;
2. knockoff covariance matching;
3. closeness to the oracle-complete STABL selection set;
4. realized FDP, power, and selected-feature count under original STABL.
