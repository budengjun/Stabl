# Experiment A V6: multiple-posterior pooling

## 1. Scientific purpose

V5 established that posterior completion restores covariance and knockoff
geometry better than median imputation, but a single posterior draw did not
improve original STABL feature selection.  V6 tests whether pooling multiple
posterior-completed STABL score paths reduces Monte Carlo variation.

The primary rule remains the published STABL `stabl_min` rule.

## 2. Install

Upload `experiments_A_B_v6_posterior_pooling.zip` to:

```text
/data/yhu94/Stabl/experiments_A_B_v6_posterior_pooling.zip
```

Then run:

```bash
cd /data/yhu94/Stabl

export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"

mv \
  "$EXP_ROOT/experiments_A_B" \
  "$EXP_ROOT/experiments_A_B_before_v6_$(date +%Y%m%d_%H%M%S)"

unzip -q \
  experiments_A_B_v6_posterior_pooling.zip \
  -d "$EXP_ROOT"

export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"
```

## 3. Static tests

```bash
python -u experiments_A_B/test_v6_updates.py
```

Expected final line:

```text
[ok] all V6 update tests passed
```

## 4. Integration smoke

This smoke runs `M=1,2`, one replicate, MCAR only, all four completions, and
sample-Sigma Gaussian knockoffs.  It intentionally does not reuse V5 scores
because it uses only 20 bootstraps.

```bash
mkdir -p \
  experiment_A/posterior_pooling_smoke_v6 \
  pair_cache/experiment_A_v6 \
  logs

python -u experiments_A_B/run_experiment_a_posterior_pooling.py \
  --out-dir experiment_A/posterior_pooling_smoke_v6 \
  --pair-cache-dir pair_cache/experiment_A_v6 \
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
  --completion-methods oracle_complete,median,exact_gaussian_posterior,bayesianridge_posterior \
  --generators gaussian_equicorrelated \
  --pool-sizes 1,2 \
  --pooling-aggregators mean \
  --stabl-seed-mode shared \
  --n-bootstraps 20 \
  --stabl-grid-size 30 \
  --stabl-c-min 0.01 \
  --stabl-c-max 1.0 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --skip-precision-recovery \
  --fail-fast \
  --overwrite \
  2>&1 | tee logs/posterior_pooling_smoke_v6.log
```

Expected rows:

```text
pooled_selection_results.csv          8
component_draw_diagnostics.csv        8
component_imputation_recovery.csv     8
```

Summarize:

```bash
python -u experiments_A_B/summarize_posterior_pooling.py \
  --out-dir experiment_A/posterior_pooling_smoke_v6
```

## 5. Recommended 10-replicate pilot: M=1 versus M=5

This run uses the exact V5 scientific settings and can import component draw 0
from the completed V5 main experiment.

Start tmux:

```bash
tmux new -s stabl_pooling_v6_pilot
```

Inside tmux:

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"

mkdir -p \
  experiment_A/posterior_pooling_pilot_v6 \
  pair_cache/experiment_A_v5 \
  logs

nice -n 10 python -u experiments_A_B/run_experiment_a_posterior_pooling.py \
  --out-dir experiment_A/posterior_pooling_pilot_v6 \
  --pair-cache-dir pair_cache/experiment_A_v5 \
  --reuse-v5-out-dir experiment_A/posterior_vs_median_main_v5 \
  --n-replicates 10 \
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
  --generators gaussian_equicorrelated \
  --pool-sizes 1,5 \
  --pooling-aggregators mean \
  --stabl-seed-mode shared \
  --n-bootstraps 100 \
  --stabl-grid-size 30 \
  --stabl-c-min 0.01 \
  --stabl-c-max 1.0 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --skip-precision-recovery \
  --fail-fast \
  --overwrite \
  2>&1 | tee logs/posterior_pooling_pilot_v6.log
```

Expected rows:

```text
component_draw_diagnostics.csv        400
component_imputation_recovery.csv     400
pooled_selection_results.csv          160
```

Draw 0 accounts for 80 component rows and should show `score_source=v5_score_reuse`.
The run therefore fits 320 new component pipelines rather than 400.

Summarize:

```bash
python -u experiments_A_B/summarize_posterior_pooling.py \
  --out-dir experiment_A/posterior_pooling_pilot_v6 \
  | tee logs/posterior_pooling_pilot_v6_summary.log
```

Detach tmux with `Ctrl+B`, then `D`.

## 6. What determines whether M=10 is justified

Proceed to `M=10` only if the M=5 pilot shows at least one of the following:

1. Exact or BayesianRidge posterior has improved power without a material FDP increase.
2. Oracle-selection Jaccard improves.
3. Component-score variability declines and the gain is larger than the median control.
4. The difference-in-differences contrast favors posterior pooling.

Do not interpret a generic improvement shared equally by median and posterior as
posterior-specific.  That would indicate ordinary knockoff derandomization.

## 7. 20-replicate M=1,5,10 confirmation

```bash
tmux new -s stabl_pooling_v6_confirm
```

Inside tmux, use the pilot command with these replacements:

```text
--out-dir experiment_A/posterior_pooling_confirmation_v6
--n-replicates 20
--pool-sizes 1,5,10
```

Expected rows:

```text
component_draw_diagnostics.csv       1600
component_imputation_recovery.csv    1600
pooled_selection_results.csv          480
```

Because V5 draw 0 is reused, 1,440 new component fits are required.

## 8. Resume

Use the identical scientific arguments and replace `--overwrite` with:

```bash
--resume
```

Append logs with `tee -a`.

Do not use `--refresh-pair-cache` unless deliberate regeneration is required.

## 9. Primary output interpretation

The main files are:

```text
pooling_summary.csv
pooling_paired_contrasts.csv
pooling_oracle_selection_agreement_summary.csv
figures/
```

The most important contrast is:

```text
posterior-specific pooling gain DID
= (posterior_M - posterior_1) - (median_M - median_1)
```

For FDP, a negative value is favorable.  For power and oracle Jaccard, a
positive value is favorable.
