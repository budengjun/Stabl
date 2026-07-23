# V6.1 validity-aware posterior pooling

## 1. Install

Upload `experiments_A_B_v61_validity_aware_pooling.zip` to:

```text
/data/yhu94/Stabl/
```

Then:

```bash
cd /data/yhu94/Stabl

export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"

mv \
  "$EXP_ROOT/experiments_A_B" \
  "$EXP_ROOT/experiments_A_B_before_v61_$(date +%Y%m%d_%H%M%S)"

unzip -q \
  experiments_A_B_v61_validity_aware_pooling.zip \
  -d "$EXP_ROOT"

export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"
```

## 2. Tests

```bash
python -u experiments_A_B/test_v61_updates.py
```

Expected:

```text
[ok] V6.1 rule and vote parsing
[ok] M=1 validity-aware rule reproduces original stabl_min
[ok] component-wise artificial counts prevent identity dilution
[ok] component-selection vote sensitivity rules
[ok] V6.1 runner, reanalysis, and summarizer available
[ok] all V6.1 update tests passed
```

## 3. First action: reanalyze the existing V6 pilot

This does not regenerate knockoffs and does not refit STABL. It reads the existing 400 component score files.

```bash
mkdir -p \
  experiment_A/posterior_pooling_pilot_v61_reanalysis \
  logs

python -u experiments_A_B/reanalyze_v6_validity_aware_pooling.py \
  --source-dir experiment_A/posterior_pooling_pilot_v6 \
  --out-dir experiment_A/posterior_pooling_pilot_v61_reanalysis \
  --pool-sizes 1,5 \
  --pooling-aggregators mean \
  --pooling-rules mean_path_naive,mean_real_component_artificial_count,component_selection_vote \
  --vote-fractions 0.5,0.8 \
  --overwrite \
  2>&1 | tee logs/posterior_pooling_pilot_v61_reanalysis.log
```

Expected:

```text
validity_aware_selection_results.csv      640 rows
validity_aware_summary.csv                 64 rows
```

The 640 rows are:

```text
10 replicates × 2 mechanisms × 4 completions × 2 pool sizes × 4 rule variants
```

The four variants are naive mean path, validity-aware count, 50% vote and 80% vote.

Generate contrasts, oracle agreement and figures:

```bash
python -u experiments_A_B/summarize_validity_aware_pooling.py \
  --out-dir experiment_A/posterior_pooling_pilot_v61_reanalysis \
  2>&1 | tee logs/posterior_pooling_pilot_v61_summary.log
```

## 4. What to inspect

Primary files:

```text
validity_aware_summary.csv
validity_aware_paired_contrasts.csv
validity_aware_oracle_agreement_summary.csv
figures/
```

Primary calibration rule:

```text
mean_real_component_artificial_count
```

Negative control:

```text
mean_path_naive
```

### Required sanity checks

1. At M=1, naive and validity-aware results must be exactly identical.
2. At M=5, the validity-aware artificial count should exceed the naive pooled-artificial count when artificial identities rotate.
3. Oracle complete must no longer show an estimated FDP+ near 0.1 while realized FDP remains around 0.6 merely because of artificial-score dilution.
4. Vote rules have `estimated_fdp = NaN` by design. Evaluate them only by empirical FDP, power and oracle agreement.

### Fields that diagnose dilution

```text
mean_component_artificial_count_at_threshold
naive_pooled_artificial_count_at_threshold
artificial_dilution_at_threshold
calibration_gap_mean
```

## 5. Full V6.1 integration smoke

The reanalysis is the fastest scientific test. The command below separately confirms that the new full runner can generate components and apply all rules in one pass.

```bash
mkdir -p \
  experiment_A/posterior_pooling_smoke_v61 \
  pair_cache/experiment_A_v61 \
  logs

python -u experiments_A_B/run_experiment_a_validity_aware_pooling.py \
  --out-dir experiment_A/posterior_pooling_smoke_v61 \
  --pair-cache-dir pair_cache/experiment_A_v61 \
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
  --pooling-rules mean_path_naive,mean_real_component_artificial_count,component_selection_vote \
  --vote-fractions 0.5,0.8 \
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
  2>&1 | tee logs/posterior_pooling_smoke_v61.log
```

Expected:

```text
component_draw_diagnostics.csv          8 rows
component_imputation_recovery.csv       8 rows
validity_aware_selection_results.csv   32 rows
```

Then:

```bash
python -u experiments_A_B/summarize_validity_aware_pooling.py \
  --out-dir experiment_A/posterior_pooling_smoke_v61
```

## 6. Do not run M=10 yet

Use the existing V6 pilot reanalysis to decide whether the validity-aware rule improves calibration without destroying power or oracle agreement. The rule is an empirical proposal and does not yet have a formal FDR guarantee.

Proceed to a larger M only when the oracle-complete calibration check is acceptable and the posterior pooling gain exceeds ordinary median/knockoff averaging.
