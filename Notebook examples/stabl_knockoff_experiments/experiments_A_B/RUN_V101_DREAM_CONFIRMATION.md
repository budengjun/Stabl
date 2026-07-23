# V10.1 DREAM confirmation and SSI diagnosis

## Scientific scope

V10.1 follows the V10 external pilot. DREAM Phylotype passed the directional progression gate under both MCAR and MAR. SSI Proteomics showed mixed final-selection behavior and is therefore diagnosed rather than immediately expanded.

The original STABL `stabl_min` selection rule remains unchanged. The primary comparison is BayesianRidge conditional mean minus Median.

## Install and test

```bash
cd /data/yhu94/Stabl

export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"

mv \
  "$EXP_ROOT/experiments_A_B" \
  "$EXP_ROOT/experiments_A_B_before_v101_$(date +%Y%m%d_%H%M%S)"

unzip -q \
  experiments_A_B_v101_dream_confirmation.zip \
  -d "$EXP_ROOT"

export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"

python -u experiments_A_B/test_v101_updates.py
```

Expected final line:

```text
[ok] all V10.1 update tests passed
```

## Stage A smoke: DREAM convergence guard

```bash
mkdir -p experiment_B pair_cache logs

python -u experiments_A_B/run_v101_dream_convergence_guard.py \
  --data-path "$STABL_ROOT/Sample Data/Dream" \
  --out-dir experiment_B/v101_dream_max_iter_guard_smoke \
  --pair-cache-dir pair_cache/experiment_B_v101_guard_smoke \
  --n-replicates 1 \
  --n-downstream-draws 2 \
  --max-iters 10,25 \
  --n-bootstraps 20 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/v101_dream_max_iter_guard_smoke.log
```

Expected rows:

```text
completion diagnostics: 1 × 2 mechanisms × 2 max_iter = 4
knockoff diagnostics: 1 × 2 × 3 variants × 2 draws = 12
selection results: 12
```

## Stage A main: five fresh guard replicates

```bash
tmux new -s stabl_v101_guard
```

Inside tmux:

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"

nice -n 10 python -u experiments_A_B/run_v101_dream_convergence_guard.py \
  --data-path "$STABL_ROOT/Sample Data/Dream" \
  --out-dir experiment_B/v101_dream_max_iter_guard \
  --pair-cache-dir pair_cache/experiment_B_v101_guard \
  --n-replicates 5 \
  --n-downstream-draws 3 \
  --max-iters 10,25,50 \
  --n-bootstraps 50 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/v101_dream_max_iter_guard.log
```

Primary output:

```text
v101_guard_integrity_audit.csv
v101_guard_stage1a_summary.csv
v101_guard_stage1a_directional_contrasts.csv
v101_guard_stage1b_exploratory_summary.csv
v101_dream_guard_report.md
```

Interpretation rule:

- Keep `max_iter=10` when completion and knockoff geometry are stable across 10, 25, and 50.
- Do not select an iteration budget solely because its exploratory FDP is lowest.
- If 25 and 50 materially change geometry, select the first stable plateau and record the decision before confirmation.

## Stage B smoke: DREAM confirmation pipeline

```bash
python -u experiments_A_B/run_v101_dream_confirmation.py \
  --analysis-status smoke \
  --data-path "$STABL_ROOT/Sample Data/Dream" \
  --out-dir experiment_B/v101_dream_confirmation_smoke \
  --pair-cache-dir pair_cache/experiment_B_v101_confirmation_smoke \
  --n-replicates 2 \
  --n-completion-draws 2 \
  --n-bootstraps 20 \
  --iterative-max-iter 10 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/v101_dream_confirmation_smoke.log
```

The smoke output is always labelled `smoke_only_no_scientific_decision`.

## Stage B main: fresh 50-replicate confirmation

Run only after reviewing Stage A.

```bash
tmux new -s stabl_v101_confirm
```

Inside tmux:

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"

nice -n 10 python -u experiments_A_B/run_v101_dream_confirmation.py \
  --analysis-status confirmatory \
  --data-path "$STABL_ROOT/Sample Data/Dream" \
  --out-dir experiment_B/v101_dream_confirmation \
  --pair-cache-dir pair_cache/experiment_B_v101_confirmation \
  --n-replicates 50 \
  --n-completion-draws 5 \
  --n-bootstraps 100 \
  --iterative-max-iter 10 \
  --power-noninferiority-margin 0.03 \
  --fdp-nonworsening-margin 0.03 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/v101_dream_confirmation.log
```

Expected selection rows:

```text
50 replicates × 2 mechanisms × 3 completions × 5 runs = 1500
```

Resume by replacing `--overwrite` with `--resume` and using `tee -a`.

Confirmatory mechanism-level gate:

```text
Oracle Jaccard lower 95% CI > 0
Average precision lower 95% CI > 0
Power lower 95% CI >= -0.03
FDP upper 95% CI <= +0.03
```

Primary output:

```text
v101_preregistration.json
v101_integrity_audit.csv
v101_primary_and_supportive_contrasts.csv
v101_confirmatory_decision.csv
v101_dream_confirmation_report.md
v101_threshold_independent_ranking/
```

## Stage C: SSI Proteomics threshold-path diagnosis

This stage reads the existing V10 SSI pilot scores and performs no model fitting.

```bash
python -u experiments_A_B/reanalyze_v101_ssi_threshold_path.py \
  --ssi-pilot-dir experiment_B/v10_external_generalization_pilot/ssi_proteomics \
  --out-dir experiment_B/v10_external_generalization_pilot/ssi_proteomics/v101_ssi_threshold_path_diagnosis \
  2>&1 | tee logs/v101_ssi_threshold_path_diagnosis.log
```

Primary output:

```text
ssi_threshold_path_integrity_audit.csv
ssi_stabl_min_paired_contrasts.csv
ssi_argmin_variability.csv
ssi_br_at_median_selection_size_contrasts.csv
v101_ssi_threshold_path_report.md
```

## Pair-cache cleanup

After a result passes its integrity audit and is backed up:

```bash
rm -rf pair_cache/experiment_B_v101_guard
rm -rf pair_cache/experiment_B_v101_confirmation
```

Do not delete the confirmation `scores/` directory. It is required for threshold-independent ranking and future audits.
