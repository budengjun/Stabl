# V9.1 Run Guide: Convergence, Over-Smoothing, and Ranking

## 1. Install

```bash
cd /data/yhu94/Stabl

export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"

mv \
  "$EXP_ROOT/experiments_A_B" \
  "$EXP_ROOT/experiments_A_B_before_v91_$(date +%Y%m%d_%H%M%S)"

unzip -q \
  experiments_A_B_v91_convergence_ranking.zip \
  -d "$EXP_ROOT"

export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"

python -u experiments_A_B/test_v91_updates.py
```

Expected final line:

```text
[ok] all V9.1 update tests passed
```

## 2. Stage 1 smoke

```bash
mkdir -p experiment_B pair_cache/experiment_B_v91 logs

python -u experiments_A_B/run_v91_convergence_sensitivity.py \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --out-dir experiment_B/ssi_br_mean_max_iter_smoke_v91 \
  --pair-cache-dir pair_cache/experiment_B_v91 \
  --n-replicates 1 \
  --n-downstream-draws 2 \
  --max-iters 10,25 \
  --n-bootstraps 20 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/ssi_br_mean_max_iter_smoke_v91.log
```

Expected rows:

```text
completion_diagnostics: 1 × 2 = 2
knockoff_diagnostics: 1 × (1 Oracle + 2 iteration budgets) × 2 draws = 6
selection_results: 6
```

Check:

```bash
column -s, -t < \
  experiment_B/ssi_br_mean_max_iter_smoke_v91/v91_integrity_audit.csv
```

Every row must show `complete=True`.

## 3. Stage 1 main directional experiment

This is the preregistered 10-replicate MCAR experiment. Geometry and completion
metrics are the directional mechanism test. Downstream FDP, power, and Oracle
Jaccard are exploratory only at this sample size.

```bash
tmux new -s stabl_v91_iter
```

Inside tmux:

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"

mkdir -p experiment_B pair_cache/experiment_B_v91 logs

nice -n 10 python -u experiments_A_B/run_v91_convergence_sensitivity.py \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --out-dir experiment_B/ssi_br_mean_max_iter_v91 \
  --pair-cache-dir pair_cache/experiment_B_v91 \
  --n-replicates 10 \
  --n-downstream-draws 5 \
  --max-iters 10,25,50 \
  --n-bootstraps 100 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/ssi_br_mean_max_iter_v91.log
```

Expected work:

```text
30 BayesianRidge completion fits
200 Gaussian knockoff draws
200 ordinary STABL fits
```

Resume with the identical command after replacing `--overwrite` with
`--resume`.

### Geometry-only option

To run the cheap Stage 1A direction check without STABL:

```bash
python -u experiments_A_B/run_v91_convergence_sensitivity.py \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --out-dir experiment_B/ssi_br_mean_max_iter_geometry_v91 \
  --pair-cache-dir pair_cache/experiment_B_v91 \
  --n-replicates 10 \
  --n-downstream-draws 5 \
  --max-iters 10,25,50 \
  --no-run-downstream \
  --overwrite
```

## 4. Optional 25-pair downstream follow-up

Only use this after reviewing Stage 1A. It increases downstream precision and
uses the two extreme iteration budgets.

```bash
nice -n 10 python -u experiments_A_B/run_v91_convergence_sensitivity.py \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --out-dir experiment_B/ssi_br_mean_max_iter_25pairs_v91 \
  --pair-cache-dir pair_cache/experiment_B_v91_25pairs \
  --n-replicates 25 \
  --n-downstream-draws 5 \
  --max-iters 10,50 \
  --n-bootstraps 100 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/ssi_br_mean_max_iter_25pairs_v91.log
```

## 5. Stage 2 threshold-independent ranking

This step reuses the existing V9 score files. It does not rerun STABL.

```bash
python -u experiments_A_B/reanalyze_v91_threshold_independent_ranking.py \
  --v9-out-dir experiment_B/ssi_br_mean_confirmation_v9 \
  --out-dir experiment_B/ssi_br_mean_confirmation_v9/v91_threshold_independent_ranking \
  --k-values 5,10,15,20 \
  2>&1 | tee logs/v91_threshold_independent_ranking.log
```

## 6. Main outputs

Stage 1:

```text
config.json
completion_diagnostics.csv
knockoff_diagnostics.csv
selection_results.csv
stage1a_replicate_metrics.csv
stage1a_summary.csv
stage1a_directional_contrasts.csv
stage1b_replicate_metrics.csv
stage1b_exploratory_summary.csv
v91_integrity_audit.csv
v91_stage1_report.md
figures_v91/
```

Stage 2:

```text
ranking_run_metrics.csv
ranking_replicate_metrics.csv
ranking_primary_contrasts.csv
ranking_reference_contrasts.csv
ranking_completion_summary.csv
ranking_consistency_summary.csv
ranking_integrity_audit.csv
v91_ranking_report.md
figures_ranking/
```

## 7. Interpretation guardrails

* `ConvergenceWarning` means the stopping criterion was not reached within the
  iteration budget. It is not itself proof of over-smoothing.
* The over-smoothing prediction is supported only if the directional pattern
  is coherent across completion variance, pair correlation, `s_relative`, and
  conditioning diagnostics.
* Default 10-replicate downstream results are exploratory. Do not call them
  confirmed or robust.
* Fixed-k TP and recall are deterministic transforms of Precision@k and are not
  independent evidence.
* Stage 2 conclusions are independent of `stabl_min` threshold selection but
  remain specific to the saved V9 STABL score paths.
