# Run V11.1 threshold diagnosis

## Scientific question

Determine whether V11 mixed cells reflect a failure of BR mean feature ranking,
or an interaction between improved ranking and the original `stabl_min` threshold
and selected-set size.

## Inputs

A completed V11 pilot directory containing, for each dataset and missingness rate:

```text
scores/*.npz
selection_results.csv
v11_threshold_independent_ranking/ranking_replicate_metrics.csv
```

## Command

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"

cd "$EXP_ROOT"

python -u experiments_A_B/reanalyze_v111_threshold_diagnosis.py \
  --v11-pilot-dir experiment_B/v11_missingness_stress_pilot \
  --out-dir experiment_B/v11_missingness_stress_pilot/v111_threshold_diagnosis \
  --bootstrap-seed 80260730 \
  --n-bootstrap 20000 \
  --power-loss-guard 0.05 \
  --fdp-worsening-guard 0.05 \
  2>&1 | tee logs/v111_threshold_diagnosis.log
```

This typically finishes in under a few minutes because it reads saved score arrays only.

## Main outputs

```text
v111_integrity_audit.csv
v111_cell_diagnosis.csv
v111_stabl_min_paired_contrasts.csv
v111_br_at_median_selection_size_contrasts.csv
v111_ranking_paired_contrasts.csv
v111_argmin_variability.csv
v111_threshold_path_summary.csv
v111_matched_size_auc_contrasts.csv
v111_threshold_diagnosis_report.md
```

## Diagnostic labels

The labels are descriptive, not confirmatory:

```text
ranking_and_stabl_min_support
threshold_interaction_strong
threshold_interaction_directional
ranking_support_selected_set_tradeoff
ranking_limitation
mixed_inconclusive
```

A threshold-interaction label requires favorable threshold-independent ranking,
a favorable or materially improved matched-selection-size comparison, and evidence
that the original `stabl_min` result is worsened by selected-count or threshold behavior.

## Optional full path output

The aggregated threshold path is sufficient for the default diagnosis. To also save
all 194,400 per-run threshold rows:

```bash
... --save-full-path-runs
```
