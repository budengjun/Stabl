# V6 change log: multiple-posterior pooling

## Scientific reset

V6 keeps the published STABL `stabl_min` rule as the only primary selection
rule.  It does not use `stabl_q` or LCD as a pass/fail criterion.

The new question is whether averaging full STABL score paths across multiple
posterior-completed datasets can convert V5's distributional improvement into
more stable downstream feature selection.

## New files

- `run_experiment_a_posterior_pooling.py`
- `posterior_pooling_common.py`
- `summarize_posterior_pooling.py`
- `test_v6_updates.py`
- `RUN_V6_POSTERIOR_POOLING.md`

## Primary pooling estimator

For the first `M` component pipelines, V6 averages the complete STABL score
matrices elementwise for both real and knockoff variables, then applies the
original STABL FDP+ argmin threshold once to the pooled paths.

Pool sizes are nested prefixes.  For example, `M=1,5,10` uses components
`0`, `0..4`, and `0..9`.  This makes within-replicate comparisons efficient
and paired.

## Critical controls

- Oracle-complete and median are repeated at the same `M` values.
- Their completion matrices are deterministic, but knockoff draws vary.
- They therefore quantify generic knockoff derandomization.
- A posterior-specific benefit must exceed the median gain from `M=1` to `M>1`.
- The summarizer reports this as a difference-in-differences contrast.

## Randomness design

Default `--stabl-seed-mode shared` holds the STABL bootstrap subsamples fixed
across component draws.  Completion and knockoff randomness vary, so pooling
is not confounded by merely averaging different bootstrap samples.

`--stabl-seed-mode independent` is available as a sensitivity analysis.

## V5 cache reuse

`--reuse-v5-out-dir` imports scientifically compatible V5 draw-0 score paths,
diagnostics, and recovery rows.  It validates the V5 configuration before
reuse.  Component draw 0 is designed to reproduce the V5 single-draw baseline.

## New outputs

- `pooled_selection_results.csv`
- `component_draw_diagnostics.csv`
- `component_imputation_recovery.csv`
- `component_scores/`
- `pooled_scores/`
- `pooling_summary.csv`
- `pooling_paired_contrasts.csv`
- `pooling_oracle_selection_agreement.csv`
- `pooling_oracle_selection_agreement_summary.csv`
- `figures/`
