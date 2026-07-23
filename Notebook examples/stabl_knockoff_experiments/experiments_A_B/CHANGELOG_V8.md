# V8 Changelog

## Main scientific change

V8 stops expanding posterior-pooling rules and returns to a clean drop-in replacement question under original STABL `stabl_min`.

## Added

- `run_experiment_b_oracle_calibration.py`
  - Runs an oracle-complete grid over feature count and signal strength.
  - Uses common random numbers across grid cells.
  - Reuses compatible knockoff pairs across signal strengths.
- `summarize_oracle_calibration.py`
  - Produces replicate-level metrics, stability, ranked grid summary, recommendation, and figures.
- `bayesianridge_mean` completion
  - `IterativeImputer(BayesianRidge, sample_posterior=False)`.
  - Fixed within replicate/mechanism across ordinary STABL draws.
- V8 replacement contrasts
  - BayesianRidge mean vs median.
  - BayesianRidge posterior vs median.
  - BayesianRidge posterior vs BayesianRidge mean.
  - Oracle complete vs median.
- `test_v8_updates.py`.

## Modified

- `run_experiment_b_completion_replacement.py`
  - Version increased to 8.
  - Supports `--experiment-mode completion_replacement` and `oracle_calibration`.
  - Default replacement set includes BayesianRidge conditional mean.
  - Keeps original STABL `stabl_min`; no pooling or alternative threshold rule.
- `research_common.py`
  - Added deterministic BayesianRidge conditional-mean completion.
- `summarize_completion_replacement.py`
  - Added mean completion to summaries, figures, stability, oracle agreement, and paired contrasts.

## Scientific rationale

The V7 pilot used a difficult SSI setting where oracle-complete STABL had low power and high FDP. V8 first calibrates outcome difficulty on complete real X. It then separates two questions:

1. Does a BayesianRidge conditional model improve over median?
2. Does posterior sampling add value or mainly add instability beyond the conditional mean?
