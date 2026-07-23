# V9 Changelog

## Scientific scope

V9 narrows the project to a confirmatory paired comparison of BayesianRidge conditional mean against Median under original STABL `stabl_min`.

## Primary design

* SSI CyTOF real-X
* n = 93, p = 100
* 15 simulated signals, signal strength = 4
* 20% MCAR and 20% MAR
* 50 simulation replicates
* 5 repeated ordinary STABL runs per replicate and completion
* Oracle complete, Median, BayesianRidge conditional mean
* Sample-Sigma Gaussian equicorrelated knockoffs
* 100 STABL bootstraps and 30 C values

## Added

* `run_experiment_b_mean_confirmation.py`
  * Writes a preregistration sidecar before execution.
  * Locks the primary scientific configuration.
  * Runs the V9 real-X runner and both summary layers.
* `summarize_mean_confirmation.py`
  * Audits completeness and duplicate keys.
  * Averages repeated runs within simulation replicate.
  * Tests FDP, power, and Oracle-selection Jaccard as primary endpoints.
  * Uses paired bootstrap confidence intervals and paired Wilcoxon tests.
  * Applies Holm adjustment across primary tests.
  * Produces a predefined decision gate.
* `test_v9_updates.py`
* `RUN_V9_BR_MEAN_CONFIRMATION.md`
* `README_V9_FIRST.md`

## Modified

* `run_experiment_b_completion_replacement.py`
  * Version increased to 9.
  * Added `mean_confirmation` mode.
  * Confirmation mode requires exactly Oracle complete, Median, and BayesianRidge mean.
  * Posterior completion is no longer required for general replacement experiments.
* `summarize_completion_replacement.py`
  * Labels updated to V9 and remains compatible with the three-branch confirmation.

## Decision rule

For each missingness mechanism, support requires:

1. The upper 95% CI for delta FDP does not exceed the preregistered noninferiority margin.
2. The lower 95% CI for delta power does not cross below the preregistered noninferiority margin.
3. At least one primary endpoint shows a stable improvement by its 95% CI.

Default noninferiority margins are zero. Change them only before starting the main experiment.
