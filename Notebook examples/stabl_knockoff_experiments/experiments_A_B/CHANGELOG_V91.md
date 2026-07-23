# V9.1 Changelog

V9.1 does not alter or overwrite the V9 confirmatory analysis. It adds two
separate, preregistered follow-up components.

## Stage 1: convergence and over-smoothing sensitivity

* Adds `run_v91_convergence_sensitivity.py`.
* Uses SSI CyTOF real-X, p=100, 15 signals, strength 4, and 20% MCAR.
* Compares BayesianRidge conditional mean at `max_iter=10,25,50` with fully
  paired X, outcome, mask, completion seed, knockoff seed, and STABL seed.
* Records `IterativeImputer.n_iter_`, convergence warnings, masked-entry
  variance ratios, completed variance ratios, effective/ridge condition
  numbers, minimum eigenvalues, completion recovery, and knockoff geometry.
* Runs complete-data STABL only as the Oracle-selection reference.
* Labels default 10-replicate FDP, power, and Oracle-Jaccard results as
  exploratory/descriptive only.

## Stage 2: threshold-independent support ranking

* Adds `reanalyze_v91_threshold_independent_ranking.py`.
* Reuses existing V9 score files and does not rerun STABL.
* Defines each real-feature score as the maximum selection frequency across
  the regularization grid.
* Reports tie-aware Precision@k for k=5,10,15,20, average precision, median
  signal rank, and reference AUROC.
* Performs replicate-level paired BR-mean-minus-Median bootstrap inference.

## Engineering

* Adds `v91_common.py` for convergence diagnostics, tie-aware fixed-k ranking,
  paired bootstrap intervals, Wilcoxon tests, and Holm adjustment.
* Adds `summarize_v91_convergence.py`, V9.1 run documentation, and tests.


## Hotfix 2026-07-18

* Removed the runtime requirement on the optional `tabulate` package when
  writing Markdown reports. CSV results and scientific calculations are
  unchanged.
* Test subprocess failures now print the captured child stdout and stderr.
