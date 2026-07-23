# V7 changelog: real-X posterior replacement benchmark

## Scientific scope

V7 returns to the primary research question:

> Does BayesianRidge posterior-sampling imputation work better than median
> imputation as a drop-in replacement inside the original STABL pipeline?

V7 does **not** pool score paths and does not introduce a new selection rule.
The only primary selection rule is the original STABL `stabl_min`.

## New files

- `run_experiment_b_completion_replacement.py`
- `summarize_completion_replacement.py`
- `test_v7_updates.py`
- `RUN_V7_REAL_X_POSTERIOR_REPLACEMENT.md`
- `README_V7_FIRST.md`

## Design changes

1. Uses a complete reference matrix drawn from a real omics dataset.
   Native-missing columns are dropped before controlled missingness is injected.
2. Simulates sparse outcomes from the complete real-X reference, so the support
   is known and empirical FDP/power are measurable.
3. Compares ordinary pipelines:
   - oracle complete -> knockoff -> STABL `stabl_min`
   - median -> knockoff -> STABL `stabl_min`
   - BayesianRidge posterior draw -> knockoff -> STABL `stabl_min`
4. Posterior draws are repeated ordinary STABL runs. They are never pooled.
5. Median and posterior runs share the same knockoff and STABL seeds within each
   replicate/draw slot, giving a paired common-random-number comparison.
6. Statistical inference is performed at the simulation-replicate level after
   averaging repeated completion draws within replicate.
7. Reports empirical FDP, power, selected count, imputation recovery, knockoff
   diagnostics, agreement with complete-data STABL, and selection stability.
