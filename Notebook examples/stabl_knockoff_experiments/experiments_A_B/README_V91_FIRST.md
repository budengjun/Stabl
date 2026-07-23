# Read This First: V9.1

V9.1 has two independent stages.

1. `run_v91_convergence_sensitivity.py` is a prospective MCAR max-iteration
   experiment. Its default 10-replicate downstream outputs are exploratory.
2. `reanalyze_v91_threshold_independent_ranking.py` reads the completed V9
   score files and tests support ranking without the `stabl_min` threshold.

Run `python -u experiments_A_B/test_v91_updates.py` before starting either
stage. Full commands and interpretation rules are in
`RUN_V91_CONVERGENCE_AND_RANKING.md`.
