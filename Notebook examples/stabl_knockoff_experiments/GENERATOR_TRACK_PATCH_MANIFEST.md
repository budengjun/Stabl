# Generator Track patch manifest

Base archive SHA256: `e20c5324ec4c1c028b2c2c608fc894425634f827af4e8fba16e5671f1baa1f7b`

## Existing files modified

1. `experiments_A_B/research_common.py`
   Added support Jaccard, ranking metrics, selected index serialization, draw Jaccard helpers, near constant knockoff diagnostics, and marginal plus swap C2ST.
2. `experiments_A_B/run_experiment_b_semisynthetic.py`
   Added DREAM CLI support, paired generator seeds, a separate shared STABL seed, optional C2ST, ranking metrics, calibration gap, and selected feature recording.
3. `experiments_A_B/check_environment_ab.py`
   Added the official PLSKO tuning bridge check.
4. `experiments_A_B/RUN_COMMANDS.md`
   Added Generator Track launch guidance and clarified per replicate PLSKO pair banks.
5. `README.md`
   Added the integrated Generator Track entry point.

## Files added

1. `experiments_A_B/RUN_GENERATOR_TRACK_G0_G3.md`
2. `experiments_A_B/generator_track/`
3. `experiments_A_B/test_generator_track_updates.py`

## Compatibility

All prior V3 through V11.1 scripts, results, logs, caches, and snapshots are retained. The new Generator Track reuses the existing Experiment B runner and shared utilities. It does not replace or delete the BR mean experiment line.
