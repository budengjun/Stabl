# Start here: V6.1

Run the lightweight tests first:

```bash
python -u experiments_A_B/test_v61_updates.py
```

Then reanalyze the existing V6 pilot without rerunning STABL:

```bash
python -u experiments_A_B/reanalyze_v6_validity_aware_pooling.py \
  --source-dir experiment_A/posterior_pooling_pilot_v6 \
  --out-dir experiment_A/posterior_pooling_pilot_v61_reanalysis \
  --pool-sizes 1,5 \
  --pooling-rules mean_path_naive,mean_real_component_artificial_count,component_selection_vote \
  --vote-fractions 0.5,0.8 \
  --overwrite

python -u experiments_A_B/summarize_validity_aware_pooling.py \
  --out-dir experiment_A/posterior_pooling_pilot_v61_reanalysis
```

Read `RUN_V61_VALIDITY_AWARE_POOLING.md` for installation, validation and full-run commands.
