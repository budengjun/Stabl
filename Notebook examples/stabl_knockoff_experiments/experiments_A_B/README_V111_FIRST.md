# V11.1 first steps

V11.1 is a score-only reanalysis of an already completed V11 missingness-stress pilot.
It does not rerun imputation, knockoff generation, or STABL.

Run the regression test first:

```bash
cd "$EXP_ROOT"
python -u experiments_A_B/test_v111_updates.py
```

Then run the diagnosis:

```bash
python -u experiments_A_B/reanalyze_v111_threshold_diagnosis.py \
  --v11-pilot-dir experiment_B/v11_missingness_stress_pilot \
  --out-dir experiment_B/v11_missingness_stress_pilot/v111_threshold_diagnosis \
  2>&1 | tee logs/v111_threshold_diagnosis.log
```

Expected integrity for the current pilot:

```text
score_files = 2160
threshold_path_rows = 194400
matched_runs = 720
diagnostic_cells = 24
complete = True
```

The default run writes an aggregated threshold-path table. Add
`--save-full-path-runs` only when the large per-run path table is needed.
