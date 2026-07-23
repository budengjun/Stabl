# STABL Generator Track G2.2 Python only patch

This patch adds the next generator experiments directly under the existing
`stabl_knockoff_experiments/experiments_A_B/generator_track` directory.
It contains no shell runner scripts.

## Files added

```text
generator_track/
  g22_common.py
  run_g22a_validity_calibration.py
  run_g22a_ssi_validity.py
  run_g22b_stabl_aware_screening.py
  run_g22b_ssi_screening.py
  run_g22_self_test.py
  test_g22_python.py
  RUN_G22_PYTHON_ONLY.md
  configs/
    g22_plsko_screen_grid.json
```

No existing G0, G1, G2, or G2.1 result file is modified.

## Apply

Place the zip in:

```text
/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments/experiments_A_B
```

Extract it with overwrite enabled, then activate the existing Conda environment.

## Run order

First run the Python self test:

```bash
python generator_track/run_g22_self_test.py
```

Then run G2.2A:

```bash
python generator_track/run_g22a_ssi_validity.py
```

After G2.2A completes, run G2.2B:

```bash
python generator_track/run_g22b_ssi_screening.py
```

The convenience launchers contain the existing fenn10 SSI path and fresh random
seeds. Full command line interfaces are available in the corresponding main
Python files.
