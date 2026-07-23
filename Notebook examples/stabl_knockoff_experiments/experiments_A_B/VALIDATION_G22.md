# Validation record

Completed before packaging:

1. All Python files passed `py_compile`.
2. Both main command line interfaces loaded successfully against the integrated
   `experiments_A_B` source tree.
3. `run_g22_self_test.py` passed with an exchangeable independent copy near AUC
   0.5 and a mean shifted invalid control above AUC 0.75.
4. Three unit tests passed for deterministic top k selection and both validity
   and downstream gate logic.
5. The package contains no shell script.

The local runtime does not contain the user's R PLSKO installation, SSI data,
knockpy environment, or local editable STABL package. Therefore official PLSKO
and full STABL fitting were not claimed to have run locally. Those paths are
exercised through the already validated project interfaces on fenn10.
