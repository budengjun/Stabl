# Experiment A v3 update and calibration pilot

This update adds two requested changes.

1. A true population covariance Gaussian equicorrelated generator named `gaussian_equicorrelated_true_sigma`
2. STABL threshold comparisons now use `>=`, matching the paper definition

It also renames the recovery field `posterior_theory_aligned` to `missingness_assumption_compatible`.

## 1. Replace the code folder in the existing experiment directory

The expected project directory is:

```text
/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments
```

From the STABL repository root:

```bash
cd /data/yhu94/Stabl

EXP_ROOT="/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments"

mv "$EXP_ROOT/experiments_A_B" \
   "$EXP_ROOT/experiments_A_B_before_v3_$(date +%Y%m%d_%H%M%S)"

cd "$EXP_ROOT"
unzip /path/to/experiments_A_B_v3_true_sigma.zip
```

The zip contains one folder named `experiments_A_B`.

Do not delete the existing `pair_cache` directory. Version 1 sample covariance and PLSKO cache files remain compatible. Only true Sigma cache entries use the new version 2 metadata.

## 2. Set paths and run the lightweight update test

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"

cd "$EXP_ROOT"

python -u experiments_A_B/test_v3_updates.py
```

Expected output:

```text
[ok] known Sigma Gaussian sampler
[ok] covariance checks: ...
[ok] STABL threshold comparisons use >=
```

Then run the environment check:

```bash
python -u experiments_A_B/check_environment_ab.py
```

## 3. Run the five replicate calibration pilot

This pilot is deliberately stronger than the first smoke test so that `stabl_q` and `knockoff_plus` have a realistic chance to return nonempty sets.

```bash
cd "$EXP_ROOT"

mkdir -p \
  experiment_A/calibration_pilot_v3 \
  pair_cache/experiment_A \
  logs

nice -n 10 python -u experiments_A_B/run_experiment_a_oracle_completion.py \
  --out-dir experiment_A/calibration_pilot_v3 \
  --pair-cache-dir pair_cache/experiment_A \
  --n-replicates 5 \
  --n 150 \
  --p 200 \
  --n-signal 25 \
  --block-size 25 \
  --rho 0.60 \
  --task classification \
  --signal-strength 2.0 \
  --mechanisms MCAR,MAR \
  --missing-rate 0.20 \
  --completion-methods oracle_complete,median,exact_gaussian_posterior,bayesianridge_posterior \
  --generators gaussian_equicorrelated,gaussian_equicorrelated_true_sigma \
  --n-knockoff-draws 1 \
  --n-bootstraps 50 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --skip-precision-recovery \
  --fail-fast \
  --overwrite \
  2>&1 | tee logs/experiment_A_calibration_pilot_v3.log
```

The old sample covariance pair cache can be reused when the scaled X matrix, generator label, and seed match. New true Sigma pairs will be generated once and cached separately under the label `equicorr_true_sigma`.

## 4. Check the pilot automatically

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("experiment_A/calibration_pilot_v3")
selection = pd.read_csv(base / "selection_results.csv")
diagnostics = pd.read_csv(base / "draw_diagnostics.csv")
recovery = pd.read_csv(base / "imputation_recovery.csv")
summary = pd.read_csv(base / "summary.csv")

print("selection rows:", len(selection))
print("diagnostic rows:", len(diagnostics))
print("recovery rows:", len(recovery))
print("selection errors:", selection["error"].fillna("").ne("").sum())
print("diagnostic errors:", diagnostics["error"].fillna("").ne("").sum())

columns = [
    "mechanism",
    "completion",
    "generator_label",
    "selection_rule",
    "n_success",
    "empirical_fdr",
    "power_mean",
    "selected_mean",
    "fdp_exceedance_probability",
]
print(summary[columns].to_string(index=False))
PY
```

Expected row counts are:

```text
selection_results.csv: 120

draw_diagnostics.csv: 40

imputation_recovery.csv: 40
```

The selection count is calculated as 5 replicates times 2 mechanisms times 4 completions times 2 generators times 3 selection rules.

## 5. Interpretation guard for the true Sigma generator

The true Sigma generator is a clean oracle covariance reference for:

```text
oracle_complete
exact_gaussian_posterior under MCAR or MAR
```

For median and BayesianRidge completion, the completed distribution need not equal the original Gaussian population. Using the original true Sigma in those branches is therefore a diagnostic intervention, not a claim that the resulting knockoff is oracle valid.

The output column `true_sigma_exactly_applicable` records this distinction.

## 6. Resume after interruption

Use exactly the same scientific arguments and replace `--overwrite` with `--resume`.

```bash
--resume
```

Do not use `--refresh-pair-cache` unless you intentionally want to regenerate valid cached pairs.
