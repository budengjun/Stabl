# V7: real-X semisynthetic posterior replacement benchmark

## Research question

V7 tests one focused question:

> When every downstream choice is held fixed, does BayesianRidge posterior-
> sampling imputation perform better than median imputation inside the original
> STABL `stabl_min` pipeline?

There is no score pooling, voting, `stabl_q`, or LCD comparator in the primary
experiment.

## Data design

The primary dataset is SSI CyTOF.

1. Load the real omics matrix.
2. Keep only columns without native missingness to create an observed complete
   real-X reference.
3. Select the requested number of high-variance features.
4. Simulate a sparse outcome from the standardized complete real-X matrix.
5. Inject controlled 20% MCAR or MAR missingness.
6. Compare oracle complete, median, and BayesianRidge posterior completion.
7. Run the original STABL `stabl_min` with paired knockoff/STABL seeds.

The complete reference is not a claim that SSI is naturally complete. It is a
controlled semisynthetic benchmark that preserves real omics dependence while
providing ground truth for missing values and signal support.

## Install V7

Upload the package to:

```text
/data/yhu94/Stabl/experiments_A_B_v7_real_x_completion_replacement.zip
```

Then:

```bash
cd /data/yhu94/Stabl

export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"

mv \
  "$EXP_ROOT/experiments_A_B" \
  "$EXP_ROOT/experiments_A_B_before_v7_$(date +%Y%m%d_%H%M%S)"

unzip -q \
  experiments_A_B_v7_real_x_completion_replacement.zip \
  -d "$EXP_ROOT"

export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"
```

## Test

```bash
python -u experiments_A_B/test_v7_updates.py
```

Expected:

```text
[ok] paired downstream seeds and varying completion seeds
[ok] median is deterministic and posterior draws vary without pooling
[ok] original STABL stabl_min is the only selection rule
[ok] V7 replicate-level summarizer and stability analysis
[ok] all V7 update tests passed
```

## Integration smoke

The smoke uses SSI CyTOF, 100 complete real-X features, one replicate, two
completion-draw slots, and 20 bootstraps.

```bash
cd "$EXP_ROOT"

mkdir -p \
  experiment_B/ssi_completion_replacement_smoke_v7 \
  pair_cache/experiment_B_v7 \
  logs

python -u experiments_A_B/run_experiment_b_completion_replacement.py \
  --dataset ssi \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --omic CyTOF \
  --subject-mode first \
  --reference-complete-strategy drop-columns \
  --p-values 100 \
  --out-dir experiment_B/ssi_completion_replacement_smoke_v7 \
  --pair-cache-dir pair_cache/experiment_B_v7 \
  --n-replicates 1 \
  --n-completion-draws 2 \
  --n-signal 10 \
  --task classification \
  --signal-strength 2.0 \
  --mechanisms MCAR,MAR \
  --missing-rate 0.20 \
  --completion-methods oracle_complete,median,bayesianridge_posterior \
  --generators gaussian_equicorrelated \
  --n-bootstraps 20 \
  --stabl-grid-size 30 \
  --stabl-c-min 0.01 \
  --stabl-c-max 1.0 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --iterative-max-iter 10 \
  --iterative-nearest-features 50 \
  --skip-precision-recovery \
  --fail-fast \
  --overwrite \
  2>&1 | tee logs/ssi_completion_replacement_smoke_v7.log
```

Expected row counts:

```text
selection_results.csv      12
imputation_recovery.csv    12
draw_diagnostics.csv       12
```

Calculation:

```text
1 replicate × 2 mechanisms × 3 completions × 2 draw slots = 12
```

Summarize:

```bash
python -u experiments_A_B/summarize_completion_replacement.py \
  --out-dir experiment_B/ssi_completion_replacement_smoke_v7
```

Check:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("experiment_B/ssi_completion_replacement_smoke_v7")
for name, expected in {
    "selection_results.csv": 12,
    "imputation_recovery.csv": 12,
    "draw_diagnostics.csv": 12,
}.items():
    df = pd.read_csv(base / name)
    errors = df["error"].fillna("").ne("").sum()
    print(name, len(df), "expected", expected, "errors", errors)
PY
```

## Paired pilot

The pilot uses 10 simulation replicates and 5 repeated ordinary STABL runs per
completion. Median and posterior receive the same knockoff/STABL seed in every
replicate/draw slot. The repeated posterior results are averaged within
replicate for inference; they are not pooled into a new score path.

```bash
tmux new -s stabl_v7_ssi_pilot
```

Inside tmux:

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"

mkdir -p \
  experiment_B/ssi_completion_replacement_pilot_v7 \
  pair_cache/experiment_B_v7 \
  logs

nice -n 10 python -u experiments_A_B/run_experiment_b_completion_replacement.py \
  --dataset ssi \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --omic CyTOF \
  --subject-mode first \
  --reference-complete-strategy drop-columns \
  --p-values 250 \
  --out-dir experiment_B/ssi_completion_replacement_pilot_v7 \
  --pair-cache-dir pair_cache/experiment_B_v7 \
  --n-replicates 10 \
  --n-completion-draws 5 \
  --n-signal 15 \
  --task classification \
  --signal-strength 2.0 \
  --mechanisms MCAR,MAR \
  --missing-rate 0.20 \
  --completion-methods oracle_complete,median,bayesianridge_posterior \
  --generators gaussian_equicorrelated \
  --n-bootstraps 100 \
  --stabl-grid-size 30 \
  --stabl-c-min 0.01 \
  --stabl-c-max 1.0 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --iterative-max-iter 10 \
  --iterative-nearest-features 50 \
  --skip-precision-recovery \
  --fail-fast \
  --overwrite \
  2>&1 | tee logs/ssi_completion_replacement_pilot_v7.log
```

Expected row counts:

```text
selection_results.csv      300
imputation_recovery.csv    300
draw_diagnostics.csv       300
```

Calculation:

```text
10 replicates × 2 mechanisms × 3 completions × 5 draw slots = 300
```

Summarize:

```bash
python -u experiments_A_B/summarize_completion_replacement.py \
  --out-dir experiment_B/ssi_completion_replacement_pilot_v7 \
  | tee logs/ssi_completion_replacement_pilot_v7_summary.log
```

Primary outputs:

```text
completion_replacement_summary.csv
replicate_level_metrics.csv
paired_completion_contrasts.csv
oracle_selection_agreement.csv
oracle_selection_agreement_summary.csv
selection_stability.csv
selection_stability_summary.csv
imputation_recovery_summary.csv
knockoff_diagnostics_summary.csv
figures/
```

## Interpretation

The primary comparison is:

```text
BayesianRidge posterior mean across repeated ordinary runs
minus
median mean across the same paired downstream randomizations
```

Main criteria:

1. Lower empirical FDP or false-positive burden.
2. Equal or higher power.
3. Higher Jaccard with complete-data STABL.
4. Equal or higher selection stability across repeated runs.
5. Better covariance and knockoff diagnostics.

Posterior may improve distribution recovery without improving support recovery.
That is a valid outcome and must be reported honestly.

## Main experiment

Run the main only after inspecting the pilot. Keep the scientific parameters
unchanged and change:

```text
--n-replicates 50
--out-dir experiment_B/ssi_completion_replacement_main_v7
```

Fifty replicates with five draw slots produces:

```text
50 × 2 × 3 × 5 = 1,500 rows per primary result file
```

## OOL replication

After SSI, use OOL with one row per subject:

```text
--dataset ool
--data-path "$STABL_ROOT/Sample Data"
--omic CyTOF
--subject-mode first
--p-values 100,250
```

Because OOL first-subject has a much smaller n, use it as a secondary
cross-dataset replication rather than the primary benchmark.

## Resume

Use the exact same scientific parameters and output directory, remove
`--overwrite`, and add:

```text
--resume
```
