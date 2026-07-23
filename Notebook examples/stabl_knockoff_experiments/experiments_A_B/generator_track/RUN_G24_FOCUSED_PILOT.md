# G2.4 Focused Gaussian Knockoff Pilot

## Scientific scope

G2.4 carries forward only the two candidates that survived the G2.3 smoke screen, alongside the current baseline:

1. `equicorr`: current Ledoit Wolf covariance plus equicorrelated Gaussian knockoffs.
2. `mvr`: current covariance estimate plus MVR knockoffs.
3. `factor_r20_mvr`: rank 20 factor covariance plus MVR knockoffs.

PLSKO, SDP, factor rank 10 MVR, and the wider factor grid are not rerun.

Stage 1 is outcome free. It uses 10 knockoff draws, five repeated paired C2ST evaluations, five folds, marginal AUC, global swap AUC, moment diagnostics, and explicit convergence tracking.

Stage 2 uses 10 fresh semi synthetic outcomes and five knockoff draws per outcome. All three generators share outcome seeds, knockoff seeds, and STABL seeds. With the pilot defaults, this is 150 STABL fits.

The experiment may nominate a confirmation candidate. It cannot authorize a final generator replacement.

## Correct installation location

Place `G24_focused_pilot_patch.zip` in the project root:

```text
/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments
```

Then run:

```bash
PROJECT_ROOT="/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments"
cd "$PROJECT_ROOT"
unzip -o G24_focused_pilot_patch.zip -d "$PROJECT_ROOT"
```

The ZIP begins with `experiments_A_B/`. It does not contain an outer `stabl_knockoff_experiments/` folder, so it will not create a nested project.

## Environment and self tests

```bash
cd "/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments"
. ~/.profile
conda activate stabl

python -u experiments_A_B/generator_track/run_g24_self_test.py
python -u experiments_A_B/generator_track/check_g24_environment.py
```

Expected generator checks:

```text
[ok] equicorr
[ok] mvr
[ok] factor_r20_mvr
```

## Smoke test

Run this first to validate the complete server workflow:

```bash
cd "/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments"
mkdir -p logs
set -o pipefail

python -u experiments_A_B/generator_track/run_g24_focused_pilot.py \
  --ssi-data-path "/data/yhu94/Stabl/Sample Data/Biobank SSI" \
  --ssi-omic CyTOF \
  --out-dir experiment_B/G24_ssi_focused_smoke \
  --pair-cache-dir pair_cache/G24_ssi_focused_smoke \
  --p 50 \
  --validity-draws 2 \
  --c2st-repeats 2 \
  --c2st-folds 4 \
  --screen-replicates 2 \
  --screen-draws 2 \
  --n-signal 10 \
  --signal-strength 4 \
  --n-bootstraps 20 \
  --n-jobs 8 \
  --bootstrap-samples 1000 \
  --overwrite \
  2>&1 | tee logs/g24_ssi_focused_smoke.log
```

## Focused pilot

After the smoke completes successfully:

```bash
cd "/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments"
mkdir -p logs
set -o pipefail

python -u experiments_A_B/generator_track/run_g24_focused_pilot.py \
  --ssi-data-path "/data/yhu94/Stabl/Sample Data/Biobank SSI" \
  --ssi-omic CyTOF \
  --out-dir experiment_B/G24_ssi_focused_pilot \
  --pair-cache-dir pair_cache/G24_ssi_focused_pilot \
  --p 100 \
  --validity-draws 10 \
  --c2st-repeats 5 \
  --c2st-folds 5 \
  --screen-replicates 10 \
  --screen-draws 5 \
  --n-signal 15 \
  --signal-strength 4 \
  --n-bootstraps 100 \
  --n-jobs 8 \
  --bootstrap-samples 10000 \
  --overwrite \
  2>&1 | tee logs/g24_ssi_focused_pilot.log
```

The values above are also the script defaults except for the input and output paths.

## Resume after interruption

Use exactly the same scientific arguments. Replace `--overwrite` with `--resume`:

```bash
python -u experiments_A_B/generator_track/run_g24_focused_pilot.py \
  --ssi-data-path "/data/yhu94/Stabl/Sample Data/Biobank SSI" \
  --ssi-omic CyTOF \
  --out-dir experiment_B/G24_ssi_focused_pilot \
  --pair-cache-dir pair_cache/G24_ssi_focused_pilot \
  --p 100 \
  --validity-draws 10 \
  --c2st-repeats 5 \
  --c2st-folds 5 \
  --screen-replicates 10 \
  --screen-draws 5 \
  --n-signal 15 \
  --signal-strength 4 \
  --n-bootstraps 100 \
  --n-jobs 8 \
  --bootstrap-samples 10000 \
  --resume \
  2>&1 | tee -a logs/g24_ssi_focused_pilot.log
```

Do not combine `--resume` with `--refresh-pairs` or `--refresh-scores`.

## Directory layout

Results and caches are intentionally separated:

```text
stabl_knockoff_experiments/
├── experiment_B/
│   └── G24_ssi_focused_pilot/
├── pair_cache/
│   └── G24_ssi_focused_pilot/
├── logs/
│   └── g24_ssi_focused_pilot.log
└── experiments_A_B/
    └── generator_track/
```

Relative paths are resolved from the project root, not from the shell current working directory.

## Main outputs

```text
g24_stage1_validity_runs.csv
g24_stage1_validity_summary.csv
g24_stage1_selection.json
g24_stage2_stabl_runs.csv
g24_stage2_draw_stability.csv
g24_stage2_replicate_metrics.csv
g24_stage2_matched_size_runs.csv
g24_stage2_matched_size_replicates.csv
g24_stage2_paired_contrasts.csv
g24_candidate_decisions.csv
g24_recommendation.json
g24_status.json
```

`score_file` entries are stored relative to the result directory, so the result folder remains portable.

## C2ST convergence handling

Each logistic C2ST fold starts with `max_iter=10000`. A fold that emits `ConvergenceWarning` is retried using the same `liblinear` solver with `max_iter=1000000`.

The run records:

```text
marginal_c2st_warning_count
marginal_c2st_retry_count
marginal_c2st_persistent_folds
global_swap_c2st_warning_count
global_swap_c2st_retry_count
global_swap_c2st_persistent_folds
```

Any persistent nonconverged fold fails the validity gate. It is not silently included as a valid AUC.

## Preregistered downstream gates

MVR is compared with equicorr.

Factor rank 20 MVR is compared with both equicorr and MVR.

Default noninferiority thresholds are:

```text
FDP delta <= 0.05
Average precision delta >= -0.02
Matched size power delta >= -0.02
Draw Jaccard delta >= -0.05
```

A successful pilot writes `status: confirmation_candidate`. The recommendation always keeps:

```json
"holdout_required": true,
"final_replacement_authorized": false
```

## Package the completed pilot

```bash
cd "/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments"

zip -r G24_ssi_focused_pilot_results.zip \
  experiment_B/G24_ssi_focused_pilot \
  logs/g24_ssi_focused_pilot.log
```
