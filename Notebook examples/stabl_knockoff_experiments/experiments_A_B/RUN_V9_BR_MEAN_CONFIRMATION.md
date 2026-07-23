# V9 Run Guide: BayesianRidge Mean Confirmation

## 1. Install

Upload the V9 archive to `/data/yhu94/Stabl`, then run:

```bash
cd /data/yhu94/Stabl

export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"

mv \
  "$EXP_ROOT/experiments_A_B" \
  "$EXP_ROOT/experiments_A_B_before_v9_$(date +%Y%m%d_%H%M%S)"

unzip -q \
  experiments_A_B_v9_br_mean_confirmation.zip \
  -d "$EXP_ROOT"

export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"
```

Run tests:

```bash
python -u experiments_A_B/test_v9_updates.py
```

Expected final line:

```text
[ok] all V9 update tests passed
```

## 2. Smoke test

```bash
cd "$EXP_ROOT"
mkdir -p experiment_B pair_cache/experiment_B_v9 logs

python -u experiments_A_B/run_experiment_b_mean_confirmation.py \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --out-dir experiment_B/ssi_br_mean_confirmation_smoke_v9 \
  --pair-cache-dir pair_cache/experiment_B_v9 \
  --n-replicates 1 \
  --n-completion-draws 2 \
  --n-bootstraps 20 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/ssi_br_mean_confirmation_smoke_v9.log
```

Expected successful selection rows:

```text
1 replicate × 2 mechanisms × 3 completions × 2 repeated runs = 12 rows
```

Check the audit:

```bash
column -s, -t < \
  experiment_B/ssi_br_mean_confirmation_smoke_v9/confirmatory_integrity_audit.csv
```

Every row must show `complete=True`.

## 3. Main 50-replicate confirmation

Start tmux:

```bash
tmux new -s stabl_v9_br_mean
```

Inside tmux:

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"

mkdir -p experiment_B pair_cache/experiment_B_v9 logs

nice -n 10 python -u experiments_A_B/run_experiment_b_mean_confirmation.py \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --out-dir experiment_B/ssi_br_mean_confirmation_v9 \
  --pair-cache-dir pair_cache/experiment_B_v9 \
  --n-replicates 50 \
  --n-completion-draws 5 \
  --n-bootstraps 100 \
  --n-jobs 8 \
  --fdp-noninferiority-margin 0 \
  --power-noninferiority-margin 0 \
  --overwrite \
  2>&1 | tee logs/ssi_br_mean_confirmation_v9.log
```

Expected successful selection rows:

```text
50 replicates × 2 mechanisms × 3 completions × 5 repeated runs
= 1,500 ordinary STABL fits
```

The independent inferential sample size is 50 replicates per mechanism, not 250 repeated runs.

Detach from tmux:

```text
Ctrl+B, then D
```

## 4. Resume after interruption

Use the identical command and replace `--overwrite` with:

```bash
--resume
```

The runner checks both the experiment config fingerprint and the V9 preregistration fingerprint. Do not change scientific parameters when resuming.

## 5. Main outputs

```text
v9_preregistration.json
confirmatory_integrity_audit.csv
confirmatory_replicate_metrics.csv
confirmatory_primary_contrasts.csv
confirmatory_secondary_contrasts.csv
confirmatory_decision.csv
confirmatory_report.md
figures_confirmatory/
```

Review the decision:

```bash
column -s, -t < \
  experiment_B/ssi_br_mean_confirmation_v9/confirmatory_decision.csv \
  | less -S

cat experiment_B/ssi_br_mean_confirmation_v9/confirmatory_report.md
```

## 6. Scientific interpretation

The primary contrast is always:

```text
BayesianRidge conditional mean minus Median
```

Primary endpoints are replicate-level mean FDP, power, and Jaccard agreement with complete-data STABL.

The default confirmation gate is strict. BR mean must show no confidence-interval evidence of worse FDP or power, and at least one primary endpoint must show a confidence-interval improvement. Directionally favorable point estimates without this gate are reported as promising but not confirmed.
