# V8 Run Guide: SSI Oracle Calibration + BayesianRidge Mean/Posterior Ablation

## Scientific purpose

V8 keeps the original STABL pipeline and `stabl_min` rule. It makes two focused changes:

1. **Oracle difficulty calibration** on real SSI X before comparing completion methods.
2. **BayesianRidge mean vs posterior draw ablation** to separate the value of the conditional model from the effect of posterior-sampling noise.

No score pooling, voting, `stabl_q`, or LCD is used.

---

## 1. Install V8

Upload the archive to:

```text
/data/yhu94/Stabl/experiments_A_B_v8_oracle_calibration_mean_ablation.zip
```

Then run:

```bash
cd /data/yhu94/Stabl

export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"

mv \
  "$EXP_ROOT/experiments_A_B" \
  "$EXP_ROOT/experiments_A_B_before_v8_$(date +%Y%m%d_%H%M%S)"

unzip -q \
  experiments_A_B_v8_oracle_calibration_mean_ablation.zip \
  -d "$EXP_ROOT"

export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"
```

Run tests:

```bash
python -u experiments_A_B/test_v8_updates.py
```

Expected final line:

```text
[ok] all V8 update tests passed
```

---

## 2. Oracle calibration smoke

This confirms the new oracle-only mode and grid driver in the real server environment.

```bash
cd "$EXP_ROOT"

mkdir -p \
  experiment_B/ssi_oracle_calibration_smoke_v8 \
  pair_cache/experiment_B_v8 \
  logs

python -u experiments_A_B/run_experiment_b_oracle_calibration.py \
  --dataset ssi \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --omic CyTOF \
  --subject-mode first \
  --reference-complete-strategy drop-columns \
  --p-values 100 \
  --signal-strengths 4 \
  --out-dir experiment_B/ssi_oracle_calibration_smoke_v8 \
  --pair-cache-dir pair_cache/experiment_B_v8 \
  --n-replicates 1 \
  --n-completion-draws 2 \
  --n-signal 15 \
  --task classification \
  --generators gaussian_equicorrelated \
  --n-bootstraps 20 \
  --stabl-grid-size 30 \
  --stabl-c-min 0.01 \
  --stabl-c-max 1.0 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --skip-precision-recovery \
  --fail-fast \
  --overwrite \
  2>&1 | tee logs/ssi_oracle_calibration_smoke_v8.log
```

Expected subdirectory:

```text
experiment_B/ssi_oracle_calibration_smoke_v8/p_100__strength_4/
```

The root output should also contain:

```text
oracle_calibration_summary.csv
oracle_calibration_recommendation.csv
oracle_calibration_replicate_metrics.csv
oracle_calibration_stability.csv
figures/
```

---

## 3. Oracle calibration grid

This is the next scientific experiment. It uses complete SSI X only and asks which outcome setting avoids the severe floor observed in V7.

```bash
tmux new -s stabl_v8_oracle_calibration
```

Inside tmux:

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"

cd "$EXP_ROOT"

mkdir -p \
  experiment_B/ssi_oracle_calibration_v8 \
  pair_cache/experiment_B_v8 \
  logs

nice -n 10 python -u experiments_A_B/run_experiment_b_oracle_calibration.py \
  --dataset ssi \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --omic CyTOF \
  --subject-mode first \
  --reference-complete-strategy drop-columns \
  --p-values 100,250 \
  --signal-strengths 4,6 \
  --out-dir experiment_B/ssi_oracle_calibration_v8 \
  --pair-cache-dir pair_cache/experiment_B_v8 \
  --n-replicates 10 \
  --n-completion-draws 3 \
  --n-signal 15 \
  --task classification \
  --generators gaussian_equicorrelated \
  --n-bootstraps 100 \
  --stabl-grid-size 30 \
  --stabl-c-min 0.01 \
  --stabl-c-max 1.0 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --skip-precision-recovery \
  --fail-fast \
  --overwrite \
  2>&1 | tee logs/ssi_oracle_calibration_v8.log
```

This grid has:

```text
2 p settings × 2 signal strengths × 10 replicates × 3 downstream draws
= 120 ordinary oracle-complete STABL fits
```

Compatible knockoff pairs are reused across signal strengths because X is unchanged and knockoff generation does not use y.

Detach from tmux:

```text
Ctrl+B, then D
```

Review the ranking:

```bash
column -s, -t < \
  experiment_B/ssi_oracle_calibration_v8/oracle_calibration_summary.csv \
  | less -S
```

The automatically ranked first row is saved in:

```text
oracle_calibration_recommendation.csv
```

The recommendation is a calibration ranking, not a claim of FDR control. Prefer a setting with oracle power roughly 0.30–0.60 and no extreme selection floor.

---

## 4. Completion-replacement smoke with BayesianRidge mean ablation

After installation, the original V7 runner is upgraded to V8 and supports:

```text
oracle_complete
median
bayesianridge_mean
bayesianridge_posterior
```

Run a one-replicate integration smoke:

```bash
cd "$EXP_ROOT"

mkdir -p \
  experiment_B/ssi_completion_ablation_smoke_v8 \
  pair_cache/experiment_B_v8 \
  logs

python -u experiments_A_B/run_experiment_b_completion_replacement.py \
  --experiment-mode completion_replacement \
  --dataset ssi \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --omic CyTOF \
  --subject-mode first \
  --reference-complete-strategy drop-columns \
  --p-values 100 \
  --out-dir experiment_B/ssi_completion_ablation_smoke_v8 \
  --pair-cache-dir pair_cache/experiment_B_v8 \
  --n-replicates 1 \
  --n-completion-draws 2 \
  --n-signal 15 \
  --task classification \
  --signal-strength 4 \
  --mechanisms MCAR,MAR \
  --missing-rate 0.20 \
  --completion-methods oracle_complete,median,bayesianridge_mean,bayesianridge_posterior \
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
  2>&1 | tee logs/ssi_completion_ablation_smoke_v8.log
```

Expected rows:

```text
1 replicate × 2 mechanisms × 4 completions × 2 draws = 16
```

Generate the summary:

```bash
python -u experiments_A_B/summarize_completion_replacement.py \
  --out-dir experiment_B/ssi_completion_ablation_smoke_v8
```

---

## 5. Replacement pilot after calibration

Use the `p` and signal strength selected by `oracle_calibration_recommendation.csv`.

Example below assumes the selected setting is `p=100`, `signal_strength=4`. Replace these two values if the calibration chooses another cell.

```bash
tmux new -s stabl_v8_completion_pilot
```

Inside tmux:

```bash
export STABL_ROOT="/data/yhu94/Stabl"
export EXP_ROOT="$STABL_ROOT/Notebook examples/stabl_knockoff_experiments"
export PYTHONPATH="$STABL_ROOT:$EXP_ROOT:$EXP_ROOT/experiments_A_B:$PYTHONPATH"
cd "$EXP_ROOT"

mkdir -p \
  experiment_B/ssi_completion_ablation_pilot_v8 \
  pair_cache/experiment_B_v8 \
  logs

nice -n 10 python -u experiments_A_B/run_experiment_b_completion_replacement.py \
  --experiment-mode completion_replacement \
  --dataset ssi \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --omic CyTOF \
  --subject-mode first \
  --reference-complete-strategy drop-columns \
  --p-values 100 \
  --out-dir experiment_B/ssi_completion_ablation_pilot_v8 \
  --pair-cache-dir pair_cache/experiment_B_v8 \
  --n-replicates 10 \
  --n-completion-draws 5 \
  --n-signal 15 \
  --task classification \
  --signal-strength 4 \
  --mechanisms MCAR,MAR \
  --missing-rate 0.20 \
  --completion-methods oracle_complete,median,bayesianridge_mean,bayesianridge_posterior \
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
  2>&1 | tee logs/ssi_completion_ablation_pilot_v8.log
```

Expected rows:

```text
10 replicates × 2 mechanisms × 4 completions × 5 draws = 400
```

Summarize:

```bash
python -u experiments_A_B/summarize_completion_replacement.py \
  --out-dir experiment_B/ssi_completion_ablation_pilot_v8 \
  | tee logs/ssi_completion_ablation_pilot_v8_summary.log
```

Primary contrasts in `paired_completion_contrasts.csv`:

```text
BayesianRidge mean vs Median
BayesianRidge posterior vs Median
BayesianRidge posterior vs BayesianRidge mean
Oracle complete vs Median
```

Interpretation:

- **Mean better than median**: BayesianRidge conditional model adds value without posterior sampling.
- **Posterior worse than mean**: posterior-sampling noise is the main instability source.
- **Posterior better than mean**: restoring conditional uncertainty provides additional downstream benefit.
- **Both mean and posterior fail**: the conditional model itself is not a useful replacement in this setting.

---

## 6. Resume

For an interrupted cell or pilot, keep all scientific parameters and paths identical and replace `--overwrite` with:

```bash
--resume
```

Do not use `--refresh-pair-cache` unless deliberate regeneration is required.
