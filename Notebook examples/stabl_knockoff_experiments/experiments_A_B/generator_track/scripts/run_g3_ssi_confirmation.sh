#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
CONFIG="generator_track_results/G1_ssi_pilot/g2_generators.json"
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing $CONFIG. Run G1 first." >&2
  exit 2
fi
python -u run_experiment_b_semisynthetic.py \
  --dataset ssi \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --omic CyTOF \
  --subject-mode first \
  --complete-strategy drop-columns \
  --p-values 100 \
  --generator-configs-json "$CONFIG" \
  --n-replicates 50 \
  --n-knockoff-draws 5 \
  --n-signal 15 \
  --signal-strength 4 \
  --task classification \
  --n-bootstraps 100 \
  --stabl-grid-size 30 \
  --selection-rules stabl_min \
  --pair-bank-scope per-replicate \
  --c2st-every 10 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --random-state 20260822 \
  --out-dir generator_track_results/G3_ssi_confirmation \
  --overwrite
python -u generator_track/summarize_g2_generator_results.py \
  --out-dir generator_track_results/G3_ssi_confirmation
