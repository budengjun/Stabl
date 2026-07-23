#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
CONFIG="generator_track_results/G1_dream_pilot/g2_generators.json"
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing $CONFIG. Run generator_track/scripts/run_g1_dream_pilot.sh first." >&2
  exit 2
fi
python -u run_experiment_b_semisynthetic.py \
  --dataset dream \
  --data-path "$STABL_ROOT/Sample Data/Dream" \
  --omic Phylotype \
  --subject-mode first \
  --complete-strategy drop-columns \
  --p-values 500 \
  --generator-configs-json "$CONFIG" \
  --n-replicates 10 \
  --n-knockoff-draws 3 \
  --n-signal 20 \
  --signal-strength 4 \
  --task classification \
  --n-bootstraps 100 \
  --stabl-grid-size 30 \
  --selection-rules stabl_min \
  --pair-bank-scope per-replicate \
  --c2st-every 5 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --random-state 20260723 \
  --out-dir generator_track_results/G2_dream_pilot \
  --overwrite
python -u generator_track/summarize_g2_generator_results.py \
  --out-dir generator_track_results/G2_dream_pilot
