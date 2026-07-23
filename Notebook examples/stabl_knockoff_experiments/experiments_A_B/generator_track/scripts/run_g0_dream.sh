#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
python -u generator_track/run_g0_integration_smoke.py \
  --dataset dream \
  --data-path "$STABL_ROOT/Sample Data/Dream" \
  --omic Phylotype \
  --p 500 \
  --generator-configs-json generator_track/configs/g0_generators.json \
  --out-dir generator_track_results/G0_dream_smoke \
  --overwrite
