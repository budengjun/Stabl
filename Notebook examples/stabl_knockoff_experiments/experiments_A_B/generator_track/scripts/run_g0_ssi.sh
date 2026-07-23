#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
python -u generator_track/run_g0_integration_smoke.py \
  --dataset ssi \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --omic CyTOF \
  --p 100 \
  --generator-configs-json generator_track/configs/g0_generators.json \
  --out-dir generator_track_results/G0_ssi_smoke \
  --overwrite
