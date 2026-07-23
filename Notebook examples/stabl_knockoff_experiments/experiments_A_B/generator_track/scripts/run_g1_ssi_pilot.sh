#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
python -u generator_track/run_g1_plsko_tuning.py \
  --dataset ssi \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --omic CyTOF \
  --p 100 \
  --n-ko 10 \
  --p-s 20 \
  --ncomp 2,5,10 \
  --threshold-abs 0,0.1,0.2,0.3 \
  --sparsity 0.5,0.8,1.0 \
  --n-cores 4 \
  --out-dir generator_track_results/G1_ssi_pilot \
  --frozen-label plsko_tuned_ssi \
  --overwrite
python -u generator_track/build_g2_generator_config.py \
  --frozen-plsko generator_track_results/G1_ssi_pilot/frozen_plsko_generator.json \
  --out generator_track_results/G1_ssi_pilot/g2_generators.json \
  --include-default-plsko
