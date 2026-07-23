#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
python -u generator_track/run_g1_plsko_tuning.py \
  --dataset dream \
  --data-path "$STABL_ROOT/Sample Data/Dream" \
  --omic Phylotype \
  --p 500 \
  --n-ko 10 \
  --p-s 20 \
  --ncomp 2,5,10 \
  --threshold-abs 0,0.1,0.2,0.3 \
  --sparsity 0.5,0.8,1.0 \
  --n-cores 4 \
  --out-dir generator_track_results/G1_dream_pilot \
  --frozen-label plsko_tuned_dream \
  --overwrite
python -u generator_track/build_g2_generator_config.py \
  --frozen-plsko generator_track_results/G1_dream_pilot/frozen_plsko_generator.json \
  --out generator_track_results/G1_dream_pilot/g2_generators.json \
  --include-default-plsko
