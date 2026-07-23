#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
python -u generator_track/run_g1_plsko_tuning.py \
  --dataset ssi \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --omic CyTOF \
  --p 100 \
  --n-ko 2 \
  --p-s 10 \
  --ncomp 2,5 \
  --threshold-abs 0.1,0.2 \
  --sparsity 0.5,1.0 \
  --n-cores 4 \
  --out-dir generator_track_results/G1_ssi_smoke \
  --frozen-label plsko_tuned_ssi_smoke \
  --overwrite
