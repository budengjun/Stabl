# Generator Track G0 to G3

The Generator Track is integrated under `experiments_A_B/generator_track`. It reuses `research_common.py` and the existing `run_experiment_b_semisynthetic.py`; it is not a separate project.

## 1. Enter the existing experiment directory

```bash
cd "/data/yhu94/Stabl/Notebook examples/stabl_knockoff_experiments/experiments_A_B"
source /data/yhu94/Stabl/stabl/bin/activate
```

The scripts infer `STABL_ROOT` from the current repository layout. It can also be supplied explicitly:

```bash
export STABL_ROOT=/data/yhu94/Stabl
```

## 2. Install and check PLSKO

```bash
Rscript generator_track/install_plsko.R
python check_environment_ab.py
```

## 3. G0 integration smoke

```bash
bash generator_track/scripts/run_g0_ssi.sh 2>&1 | tee ../logs/g0_ssi_generator_smoke.log
```

Proceed only when all four generator configurations finish without errors, same seed reproduction passes, and no near constant knockoff columns appear.

## 4. G1 tuning

Cheap bridge smoke:

```bash
bash generator_track/scripts/run_g1_ssi_smoke.sh 2>&1 | tee ../logs/g1_ssi_plsko_tuning_smoke.log
```

Formal pilot tuning:

```bash
bash generator_track/scripts/run_g1_ssi_pilot.sh 2>&1 | tee ../logs/g1_ssi_plsko_tuning_pilot.log
```

The second command creates:

```text
generator_track_results/G1_ssi_pilot/frozen_plsko_generator.json
generator_track_results/G1_ssi_pilot/g2_generators.json
```

## 5. G2 paired STABL pilot

```bash
bash generator_track/scripts/run_g2_ssi_pilot.sh 2>&1 | tee ../logs/g2_ssi_generator_pilot.log
```

The runner compares equicorrelated Gaussian, MVR, tuned PLSKO, and an untuned PLSKO diagnostic. It uses 10 independent semi synthetic outcomes, 3 fresh knockoff draws per outcome, 100 STABL bootstraps, original `stabl_min`, and a per replicate pair bank.

## 6. G3 confirmation

Run only after G2 progression is justified:

```bash
bash generator_track/scripts/run_g3_ssi_confirmation.sh 2>&1 | tee ../logs/g3_ssi_generator_confirmation.log
```

G3 uses fresh seeds, 50 replicates, and 5 knockoff draws.

## 7. DREAM external pilot

```bash
bash generator_track/scripts/run_g0_dream.sh
bash generator_track/scripts/run_g1_dream_pilot.sh
bash generator_track/scripts/run_g2_dream_pilot.sh
```

## Manual G2 invocation

```bash
python -u run_experiment_b_semisynthetic.py \
  --dataset ssi \
  --data-path "$STABL_ROOT/Sample Data/Biobank SSI" \
  --omic CyTOF \
  --p-values 100 \
  --generator-configs-json generator_track_results/G1_ssi_pilot/g2_generators.json \
  --n-replicates 10 \
  --n-knockoff-draws 3 \
  --n-signal 15 \
  --signal-strength 4 \
  --n-bootstraps 100 \
  --selection-rules stabl_min \
  --pair-bank-scope per-replicate \
  --c2st-every 5 \
  --target-fdr 0.10 \
  --out-dir generator_track_results/G2_ssi_pilot \
  --resume
```

Do not reuse G1 tuning outcomes as G2 or G3 confirmation outcomes. Do not average artificial STABL score identities across PLSKO draws.
