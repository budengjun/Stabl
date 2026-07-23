# G2.2 validity calibration and STABL aware PLSKO screening

## Scientific purpose

G2 and G2.1 showed that the officially tuned PLSKO configuration increased raw
power mainly by lowering the STABL reliability threshold and expanding the
selected set. They also showed persistent swap C2ST AUC near 0.98. G2.2 does
not proceed directly to confirmation. It first calibrates the validity tests,
then screens PLSKO configurations with validity and reliability gates before
power is considered.

## G2.2A

`run_g22a_validity_calibration.py` evaluates:

1. Gaussian oracle true covariance equicorrelated knockoffs
2. Sample covariance equicorrelated knockoffs
3. MVR knockoffs
4. PLSKO default
5. The G1 frozen PLSKO configuration
6. Columnwise permutation invalid controls
7. Mean shifted invalid controls

It runs both a Gaussian design with known population covariance and the SSI
CyTOF real design. The primary tests are repeated paired marginal C2ST and
repeated paired global swap C2ST. A subset of runs also receives single feature
and block swap audits.

Primary outputs:

```text
generator_track_results/G22A_validity_calibration/
  g22a_c2st_runs.csv
  g22a_c2st_summary.csv
  g22a_pair_diagnostics.csv
  g22a_pair_summary.csv
  g22a_recommended_c2st_gate.json
  g22a_status.json
  pairs/
```

Run the default experiment:

```bash
python generator_track/run_g22a_ssi_validity.py
```

A lighter smoke run can be launched directly:

```bash
python generator_track/run_g22a_validity_calibration.py \
  --ssi-data-path "/data/yhu94/Stabl/Sample Data/Biobank SSI" \
  --tuned-config-json generator_track_results/G1_ssi_pilot/frozen_plsko_generator.json \
  --out-dir generator_track_results/G22A_smoke \
  --synthetic-replicates 1 \
  --draws 1 \
  --ssi-draws 2 \
  --c2st-repeats 2 \
  --n-single-features 2 \
  --overwrite
```

## G2.2B

`run_g22b_stabl_aware_screening.py` has two stages.

### Stage 1

The complete PLSKO grid is audited without using outcomes. The default grid has
36 absolute threshold configurations plus the package default configuration.
Only configurations passing the G2.2A marginal and global swap gates are
eligible to become a frozen candidate.

When no PLSKO configuration passes validity, the closest three configurations
still enter Stage 2 as diagnostic only. The script will not freeze one of them.

### Stage 2

Selected PLSKO configurations are compared with equicorrelated and MVR
knockoffs using paired semi synthetic SSI outcomes. The gate is applied in this
order:

1. G2.2A validity pass
2. Mean realized FDP difference no greater than 0.05 versus equicorr
3. Average precision difference no lower than negative 0.02
4. Exact matched selection size power difference no lower than negative 0.02
5. Across draw Jaccard difference no lower than negative 0.05
6. Raw power only ranks candidates after all previous gates pass

Primary outputs:

```text
generator_track_results/G22B_stabl_aware_screening/
  g22b_stage1_validity_runs.csv
  g22b_stage1_validity_summary.csv
  g22b_stage1_selection.json
  g22b_stage2_stabl_runs.csv
  g22b_stage2_draw_stability.csv
  g22b_stage2_replicate_metrics.csv
  g22b_stage2_matched_size_runs.csv
  g22b_stage2_matched_size_replicates.csv
  g22b_stage2_paired_contrasts.csv
  g22b_candidate_decisions.csv
  frozen_plsko_stabl_candidate.json
  or g22b_no_candidate.json
  g22b_status.json
```

Run the default experiment after G2.2A:

```bash
python generator_track/run_g22b_ssi_screening.py
```

A cheaper smoke run can be launched directly:

```bash
python generator_track/run_g22b_stabl_aware_screening.py \
  --ssi-data-path "/data/yhu94/Stabl/Sample Data/Biobank SSI" \
  --plsko-grid-json generator_track/configs/g22_plsko_screen_grid.json \
  --c2st-gate-json generator_track_results/G22A_validity_calibration/g22a_recommended_c2st_gate.json \
  --out-dir generator_track_results/G22B_smoke \
  --validity-draws 1 \
  --max-downstream-configs 2 \
  --diagnostic-top-k-if-none 2 \
  --screen-replicates 2 \
  --screen-draws 1 \
  --n-bootstraps 10 \
  --n-jobs 4 \
  --overwrite
```

## Resume

Both main scripts support `--resume`. Pair caches and run level CSV checkpoints
are reused. The scientific configuration must match the existing output
configuration.

## Decision rule

A G2.2B result with `g22b_no_candidate.json` is a valid scientific result. It
means no searched PLSKO configuration met the predeclared validity and STABL
reliability requirements. A configuration should advance to fresh holdout
confirmation only when `frozen_plsko_stabl_candidate.json` exists.
