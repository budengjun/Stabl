# STABL Generator Track, integrated edition

This directory is embedded inside the existing `stabl_knockoff_experiments/experiments_A_B` tree. It does not duplicate the Experiment A and B infrastructure. Gaussian equicorrelated knockoffs, Gaussian MVR, official PLSKO, STABL score fitting, caching, dataset loading, and metric computation all use the shared `research_common.py` implementation.

## Scientific scope

The first Generator Track experiments use complete real omic covariates and semi synthetic outcomes with known support. Completion is held fixed. The primary question is whether a tuned official PLSKO generator improves STABL reliability relative to Gaussian equicorrelated Model X knockoffs and Gaussian MVR.

Primary downstream rule: original `stabl_min`.

Primary downstream metrics: realized FDP, power, support Jaccard, average precision, ranking AUROC, mean null score, calibration gap, and across draw selected set Jaccard.

Primary generator validity diagnostics: marginal C2ST and paired swap C2ST. Pair correlation and covariance diagnostics remain secondary mechanism measures.

## Important integration changes

`run_experiment_b_semisynthetic.py` now:

1. supports the DREAM loader directly
2. uses the same numeric knockoff seed across generators within each paired replicate and draw
3. uses a separate STABL bootstrap seed shared across generators
4. records support Jaccard and threshold independent ranking metrics
5. records selected feature indices for draw stability analysis
6. optionally computes marginal and swap C2ST with `--c2st-every`
7. warns when official PLSKO is used with a shared pair bank

For the primary Generator Track, always use:

```bash
--pair-bank-scope per-replicate
--selection-rules stabl_min
```

A shared pair bank is still available for legacy and low cost diagnostics, but it underrepresents PLSKO draw variability across independent synthetic outcomes.

## Run order

Run from the `experiments_A_B` directory.

```bash
Rscript generator_track/install_plsko.R
python check_environment_ab.py
bash generator_track/scripts/run_g0_ssi.sh
bash generator_track/scripts/run_g1_ssi_smoke.sh
bash generator_track/scripts/run_g1_ssi_pilot.sh
bash generator_track/scripts/run_g2_ssi_pilot.sh
```

After G2 supports progression:

```bash
bash generator_track/scripts/run_g3_ssi_confirmation.sh
```

DREAM pilot commands:

```bash
bash generator_track/scripts/run_g0_dream.sh
bash generator_track/scripts/run_g1_dream_pilot.sh
bash generator_track/scripts/run_g2_dream_pilot.sh
```

## Stage outputs

### G0

`g0_results.csv` contains shape checks, same seed reproducibility, different seed variability, near constant column checks, pair geometry, C2ST, and a small STABL integration run.

### G1

The official R `PLSKO::plsko_tuning()` result is saved under `official_tuning/`. The selected parameters are frozen as `frozen_plsko_generator.json`. G1 tuning seeds must not be reused for G2 or G3 confirmation.

### G2 and G3

The existing Experiment B runner writes:

```text
selection_results.csv
draw_diagnostics.csv
summary.csv
scores/
pair_cache/
truth/
```

The Generator Track summarizer additionally writes:

```text
generator_draw_stability.csv
generator_replicate_aggregates.csv
generator_summary_replicate_level.csv
generator_paired_contrasts.csv
generator_summary_status.json
```

## Resume rather than overwrite

The supplied shell scripts use `--overwrite` for a clean first launch. For a long server run, after the initial launch replace `--overwrite` with `--resume`. The configuration fingerprint prevents accidental resume with changed scientific settings.

## PLSKO bridge

The existing `official_plsko_bridge.py` and `official_plsko_bridge.R` remain the single generation interface. G1 adds only `official_plsko_tuning_bridge.R`. The frozen G1 JSON is accepted directly by `run_experiment_b_semisynthetic.py` after it is merged with Gaussian baselines by `build_g2_generator_config.py`.
