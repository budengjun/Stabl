# STABL knockoff timing experiment suite

This directory contains a conservative experimental implementation for the current STABL timing project. It keeps generator diagnostics, timing comparisons, STABL selection, and repeated knockoff aggregation as separate layers.

## 1. Main design decisions

1. The original Python PLSKO port is not used for scientific comparisons. The suite calls the authors' official R package through `official_plsko_bridge.py` and `official_plsko_bridge.R`.

2. Different outer cross validation folds are never treated as repeated knockoff draws. Repeated draws must be generated on the same training matrix.

3. The unvalidated observed only PLS implementation is excluded. The supported missing data timings are:

   `post_median`

   Median completion, standardization, then knockoff generation.

   `posterior_then_knockoff`

   Stochastic IterativeImputer completion, knockoff generation, then joint standardization.

   `posterior_remask_negative_control`

   Posterior completion, knockoff generation, remasking of the knockoff block, deterministic second completion, then joint standardization. This is deliberately retained as a negative control and must not be described as a valid method.

4. OOL missingness masks use the same MCAR, MAR, and MNAR implementations as `imputation_effect_on_stabl.py`.

5. OOL defaults to at most 1500 features per omic layer, selected by outcome independent variance ranking while preserving original column order. This matches the Direction 2 configuration and avoids an incomparable 3529 feature Metabolomics run.

6. The classifier diagnostic defaults to regularized logistic regression with three repeats. This avoids the LightGBM bottleneck in the initial audit script.

## 2. Files

### `knockoff_exchangeability_v2.py`

Runs the marginal X versus X tilde classifier diagnostic from the nonexchangeability paper, a group aware paired swap classifier diagnostic, and normalized second moment diagnostics. A non significant classifier result does not prove validity. Use it together with the half synthetic empirical FDP benchmark.

### `official_plsko_bridge.py` and `official_plsko_bridge.R`

Call the authors' official PLSKO R package from Python. The bridge requires a complete finite matrix, so it can be used with `post_median` and `posterior_then_knockoff`.

### `run_ool_knockoff_audit_v2.py`

Runs generator and timing audits for OOL CyTOF, Proteomics, and Metabolomics. It supports repeated knockoff copies on each fixed fold and can optionally fit STABL.

### `run_ssi_timing_audit_v2.py`

Runs the corresponding timing audit on the SSI CyTOF dataset with its native missingness and the existing five fold by three repeat outer split structure.

### `derandomized_stabl_v2.py`

Computes paired STABL statistics, knockoff e values, averaged e values, and e BH selection. It also implements classic STABL FDP plus selection for comparison.

### `analyze_score_draws.py`

Reads the score NPZ files produced by the OOL and SSI runners. It compares single draw FDP plus, score averaged FDP plus, and derandomized e value aggregation for each fixed fold separately.

### `run_half_synthetic_benchmark.py`

Generates block correlated Gaussian covariates with known sparse support, injects missingness, runs the timing and generator cells, and reports empirical FDP and power. This is the primary experiment for deciding whether a method actually controls false discoveries.

### `audit_stabl_chunking_bug.py`

Reproduces the source identity failure in the current `p > 3000` artificial feature branch without running an expensive knockoff sampler.

### `stabl_pairing_guard.patch`

A fail fast patch for the legacy chunking branch. It prevents silent use of an unpaired artificial block when an omic layer has more than 3000 features. Apply it from the STABL repository root with `patch -p1 < stabl_pairing_guard.patch`.

## 3. Environment

Activate the existing STABL environment first.

```bash
conda activate stabl
python -c "import stabl, knockpy, sklearn, pandas, numpy"
```

For official PLSKO, install the R package once.

```r
install.packages("devtools")
devtools::install_github("guannan-yang/PLSKO/PLSKO", quiet = TRUE, upgrade = "never")
```

The official repository also lists package dependencies such as `knockoff`, `progress`, `parallel`, `doParallel`, `foreach`, and `mixOmics`.

Check installation from the shell.

```bash
Rscript -e "stopifnot(requireNamespace('PLSKO', quietly=TRUE))"
```

## 4. Recommended execution order

### Experiment 0: reproduce the chunking defect

```bash
python audit_stabl_chunking_bug.py \
  --p 3529 \
  --chunk-size 3000 \
  --seed 42 \
  --out-dir chunking_bug_audit
```

The important outputs are `n_sources_absent`, `n_sources_repeated`, and `correct_pair_fraction_at_final_position` in `summary.json`.

### Experiment A: fast OOL generator health audit

Start with one fold and one omic.

```bash
python run_ool_knockoff_audit_v2.py \
  --data-path "../Sample Data" \
  --out-dir OOL_smoke \
  --omics CyTOF \
  --generators gaussian_equicorrelated,gaussian_mvr \
  --timings post_median \
  --mechanism MAR \
  --missing-rate 0.20 \
  --n-splits 1 \
  --ctst-permutation-mode fast
```

Then run all OOL omics without STABL.

```bash
python run_ool_knockoff_audit_v2.py \
  --data-path "../Sample Data" \
  --out-dir OOL_generator_audit \
  --generators gaussian_equicorrelated,gaussian_mvr \
  --timings post_median,posterior_then_knockoff,posterior_remask_negative_control \
  --mechanism MAR \
  --missing-rate 0.20 \
  --n-splits 10
```

Read `summary.csv`. The most useful generator quantities are `pair_corr_mean`, `s_relative_mean`, `cov_kk_fro_relative`, and `cov_xk_offdiag_rmse`. The classifier AUC is a falsification diagnostic, not a certificate.

### Experiment B: OOL timing plus STABL scores

Use at least five knockoff copies on the same fold if the goal includes aggregation.

```bash
python run_ool_knockoff_audit_v2.py \
  --data-path "../Sample Data" \
  --out-dir OOL_timing_stabl \
  --generators gaussian_equicorrelated,gaussian_mvr \
  --mechanism MAR \
  --missing-rate 0.20 \
  --n-splits 10 \
  --n-knockoff-draws 5 \
  --fit-stabl \
  --n-bootstraps 50 \
  --n-jobs 8
```

For the official PLSKO comparison, run it separately because its runtime profile differs.

```bash
python run_ool_knockoff_audit_v2.py \
  --data-path "../Sample Data" \
  --out-dir OOL_plsko \
  --generators official_plsko \
  --timings post_median,posterior_then_knockoff \
  --mechanism MAR \
  --missing-rate 0.20 \
  --n-splits 10 \
  --n-knockoff-draws 5 \
  --fit-stabl \
  --n-bootstraps 50
```

### Experiment C: SSI timing audit

```bash
python run_ssi_timing_audit_v2.py \
  --data-path "../Sample Data" \
  --out-dir SSI_timing_audit \
  --generators gaussian_equicorrelated,gaussian_mvr \
  --timings post_median,posterior_then_knockoff,posterior_remask_negative_control \
  --models lasso,alasso,elasticnet \
  --n-knockoff-draws 5 \
  --n-bootstraps 50 \
  --n-jobs 8
```

### Experiment D: compare repeated draw aggregation rules

```bash
python analyze_score_draws.py \
  OOL_timing_stabl/scores \
  --out-dir OOL_aggregation_analysis \
  --alpha-ebh 0.10 \
  --alpha-kn 0.05
```

Run the same command on `SSI_timing_audit/scores`.

The e value method can be conservative when the number of discoveries is small. Treat it as a formally motivated comparator, not as an automatic replacement for the classic STABL rule.

### Experiment E: half synthetic empirical FDP and power

Start with a small smoke test.

```bash
python run_half_synthetic_benchmark.py \
  --out-dir half_synthetic_smoke \
  --n-replicates 2 \
  --n 150 \
  --p 200 \
  --n-signal 10 \
  --n-knockoff-draws 3 \
  --n-bootstraps 20 \
  --mechanism MAR \
  --missing-rate 0.20
```

Then run the main Gaussian benchmark.

```bash
python run_half_synthetic_benchmark.py \
  --out-dir half_synthetic_main \
  --n-replicates 50 \
  --n 150 \
  --p 500 \
  --n-signal 20 \
  --block-size 25 \
  --rho 0.60 \
  --task classification \
  --mechanism MAR \
  --missing-rate 0.20 \
  --generators gaussian_equicorrelated,gaussian_mvr \
  --n-knockoff-draws 10 \
  --n-bootstraps 50 \
  --alpha-ebh 0.10 \
  --alpha-kn 0.05
```

Repeat this benchmark for MCAR and MNAR. The main decision variables are empirical `fdp`, `power`, and `n_selected` in `summary.csv`.

## 5. Interpretation rules

1. High `pair_corr_mean` is primarily a power warning. It does not by itself prove invalidity.

2. A significant swap classifier result is evidence against exchangeability. A non significant result is inconclusive in small sample omics data.

3. The decisive validity evidence is empirical FDP on known null features in the half synthetic benchmark.

4. The `posterior_then_knockoff` timing only approximates the posterior conditional sampling assumption. `IterativeImputer` with Bayesian Ridge is not an oracle conditional sampler.

5. The `posterior_remask_negative_control` cell is expected to fail or become biased. It exists to localize the effect of remasking and deterministic second imputation.

6. Never average score files from different outer folds and submit them to e value aggregation. Analyse every fold independently, then summarize selected sets across folds descriptively.

7. When comparing generators, use the same missingness mask, outer split, bootstrap count, and random seed schedule.


## Integrated Generator Track

The PLSKO replacement study is now integrated into the original experiment
tree under `experiments_A_B/generator_track`. Start with
`experiments_A_B/RUN_GENERATOR_TRACK_G0_G3.md`. The implementation reuses the
existing Experiment B runner and shared research utilities rather than creating
a separate codebase.

## G2.3 Gaussian family follow up

After the official PLSKO exchangeability failure on SSI, the next generator experiment compares default covariance equicorrelated, MVR, and SDP knockoffs against low rank plus diagonal factor covariance variants. The experiment uses validity first screening followed by paired semi synthetic STABL evaluation.

Read:

`experiments_A_B/generator_track/RUN_G23_GAUSSIAN_FAMILY.md`

Run the fast code test first:

```bash
cd experiments_A_B
python -u generator_track/run_g23_self_test.py
```
