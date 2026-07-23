# G2.1 Saved Score Diagnosis

This patch is Python only. It does not contain or require shell launch scripts.
It reads the completed G2 score files and pair cache. It does not rerun STABL,
PLSKO, or Gaussian knockoff generation.

## Primary command

Run from the existing `experiments_A_B` directory:

```bash
conda activate stabl
python generator_track/run_g21_ssi_diagnosis.py
```

The convenience launcher uses:

* source directory: `generator_track_results/G2_ssi_pilot`
* output directory: `generator_track_results/G2_ssi_pilot/G21_saved_score_diagnosis`
* 20,000 paired bootstrap samples
* all 120 stored score files
* all 120 stored pair-cache files for expanded C2ST

## Generic command

```bash
python generator_track/run_g21_saved_score_diagnosis.py \
  --g2-dir generator_track_results/G2_ssi_pilot \
  --out-dir generator_track_results/G2_ssi_pilot/G21_saved_score_diagnosis \
  --comparisons "plsko_tuned_ssi:equicorr,plsko_tuned_ssi:mvr" \
  --expanded-c2st \
  --save-full-threshold-path \
  --make-figures
```

## Main outputs

* `g21_reconstruction_validation.csv`
* `g21_threshold_path_runs.csv`
* `g21_threshold_path_summary.csv`
* `g21_matched_size_runs.csv`
* `g21_matched_size_replicates.csv`
* `g21_paired_contrast_summary.csv`
* `g21_score_distribution_runs.csv`
* `g21_score_distribution_summary.csv`
* `g21_c2st_all_draws.csv`
* `g21_c2st_summary.csv`
* `g21_status.json`
* `figures/`

## Diagnostic decomposition

For every paired run, the code reports two exact decompositions.

### Matched selected size

`raw difference = matched size ranking component + selection size component`

The candidate top k is chosen only from the candidate real score ranking, where
k is the baseline selected count. Support labels are used only after selection
to evaluate FDP, power, and Jaccard.

### Common threshold

`raw difference = common threshold score component + threshold choice component`

This evaluates both generators at the baseline threshold and then quantifies the
additional change caused by allowing the candidate to use its own `stabl_min`
argmin threshold.

## Test

```bash
python -m pytest generator_track/test_g21_saved_score_diagnosis.py -q
```

A pure Python self-test is also available:

```bash
python generator_track/run_g21_self_test.py
```
