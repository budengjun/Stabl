# V11 changelog

V11 adds a prospective missingness stress map for the BayesianRidge conditional-mean replacement.

## Scientific design

- Datasets: SSI CyTOF and DREAM Phylotype.
- Fixed real-X semisynthetic design: `p=100`, 15 signals, signal strength 4.
- Completion methods: Oracle complete, Median, BayesianRidge conditional mean.
- Missingness rates: 10%, 20%, 30%, 40%.
- Mechanisms: MCAR, MAR, and rectangular block missingness.
- Original STABL `stabl_min`, Gaussian equicorrelated knockoffs, and existing score-ranking analysis remain unchanged.
- Outcomes, supports, and mask seeds are paired across rates. Rate masks are nested within replicate.

## New code

- `run_v11_missingness_stress.py`
- `summarize_v11_missingness_stress.py`
- `v11_common.py`
- `v11_stress_blocks.example.json`
- `test_v11_updates.py`

## Core compatibility changes

- `research_common.make_missing_mask` accepts `BLOCK` and `block_feature_blocks`.
- `run_experiment_b_completion_replacement.py` accepts `--block-feature-blocks` and records mask-shape diagnostics.
- V9.1 score reanalysis accepts `BLOCK` filenames.
- V10 mechanism validation accepts `BLOCK` without changing prior defaults.

## Interpretation safeguards

- Smoke runs never produce scientific decisions.
- Pilot runs generate a directional boundary map only.
- Stress mode maps tested cells and does not imply universal superiority.

## V11.0.1 hotfix

- Fixed one-rate smoke summarization. AUC-over-rate and rate-slope outputs now become schema-valid empty CSV files when fewer than two missingness rates are present.
- Added an explicit regression test for the documented one-rate smoke command.
- No data generation, completion, knockoff, STABL, ranking, or scientific decision logic changed.
