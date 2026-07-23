# V10 first

V10 is the external omic generalization pilot following V9.1.

The default design deliberately uses two different types of external evidence:

1. SSI Proteomics: same cohort, different omic modality.
2. DREAM Phylotype: different cohort and different omic modality.

The workflow has two independent stages.

## Stage 0: Oracle calibration

Each block is calibrated with Oracle complete data only. Candidate signal
strengths are evaluated using a calibration seed. The chosen strength is locked
before the external evaluation pilot.

Calibration is not evidence for completion superiority.

## Stage 1: External pilot

A fresh evaluation seed is used. Each block runs:

- Oracle complete
- Median
- BayesianRidge conditional mean
- MCAR and MAR at 20%
- original `stabl_min`
- 10 simulation replicates
- 5 paired ordinary STABL runs per replicate
- 100 STABL bootstraps

The default pilot progression gate requires favorable point estimates for FDP,
Oracle Jaccard, all configured Precision@k values, and no absolute power loss
larger than 0.05. Passing the gate only justifies a fresh 50-replicate run. It is
not itself a confirmatory result.

Read `RUN_V10_EXTERNAL_GENERALIZATION.md` for exact commands.
