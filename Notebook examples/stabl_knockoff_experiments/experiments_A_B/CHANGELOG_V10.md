# V10 changelog

V10 adds a prospective external-omic generalization workflow for the confirmed
V9/V9.1 BayesianRidge conditional-mean candidate.

## Scientific scope

- Primary question: does BR mean's reliability advantage generalize beyond SSI
  CyTOF while original STABL `stabl_min` remains unchanged?
- Default blocks: SSI Proteomics and DREAM Phylotype.
- Evaluation completions: Oracle complete, Median, BR conditional mean.
- Primary missingness: MCAR and MAR at 20%.
- Primary downstream: realized FDP, power, Oracle Jaccard.
- Threshold-independent ranking: Precision@5/10/15/20 and average precision.
- The default 10-replicate run is a directional pilot, not confirmatory.

## Engineering changes

- Added DREAM support to `research_common.load_real_dataset`.
- Added independent per-block oracle calibration using a seed distinct from the
  evaluation seed.
- Added multi-block orchestration, preregistration sidecar, integrity audit,
  threshold-independent ranking, cross-block consistency, and progression gate.
- Pair caches remain resumability artifacts and can be deleted automatically
  after successful completion with `--delete-pair-cache-after-success`.
