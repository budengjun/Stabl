# V5 change log: posterior versus median under original STABL

## Scientific scope correction

V5 restores the primary research question:

> Does posterior-sampling completion preserve complete-data structure and original STABL biomarker-selection behavior better than deterministic median imputation?

The published STABL FDP+ argmin rule (`stabl_min`) is the primary selection rule. `stabl_q`, STABL-frequency knockoff+, and LCD knockoff+ remain available only as optional diagnostics and are no longer decision gates for the posterior-versus-median experiment.

## Code changes

1. Experiment version changed to `5` and the experiment label changed to `A_posterior_vs_median_original_stabl`.
2. Default selection rule changed to `stabl_min` only.
3. Default generators include sample-Sigma equicorrelated and known-Sigma equicorrelated knockoffs.
4. Misspecified true-Sigma combinations are skipped by default. True Sigma is run only for `oracle_complete` and `exact_gaussian_posterior` under MCAR/MAR.
5. Knockoff and STABL random seeds are now paired across completion methods within each replicate, mechanism, generator, and draw. Completion is the intended changing factor.
6. Added `primary_analysis=1` for `stabl_min` rows.
7. Added `summarize_posterior_vs_median.py`, which writes paired completion contrasts, oracle-selection agreement, primary STABL summaries, and figures.
8. Added `test_v5_updates.py`.

## Interpretation of existing evidence

The existing five-replicate calibration provides strong preliminary evidence that exact posterior completion improves distribution recovery and sample-Sigma knockoff covariance matching relative to median. It does not yet establish superior downstream STABL FDP, power, or biomarker-set recovery because the selection results were mixed, the replicate count was small, and V3 did not pair downstream random seeds across completion methods.
