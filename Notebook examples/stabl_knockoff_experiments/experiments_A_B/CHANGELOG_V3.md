# Changelog v3

## Added

* `gaussian_equicorrelated_true_sigma` for Experiment A
* Exact Gaussian conditional knockoff sampling using known population mean and covariance
* Population Gaussian parameter transformation into the fitted StandardScaler space
* Cache fingerprints for known mean and covariance
* `uses_true_sigma`, `true_sigma_exactly_applicable`, and `oracle_scaled_cov_condition_number` diagnostics
* `test_v3_updates.py`

## Corrected

* STABL FDP plus counts now use `score >= threshold`
* STABL selected sets now use `score >= threshold`
* If the minimum FDP plus exceeds 1, `stabl_min` explicitly returns an empty set
* `posterior_theory_aligned` was renamed to `missingness_assumption_compatible`

## Cache compatibility

Existing version 1 pair caches for sample covariance Gaussian generators and PLSKO remain reusable. The true Sigma generator uses version 2 cache metadata because the known Gaussian parameters must be fingerprinted.
