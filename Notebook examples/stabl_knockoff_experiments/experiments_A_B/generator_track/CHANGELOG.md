# Generator Track integration changelog

## Integrated G0 to G3 package

1. Added official PLSKO tuning bridge and frozen configuration export.
2. Added G0 generator and STABL integration smoke.
3. Extended Experiment B to load DREAM directly.
4. Corrected paired seeding so downstream STABL bootstrap seeds are shared across generators.
5. Added per replicate ranking metrics, support Jaccard, calibration gap, and selected feature indices.
6. Added optional marginal and swap C2ST diagnostics.
7. Added Generator Track replicate aggregation, paired contrasts, and across draw Jaccard summaries.
8. Added SSI and DREAM shell launchers while retaining all earlier V3 to V11.1 code and results.

## G2.3 Gaussian family screening

Added SDP Gaussian knockoffs and low rank plus diagonal factor covariance variants for equicorrelated, MVR, and SDP S matrices. Added a validity first SSI screening runner, paired semi synthetic STABL evaluation, environment check, self test, configuration grid, and calibrated gate file. PLSKO remains available only in the earlier generator track and is not included in the G2.3 candidate grid.
