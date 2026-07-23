# G2.3 patch manifest

## Scientific purpose

This patch adds a Gaussian family follow up after the official PLSKO negative result on SSI. It compares default covariance Gaussian knockoffs and low rank plus diagonal factor covariance knockoffs under equicorrelated, MVR, and SDP S matrix constructions.

## Existing files modified

`experiments_A_B/research_common.py`

Added `gaussian_sdp`, `gaussian_factor_equicorrelated`, `gaussian_factor_mvr`, and `gaussian_factor_sdp`. Added explicit factor covariance configuration and a deterministic low rank plus diagonal covariance estimator.

`README.md`

Added the G2.3 entry point.

## Files added

`experiments_A_B/generator_track/run_g23_gaussian_family_screening.py`

`experiments_A_B/generator_track/check_g23_environment.py`

`experiments_A_B/generator_track/run_g23_self_test.py`

`experiments_A_B/generator_track/test_g23_gaussian_family.py`

`experiments_A_B/generator_track/RUN_G23_GAUSSIAN_FAMILY.md`

`experiments_A_B/generator_track/configs/g23_gaussian_family_grid.json`

`experiments_A_B/generator_track/configs/g23_c2st_gate.json`

## Compatibility

All existing G0 through G2.2 code is retained. Existing generator configurations remain valid because the new factor fields have defaults and are included in cache fingerprints through the existing dataclass serialization.
