# Review of the supplied initial scripts

## `knockoff_exchangeability.py`

The initial script correctly separates knockoff diagnostics from STABL and correctly recognizes that repeated OOL measurements require group aware splitting. It also uses one vector per sample in the swap classifier dataset, which avoids a known high dimensional pairing artefact.

The main changes in version 2 are:

1. The default classifier is always regularized logistic regression. The initial `auto` setting selected LightGBM for OOL and caused a severe runtime bottleneck.

2. The classifier output reports oriented AUC and absolute deviation from chance.

3. Permutation testing has `fast`, `refit`, and `none` modes. The refit mode is preferable for a final report because it repeats the complete learning procedure under each permuted label vector.

4. Second moment errors are normalized, making comparisons across omics with different scales easier.

5. The documentation now states explicitly that a non significant classifier result does not prove exchangeability.

## `plsko.py`

The supplied Python port is not suitable for scientific comparison without substantial validation.

The main risk is in sample PLS overfitting. In the small sample, high dimensional setting, a PLS model can explain noise in the same observations used to compute residuals. The resulting permuted residuals may be too small, producing knockoffs that are close copies of the original variables. Correlation based neighbour screening on the full dataset adds another source of selection leakage.

The experiment suite therefore does not import this file. It uses the authors' official R package through `official_plsko_bridge.py` and `official_plsko_bridge.R`.

## `derandomized_stabl.py`

The initial implementation captures the central e value aggregation structure, but two distinctions must be enforced operationally.

1. Repeated knockoff copies must be generated on the same observed training data. Different outer folds are not repeated knockoff copies and cannot be pooled as the draws in the derandomized theorem.

2. The theorem still requires valid knockoffs and an antisymmetric statistic. Repeated aggregation cannot repair systematic nonexchangeability.

Version 2 adds metadata aware NPZ storage, loading, method comparison, input validation, and explicit warnings against fold pooling.

## `run_ool_knockoff_audit.py`

The initial runner contains several useful ideas, including direct generation outside `_make_artificial_features`, subject aware diagnostics, and separate generator and timing cells.

The main changes in version 2 are:

1. OOL Metabolomics is capped at 1500 features by default to match the existing Direction 2 configuration.

2. The exact MAR mask from `imputation_effect_on_stabl.py` is included.

3. The unvalidated Python PLSKO and observed only PLSKO cells are removed.

4. The official PLSKO implementation is optional.

5. The classifier diagnostic is made fast enough for routine use.

6. Multiple knockoff draws are generated inside each fixed fold and saved for later aggregation.

7. Progress, checkpointing, configuration capture, and resumable CSV output are added.

8. The remasking path is retained only as an explicitly named negative control.
