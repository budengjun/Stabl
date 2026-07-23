# V10.1 changelog

## V10.1A DREAM convergence guard

- Adds fresh-seed DREAM max_iter comparison at 10, 25, and 50.
- Supports MCAR and MAR with paired outcomes, masks, completion seeds, knockoff seeds, and STABL seeds.
- Records IterativeImputer convergence, variance, covariance, correlation, condition-number, and knockoff-geometry diagnostics.
- Labels downstream FDP, power, and Oracle Jaccard as exploratory.

## V10.1B DREAM fresh confirmation

- Locks DREAM Phylotype, p=100, 15 signals, strength=4, MCAR/MAR 20%.
- Uses 50 fresh simulation replicates, five paired STABL runs, 100 bootstraps, and original stabl_min.
- Fixes practical power noninferiority and FDP non-worsening margins at 0.03.
- Uses Oracle-selection Jaccard and average precision as primary superiority endpoints.
- Prevents nonstandard run sizes from being labelled confirmatory.

## V10.1C SSI threshold-path diagnosis

- Reconstructs the full FDP+ and realized FDP path from saved V10 score files.
- Quantifies stabl_min threshold and selected-count variability.
- Evaluates BR mean near Median's stabl_min selection size using an outcome-independent matching rule.
- Does not rerun completion, knockoffs, or STABL.
