# V4 change log: paper-aligned STABL and LCD comparator

## Scientific changes

1. **Paper-aligned STABL regularization path**
   - Classification now uses an explicit 30-value linear grid `C = 0.01 ... 1.0` by default.
   - Regression uses an explicit 30-value logarithmic alpha grid by default.
   - The grid is written into `config.json` through the command-line parameters, so later package-default changes cannot silently alter the experiment.

2. **100 bootstrap default**
   - Experiment A and Experiment B now default to `--n-bootstraps 100`.
   - Smaller values remain available for smoke tests only.

3. **Standard LCD knockoff-plus comparator**
   - New rule: `lcd_knockoff_plus`.
   - Fits a symmetric L1 model to `[X, X_tilde]`.
   - Uses `W_j = |beta_j| - |beta_tilde_j|`.
   - Applies the standard knockoff-plus threshold at the requested `q`.
   - Classification uses cross-validated logistic L1; regression uses cross-validated Lasso.

4. **Explicit name for the previous frequency comparator**
   - New preferred name: `stabl_knockoff_plus`.
   - Legacy name `knockoff_plus` remains an alias so old score-reanalysis commands continue to work.

5. **Stable subset-independent seeds in Experiment A**
   - Mechanism and completion seeds now use fixed canonical indices.
   - Running only `oracle_complete,exact_gaussian_posterior` preserves the same branch seeds as running all four completion branches.
   - Compatible V3 pair caches can therefore still be reused when matrix, generator, and seed match exactly.

## Engineering changes

- Score NPZ files now optionally include `lcd_W` and `lcd_best_parameter`.
- Draw diagnostics include LCD fit time, selected tuning parameter, and counts of positive/negative W statistics.
- Experiment B rejects `lcd_knockoff_plus` when repeated rows are retained, because the current LCD CV comparator is iid-aligned. Use OOL `--subject-mode first` for the primary benchmark.
- Added `test_v4_updates.py`.
- Added `summarize_v4_results.py`.

## New default rules

```text
stabl_min
stabl_q
stabl_knockoff_plus
lcd_knockoff_plus
```

## Important interpretation

`stabl_knockoff_plus` and `lcd_knockoff_plus` are not the same statistic:

- `stabl_knockoff_plus` uses the difference between paired maximum STABL selection frequencies.
- `lcd_knockoff_plus` uses paired L1 coefficient differences from a single augmented model.
