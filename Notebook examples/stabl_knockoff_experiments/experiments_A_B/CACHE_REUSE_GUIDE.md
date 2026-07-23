# Cache reuse guide

## What can be reused from the completed OOL audit

The old `OOL_all_generators_formal_v2/scores/*.npz` and
`score_checkpoints/*.npz` files contain fitted real and knockoff STABL score
paths. They can be reused for threshold and aggregation analyses on the same
clinical outcome without regenerating knockoffs or refitting STABL.

Run:

```bash
python -u stabl_experiments_A_B_cached/reanalyze_legacy_ool_scores.py \
  --score-dir stabl_knockoff_experiments/OOL_all_generators_formal_v2/scores \
  --out-csv results/legacy_ool_threshold_reanalysis.csv \
  --target-fdr 0.10 \
  --include-selected-features
```

This produces selection counts and selected feature sets for `stabl_min`,
`stabl_q`, and `knockoff_plus`. It does not produce empirical FDR or power,
because the old clinical outcome has no known support.

## Why old OOL scores cannot replace Experiment A or B

Experiment A uses newly simulated Gaussian X, missing masks, and known support.
Experiment B uses newly simulated outcomes and support on real X. STABL score
paths depend on y, so old clinical outcome score paths cannot be used as
semi-synthetic score paths.

The old checkpoints do not contain the generated knockoff matrix `X_tilde`.
They contain only score paths, feature names, and metadata. Therefore they
cannot serve as a knockoff-pair cache for a new outcome.

## New pair cache

Both Experiment A and B now persist validated knockoff pairs. A cache is
accepted only when all of the following match exactly:

1. The scaled X matrix and column order
2. Generator type and all generator parameters
3. Knockoff seed
4. Matrix shape and finite-value checks

By default the cache is stored under each output directory in `pair_cache/`.
A shared persistent directory can be supplied with:

```bash
--pair-cache-dir /data/yhu94/stabl_pair_cache
```

Do not use `--refresh-pair-cache` unless the cache must be regenerated.

## Experiment B shared pair bank

Experiment B defaults to:

```bash
--pair-bank-scope shared
```

For each p setting, generator configuration, and draw index, it generates one
knockoff pair and reuses it across all independently simulated outcomes. This
is valid because a Model X knockoff is generated from X without using y. It
changes Experiment B into a conditional benchmark over outcomes given the real
X and a fixed knockoff bank.

Use several draws to avoid conditioning on a single unusually favorable or
unfavorable pair:

```bash
--n-knockoff-draws 5 --pair-bank-scope shared
```

This reduces PLSKO generation from `replicates × draws` calls to only `draws`
calls per p setting and PLSKO configuration. STABL still needs to be fitted for
each synthetic outcome because its score paths depend on y.

For a secondary analysis that also averages over new knockoff randomness for
every outcome, use:

```bash
--pair-bank-scope per-replicate
```

## Score cache for future reanalysis

Score saving is now enabled by default. New runs store fitted score paths under
`scores/`. These can later be re-thresholded without rerunning the expensive
STABL fit. Use `--no-save-scores` only when disk space is a serious constraint.
