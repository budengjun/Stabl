# V6.1 changelog: validity-aware artificial-feature aggregation

## Why V6.1 exists

The V6 pilot showed a systematic failure of naive path averaging. Real feature identities remain fixed across component draws, whereas regenerated artificial-feature identities are unstable. Averaging artificial score paths by feature identity therefore diluted artificial evidence and drove the reported FDP+ far below realized FDP, including in the oracle-complete control.

## New primary rule

`mean_real_component_artificial_count`

For pool size M, V6.1 still pools the real STABL path:

\[
\bar\Pi_j(\lambda)=M^{-1}\sum_m \Pi_j^{(m)}(\lambda).
\]

At threshold t, however, the artificial burden is

\[
\bar A(t)=M^{-1}\sum_m \#\{j:\max_\lambda \widetilde\Pi_j^{(m)}(\lambda)\ge t\}.
\]

The V6.1 curve is

\[
\widehat{FDP}_{multi}^{+}(t)=
\frac{1+\bar A(t)}
{\#\{j:\max_\lambda \bar\Pi_j(\lambda)\ge t\}\vee 1}.
\]

This is a simulation-tested empirical extension, not a new theoretical guarantee.

## Rules included

- `mean_path_naive`: original V6 rule, retained as a negative control.
- `mean_real_component_artificial_count`: V6.1 primary validity-aware rule.
- `component_selection_vote`: 50% and 80% vote sensitivity rules by default. No FDP+ estimate is claimed for vote rules.

## New files

- `validity_aware_pooling.py`
- `run_experiment_a_validity_aware_pooling.py`
- `reanalyze_v6_validity_aware_pooling.py`
- `summarize_validity_aware_pooling.py`
- `test_v61_updates.py`
- `RUN_V61_VALIDITY_AWARE_POOLING.md`

## Reuse

Existing V6 component score banks can be reanalyzed directly. No STABL fitting or knockoff regeneration is required.
