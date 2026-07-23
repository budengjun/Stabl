# V11.0.1 single-rate smoke summary hotfix

This hotfix fixes the documented V11 smoke command when only one missingness rate is supplied.

Cause: AUC-over-rate and rate-slope summaries require at least two rates. The original summarizer created an empty DataFrame without schema and then attempted `groupby(["block_label", ...])`, causing `KeyError: 'block_label'` after all STABL cells had already completed.

Changes:

- write schema-valid empty AUC and slope CSVs for one-rate smoke runs;
- skip AUC and slope groupby operations when these tables are empty;
- add a regression test for the one-rate smoke configuration.

No data generation, completion, knockoff, STABL, ranking, gate, or scientific decision logic changed.

After installation, rerun only `summarize_v11_missingness_stress.py` on the existing smoke output. The 36 STABL fits do not need to be rerun.
