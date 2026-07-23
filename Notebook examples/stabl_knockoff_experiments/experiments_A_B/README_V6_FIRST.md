# Start here: V6 posterior pooling

Run in this order:

1. `python -u experiments_A_B/test_v6_updates.py`
2. V6 integration smoke from `RUN_V6_POSTERIOR_POOLING.md`
3. 10-replicate `M=1,5` pilot with V5 score reuse
4. `python -u experiments_A_B/summarize_posterior_pooling.py --out-dir ...`
5. Run `M=10` only if the pilot shows a posterior-specific gain beyond median

The primary method is still original STABL `stabl_min`.  V6 changes only how
multiple completed-data STABL score paths are aggregated.
