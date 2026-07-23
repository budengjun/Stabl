# V9 First

V9 is the confirmatory experiment for the only remaining primary replacement candidate:

```text
BayesianRidge conditional mean vs Median
```

The primary experiment keeps the original STABL `stabl_min` rule and excludes posterior sampling, score pooling, voting, `stabl_q`, and LCD knockoff+.

Run the tests first:

```bash
python -u experiments_A_B/test_v9_updates.py
```

Then run the smoke command in `RUN_V9_BR_MEAN_CONFIRMATION.md`. After the smoke passes, run the 50-replicate main confirmation.
