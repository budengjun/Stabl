# Start Here: V8

Run in this order:

```text
1. test_v8_updates.py
2. oracle calibration smoke
3. SSI oracle calibration grid: p={100,250}, strength={4,6}
4. inspect oracle_calibration_recommendation.csv
5. completion-ablation smoke
6. 10-replicate replacement pilot at the calibrated setting
```

The primary research question remains:

> Can BayesianRidge-based completion replace median in original STABL?

V8 additionally separates the conditional-model effect from posterior-sampling noise by comparing:

```text
Median
BayesianRidge conditional mean
BayesianRidge posterior draw
```

See `RUN_V8_ORACLE_CALIBRATION_AND_MEAN_ABLATION.md` for commands.
