# Read this first: V11 missingness stress map

V11 asks where the BR conditional-mean advantage remains reliable as missingness becomes more severe and structured.

The experiment does not alter the original STABL selection rule. It changes only the completion method and controlled missingness condition.

## Stages

1. Smoke: engineering validation only.
2. Pilot: 10 replicates, 3 paired STABL runs, 50 bootstraps. Directional boundary screen.
3. Stress: 20 replicates, 5 paired STABL runs, 100 bootstraps. Full tested-cell map.

The full stress grid is expensive. Run the pilot first and inspect its cell map before starting stress mode.

## Rectangular block missingness

Features are shuffled with a fixed seed and split into five feature blocks. For each feature block, an independently seeded subset of samples loses the entire block. The same seed is reused at each missingness rate, so 10% masks are subsets of 20%, 30%, and 40% masks.

## Primary outputs

- `v11_integrity_audit.csv`
- `v11_replicate_metrics.csv`
- `v11_paired_contrasts.csv`
- `v11_rate_trajectory_summary.csv`
- `v11_cell_progression_gate.csv`
- `v11_support_boundary_summary.csv`
- `v11_auc_over_missing_rates.csv`
- `v11_rate_slope_contrasts.csv`
- `v11_mapping_decision.csv`
- `v11_missingness_stress_report.md`
