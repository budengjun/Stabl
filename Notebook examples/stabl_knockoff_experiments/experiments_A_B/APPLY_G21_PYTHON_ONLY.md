# Apply the G2.1 Python patch

Place this ZIP in the existing `experiments_A_B` directory and extract it there.
The archive contains only Python and Markdown files. It does not contain shell
scripts and does not replace any G0, G1, or G2 result files.

Run the self-test:

```bash
conda activate stabl
python generator_track/run_g21_self_test.py
```

Run the complete SSI diagnosis:

```bash
python generator_track/run_g21_ssi_diagnosis.py
```

The launcher reads:

`generator_track_results/G2_ssi_pilot`

and writes:

`generator_track_results/G2_ssi_pilot/G21_saved_score_diagnosis`
