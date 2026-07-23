# V11.1 changelog

Added `reanalyze_v111_threshold_diagnosis.py`.

The new analysis:

1. reads all saved V11 score arrays;
2. reconstructs original `stabl_min` and the complete threshold grid;
3. compares estimated FDP+ with realized FDP;
4. evaluates BR mean at the Median selected-set size without using truth labels to choose the threshold;
5. combines matched-size results with threshold-independent ranking metrics;
6. produces descriptive cell-level diagnoses and integrated missing-rate summaries.

No completion, knockoff, STABL, mask, or data-generation code was changed.
The V11 scientific results remain unchanged.
