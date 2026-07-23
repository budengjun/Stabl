# Validation completed before packaging

The Python analysis was run against the uploaded `G2_ssi_pilot` result bundle.

* 120 saved score files loaded successfully
* 10,800 threshold-path rows reconstructed
* 120 matched-size comparison rows generated
* all 120 reconstructed `stabl_min` rows matched the stored G2 results
* maximum numerical reconstruction discrepancy was below `1.2e-16`
* exact matched-size and common-threshold decomposition identities passed
* expanded C2ST file loading and execution were tested end to end on a synthetic
  CSV fixture with stored pair-cache files

The server run will additionally perform expanded C2ST on all 120 SSI pair-cache
files because the original SSI data path is available there.
