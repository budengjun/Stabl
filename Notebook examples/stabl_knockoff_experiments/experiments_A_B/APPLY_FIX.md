# G1 PLSKO parallel fix

Copy the contents of this archive into `experiments_A_B`, preserving paths.

The fix removes the redundant outer PSOCK cluster, passes `parallel` and
`ncore` directly to `PLSKO::plsko_tuning`, limits BLAS/OpenMP threads to one
per worker, and uses four workers for pilot scripts.
