#!/usr/bin/env Rscript

cran_packages <- c("remotes", "progress", "parallel", "doParallel", "foreach", "jsonlite")
missing_cran <- cran_packages[!vapply(cran_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_cran) > 0) {
  install.packages(missing_cran, repos = "https://cloud.r-project.org")
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}
if (!requireNamespace("mixOmics", quietly = TRUE)) {
  BiocManager::install("mixOmics", ask = FALSE, update = FALSE)
}
if (!requireNamespace("PLSKO", quietly = TRUE)) {
  remotes::install_github("guannan-yang/PLSKO/PLSKO", upgrade = "never")
}
cat("Installed PLSKO version:", as.character(packageVersion("PLSKO")), "\n")
