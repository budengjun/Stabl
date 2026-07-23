#!/usr/bin/env Rscript

# Command line bridge to the authors' official PLSKO R package.
#
# Example:
#   Rscript official_plsko_bridge.R \
#     --input X.csv --output X_knockoff.csv --seed 1 \
#     --threshold-q 0.8 --ncomp 5 --sparsity 1

args <- commandArgs(trailingOnly = TRUE)

read_flag <- function(name, default = NULL) {
  pos <- which(args == name)
  if (length(pos) == 0) return(default)
  if (pos[1] == length(args)) stop(sprintf("Missing value after %s", name))
  args[pos[1] + 1]
}

input_path <- read_flag("--input")
output_path <- read_flag("--output")
seed <- as.integer(read_flag("--seed", "1"))
threshold_abs_raw <- read_flag("--threshold-abs", NA_character_)
threshold_q_raw <- read_flag("--threshold-q", "0.8")
ncomp_raw <- read_flag("--ncomp", NA_character_)
sparsity_raw <- read_flag("--sparsity", "1")

if (is.null(input_path) || is.null(output_path)) {
  stop("Both --input and --output are required")
}
if (!requireNamespace("PLSKO", quietly = TRUE)) {
  stop(
    paste0(
      "The PLSKO package is not installed. Run in R:\n",
      "  install.packages('devtools')\n",
      "  devtools::install_github('guannan-yang/PLSKO/PLSKO', ",
      "quiet = TRUE, upgrade = 'never')"
    )
  )
}

X_df <- read.csv(input_path, row.names = 1, check.names = FALSE)
X <- as.matrix(X_df)
storage.mode(X) <- "double"
if (any(!is.finite(X))) {
  stop("Official PLSKO bridge currently requires a complete finite matrix")
}

set.seed(seed)
call_args <- list(X = X)
if (!is.na(threshold_abs_raw)) call_args$threshold.abs <- as.numeric(threshold_abs_raw)
if (!is.na(threshold_q_raw)) call_args$threshold.q <- as.numeric(threshold_q_raw)
if (!is.na(ncomp_raw)) call_args$ncomp <- as.integer(ncomp_raw)
if (!is.na(sparsity_raw)) call_args$sparsity <- as.numeric(sparsity_raw)

Xk <- do.call(PLSKO::plsko, call_args)
Xk <- as.matrix(Xk)
if (!all(dim(Xk) == dim(X))) {
  stop(sprintf(
    "PLSKO returned shape %s but expected %s",
    paste(dim(Xk), collapse = " x "),
    paste(dim(X), collapse = " x ")
  ))
}
rownames(Xk) <- rownames(X)
colnames(Xk) <- colnames(X)
write.csv(Xk, output_path, row.names = TRUE, quote = FALSE)
