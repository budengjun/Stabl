#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
read_flag <- function(name, default = NULL) {
  position <- which(args == name)
  if (length(position) == 0) return(default)
  if (position[1] == length(args)) stop(sprintf("Missing value after %s", name))
  args[position[1] + 1]
}

input_path <- read_flag("--input")
output_path <- read_flag("--output")
seed <- as.integer(read_flag("--seed", "1"))
threshold_abs_raw <- read_flag("--threshold-abs", NA_character_)
threshold_q_raw <- read_flag("--threshold-q", NA_character_)
ncomp_raw <- read_flag("--ncomp", NA_character_)
sparsity_raw <- read_flag("--sparsity", "1")

if (is.null(input_path) || is.null(output_path)) stop("--input and --output are required")
if (!requireNamespace("PLSKO", quietly = TRUE)) stop("R package PLSKO is not installed")

X_df <- read.csv(input_path, row.names = 1, check.names = FALSE)
X <- as.matrix(X_df)
storage.mode(X) <- "double"
if (any(!is.finite(X))) stop("PLSKO requires complete finite X")

set.seed(seed)
call_args <- list(X = X)
formals_names <- names(formals(PLSKO::plsko))
if ("seed" %in% formals_names) call_args$seed <- seed
if (!is.na(threshold_abs_raw)) call_args$threshold.abs <- as.numeric(threshold_abs_raw)
if (!is.na(threshold_q_raw)) call_args$threshold.q <- as.numeric(threshold_q_raw)
if (!is.na(ncomp_raw)) call_args$ncomp <- as.integer(ncomp_raw)
if (!is.na(sparsity_raw)) call_args$sparsity <- as.numeric(sparsity_raw)

Xk <- do.call(PLSKO::plsko, call_args)
Xk <- as.matrix(Xk)
if (!all(dim(Xk) == dim(X))) stop("PLSKO returned an incompatible matrix")
rownames(Xk) <- rownames(X)
colnames(Xk) <- colnames(X)
write.csv(Xk, output_path, row.names = TRUE, quote = FALSE)
