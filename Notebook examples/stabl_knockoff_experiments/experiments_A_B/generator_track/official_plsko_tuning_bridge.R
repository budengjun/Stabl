#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

# Prevent each PSOCK worker from spawning its own BLAS/OpenMP thread pool.
# This is important on shared servers and avoids CPU/RAM oversubscription.
Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)
read_flag <- function(name, default = NULL) {
  position <- which(args == name)
  if (length(position) == 0) return(default)
  if (position[1] == length(args)) stop(sprintf("Missing value after %s", name))
  args[position[1] + 1]
}
parse_numeric_csv <- function(value) {
  as.numeric(strsplit(value, ",", fixed = TRUE)[[1]])
}
parse_integer_csv <- function(value) {
  as.integer(strsplit(value, ",", fixed = TRUE)[[1]])
}
parse_bool <- function(value) {
  tolower(value) %in% c("true", "1", "yes")
}
safe_name <- function(value) {
  gsub("[^A-Za-z0-9_.]+", "_", value)
}
flatten_one_row <- function(value) {
  if (is.null(value)) return(NULL)
  if (is.data.frame(value)) return(value)
  if (is.matrix(value)) return(as.data.frame(value))
  if (is.atomic(value) && !is.null(names(value))) {
    return(as.data.frame(as.list(value), stringsAsFactors = FALSE))
  }
  if (is.list(value) && length(value) > 0 && all(lengths(value) == 1)) {
    return(as.data.frame(value, stringsAsFactors = FALSE))
  }
  NULL
}

input_path <- read_flag("--input")
out_dir <- read_flag("--out-dir")
seed <- as.integer(read_flag("--seed", "1"))
n_ko <- as.integer(read_flag("--n-ko", "10"))
p_s <- as.integer(read_flag("--p-s", "20"))
q <- as.numeric(read_flag("--q", "0.05"))
ncomp <- parse_integer_csv(read_flag("--ncomp", "2,5,10"))
threshold_abs <- parse_numeric_csv(read_flag("--threshold-abs", "0,0.1,0.2,0.3"))
sparsity <- parse_numeric_csv(read_flag("--sparsity", "0.5,0.8,1"))
fdp_measure <- read_flag("--fdp-measure", "median")
early_stop <- parse_bool(read_flag("--early-stop", "false"))
n_cores <- as.integer(read_flag("--n-cores", "1"))

if (is.null(input_path) || is.null(out_dir)) stop("--input and --out-dir are required")
if (!requireNamespace("PLSKO", quietly = TRUE)) stop("R package PLSKO is not installed")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

X_df <- read.csv(input_path, row.names = 1, check.names = FALSE)
X <- as.matrix(X_df)
storage.mode(X) <- "double"
if (any(!is.finite(X))) stop("PLSKO tuning requires complete finite X")
if (p_s < 1 || p_s >= ncol(X)) stop("p_s must lie between 1 and p minus 1")

if (!is.finite(n_cores) || n_cores < 1) stop("n_cores must be at least 1")
n_cores <- as.integer(n_cores)

# PLSKO::plsko_tuning creates and manages its own PSOCK cluster through its
# `parallel` and `ncore` arguments. Do not create a second outer cluster here.
# Keep an exit cleanup guard in case the package errors before its own cleanup.
if (requireNamespace("doParallel", quietly = TRUE)) {
  on.exit(try(doParallel::stopImplicitCluster(), silent = TRUE), add = TRUE)
}

set.seed(seed)
formals_names <- names(formals(PLSKO::plsko_tuning))
call_args <- list(X = X)
set_supported <- function(candidates, value, required = TRUE) {
  found <- candidates[candidates %in% formals_names]
  if (length(found) > 0) {
    call_args[[found[1]]] <<- value
  } else if (required) {
    stop(sprintf(
      "PLSKO::plsko_tuning does not expose any of the expected arguments: %s",
      paste(candidates, collapse = ", ")
    ))
  }
}
set_supported(c("n_ko", "n.ko", "nko"), n_ko)
set_supported(c("p_s", "p.s", "ps"), p_s)
set_supported(c("q", "fdr", "target.fdr"), q)
set_supported(c("ncomp", "n.comp"), ncomp)
set_supported(c("threshold.abs", "threshold_abs"), threshold_abs)
set_supported(c("sparsity"), sparsity)
set_supported(c("fdp.measure", "fdp_measure"), fdp_measure, required = FALSE)
set_supported(c("early.stop", "early_stop"), early_stop, required = FALSE)
set_supported(c("seed", "random.seed"), seed, required = FALSE)
# Critical: pass the requested worker count to the official implementation.
# Without this, PLSKO defaults to detectCores() - 2 and can launch far more
# workers than requested, while the old bridge also kept another cluster alive.
set_supported(c("parallel"), n_cores > 1, required = FALSE)
set_supported(c("ncore", "ncores"), n_cores, required = FALSE)

start_time <- Sys.time()
result <- do.call(PLSKO::plsko_tuning, call_args)
end_time <- Sys.time()
saveRDS(result, file.path(out_dir, "tuning_result.rds"))
capture.output(str(result, max.level = 4), file = file.path(out_dir, "result_structure.txt"))
capture.output(sessionInfo(), file = file.path(out_dir, "session_info.txt"))
writeLines(as.character(utils::packageVersion("PLSKO")), file.path(out_dir, "PLSKO_version.txt"))
writeLines(as.character(difftime(end_time, start_time, units = "secs")), file.path(out_dir, "elapsed_seconds.txt"))

if (is.list(result) && !is.null(names(result))) {
  for (name in names(result)) {
    table <- flatten_one_row(result[[name]])
    if (!is.null(table)) {
      write.csv(table, file.path(out_dir, paste0("component_", safe_name(name), ".csv")), row.names = FALSE)
    }
  }
}

optimal <- NULL
if (is.list(result) && "optimal" %in% names(result)) {
  optimal <- flatten_one_row(result$optimal)
}

# Fallback for package versions that do not expose $optimal in a flat form.
if (is.null(optimal)) {
  candidates <- NULL
  if (is.list(result) && "median" %in% names(result) && is.data.frame(result$median)) {
    candidates <- result$median
  } else if (is.list(result) && "mean" %in% names(result) && is.data.frame(result$mean)) {
    candidates <- result$mean
  }
  if (!is.null(candidates) && nrow(candidates) > 0) {
    fdp_columns <- intersect(c("median.fdp", "mean.fdp", "fdp", "FDP"), names(candidates))
    power_columns <- intersect(c("mean.power", "median.power", "power", "Power"), names(candidates))
    if (length(fdp_columns) > 0) {
      fdp_values <- as.numeric(candidates[[fdp_columns[1]]])
      valid <- which(is.finite(fdp_values) & fdp_values <= q)
      if (length(valid) > 0 && length(power_columns) > 0) {
        power_values <- as.numeric(candidates[[power_columns[1]]])
        index <- valid[which.max(power_values[valid])]
      } else {
        index <- which.min(fdp_values)
      }
      optimal <- candidates[index, , drop = FALSE]
      optimal$selection_source <- "bridge_fallback"
    }
  }
}

if (is.null(optimal) || nrow(optimal) < 1) {
  stop("Could not extract an optimal PLSKO configuration. Inspect tuning_result.rds")
}
write.csv(optimal[1, , drop = FALSE], file.path(out_dir, "optimal.csv"), row.names = FALSE)

manifest <- data.frame(
  package = "PLSKO",
  package_version = as.character(utils::packageVersion("PLSKO")),
  seed = seed,
  n = nrow(X),
  p = ncol(X),
  n_ko = n_ko,
  p_s = p_s,
  q = q,
  fdp_measure = fdp_measure,
  early_stop = early_stop,
  n_cores = n_cores,
  elapsed_seconds = as.numeric(difftime(end_time, start_time, units = "secs")),
  stringsAsFactors = FALSE
)
write.csv(manifest, file.path(out_dir, "tuning_manifest.csv"), row.names = FALSE)
