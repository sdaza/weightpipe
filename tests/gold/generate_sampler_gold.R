#!/usr/bin/env Rscript
# Generate frozen gold CSVs from R sampler for planning parity.
#
# Usage:
#   Rscript tests/gold/generate_sampler_gold.R
#
# Requires R package sampler (GitHub: sdaza/sampler), or local sources via:
#   SAMPLER_R_DIR=.tmp/sampler/R Rscript tests/gold/generate_sampler_gold.R
#
# Install package (optional):
#   remotes::install_github("sdaza/sampler")

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
out_dir <- if (length(file_arg) == 1) {
  dirname(normalizePath(sub("^--file=", "", file_arg)))
} else {
  "tests/gold"
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

load_sampler <- function() {
  src_dir <- Sys.getenv("SAMPLER_R_DIR", unset = "")
  if (nzchar(src_dir)) {
    for (f in c("ssize.R", "serr.R", "astrata.R", "serrst.R")) {
      path <- file.path(src_dir, f)
      if (!file.exists(path)) stop("missing ", path)
      source(path, local = FALSE)
    }
    return(invisible(TRUE))
  }
  if (!requireNamespace("sampler", quietly = TRUE)) {
    stop(
      "Install sampler: remotes::install_github('sdaza/sampler')\n",
      "Or set SAMPLER_R_DIR to a directory containing ssize.R/serr.R/astrata.R/serrst.R"
    )
  }
  suppressPackageStartupMessages(library(sampler))
  invisible(TRUE)
}

load_sampler()

# ---------------------------------------------------------------------------
# Scalar ssize / serr cases (blog + FPC)
# ---------------------------------------------------------------------------
cases <- data.frame(
  case = c(
    "e05",
    "e05_deff_rr",
    "e05_deff_rr_N1000",
    "e05_deff_rr_N100",
    "serr_384",
    "serr_512_deff_rr",
    "serr_370_deff_rr_N1000",
    "serr_100_deff_rr_N100"
  ),
  kind = c(rep("ssize", 4), rep("serr", 4)),
  e = c(0.05, 0.05, 0.05, 0.05, NA, NA, NA, NA),
  n = c(NA, NA, NA, NA, 384, 512, 370, 100),
  deff = c(1, 1.2, 1.2, 1.2, 1, 1.2, 1.2, 1.2),
  rr = c(1, 0.9, 0.9, 0.9, 1, 0.9, 0.9, 0.9),
  N = c(NA, NA, 1000, 100, NA, NA, 1000, 100),
  p = 0.5,
  cl = 0.95,
  stringsAsFactors = FALSE
)

result <- numeric(nrow(cases))
for (i in seq_len(nrow(cases))) {
  row <- cases[i, ]
  if (row$kind == "ssize") {
    if (is.na(row$N)) {
      result[i] <- ssize(e = row$e, deff = row$deff, rr = row$rr, cl = row$cl, p = row$p)
    } else {
      result[i] <- ssize(e = row$e, deff = row$deff, rr = row$rr, N = row$N, cl = row$cl, p = row$p)
    }
  } else {
    if (is.na(row$N)) {
      result[i] <- serr(n = row$n, deff = row$deff, rr = row$rr, cl = row$cl, p = row$p)
    } else {
      result[i] <- serr(n = row$n, deff = row$deff, rr = row$rr, N = row$N, cl = row$cl, p = row$p)
    }
  }
}
cases$result_r_sampler <- result
write.csv(cases, file.path(out_dir, "planning_ssize_serr_r_sampler.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# Chile strata allocation + serrst (sampler::chile / blog)
# ---------------------------------------------------------------------------
chile <- data.frame(
  reg = 1:15,
  pob = c(
    328782, 613328, 308247, 759228, 1808300, 910577, 1035593, 2100494,
    983499, 834714, 107334, 163748, 7228581, 401548, 235081
  ),
  pr = c(0.3, 0.4, 0.5, 0.5, 0.5, 0.6, 0.3, 0.1, 0.2, 0.5, 0.5, 0.4, 0.6, 0.2, 0.3)
)

chile$aprop <- astrata(1000, wp = 1, N = chile$pob)
chile$afixed <- astrata(1000, wp = 0, N = chile$pob)
chile$a40 <- astrata(1000, wp = 0.4, N = chile$pob)
chile$a60 <- astrata(1000, wp = 0.6, N = chile$pob)
chile$aroot <- astrata(1000, method = "root", N = chile$pob)
chile$aneyman <- astrata(1000, method = "neyman", N = chile$pob, p = chile$pr)
chile$astdev <- astrata(1000, method = "stdev", N = chile$pob, p = chile$pr)
# method=error does not need samplesize in practice when called with named args;
# pass a dummy samplesize for strict formals.
chile$aerr <- astrata(samplesize = 1, e = 0.11, method = "error", N = chile$pob, p = chile$pr)

write.csv(chile, file.path(out_dir, "planning_chile_alloc_r_sampler.csv"), row.names = FALSE)

moe <- data.frame(
  allocation = c("aprop", "afixed", "a40", "a60", "aroot", "aneyman", "astdev", "aerr"),
  moe_r_sampler = c(
    serrst(n = chile$aprop, N = chile$pob, p = chile$pr),
    serrst(n = chile$afixed, N = chile$pob, p = chile$pr),
    serrst(n = chile$a40, N = chile$pob, p = chile$pr),
    serrst(n = chile$a60, N = chile$pob, p = chile$pr),
    serrst(n = chile$aroot, N = chile$pob, p = chile$pr),
    serrst(n = chile$aneyman, N = chile$pob, p = chile$pr),
    serrst(n = chile$astdev, N = chile$pob, p = chile$pr),
    serrst(n = chile$aerr, N = chile$pob, p = chile$pr)
  ),
  stringsAsFactors = FALSE
)
write.csv(moe, file.path(out_dir, "planning_chile_moe_r_sampler.csv"), row.names = FALSE)

message("Wrote sampler gold CSVs to ", normalizePath(out_dir))
