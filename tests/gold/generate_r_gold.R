#!/usr/bin/env Rscript
# Generate frozen gold CSVs from R survey + weightflow for all comparable methods.
#
# Usage:  Rscript tests/gold/generate_r_gold.R
# Requires: survey, weightflow

suppressPackageStartupMessages({
  if (!requireNamespace("survey", quietly = TRUE)) {
    stop("install.packages('survey')")
  }
  if (!requireNamespace("weightflow", quietly = TRUE)) {
    stop("install.packages('weightflow')")
  }
  library(survey)
  library(weightflow)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
out_dir <- if (length(file_arg) == 1) {
  dirname(normalizePath(sub("^--file=", "", file_arg)))
} else {
  "tests/gold"
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

join_active <- function(df, out, wcol = ".weight") {
  # weightflow collect_weights keeps active rows only
  w <- rep(0, nrow(df))
  w[match(out$unit_id, df$unit_id)] <- as.numeric(out[[wcol]])
  w
}

# ---------------------------------------------------------------------------
# weightflow: unknown eligibility
# ---------------------------------------------------------------------------
df_ue <- data.frame(
  unit_id = 1:5,
  region = factor(c("N", "N", "N", "S", "S"), levels = c("N", "S")),
  unknown = c(0L, 0L, 1L, 0L, 1L),
  pw = c(1, 1, 1, 2, 2)
)
wf <- weighting_spec(df_ue, base_weights = pw) |>
  step_unknown_eligibility(unknown = unknown, by = "region") |>
  prep()
out <- collect_weights(wf)
df_ue$weight_weightflow <- join_active(df_ue, out)
write.csv(df_ue, file.path(out_dir, "unknown_eligibility_weightflow.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# weightflow: drop ineligible
# ---------------------------------------------------------------------------
df_di <- data.frame(
  unit_id = 1:4,
  ineligible = c(0L, 0L, 1L, 0L),
  pw = c(1, 1, 1, 1),
  y = 1:4
)
wf <- weighting_spec(df_di, base_weights = pw) |>
  step_drop_ineligible(ineligible = ineligible) |>
  prep()
out <- collect_weights(wf)
df_di$weight_weightflow <- join_active(df_di, out)
write.csv(df_di, file.path(out_dir, "drop_ineligible_weightflow.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# weightflow: weighting-class NR
# ---------------------------------------------------------------------------
df_nr <- data.frame(
  unit_id = 1:5,
  region = factor(c("N", "N", "N", "S", "S"), levels = c("N", "S")),
  responded = c(1L, 1L, 0L, 1L, 0L),
  pw = c(1, 1, 1, 2, 2)
)
wf <- weighting_spec(df_nr, base_weights = pw) |>
  step_nonresponse(respondent = responded, method = "weighting_class", by = "region") |>
  prep()
out <- collect_weights(wf)
df_nr$weight_weightflow <- join_active(df_nr, out)
write.csv(df_nr, file.path(out_dir, "nr_weighting_class_weightflow.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# weightflow: logit propensity NR (saturated formula ⇒ cell 1/p)
# ---------------------------------------------------------------------------
df_logit <- data.frame(
  unit_id = 1:20,
  region = factor(rep(c("N", "S"), each = 10), levels = c("N", "S")),
  sex = factor(rep(c("M", "F"), 10), levels = c("M", "F")),
  responded = c(1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1),
  pw = rep(1, 20)
)
wf <- weighting_spec(df_logit, base_weights = pw) |>
  step_nonresponse(
    respondent = responded, method = "propensity", engine = "logit",
    formula = ~ region + sex, num_classes = NULL, weight_model = FALSE
  ) |>
  prep()
out <- collect_weights(wf)
df_logit$weight_weightflow <- join_active(df_logit, out)
write.csv(df_logit, file.path(out_dir, "nr_logit_weightflow.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# weightflow: raking
# ---------------------------------------------------------------------------
df_rk <- data.frame(
  unit_id = 1:4,
  sex = factor(c("M", "M", "F", "F"), levels = c("M", "F")),
  region = factor(c("N", "S", "N", "S"), levels = c("N", "S")),
  pw = c(1, 1, 1, 1)
)
wf <- weighting_spec(df_rk, base_weights = pw) |>
  step_calibrate(
    method = "raking",
    margins = list(sex = c(M = 60, F = 40), region = c(N = 30, S = 70))
  ) |>
  prep()
out <- collect_weights(wf)
df_rk$weight_weightflow <- join_active(df_rk, out)
write.csv(df_rk, file.path(out_dir, "raking_2x2_weightflow.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# weightflow: poststratify
# ---------------------------------------------------------------------------
df_ps <- data.frame(
  unit_id = 1:4,
  region = factor(c("N", "N", "S", "S"), levels = c("N", "S")),
  pw = c(1, 1, 1, 1)
)
wf <- weighting_spec(df_ps, base_weights = pw) |>
  step_calibrate(method = "poststratify", margins = list(region = c(N = 10, S = 30))) |>
  prep()
out <- collect_weights(wf)
df_ps$weight_weightflow <- join_active(df_ps, out)
write.csv(df_ps, file.path(out_dir, "poststrat_region_weightflow.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# weightflow: linear / GREG
# ---------------------------------------------------------------------------
df_lin <- data.frame(
  unit_id = 1:4,
  region = factor(c("N", "N", "S", "S"), levels = c("N", "S")),
  age = c(20, 40, 30, 50),
  pw = c(1, 1, 1, 1)
)
totals_lin <- c(`(Intercept)` = 100, regionS = 50, age = 3500)
wf <- weighting_spec(df_lin, base_weights = pw) |>
  step_calibrate(method = "linear", formula = ~ region + age, totals = totals_lin) |>
  prep()
out <- collect_weights(wf)
df_lin$weight_weightflow <- join_active(df_lin, out)
write.csv(df_lin, file.path(out_dir, "linear_calibrate_weightflow.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# weightflow: trim (no redistribute — exact caps)
# ---------------------------------------------------------------------------
df_tr <- data.frame(unit_id = 1:3, pw = c(1, 2, 10))
wf <- weighting_spec(df_tr, base_weights = pw) |>
  step_trim(max_ratio = 5, reference = "value", redistribute = FALSE) |>
  prep()
out <- collect_weights(wf)
df_tr$weight_weightflow <- join_active(df_tr, out)
write.csv(df_tr, file.path(out_dir, "trim_value_noredist_weightflow.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# weightflow cascade: unknown → drop → NR → rake → trim
# ---------------------------------------------------------------------------
df_cas <- data.frame(
  unit_id = 1:8,
  region = factor(c("N", "N", "N", "N", "S", "S", "S", "S"), levels = c("N", "S")),
  sex = factor(c("M", "F", "M", "F", "M", "F", "M", "F"), levels = c("M", "F")),
  unknown = c(0L, 0L, 1L, 0L, 0L, 0L, 0L, 0L),
  ineligible = c(0L, 0L, 0L, 1L, 0L, 0L, 0L, 0L),
  responded = c(1L, 1L, 1L, 1L, 1L, 0L, 1L, 1L),
  pw = rep(1, 8)
)
wf <- weighting_spec(df_cas, base_weights = pw) |>
  step_unknown_eligibility(unknown = unknown, by = "region") |>
  step_drop_ineligible(ineligible = ineligible) |>
  step_nonresponse(respondent = responded, method = "weighting_class", by = "region") |>
  step_calibrate(
    method = "raking",
    margins = list(sex = c(M = 3, F = 3), region = c(N = 3, S = 3))
  ) |>
  step_trim(max_ratio = 10, reference = "median", redistribute = FALSE) |>
  prep()
out <- collect_weights(wf)
df_cas$weight_weightflow <- join_active(df_cas, out)
write.csv(df_cas, file.path(out_dir, "cascade_full_weightflow.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# survey: raking / poststrat / linear / ratio / median / mean / total
# ---------------------------------------------------------------------------
d <- data.frame(
  unit_id = 1:4,
  sex = factor(c("M", "M", "F", "F"), levels = c("M", "F")),
  region = factor(c("N", "S", "N", "S"), levels = c("N", "S")),
  pw = c(1, 1, 1, 1)
)
des <- svydesign(ids = ~1, weights = ~pw, data = d)
pop.sex <- data.frame(sex = factor(c("M", "F"), levels = c("M", "F")), Freq = c(60, 40))
pop.region <- data.frame(region = factor(c("N", "S"), levels = c("N", "S")), Freq = c(30, 70))
raked <- rake(
  des, sample.margins = list(~sex, ~region),
  population.margins = list(pop.sex, pop.region),
  control = list(maxit = 200, epsilon = 1e-12)
)
d$weight_r_survey <- as.numeric(weights(raked))
write.csv(d, file.path(out_dir, "raking_2x2_r_survey.csv"), row.names = FALSE)

d2 <- data.frame(
  unit_id = 1:4,
  region = factor(c("N", "N", "S", "S"), levels = c("N", "S")),
  pw = c(1, 1, 1, 1)
)
des2 <- svydesign(ids = ~1, weights = ~pw, data = d2)
pop.reg <- data.frame(region = factor(c("N", "S"), levels = c("N", "S")), Freq = c(10, 30))
ps <- postStratify(des2, ~region, pop.reg)
d2$weight_r_survey <- as.numeric(weights(ps))
write.csv(d2, file.path(out_dir, "poststrat_region_r_survey.csv"), row.names = FALSE)

d3 <- data.frame(
  unit_id = 1:4,
  region = factor(c("N", "N", "S", "S"), levels = c("N", "S")),
  age = c(20, 40, 30, 50),
  pw = c(1, 1, 1, 1)
)
des3 <- svydesign(ids = ~1, weights = ~pw, data = d3)
cal <- calibrate(des3, ~region + age, population = c(`(Intercept)` = 100, regionS = 50, age = 3500))
d3$weight_r_survey <- as.numeric(weights(cal))
write.csv(d3, file.path(out_dir, "linear_calibrate_r_survey.csv"), row.names = FALSE)

d4 <- data.frame(
  unit_id = 1:8,
  y = c(10, 12, 20, 22, 11, 13, 21, 23),
  x = c(2, 2, 4, 4, 2, 2, 4, 4),
  pw = rep(1, 8)
)
des4 <- svydesign(ids = ~1, weights = ~pw, data = d4)
est <- data.frame(
  estimand = c("mean", "total", "ratio", "median"),
  estimate_r_survey = c(
    as.numeric(coef(svymean(~y, des4))),
    as.numeric(coef(svytotal(~y, des4))),
    as.numeric(coef(svyratio(~y, ~x, des4))),
    as.numeric(svyquantile(~y, des4, quantiles = 0.5, ci = FALSE))
  ),
  numerator = c("y", "y", "y", "y"),
  denominator = c(NA, NA, "x", NA)
)
write.csv(est, file.path(out_dir, "estimands_r_survey.csv"), row.names = FALSE)
# keep legacy name used by older tests
write.csv(
  est[est$estimand %in% c("ratio", "median"), ],
  file.path(out_dir, "ratio_median_r_survey.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------------------------
# survey: design-based GLM (svyglm)
# ---------------------------------------------------------------------------
d_glm <- data.frame(
  stratum = factor(c("A", "A", "A", "A", "B", "B", "B", "B")),
  psu = factor(c(1, 1, 2, 2, 3, 3, 4, 4)),
  region = factor(c("N", "N", "S", "S", "N", "N", "S", "S"), levels = c("N", "S")),
  age = c(20, 40, 30, 50, 22, 38, 28, 52),
  y = c(10, 12, 20, 22, 11, 13, 21, 23),
  employed = c(1, 0, 1, 1, 0, 1, 1, 0),
  count = c(0, 1, 2, 1, 0, 2, 3, 1),
  pw = rep(1, 8)
)
des_glm <- svydesign(ids = ~psu, strata = ~stratum, weights = ~pw, data = d_glm, nest = TRUE)
pack_svyglm <- function(fit, family, formula) {
  cf <- coef(fit)
  se <- as.numeric(SE(fit))
  data.frame(
    family = family,
    formula = formula,
    term = names(cf),
    estimate_r_survey = as.numeric(cf),
    se_r_survey = se,
    stringsAsFactors = FALSE
  )
}
glm_gold <- rbind(
  pack_svyglm(svyglm(y ~ region + age, des_glm, family = gaussian()), "gaussian", "y ~ region + age"),
  pack_svyglm(svyglm(employed ~ region, des_glm, family = quasibinomial()), "binomial", "employed ~ region"),
  pack_svyglm(svyglm(count ~ age, des_glm, family = quasipoisson()), "poisson", "count ~ age"),
  pack_svyglm(svyglm(y ~ 1, des_glm, family = gaussian()), "gaussian", "y ~ 1")
)
write.csv(glm_gold, file.path(out_dir, "glm_r_survey.csv"), row.names = FALSE)

# legacy cascade NR+rake file (still useful)
df_legacy <- data.frame(
  unit_id = 1:6,
  region = factor(c("N", "N", "N", "S", "S", "S"), levels = c("N", "S")),
  sex = factor(c("M", "F", "M", "F", "M", "F"), levels = c("M", "F")),
  responded = c(1L, 1L, 0L, 1L, 0L, 1L),
  pw = rep(1, 6),
  y = c(1, 2, 3, 4, 5, 6)
)
wf <- weighting_spec(df_legacy, base_weights = pw) |>
  step_nonresponse(respondent = responded, method = "weighting_class", by = "region") |>
  step_calibrate(method = "raking", margins = list(sex = c(M = 2, F = 4), region = c(N = 3, S = 3))) |>
  prep()
out <- collect_weights(wf)
df_legacy$weight_weightflow <- join_active(df_legacy, out)
write.csv(df_legacy, file.path(out_dir, "cascade_nr_rake_weightflow.csv"), row.names = FALSE)

message("Wrote gold CSVs under ", normalizePath(out_dir))
