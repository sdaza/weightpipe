# Changelog

## [0.1.1](https://github.com/sdaza/weightpipe/compare/v0.1.0...v0.1.1) (2026-09-04)


### Bug Fixes

* vectorize linearization variance and accept Series in balance(before=) ([ff577e8](https://github.com/sdaza/weightpipe/commit/ff577e85cb4acd729ed3baf0768892f0b1da4ae2))
* vectorize linearization variance and accept Series in balance(before=) ([9532973](https://github.com/sdaza/weightpipe/commit/9532973906293a71dff4b921b6b901e8e7ac3ccf))

## [0.1.0] - 2026-08-28

### Added

- `pipe.estimate.glm`: design-based GLM (weighted IRLS for gaussian / binomial / poisson) with the same `variance=` path as other estimands (Binder sandwich, or recipe-aware bootstrap / jackknife of coefficients). Gold vs `survey::svyglm`.
- `pipe.estimate.mean` / `.total` / `.proportion` / `.ratio` / `.median`: estimand methods, several variables at once, and `by=` domain splits. `ratio` accepts one shared denominator or a list paired with the numerators. `pipe.estimate(...)` and `pipe.estimation` still work. The result table includes `cv`.
- `estimate(..., variance="linearization")`: ultimate-cluster Taylor SE treating fitted weights as fixed (`mean` / `total` / `proportion` / `ratio`).
- `WeightPipe`: single entry point that owns the sampling design and the weighting steps, fits lazily, and exposes `weights`, `collect_weights()`, `diagnostics`, and `estimate()`. `Design` / `Recipe` remain available for lower-level use.
- When no design weight is given, `Design` / `WeightPipe` create `base_weight=1.0` for all rows and log an informational message.
- Package logging: `setup_logging()` / `set_log_level()` enable compact `weightpipe` log output; the library is silent by default.
- `margins()` / `WeightPipe.margins()`: anytime weighted category totals and proportions; optional targets (or `targets="calibrate"`) for fit checks. Calibrate diagnostics include a tidy `margin_table`.
- `balance()` / `WeightPipe.balance()`: covariate balance before vs after weighting (SMD/ASMD for continuous and categorical covariates), with optional population microdata or explicit means/proportions; returns a `BalanceReport` (`table` + `summary`).
- Recipe pipeline: unknown eligibility, drop ineligible, weighting-class / propensity NR (`logit`, `gbm`, `forest`), raking, poststrat, linear/GREG calibrate (optional propensity assist; optional `forest`/`gbm` embedding engines), ratio trim, Tukey/Potter auto trim.
- Household `cluster=` on eligibility and nonresponse; bounded (`bounds=`) and ridge (`penalty=`) linear calibration.
- Parameter-driven `Design(...)` (kind inferred from `N` / `N_h`+`strata` / `weight`+`psu` / multi-stage `probabilities=` or `stage_weights=`) and `estimate()` (mean, total, proportion, ratio, median) with bootstrap and jackknife variance.
- Python-native sample planning (`sample_size`, `margin_of_error`, stratum allocation) in the same `weightpipe` package.
- Gold CSVs + optional svy / R survey–weightflow checks; frozen R `sampler` planning gold for CI; examples and validation-gated method docs.

### Changed

- Python weighting gold and the `gold` extra use [svy](https://svylab.com/svy) (`rake` / `poststratify` / class NR / GREG). Archived samplics is no longer a reference.
- Raking caches category row indexes before IPF (no per-iteration `astype(str)`); diagnostics still run after convergence.
- Recipe-aware bootstrap/jackknife scale the base-weight vector and `prep(record=False)` instead of copying the microdata frame on every replicate.
