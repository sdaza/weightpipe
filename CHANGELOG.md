# Changelog

## [0.1.0] - unreleased

### Added

- `WeightPipe`: single entry point that owns the sampling design and the weighting steps, fits lazily, and exposes `weights`, `table()`, `diagnostics`, and `estimate()`. `Design` / `Recipe` remain available for lower-level use.
- Recipe pipeline: unknown eligibility, drop ineligible, weighting-class / propensity NR (`logit`, `gbm`, `forest`), raking, poststrat, linear/GREG calibrate (optional propensity assist; optional `forest`/`gbm` embedding engines), ratio trim, Tukey/Potter auto trim.
- Household `cluster=` on eligibility and nonresponse; bounded (`bounds=`) and ridge (`penalty=`) linear calibration.
- Parameter-driven `Design(...)` (kind inferred from `N` / `N_h`+`strata` / `weight`+`psu` / multi-stage `probabilities=` or `stage_weights=`) and `estimate()` (mean, total, proportion, ratio, median) with bootstrap and jackknife variance.
- Python-native sample planning (`sample_size`, `margin_of_error`, stratum allocation) in the same `weightpipe` package.
- Gold CSVs + optional samplics / R survey–weightflow checks; frozen R `sampler` planning gold for CI; examples and validation-gated method docs.
