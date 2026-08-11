# Changelog

## [0.1.0] - unreleased

### Added

- Recipe pipeline: unknown eligibility, drop ineligible, weighting-class / logit NR, raking, poststrat, linear/GREG calibrate, ratio trim, Tukey/Potter auto trim.
- Household `cluster=` on eligibility and nonresponse; bounded (`bounds=`) and ridge (`penalty=`) linear calibration.
- `Design` helpers (SRS / stratified / cluster / from_weights) and `estimate()` (mean, total, proportion, ratio, median) with bootstrap and jackknife variance.
- Gold CSVs + optional samplics / R survey–weightflow checks; examples and validation-gated method docs.
