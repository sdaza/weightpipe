# Design + estimate (Iteration 2)

## Goal

Separate sampling **design** (base weights, strata, PSU) from the adjustment
**recipe**, and expose a single `estimate()` for mean / proportion / total with
recipe-aware bootstrap SE/CI.

## API

- `Design.srs(data, N=...)` → `w = N/n`
- `Design.stratified(data, stratum=..., N_h=...)` → `w = N_h/n_h`
- `Design.cluster(data, weight=..., psu=..., strata=...)` → user weights + ultimate cluster
- `Design.from_weights(...)` → arbitrary precomputed weights
- `Recipe.from_design(design)`
- `estimate(recipe, var, estimand=..., replicates=...)`

## Validation

1. Analytical: SRS/stratified weights match closed forms
2. `estimate` returns finite SE and CI containing the point estimate on toy data
3. Proportion rejects non-0/1 variables
4. Cluster design uses strata/psu from `recipe.design` when calling bootstrap
