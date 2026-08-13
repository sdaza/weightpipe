# Design + estimate (Iteration 2)

## Goal

Separate sampling **design** (base weights, strata, PSU) from the adjustment
**recipe**, and expose a single `estimate()` for mean / proportion / total with
recipe-aware bootstrap SE/CI.

## API

`Design` is parameter-driven; ``kind`` is inferred from inputs:

- `Design(data, N=...)` → SRS, `w = N/n`
- `Design(data, strata=..., N_h=...)` → stratified SRS, `w = N_h/n_h`
- `Design(data, weight=..., psu=..., strata=...)` → cluster / stratified cluster
- `Design(data, probabilities=[...], psu=..., strata=...)` → multi-stage, `w = 1/∏π_k`
- `Design(data, stage_weights=[...], psu=...)` → multi-stage, `w = ∏w_k`
- `Design(data, weight=...)` → existing weights
- `Recipe.from_design(design)`
- `estimate(recipe, var, estimand=..., replicates=...)`

## Validation

1. Analytical: SRS/stratified weights match closed forms
2. `estimate` returns finite SE and CI containing the point estimate on toy data
3. Proportion rejects non-0/1 variables
4. Cluster design uses strata/psu from `recipe.design` when calling bootstrap
