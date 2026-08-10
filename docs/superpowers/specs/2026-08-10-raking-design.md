# Raking / IPF design (scaffold)

## Goal

Calibrate sample weights so weighted margins match known population totals.

## Estimand

Finite-population calibration weights under a raking (iterative proportional fitting) model.

## Inputs

- Unit base weights
- Categorical features (or dummy-expanded margins)
- Population totals / proportions per margin

## Outputs

- Calibrated weights
- Per-margin residuals
- Iteration count / convergence flag
- Diagnostics: ESS, max/min weight, weight CV

## Validation gate (must pass before `Recipe.calibrate`)

1. Analytical toy (hand-worked 2×2 or 4-cell) — `tests/test_raking_recovery.py`
2. Composition: recipe calibrate == `methods.raking.rake`
3. Recovery / R gold CSV in `tests/gold/`
4. Example twin: `examples/raking_recovery.py`

## Tolerances

- Margin match: `rtol=1e-8` on toy; document scale-relative bounds for noisy cases
- Do not loosen CI tolerances without a written diagnosis
