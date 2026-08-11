# Cascade parity (eligibility, trim, poststrat/GREG, jackknife, logit NR)

## Goal

Complete the core weightflow-like cascade beyond Iteration 1/Design:

1. `step_unknown_eligibility` — redistribute unknown among known within cells
2. `step_trim` — ratio caps (base / median / value) with optional redistribution
3. `step_calibrate(method="poststratify"|"linear")` — poststrat + unbounded GREG
4. Delete-a-PSU `jackknife_weights` + `jack_mean` / `jack_total` / `jack_proportion`
5. `step_nonresponse(method="propensity", engine="logit")`

## Follow-ons (shipped)

6. `cluster=` on unknown eligibility / nonresponse (household collapse)
7. Bounded linear calibrate (`bounds=`, `calfun=`) and ridge (`penalty=`)
8. Auto trim: `step_trim_weights(method="tukey"|"potter")`

## Validation gate

Analytical toys → composition with `prep` → recovery where relevant. Do not loosen tolerances.
