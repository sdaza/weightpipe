# weightpipe

[![CI](https://github.com/sdaza/weightpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/sdaza/weightpipe/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Declarative survey weighting recipes with **recipe-aware replicate weights**, diagnostics, and bootstrap/jackknife SE/CIs.

Methods ship behind a validation gate: analytical/composition/recovery tests must pass before a public `Recipe.step_*()` is ungated.

## Install / develop

```bash
uv sync --all-groups --extra gold
uv run pytest
uv run ruff check
uv run python examples/04_cascade_parity.py
```

CI runs `uv sync --all-groups --extra gold --locked` — commit `uv.lock`.

Cross-package gold (`@pytest.mark.gold`):

- Always-on: frozen CSVs in [`tests/gold/`](tests/gold/) (includes R `sampler` planning gold for CI — no R on Actions)
- Live samplics: optional extra `gold` ([`tests/test_gold_samplics.py`](tests/test_gold_samplics.py))
- R `survey` + `weightflow`: `Rscript tests/gold/generate_r_gold.R`; checks in [`tests/test_gold_r_packages.py`](tests/test_gold_r_packages.py) (skip if R/packages/CSVs missing). Optional `uv sync --extra r-gold` for live rpy2.
- R `sampler` planning: regenerate with `Rscript tests/gold/generate_sampler_gold.R`; CSV + optional live checks in [`tests/test_gold_sampler.py`](tests/test_gold_sampler.py)

## Quickstart

```python
from weightpipe import Design, Recipe, estimate, population_totals

design = Design(df, weight="pw", psu="psu", strata="stratum")
totals = population_totals(pop, "~ region + sex + age")

recipe = (
    Recipe.from_design(design)
    .step_unknown_eligibility(unknown="unknown", by=["region"], cluster="hh")
    .step_drop_ineligible(ineligible="ineligible")
    .step_nonresponse(
        respondent="responded",
        method="propensity",  # or "weighting_class"
        engine="logit",
        formula="~ region + sex",
        num_classes=5,
        cluster="hh",
    )
    .step_calibrate(
        method="linear",
        formula="~ region + sex + age",
        totals=totals,
        # bounds=(0.3, 3.0), calfun="linear",  # bounded Deville–Särndal
        # penalty=10.0,                        # ridge (unbounded linear only)
    )
    .step_trim(max_ratio=4.0, reference="median", redistribute=True)
    # .step_trim_weights(method="tukey")  # or "potter" — auto upper cutoff
)
fitted = recipe.prep()

estimate(recipe, "y", estimand="mean", fitted=fitted, variance="bootstrap", replicates=200, seed=1)
estimate(recipe, "y", estimand="mean", fitted=fitted, variance="jackknife")
estimate(recipe, "y", estimand="ratio", denominator="x", fitted=fitted, variance="jackknife")
estimate(recipe, "y", estimand="median", fitted=fitted, variance="jackknife")
```

## Recipe steps

| Step | Notes |
|------|--------|
| `step_unknown_eligibility` | Redistribute unknown → known within cells; optional `cluster=` (any unknown → household unknown) |
| `step_drop_ineligible` | Zero ineligible units |
| `step_nonresponse` | `weighting_class` or `propensity` (`engine="logit"`); optional `cluster=` (all members must respond) |
| `step_calibrate` | `raking`, `poststratify`, or `linear` (GREG); linear supports `bounds=`, `calfun=`, `penalty=` |
| `step_trim` | Ratio caps vs `median` / `base` / `value`; optional redistribute |
| `step_trim_weights` | Auto trim: Tukey fence or Potter MSE cutoff (`method="tukey"|"potter"`) |

## Designs and estimation

`Design` takes the sampling inputs; ``kind`` is inferred (you do not name SRS/stratified/cluster):

```python
Design(df, N=10_000)  # SRS: w = N/n
Design(df, strata="region", N_h={...})  # stratified SRS: w = N_h/n_h
Design(df, weight="pw", psu="psu")  # cluster
Design(df, weight="pw", psu="psu", strata="stratum")  # stratified cluster
Design(df, probabilities=["p1", "p2"], psu="psu", strata="stratum")  # multi-stage: w = 1/(p1*p2)
Design(df, stage_weights=["w1", "w2"], psu="psu")  # multi-stage: w = w1*w2
Design(df, weight="pw")  # existing weights
```

Multi-stage designs fold stage selection into the weight and use ``psu`` as the
ultimate cluster for bootstrap/jackknife variance.

`estimate(..., estimand=)` supports `mean`, `total`, `proportion`, `ratio` (`denominator=`), and `median`. Variance: `none`, `bootstrap` (Rao–Wu), or `jackknife` (delete-a-PSU).

## Sample planning

Planning and weighting live in the same `weightpipe` package. Use the
planning helpers before fieldwork, then pass the resulting population sizes
to `Design` after drawing the sample.

```python
from weightpipe import (
    Design,
    allocate_strata,
    allocation_table,
    margin_of_error,
    sample_size,
    stratified_margin_of_error,
)

sample_size(0.05)  # 384 cases for ±5 percentage points at 95%
sample_size(0.05, deff=1.2, response_rate=0.9, population=10_000)
margin_of_error(384)

populations = {"North": 5_000, "South": 15_000}
plan = allocation_table(populations, sample=400, method="mixed")
stratified_margin_of_error(plan["sample"], population=plan["population"])

# After drawing the planned cases into df:
design = Design(df, strata="region", N_h=populations)
```

Allocation methods are `mixed` (equal/proportional blend), `root`, `neyman`,
`stdev`, and `error` (a target margin for each stratum). Planning `deff` is an
assumption made before fieldwork; `design_effect(weights)` is a post-fieldwork
diagnostic.

## Validation policy

Do **not** loosen recovery tolerances to green CI. Diagnose failures first.
