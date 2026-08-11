# weightpipe

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

- Always-on: frozen CSVs in [`tests/gold/`](tests/gold/)
- Live samplics: optional extra `gold` ([`tests/test_gold_samplics.py`](tests/test_gold_samplics.py))
- R `survey` + `weightflow`: `Rscript tests/gold/generate_r_gold.R`; checks in [`tests/test_gold_r_packages.py`](tests/test_gold_r_packages.py) (skip if R/packages/CSVs missing). Optional `uv sync --extra r-gold` for live rpy2.

Examples (`# %%` cells): [`01`](examples/01_minimal_recipe.py) · [`02`](examples/02_nonresponse_raking.py) · [`03`](examples/03_designs_estimate.py) · [`04`](examples/04_cascade_parity.py)

## Quickstart

```python
from weightpipe import Design, Recipe, estimate, population_totals

design = Design.cluster(df, weight="pw", psu="psu", strata="stratum")
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

```python
Design.srs(df, N=10_000)  # w = N/n
Design.stratified(df, stratum="region", N_h={...})  # w = N_h/n_h
Design.cluster(df, weight="pw", psu="psu", strata=...)
Design.from_weights(df, weight="pw", strata=..., psu=...)
```

`estimate(..., estimand=)` supports `mean`, `total`, `proportion`, `ratio` (`denominator=`), and `median`. Variance: `none`, `bootstrap` (Rao–Wu), or `jackknife` (delete-a-PSU).

## Validation policy

Do **not** loosen recovery tolerances to green CI. Diagnose failures first.
