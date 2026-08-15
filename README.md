# weightpipe

[![CI](https://github.com/sdaza/weightpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/sdaza/weightpipe/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Survey weighting in Python: define a sampling design, apply a weighting recipe, then estimate means, totals, proportions, ratios, and medians with bootstrap or jackknife standard errors.

## Install

Install from GitHub (latest `main`):

```bash
pip install "git+https://github.com/sdaza/weightpipe.git"
# or
uv add "git+https://github.com/sdaza/weightpipe.git"
```

Requires Python 3.11+. Not published on PyPI yet.

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
        method="propensity",
        engine="logit",  # or "gbm" / "forest"
        formula="~ region + sex",
        num_classes=5,
        cluster="hh",
    )
    .step_calibrate(
        method="linear",
        formula="~ region + sex + age",
        totals=totals,
        # engine="forest", population=pop_df,  # tree-embedding GREG (needs population microdata)
        # assist="propensity",                 # add p̂ to linear calibration
        # assist="propensity_class",           # rake/calibrate while keeping class mass
    )
    .step_trim(max_ratio=4.0, reference="median", redistribute=True)
)
fitted = recipe.prep()

estimate(recipe, "y", estimand="mean", fitted=fitted, variance="jackknife")
estimate(recipe, "y", estimand="ratio", denominator="x", fitted=fitted, variance="jackknife")
estimate(recipe, "y", estimand="median", fitted=fitted, variance="bootstrap", replicates=200, seed=1)
```

## Sampling designs

Pass the design inputs; the design type is inferred automatically.

```python
Design(df, N=10_000)                                  # SRS: w = N / n
Design(df, strata="region", N_h={"North": 5000, ...}) # stratified SRS: w = N_h / n_h
Design(df, weight="pw", psu="psu")                    # cluster
Design(df, weight="pw", psu="psu", strata="stratum")  # stratified cluster
Design(df, probabilities=["p1", "p2"], psu="psu")     # multi-stage: w = 1 / (p1 * p2)
Design(df, stage_weights=["w1", "w2"], psu="psu")     # multi-stage: w = w1 * w2
Design(df, weight="pw")                               # existing design weights
```

For multi-stage samples, stage selection is folded into the weight. Use `psu` as the ultimate cluster for variance estimation.

## Weighting steps

Build a recipe from a design, then chain adjustments:

| Step | What it does |
|------|----------------|
| `step_unknown_eligibility` | Redistribute unknown eligibility within cells; optional household `cluster=` |
| `step_drop_ineligible` | Set ineligible units to weight 0 |
| `step_nonresponse` | Weighting-class or propensity (`engine="logit"`, `"gbm"`, or `"forest"`); optional `cluster=` |
| `step_calibrate` | Raking, post-stratification, or linear/GREG; `engine="forest"`/`"gbm"` for tree-embedding GREG (`population=` required); optional `assist=` (`propensity_class` auto-converts `proportions=` → absolute `margins` before attaching class totals) |
| `step_trim` | Cap extreme weights by ratio to median/base/value |
| `step_trim_weights` | Automatic Tukey or Potter trimming |

## Estimation

```python
estimate(recipe, "y", estimand="mean", variance="jackknife")
estimate(recipe, "employed", estimand="proportion", variance="bootstrap", replicates=200, seed=1)
estimate(recipe, "y", estimand="total", variance="jackknife")
estimate(recipe, "y", estimand="ratio", denominator="x", variance="jackknife")
estimate(recipe, "y", estimand="median", variance="jackknife")
```

Supported estimands: `mean`, `total`, `proportion`, `ratio`, `median`.  
Variance options: `bootstrap` (Rao–Wu; default) or `jackknife` (delete-a-PSU).

## Sample-size planning

Plan sample sizes before fieldwork, then build a `Design` after data collection:

```python
from weightpipe import (
    Design,
    allocate_strata,
    allocation_table,
    margin_of_error,
    sample_size,
    stratified_margin_of_error,
)

sample_size(0.05)  # n for ±5 percentage points at 95% confidence
margin_of_error(384)

populations = {"North": 5_000, "South": 15_000}
plan = allocation_table(populations, sample=400, method="mixed")
stratified_margin_of_error(plan["sample"], population=plan["population"])

# After drawing the planned cases:
design = Design(df, strata="region", N_h=populations)
```

Allocation methods: `mixed`, `root`, `neyman`, `stdev`, and `error`.

## Examples

Interactive scripts (run cell-by-cell or with Python):

- [`examples/01_minimal_recipe.py`](examples/01_minimal_recipe.py)
- [`examples/02_nonresponse_raking.py`](examples/02_nonresponse_raking.py)
- [`examples/03_designs_estimate.py`](examples/03_designs_estimate.py)
- [`examples/04_cascade_parity.py`](examples/04_cascade_parity.py)

## Gold testing

Gold tests compare weightpipe to frozen reference outputs (and optionally live samplics / R) on the same toy inputs. CSVs live in [`tests/gold/`](tests/gold/).

**Run the gold suite** (same idea as CI):

```bash
uv sync --all-groups --extra gold --locked
uv run --extra gold pytest -m gold -q
```

Or run everything (unit tests + gold):

```bash
uv run --extra gold pytest -q
```

Frozen CSV checks always run when the files are present. Live samplics needs the `gold` extra; live R checks need R packages and optionally `uv sync --extra r-gold`.

**Regenerate frozen CSVs** (local only — not CI). Do this when a reference tool or gold scenario intentionally changes, then commit the updated files under `tests/gold/`:

```bash
uv run --extra gold python tests/gold/generate_samplics_gold.py
Rscript tests/gold/generate_r_gold.R          # needs R packages survey + weightflow
Rscript tests/gold/generate_sampler_gold.R    # needs R package sampler (or SAMPLER_R_DIR)
```

## License

MIT
