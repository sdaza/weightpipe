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

`WeightPipe` is the single entry point: describe the sample, chain any adjustments, then read weights or estimates. Weights are computed on first use, so estimating with no adjustment steps works too.

```python
from weightpipe import WeightPipe, population_totals

totals = population_totals(pop, "~ region + sex + age")

pipe = (
    WeightPipe(df, weight="pw", psu="psu", strata="stratum")
    .unknown_eligibility(unknown="unknown", by=["region"], cluster="hh")
    .drop_ineligible(ineligible="ineligible")
    .nonresponse(
        respondent="responded",
        method="propensity",
        engine="logit",  # or "gbm" / "forest"
        formula="~ region + sex",
        num_classes=5,
        cluster="hh",
    )
    .calibrate(
        method="linear",
        formula="~ region + sex + age",
        totals=totals,
        # engine="forest", population=pop,  # tree-embedding GREG (needs population microdata)
        # assist="propensity",              # add p̂ to linear calibration
        # assist="propensity_class",        # rake/calibrate while keeping class mass
    )
    .trim(max_ratio=4.0, reference="median", redistribute=True)
)

pipe.weights  # final weights
pipe.collect_weights(keep_intermediate=True)  # weights and per-step factors
pipe.diagnostics  # per-step diagnostics and alerts

pipe.estimate("y", estimand="mean", variance="jackknife")
pipe.estimate("y", estimand="ratio", denominator="x", variance="jackknife")
pipe.estimate("y", estimand="median", variance="bootstrap", replicates=200, seed=1)
```

Design weights alone are enough to estimate:

```python
WeightPipe(df, N=10_000).estimate("y", estimand="mean", variance="bootstrap", seed=1)
```

Each step returns a new pipe, so you can branch from a common base and compare.

## Sampling designs

Pass the sampling inputs; the design type is inferred automatically.

```python
WeightPipe(df)                                            # no weight → base_weight=1 (logged)
WeightPipe(df, N=10_000)                                  # SRS: w = N / n
WeightPipe(df, strata="region", N_h={"North": 5000, ...}) # stratified SRS: w = N_h / n_h
WeightPipe(df, weight="pw", psu="psu")                    # cluster
WeightPipe(df, weight="pw", psu="psu", strata="stratum")  # stratified cluster
WeightPipe(df, probabilities=["p1", "p2"], psu="psu")     # multi-stage: w = 1 / (p1 * p2)
WeightPipe(df, stage_weights=["w1", "w2"], psu="psu")     # multi-stage: w = w1 * w2
WeightPipe(df, weight="pw")                               # existing design weights
```

For multi-stage samples, stage selection is folded into the weight. Use `psu` as the ultimate cluster for variance estimation. Check what was inferred with `pipe.kind`.

## Weighting steps

Chain adjustments on a pipe, in order:

| Step | What it does |
|------|----------------|
| `unknown_eligibility` | Redistribute unknown eligibility within cells; optional household `cluster=` |
| `drop_ineligible` | Set ineligible units to weight 0 |
| `nonresponse` | Weighting-class or propensity (`engine="logit"`, `"gbm"`, or `"forest"`); optional `cluster=` |
| `calibrate` | Raking, post-stratification, or linear/GREG; `engine="forest"`/`"gbm"` for tree-embedding GREG (`population=` required); optional `assist=` (`propensity_class` auto-converts `proportions=` → absolute `margins` before attaching class totals) |
| `trim` | Cap extreme weights by ratio to median/base/value |
| `trim_weights` | Automatic Tukey or Potter trimming |

## Estimation

```python
pipe.estimate("y", estimand="mean", variance="jackknife")
pipe.estimate("employed", estimand="proportion", variance="bootstrap", replicates=200, seed=1)
pipe.estimate("y", estimand="total", variance="jackknife")
pipe.estimate("y", estimand="ratio", denominator="x", variance="jackknife")
pipe.estimate("y", estimand="median", variance="jackknife")
```

Supported estimands: `mean`, `total`, `proportion`, `ratio`, `median`.  
Variance options: `bootstrap` (Rao–Wu; default) or `jackknife` (delete-a-PSU).

## Sample-size planning

Plan sample sizes before fieldwork, then build a pipe after data collection:

```python
from weightpipe import (
    WeightPipe,
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
pipe = WeightPipe(df, strata="region", N_h=populations)
```

Allocation methods: `mixed`, `root`, `neyman`, `stdev`, and `error`.

## Logging

weightpipe stays silent unless you ask for messages. Turn them on to see notes such as a missing design weight:

```python
import weightpipe as wp

wp.setup_logging("INFO")  # 2026-08-17 12:03:13  No design weight provided; ...
wp.set_log_level("WARNING")  # quieter
```

Messages go to `stderr` through the `weightpipe` logger, so your own logging config keeps working if you'd rather configure it yourself.

## Lower-level API

`WeightPipe` wraps two objects you can also use directly: `Design` (sampling inputs and base weights) and `Recipe` (the adjustment steps, run with `prep()`). Reach for them when you want to hold a design or an unfitted recipe on its own, for example to build replicate weights by hand.

```python
from weightpipe import Design, Recipe, estimate

design = Design(df, weight="pw", psu="psu", strata="stratum")
recipe = Recipe.from_design(design).step_calibrate(method="raking", proportions=props)
fitted = recipe.prep()
estimate(recipe, "y", estimand="mean", fitted=fitted, variance="jackknife")
```

## Examples

Interactive scripts (run cell-by-cell or with Python):

- [`examples/01_minimal_recipe.py`](examples/01_minimal_recipe.py) — design weights and bootstrap estimates
- [`examples/02_nonresponse_raking.py`](examples/02_nonresponse_raking.py) — NR + raking
- [`examples/03_designs_estimate.py`](examples/03_designs_estimate.py) — SRS / stratified / cluster / multi-stage
- [`examples/04_cascade_parity.py`](examples/04_cascade_parity.py) — full cascade + jackknife

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
