# weightpipe

[![CI](https://github.com/sdaza/weightpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/sdaza/weightpipe/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/sdaza/weightpipe?label=release)](https://github.com/sdaza/weightpipe/releases)
[![PyPI](https://img.shields.io/pypi/v/weightpipe?label=pypi)](https://pypi.org/project/weightpipe/)
[![Python >=3.11](https://img.shields.io/badge/Python-%3E%3D3.11-blue)](https://pypi.org/project/weightpipe/)
[![Monthly downloads](https://static.pepy.tech/badge/weightpipe/month)](https://pepy.tech/project/weightpipe)
[![Stars](https://img.shields.io/github/stars/sdaza/weightpipe?label=stars)](https://github.com/sdaza/weightpipe/stargazers)
[![License](https://img.shields.io/pypi/l/weightpipe)](https://github.com/sdaza/weightpipe/blob/main/LICENSE)

Survey weighting in Python: define a sampling design, apply a weighting recipe, then estimate means, totals, proportions, ratios, medians, and design-based GLMs with bootstrap, jackknife, or linearized standard errors.

The point is one API for the whole path — design, eligibility, nonresponse, calibration, trim, estimates, and sample-size planning — instead of stitching specialized raking, weighting, and variance packages together.

## Install

Requires Python 3.11+.

### From PyPI (recommended)

```bash
pip install weightpipe
# or
uv add weightpipe
```

### From GitHub (latest `main`)

```bash
pip install "git+https://github.com/sdaza/weightpipe.git"
# or
uv add "git+https://github.com/sdaza/weightpipe.git"
```

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

pipe.estimate.mean(["income", "food_share"], by="urban_rural")
pipe.estimate.ratio(["food", "miles"], ["income", "trips"], by="urban_rural")
pipe.estimate.median("y", variance="bootstrap", replicates=200, seed=1)
pipe.estimate.glm("employed ~ region + age", family="binomial", variance="linearization")
```

Design weights alone are enough to estimate:

```python
WeightPipe(df, N=10_000).estimate.mean("y", variance="bootstrap", seed=1)
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

Check what was inferred with `pipe.kind`.

### Multi-stage clusters (one `psu`)

You can **draw** several nested stages in the field. You still name **one** `psu` for variance: the first-stage unit (the ultimate cluster). Later stages go into the weight, not into extra PSU columns.

Example: sample **schools**, then **classes** inside those schools, then **students** inside those classes. That is a three-stage cluster sample.

| Stage | Unit | What it is for |
|-------|------|----------------|
| 1 | School | PSU — independent draws; use this in `psu=` |
| 2 | Class | Nested in school — inclusion goes in the weight |
| 3 | Student | Row in the data — inclusion goes in the weight |

```python
# w = 1 / (p_school * p_class * p_student); SEs resample schools
WeightPipe(
    df,
    probabilities=["p_school", "p_class", "p_student"],
    psu="school",
    # strata="district",  # if schools were drawn within strata
)
```

If you already have per-stage weights (`w_k = 1/π_k`), use `stage_weights=["w_school", "w_class", "w_student"]` with the same `psu="school"`. If the product is already in one column, `WeightPipe(df, weight="pw", psu="school")` is enough for variance (`kind` is then `"cluster"` rather than `"multistage"`).

**Why `psu` is school, not class.** Classes are selected independently *inside a sampled school*, not from a national list of all classes. A class can appear only if its school was selected first; two classes in the same school share that school draw. Bootstrap and jackknife therefore resample **whole schools** (every selected class and student in that school moves together). Setting `psu` to class would treat classes as independent first-stage units and understate SEs; setting it to student would treat the sample like an SRS of students.

`strata=` is separate (for example district). That is the groups *within which* schools were drawn, not a second PSU.

If you take every student in the selected classes, stage 3 has π = 1 and the sample is two-stage (schools → classes). `psu` is still `"school"`.

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

After calibrate, step diagnostics include a tidy `margin_table` (target vs achieved). You can also check weighted category margins anytime:

```python
pipe.margins("sex")  # current weighted totals / proportions
pipe.margins(targets="calibrate")  # reuse last calibrate targets
pipe.margins("sex", proportions={"sex": {"M": 0.5, "F": 0.5}})
pipe.diagnostics["steps"]["calibrate"]["margin_table"]
```

Covariate **balance** compares base vs final weights to population targets via standardized mean differences (SMD), for continuous and categorical covariates:

```python
report = pipe.balance(
    ["age", "sex", "region"],
    means={"age": 40.0},
    proportions={"sex": {"M": 0.5, "F": 0.5}, "region": {"N": 0.5, "S": 0.5}},
)
report.table  # before / after / target / smd_* / balanced
report.summary  # max |SMD|, n_imbalanced, ESS before/after
# or: pipe.balance(["age", "sex"], target=pop_df, target_weight="N")
```

## Estimation

Pass one variable or a list. `by=` splits into domains (same idea as R `svyby` or svy's `sample.estimation.mean(..., by=...)`). Replicate weights are built once and reused.

```python
pipe.estimate.mean(["income", "food_share"], by="urban_rural")
pipe.estimate.mean("y", variance="jackknife")
pipe.estimate.proportion("employed", by="region", variance="bootstrap", replicates=200, seed=1)
pipe.estimate.total("y", variance="linearization")
pipe.estimate.ratio("y", "x", variance="jackknife")
pipe.estimate.ratio(["food", "housing"], "income")  # one denominator
pipe.estimate.ratio(["food", "miles"], ["income", "trips"], by="urban_rural")  # paired
pipe.estimate.median("y", variance="jackknife")

# still valid
pipe.estimate("y", estimand="mean", variance="jackknife")
pipe.estimation.mean("y")  # alias of estimate
```

The result is one row per variable × domain, with `estimate`, `se`, `cv`, and a confidence interval.

Supported estimands: `mean`, `total`, `proportion`, `ratio`, `median`.  
Variance: `bootstrap` (Rao–Wu; default) and `jackknife` (delete-a-PSU) re-run the recipe on each replicate so SEs include estimated weights. `linearization` is the fast ultimate-cluster Taylor SE that treats the fitted weights as fixed (survey-style). Median has no linearized SE.

### Design-based GLM

`pipe.estimate.glm` is `survey::svyglm`-style regression: survey-weighted IRLS for the coefficients, then design-based SEs. It is not a `statsmodels` / sklearn wrapper with weights passed through.

```python
pipe.estimate.glm("employed ~ region + age", family="binomial", variance="linearization")
pipe.estimate.glm("y ~ region", family="gaussian", variance="jackknife")
pipe.estimate.glm("count ~ age", family="poisson")
pipe.estimate.glm("y ~ 1", family="gaussian", variance="linearization")  # intercept = Hájek mean
```

| `family=` | Link | Outcome |
|-----------|------|---------|
| `gaussian` (`normal`) | identity | continuous |
| `binomial` (`logit`, `logistic`, `quasibinomial`) | logit | 0/1 |
| `poisson` (`log`, `quasipoisson`) | log | counts |

The result is one row per coefficient (`term`), with the same `estimate` / `se` / `cv` / CI columns as other estimands. Categorical predictors use sorted levels and drop the first (R treatment contrasts with alphabetical levels). An intercept-only gaussian matches `pipe.estimate.mean`; an intercept-only binomial is `logit` of the weighted proportion, not `svymean`.

`variance=` is the same as for means: `linearization` is the Binder sandwich (ultimate-cluster, weights fixed); `bootstrap` / `jackknife` re-fit β on recipe-aware replicate weights. Gold matches `survey::svyglm`. There is no `by=` on GLM, and no tabs / Rao–Scott.

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
from weightpipe import Design, Recipe, estimate, estimate_glm

design = Design(df, weight="pw", psu="psu", strata="stratum")
recipe = Recipe.from_design(design).step_calibrate(method="raking", proportions=props)
fitted = recipe.prep()
estimate(recipe, ["y", "x"], by="region", fitted=fitted, variance="jackknife")
estimate_glm(recipe, "employed ~ region", family="binomial", fitted=fitted, variance="linearization")
```

## Examples

Interactive scripts (run cell-by-cell or with Python):

- [`examples/01_minimal_recipe.py`](examples/01_minimal_recipe.py) — design weights and bootstrap estimates
- [`examples/02_nonresponse_raking.py`](examples/02_nonresponse_raking.py) — NR + raking
- [`examples/03_designs_estimate.py`](examples/03_designs_estimate.py) — SRS / stratified / cluster / multi-stage; `estimate.mean(..., by=)`; design-based GLM
- [`examples/04_cascade_parity.py`](examples/04_cascade_parity.py) — full cascade + jackknife
- [`examples/05_balance.py`](examples/05_balance.py) — covariate balance (SMD before/after)

## Comparisons

weightpipe is meant to **integrate** the steps you usually assemble from several tools, and to make that cascade **easier** in Python: one `WeightPipe`, in order, with diagnostics and estimates at the end.

Specialized packages still do one slice well.

**Raking / IPF only.** [Weightipy](https://pypi.org/project/weightipy/0.4.2/) 0.4.2 is a fast RIM engine (dicts or census tables, nested/segmented RIM, Kish efficiency). [ipfn](https://pypi.org/project/ipfn/) is a small Python IPF helper. R [`anesrake`](https://CRAN.R-project.org/package=anesrake) is ANES-style raking. [Quantipy3](https://github.com/Quantipy/quantipy3) is the market-research stack Weightipy was forked from.

**Weighting recipes.** R [`weightflow`](https://CRAN.R-project.org/package=weightflow) is the closest recipe-style analogue (eligibility → NR → calibrate → trim, plus recipe-aware bootstrap/jackknife). R [`icarus`](https://CRAN.R-project.org/package=icarus) and [`ReGenesees`](https://github.com/DiegoZardetto/ReGenesees) are calibration-focused (raking, linear/GREG, official-statistics workflows).

**Design-based analysis.** The current Python peer is [svy](https://svylab.com/svy) ([docs](https://svylab.com/docs/svy/)): one `svy.Design` + `svy.Sample`, then `.estimation`, `.weighting`, `.glm`, tabs, and BRR/SDR. It is the successor of archived [samplics](https://pypi.org/project/samplics/) (same author) and is validated against R [`survey`](https://CRAN.R-project.org/package=survey). In R, `survey` is still the usual toolkit once weights exist (`svydesign`, `svymean`, `svyby`, `svyglm`, `calibrate`, `rake`, `as.svrepdesign`); [`srvyr`](https://CRAN.R-project.org/package=srvyr) is a dplyr front end on it.

weightpipe is the recipe-first path: eligibility → NR → calibrate → trim on one `WeightPipe`, then the same estimate shape (`pipe.estimate.mean(["income", "food_share"], by="urban_rural")`, `pipe.estimate.glm("y ~ x", family="binomial")`). Use svy or R `survey` when you need sample *selection*, categorical tests, BRR/SDR, or small-area estimation ([`svy-sae`](https://svylab.com/svy)).

**Planning and selection.** [svy](https://svylab.com/svy) draws probability samples (SRS, systematic, PPS, multi-stage) and plans sizes. R [`sampler`](https://github.com/sdaza/sampler) is weightpipe's planning gold. R [`PracTools`](https://CRAN.R-project.org/package=PracTools), [`surveyplanning`](https://CRAN.R-project.org/package=surveyplanning), and [`sampling`](https://CRAN.R-project.org/package=sampling) (Tillé) are other R kits. weightpipe plans sizes, then you attach the collected microdata — it does not draw the field sample.

**Diagnostics.** Meta [`balance`](https://github.com/facebookresearch/balance) is covariate balance (SMD / Love plots), not a weighting recipe. weightpipe's `balance()` is the SMD before/after check; `margins()` checks calibration targets.

Marks mean the package has a first-class API for that row. "via `survey`" means weightflow builds the weights (or replicate weights) and you estimate in `survey` / `srvyr`.

| | weightpipe | svy | Weightipy 0.4.2 | R `survey` | R `weightflow` | R `sampler` |
|--|:----------:|:---:|:---------------:|:----------:|:--------------:|:-----------:|
| Language | Python | Python | Python | R | R | R |
| Design object (strata / PSU / weights) | ✓ | ✓ | | ✓ | base weights¹ | |
| Multi-stage inclusion weights | ✓ | ✓ | | ✓ | | |
| Sample selection (draw PPS / SRS) | | ✓ | | | | |
| Unknown eligibility / drop ineligible | ✓ | | | | ✓ | |
| Nonresponse | class + propensity | class | | | class + propensity | |
| Raking (RIM / IPF) | ✓ | ✓ | ✓ | ✓ | ✓ | |
| Nested / segmented RIM | | | ✓ | | | |
| Post-stratification | ✓ | ✓ | | ✓ | ✓ | |
| Linear / GREG calibration | ✓ | ✓ | | ✓ | ✓ | |
| Trim | ✓ | ✓ | | ✓ | ✓ | |
| One chained recipe | ✓ | fluent steps | raking only | | ✓ | |
| Recipe-aware replicate weights | ✓ | ✓ | | | ✓ | |
| Taylor linearization SE | ✓ | ✓ | | ✓ | via `survey` | |
| Bootstrap / jackknife SE | ✓ | ✓ | | ✓ | ✓ | |
| BRR / SDR | | ✓ | | ✓ | | |
| Domain estimates (`by=`) | ✓ | ✓ | | ✓ | via `survey` | |
| Several variables at once | ✓ | ✓ | | ✓ | via `survey` | |
| Design-based GLM | ✓ | ✓ | | ✓ | via `survey` | |
| Tabs / Rao–Scott | | ✓ | | ✓ | via `survey` | |
| Covariate balance (SMD) | ✓ | | | | | |
| Sample-size planning / allocation | ✓ | ✓ | | | | ✓ |
| Small-area estimation | | svy-sae | | | | |

¹ weightflow takes design weights you already computed (`weighting_spec(..., base_weights=)`). It uses strata/PSU when it resamples replicates. It does not infer SRS / stratified / cluster from `N` / `N_h` the way `WeightPipe` / `svy.Design` / `survey::svydesign` do.

Weightipy stays a focused raking library. weightpipe uses the same class of iterative raking as one `calibrate(method="raking")` step, then continues with eligibility, nonresponse, GREG, trim, balance, estimates (including design-based GLM), and planning. [svy](https://svylab.com/svy) and R `survey` remain broader for sample selection, categorical tests, tabs, BRR/SDR, and SAE.

### Numerical gold

Shared methods are checked on the same toy frames against frozen CSVs in [`tests/gold/`](tests/gold/) (CI) and optionally live R / [svy](https://svylab.com/svy). These are **correctness** checks, not runtime speed benchmarks.

| Method | R `survey` | R `weightflow` | svy² | R `sampler` |
|--------|:----------:|:--------------:|:----:|:-----------:|
| Unknown eligibility | | ✓ | | |
| Drop ineligible | | ✓ | | |
| Weighting-class NR | | ✓ | ✓ | |
| Raking | ✓ | ✓ | ✓ | |
| Post-stratification | ✓ | ✓ | ✓ | |
| Linear / GREG | ✓ | ✓ | ✓ | |
| Trim (value cap, no redistribute) | | ✓ | | |
| NR → raking cascade | | ✓ | | |
| Full cascade (eligibility → trim) | | ✓ | | |
| Mean, total, ratio, median | ✓ | | | |
| Design-based GLM | ✓ | | | |
| Sample size / allocation / MOE | | | | ✓ |

² Frozen `*_svy.csv` plus live checks from `uv sync --extra gold`.

Tolerances are tight (`1e-12`–`1e-6` depending on the solver). Median vs `survey::svyquantile` may differ by one unique *y* value because quantile definitions can differ. Design-based GLM coefficients and linearized SEs match `survey::svyglm` (gaussian / quasibinomial / quasipoisson). Logit propensity uses the same `1/p` adjustment as weightflow, but sklearn vs R `glm` coefficients are not bit-matched.

## Gold testing

Gold tests compare weightpipe to frozen reference outputs (and optionally live svy / R) on the same toy inputs. CSVs live in [`tests/gold/`](tests/gold/). Python weighting gold is [svy](https://svylab.com/svy); estimands and GLM gold are R `survey`.

**Run the gold suite** (same idea as CI):

```bash
uv sync --all-groups --extra gold --locked
uv run --extra gold pytest -m gold -q
```

Or run everything (unit tests + gold):

```bash
uv run --extra gold pytest -q
```

Frozen CSV checks always run when the files are present. Live svy checks need the `gold` extra. Live R checks need R packages and optionally `uv sync --extra r-gold`.

**Regenerate frozen CSVs** (local only — not CI). Do this when a reference tool or gold scenario intentionally changes, then commit the updated files under `tests/gold/`:

```bash
uv run --extra gold python tests/gold/generate_svy_gold.py
Rscript tests/gold/generate_r_gold.R          # needs R packages survey + weightflow
Rscript tests/gold/generate_sampler_gold.R    # needs R package sampler (or SAMPLER_R_DIR)
```

## License

MIT
