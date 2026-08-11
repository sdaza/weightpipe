# %%
# Full cascade: unknown eligibility → NR → calibrate → trim + jackknife SE/CI
# Run cells interactively, or: `uv run python examples/04_cascade_parity.py`

import numpy as np
import pandas as pd

from weightpipe import Design, Recipe, collect_weights, estimate, population_totals

# %%
rng = np.random.default_rng(7)
n = 40
df = pd.DataFrame(
    {
        "stratum": np.repeat(["A", "B"], n // 2),
        "psu": np.repeat(np.arange(8), n // 8),
        "region": rng.choice(["North", "South"], size=n),
        "sex": rng.choice(["M", "F"], size=n),
        "age": rng.normal(40, 12, size=n).clip(18, 80),
        "unknown": (rng.random(n) < 0.1).astype(int),
        "ineligible": (rng.random(n) < 0.05).astype(int),
        "responded": (rng.random(n) < 0.75).astype(int),
        "y": rng.normal(50, 10, size=n),
        "employed": rng.integers(0, 2, size=n),
        "pw": rng.uniform(1.5, 3.0, size=n),
    }
)
# Ensure at least some response contrast
df.loc[0, "responded"] = 1
df.loc[1, "responded"] = 0

design = Design.cluster(df, weight="pw", psu="psu", strata="stratum")
pop = pd.DataFrame(
    {
        "region": ["North"] * 200 + ["South"] * 200,
        "sex": (["M", "F"] * 200),
        "age": [42.0] * 400,
    }
)
totals = population_totals(pop, "~ region + sex + age")

recipe = (
    Recipe.from_design(design)
    .step_unknown_eligibility(unknown="unknown", by=["region"])
    .step_drop_ineligible(ineligible="ineligible")
    .step_nonresponse(
        respondent="responded",
        method="propensity",
        engine="logit",
        formula="~ region + sex",
        num_classes=4,
        weight_model=True,
    )
    .step_calibrate(method="linear", formula="~ region + sex + age", totals=totals)
    .step_trim(max_ratio=4.0, reference="median", redistribute=True)
    # Alternatives:
    # .step_calibrate(..., bounds=(0.3, 3.0))
    # .step_calibrate(..., penalty=10.0)
    # .step_trim_weights(method="tukey")
)
fitted = recipe.prep(min_cell_n=5, max_factor=5.0, warn=False)
print(collect_weights(fitted).head())
print("ESS-ish n / deff path done; steps:", fitted.diagnostics["steps_applied"])

# %%
print("bootstrap mean")
print(
    estimate(recipe, "y", estimand="mean", fitted=fitted, variance="bootstrap", replicates=80, seed=1)
    .round(3)
    .to_string(index=False)
)
print("jackknife mean")
print(estimate(recipe, "y", estimand="mean", fitted=fitted, variance="jackknife").round(3).to_string(index=False))
print("jackknife proportion employed")
print(
    estimate(recipe, "employed", estimand="proportion", fitted=fitted, variance="jackknife")
    .round(3)
    .to_string(index=False)
)
