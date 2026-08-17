# %%
# Full cascade: unknown eligibility → NR → calibrate → trim + jackknife SE/CI
# Run cells interactively, or: `uv run python examples/04_cascade_parity.py`

import numpy as np
import pandas as pd

from weightpipe import WeightPipe, population_totals

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

pop = pd.DataFrame(
    {
        "region": ["North"] * 200 + ["South"] * 200,
        "sex": (["M", "F"] * 200),
        "age": [42.0] * 400,
    }
)
totals = population_totals(pop, "~ region + sex + age")

pipe = (
    WeightPipe(df, weight="pw", psu="psu", strata="stratum")
    .options(min_cell_n=5, max_factor=5.0, warn=False)
    .unknown_eligibility(unknown="unknown", by=["region"])
    .drop_ineligible(ineligible="ineligible")
    .nonresponse(
        respondent="responded",
        method="propensity",
        engine="logit",
        formula="~ region + sex",
        num_classes=4,
        weight_model=True,
    )
    .calibrate(method="linear", formula="~ region + sex + age", totals=totals)
    .trim(max_ratio=4.0, reference="median", redistribute=True)
    # Alternatives:
    # .calibrate(..., bounds=(0.3, 3.0))
    # .calibrate(..., penalty=10.0)
    # .trim_weights(method="tukey")
)
print(pipe.table().head())
print("steps:", pipe.diagnostics["steps_applied"])

# %%
print("bootstrap mean")
print(
    pipe.estimate("y", estimand="mean", variance="bootstrap", replicates=80, seed=1)
    .round(3)
    .to_string(index=False)
)
print("jackknife mean")
print(pipe.estimate("y", estimand="mean", variance="jackknife").round(3).to_string(index=False))
print("jackknife proportion employed")
print(
    pipe.estimate("employed", estimand="proportion", variance="jackknife")
    .round(3)
    .to_string(index=False)
)
