# %% [markdown]
# Nonresponse + raking + estimate SE/CI
#
# Open in Cursor/VS Code and use **Run Cell** on each `# %%` block.
# Or: `uv run python examples/02_nonresponse_raking.py`

# %%
import pandas as pd

from weightpipe import WeightPipe, design_effect, weight_factors

# %%
# Toy survey microdata
df = pd.DataFrame(
    {
        "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "psu": [1, 1, 2, 2, 3, 3, 4, 4],
        "region": ["N", "N", "N", "N", "S", "S", "S", "S"],
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
        "ineligible": [0, 0, 0, 0, 0, 0, 0, 0],
        "responded": [1, 0, 1, 1, 1, 0, 1, 1],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "pw": [1.0] * 8,
    }
)
df

# %%
# Population distributions (shares must sum to 1 within each variable)
proportions = {
    "sex": {"M": 0.5, "F": 0.5},
    "region": {"N": 0.5, "S": 0.5},
}
proportions

# %%
# Chain adjustments on one pipe (nothing computed until weights / estimate)
pipe = (
    WeightPipe(df, weight="pw", psu="psu", strata="stratum")
    .options(min_cell_n=1, max_factor=None, warn=False)
    .drop_ineligible(ineligible="ineligible")
    .nonresponse(respondent="responded", method="weighting_class", by=["region"])
    .calibrate(
        method="raking",
        proportions=proportions,
        # population_size=80.0,  # optional absolute N; default = sum(active weights)
        max_iter=50,
        tol=1e-6,
    )
)
pipe

# %%
weighted = pipe.table(keep_intermediate=True, drop_zero=True)

# %%
print("active units =", len(weighted))
print("sum(weight) =", round(float(weighted["weight"].sum()), 3))
print("Kish deff =", round(design_effect(pipe.result), 3))
print("alerts =", pipe.alerts)
print("factor columns:", list(weight_factors(pipe.result).columns))
calib = pipe.diagnostics["steps"]["calibrate"]
print("raking total_source =", calib.get("total_source"), "total =", calib.get("total"))
weighted

# %%
# Stage adjustment factors
factors = weight_factors(pipe.result)
factors

# %%
# Recipe-aware bootstrap SE/CI
print(pipe.estimate("y", estimand="mean", variance="bootstrap", replicates=100, seed=42).round(3).to_string(index=False))
print(pipe.estimate("y", estimand="total", variance="bootstrap", replicates=100, seed=42).round(3).to_string(index=False))
