# %% [markdown]
# Nonresponse + raking (proportions) + bootstrap SE/CI
#
# Open in Cursor/VS Code and use **Run Cell** on each `# %%` block.
# Or: `uv run python examples/02_nonresponse_raking.py`

# %%
import pandas as pd

from weightpipe import (
    Recipe,
    boot_mean,
    boot_total,
    bootstrap_weights,
    collect_weights,
    design_effect,
    weight_factors,
)

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
# Build inert recipe (nothing computed yet)
recipe = (
    Recipe(df, base_weight="pw")
    .step_drop_ineligible(ineligible="ineligible")
    .step_nonresponse(respondent="responded", method="weighting_class", by=["region"])
    .step_calibrate(
        method="raking",
        proportions=proportions,
        # population_size=80.0,  # optional absolute N; default = sum(active weights)
        max_iter=50,
        tol=1e-6,
    )
)
recipe.to_dict()

# %%
# prep() estimates the cascade; collect_weights() returns a DataFrame
fitted = recipe.prep(min_cell_n=1, max_factor=None, warn=False)
weighted = collect_weights(fitted, keep_intermediate=True, drop_zero=True)

print("active units =", len(weighted))
print("sum(weight) =", round(float(weighted["weight"].sum()), 3))
print("Kish deff =", round(design_effect(fitted), 3))
print("alerts =", fitted.alerts)
print("factor columns:", list(weight_factors(fitted).columns))
calib = fitted.diagnostics["steps"]["calibrate"]
print("raking total_source =", calib.get("total_source"), "total =", calib.get("total"))
weighted

# %%
# Stage adjustment factors
factors = weight_factors(fitted)
factors

# %%
# Recipe-aware bootstrap SE/CI (re-runs full recipe per replicate)
boot = bootstrap_weights(
    recipe, replicates=100, strata="stratum", psu="psu", seed=42, point=fitted
)
mean_ci = boot_mean(boot, "y")
total_ci = boot_total(boot, "y")
print("boot_mean(y)")
print(mean_ci.round(3).to_string(index=False))
print("boot_total(y)")
print(total_ci.round(3).to_string(index=False))
