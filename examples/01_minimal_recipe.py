# %% [markdown]
# Minimal recipe — base weights + bootstrap SE/CI
#
# Open in Cursor/VS Code and use **Run Cell** on each `# %%` block.
# Or: `uv run python examples/01_minimal_recipe.py`

# %%
import pandas as pd

from weightpipe import (
    Recipe,
    boot_mean,
    boot_total,
    bootstrap_weights,
    collect_weights,
    design_effect,
)

# %%
# Toy sample
df = pd.DataFrame(
    {
        "stratum": ["A", "A", "B", "B", "A", "A", "B", "B"],
        "psu": [1, 2, 3, 4, 1, 2, 3, 4],
        "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
        "pw": [1.0, 1.2, 0.8, 1.1, 1.0, 0.9, 1.3, 1.0],
    }
)
df

# %%
# Define inert recipe and prep (base weights only)
recipe = Recipe(df, base_weight="pw")
fitted = recipe.prep()
out = collect_weights(fitted)
print("n =", len(out), "sum(w) =", round(float(out["weight"].sum()), 3))
print("Kish deff =", round(design_effect(fitted), 3))
out

# %%
# Recipe-aware bootstrap: SE and 95% CI for mean and total
boot = bootstrap_weights(recipe, replicates=200, strata="stratum", psu="psu", seed=1)
mean_ci = boot_mean(boot, "y")
total_ci = boot_total(boot, "y")
print("boot_mean(y)")
print(mean_ci.round(3).to_string(index=False))
print("boot_total(y)")
print(total_ci.round(3).to_string(index=False))
mean_ci
