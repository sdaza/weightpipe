# %% [markdown]
# Minimal WeightPipe — base weights + bootstrap SE/CI
#
# Open in Cursor/VS Code and use **Run Cell** on each `# %%` block.
# Or: `uv run python examples/01_minimal_recipe.py`

# %%
import pandas as pd

from weightpipe import WeightPipe, design_effect

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
# One object: sampling inputs + (optional) steps. Weights compute on first use.
pipe = WeightPipe(df, weight="pw", psu="psu", strata="stratum")
out = pipe.collect_weights()
print("n =", len(out), "sum(w) =", round(float(out["weight"].sum()), 3))
print("Kish deff =", round(design_effect(pipe.result), 3))
out

# %%
# Estimate with recipe-aware bootstrap SE/CI
print(pipe.estimate.mean("y", variance="bootstrap", replicates=200, seed=1).round(3).to_string(index=False))
print(pipe.estimate.total("y", variance="bootstrap", replicates=200, seed=1).round(3).to_string(index=False))
