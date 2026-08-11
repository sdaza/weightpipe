# %% [markdown]
# Sampling designs + estimate() — mean / proportion / total with SE/CI
#
# Open in Cursor/VS Code and use **Run Cell** on each `# %%` block.
# Or: `uv run python examples/03_designs_estimate.py`

# %%
import pandas as pd

from weightpipe import Design, Recipe, collect_weights, estimate

# %%
# --- SRS: base weight = N / n ---
df_srs = pd.DataFrame(
    {
        "y": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        "employed": [1, 0, 1, 1, 0, 1],
    }
)
design_srs = Design.srs(df_srs, N=600)
recipe_srs = Recipe.from_design(design_srs)
print("SRS weights:", design_srs.data[design_srs.weight].unique())
estimate(recipe_srs, "y", estimand="mean", replicates=100, seed=1)

# %%
estimate(recipe_srs, "employed", estimand="proportion", replicates=100, seed=1)

# %%
# --- Stratified SRS: w = N_h / n_h ---
df_st = pd.DataFrame(
    {
        "stratum": ["North"] * 4 + ["South"] * 6,
        "y": [5, 6, 7, 8, 15, 16, 17, 18, 19, 20],
        "employed": [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    }
)
design_st = Design.stratified(df_st, stratum="stratum", N_h={"North": 40, "South": 90})
recipe_st = Recipe.from_design(design_st)
print(design_st.data.groupby("stratum")[design_st.weight].first())
estimate(recipe_st, "y", estimand="mean", replicates=100, seed=2)

# %%
# --- Cluster / multi-stage: supply design weights; declare psu (+ optional strata) ---
df_cl = pd.DataFrame(
    {
        "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "psu": [1, 1, 2, 2, 3, 3, 4, 4],
        "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
        "x": [2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0],
        "employed": [1, 0, 1, 1, 0, 1, 1, 0],
        "pw": [2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 1.5, 1.5],
    }
)
design_cl = Design.cluster(df_cl, weight="pw", psu="psu", strata="stratum")
recipe_cl = (
    Recipe.from_design(design_cl)
    # optional adjustments still compose on top of design weights
)

fitted = recipe_cl.prep()
print(collect_weights(fitted).head())

# %%
print("mean")
print(estimate(recipe_cl, "y", estimand="mean", fitted=fitted, replicates=100, seed=3).round(3).to_string(index=False))
print("proportion employed")
print(
    estimate(recipe_cl, "employed", estimand="proportion", fitted=fitted, replicates=100, seed=3)
    .round(3)
    .to_string(index=False)
)
print("total y")
print(estimate(recipe_cl, "y", estimand="total", fitted=fitted, replicates=100, seed=3).round(3).to_string(index=False))

# %%
print("ratio y/x (jackknife)")
print(
    estimate(recipe_cl, "y", estimand="ratio", denominator="x", fitted=fitted, variance="jackknife")
    .round(3)
    .to_string(index=False)
)
print("median y (jackknife)")
print(
    estimate(recipe_cl, "y", estimand="median", fitted=fitted, variance="jackknife")
    .round(3)
    .to_string(index=False)
)
