# %% [markdown]
# Sampling designs + estimate — mean / proportion / total / ratio / median / by=
#
# Open in Cursor/VS Code and use **Run Cell** on each `# %%` block.
# Or: `uv run python examples/03_designs_estimate.py`

# %%
import pandas as pd

from weightpipe import WeightPipe

# %%
# --- SRS: base weight = N / n ---
df_srs = pd.DataFrame(
    {
        "y": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        "employed": [1, 0, 1, 1, 0, 1],
    }
)
pipe_srs = WeightPipe(df_srs, N=600)
print("kind:", pipe_srs.kind, "weights:", pipe_srs.weights.unique())
pipe_srs.estimate.mean("y", replicates=100, seed=1)

# %%
pipe_srs.estimate.proportion("employed", replicates=100, seed=1)

# %%
# --- Stratified SRS: w = N_h / n_h ---
df_st = pd.DataFrame(
    {
        "stratum": ["North"] * 4 + ["South"] * 6,
        "y": [5, 6, 7, 8, 15, 16, 17, 18, 19, 20],
        "employed": [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    }
)
pipe_st = WeightPipe(df_st, strata="stratum", N_h={"North": 40, "South": 90})
print(pipe_st.collect_weights().groupby("stratum")["weight"].first())
pipe_st.estimate.mean("y", replicates=100, seed=2)

# %%
# --- Cluster / multi-stage: supply design weights; declare psu (+ optional strata) ---
df_cl = pd.DataFrame(
    {
        "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "psu": [1, 1, 2, 2, 3, 3, 4, 4],
        "urban_rural": ["urban", "urban", "rural", "rural", "urban", "urban", "rural", "rural"],
        "income": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
        "food_share": [0.40, 0.50, 0.30, 0.20, 0.45, 0.55, 0.25, 0.15],
        "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
        "x": [2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0],
        "employed": [1, 0, 1, 1, 0, 1, 1, 0],
        "pw": [2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 1.5, 1.5],
    }
)
pipe_cl = WeightPipe(df_cl, weight="pw", psu="psu", strata="stratum")
print(pipe_cl.collect_weights().head())

# %%
print("means by urban_rural (linearization)")
print(
    pipe_cl.estimate.mean(["income", "food_share"], by="urban_rural", variance="linearization")
    .round(3)
    .to_string(index=False)
)
print("mean y (bootstrap)")
print(pipe_cl.estimate.mean("y", replicates=100, seed=3).round(3).to_string(index=False))
print("proportion employed")
print(pipe_cl.estimate.proportion("employed", replicates=100, seed=3).round(3).to_string(index=False))
print("total y")
print(pipe_cl.estimate.total("y", variance="linearization").round(3).to_string(index=False))

# %%
print("ratio y/x (jackknife)")
print(
    pipe_cl.estimate.ratio("y", "x", variance="jackknife")
    .round(3)
    .to_string(index=False)
)
print("median y (jackknife)")
print(pipe_cl.estimate.median("y", variance="jackknife").round(3).to_string(index=False))

# %%
# --- Multi-stage from inclusion probabilities: w = 1/(p1*p2); psu = ultimate cluster ---
df_ms = pd.DataFrame(
    {
        "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "psu": [1, 1, 2, 2, 3, 3, 4, 4],
        "p1": [0.2] * 8,
        "p2": [0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.25, 0.25],
        "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
    }
)
pipe_ms = WeightPipe(df_ms, probabilities=["p1", "p2"], psu="psu", strata="stratum")
print("multistage kind:", pipe_ms.kind)
print("weights:", pipe_ms.weights.tolist())
print(pipe_ms.estimate.mean("y", variance="linearization").round(3).to_string(index=False))
print(pipe_ms.estimate.mean("y", variance="jackknife").round(3).to_string(index=False))
