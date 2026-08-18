# %% [markdown]
# Covariate balance before vs after weighting
#
# Open in Cursor/VS Code and use **Run Cell** on each `# %%` block.
# Or: `uv run python examples/05_balance.py`

# %%
import pandas as pd

from weightpipe import WeightPipe

# %%
# Sample over-represents males and the North relative to the targets below
df = pd.DataFrame(
    {
        "sex": ["M", "M", "M", "M", "M", "M", "F", "F"],
        "age": [22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 48.0, 52.0],
        "region": ["N", "N", "N", "N", "N", "S", "N", "S"],
        "pw": [1.0] * 8,
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    }
)
df

# %%
proportions = {
    "sex": {"M": 0.5, "F": 0.5},
    "region": {"N": 0.5, "S": 0.5},
}
pipe = WeightPipe(df, weight="pw").calibrate(
    method="raking",
    proportions=proportions,
    population_size=100.0,
    max_iter=100,
    tol=1e-10,
)
pipe

# %%
# Balance vs explicit targets (means for continuous, proportions for categorical)
report = pipe.balance(
    ["age", "sex", "region"],
    means={"age": 40.0},
    proportions=proportions,
    sds={"age": 12.0},
    threshold=0.1,
)
print(report)
print(pd.Series(report.summary).round(3).to_string())
report.table.round(3)

# %%
# Same idea with population microdata (cell sizes as target_weight)
pop = pd.DataFrame(
    {
        "sex": ["M", "M", "F", "F"],
        "age": [30.0, 50.0, 30.0, 50.0],
        "region": ["N", "S", "N", "S"],
        "N": [25.0, 25.0, 25.0, 25.0],
    }
)
report_pop = pipe.balance(["sex", "age", "region"], target=pop, target_weight="N")
print(report_pop)
report_pop.table.round(3)
