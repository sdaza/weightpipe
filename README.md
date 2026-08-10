# weightpipe

Declarative survey weighting recipes with **recipe-aware replicate weights**, diagnostics, and bootstrap SE/CIs.

Methods ship behind a validation gate: analytical/composition/recovery tests must pass before a public `Recipe.step_*()` is ungated.

## Package management (uv)

```bash
uv sync --all-groups
uv run pytest
uv run ruff check
uv run python examples/01_minimal_recipe.py
uv run python examples/02_nonresponse_raking.py
```

CI runs `uv sync --all-groups --locked` — commit `uv.lock`.

GitHub Actions [`.github/workflows/ci.yml`](.github/workflows/ci.yml): ruff check/format + pytest on Python 3.11/3.12 (push/PR). PyPI publish is left for a separate workflow.

Interactive examples use `# %%` cells (Cursor/VS Code **Run Cell**):

- [`examples/01_minimal_recipe.py`](examples/01_minimal_recipe.py)
- [`examples/02_nonresponse_raking.py`](examples/02_nonresponse_raking.py)

## Quickstart (Iteration 1)

```python
from weightpipe import (
    Recipe,
    collect_weights,
    design_effect,
    bootstrap_weights,
    boot_mean,
)

recipe = (
    Recipe(df, base_weight="pw")
    .step_drop_ineligible(ineligible="ineligible")
    .step_nonresponse(respondent="responded", method="weighting_class", by=["region"])
    .step_calibrate(
        method="raking",
        proportions={"sex": {"M": 0.48, "F": 0.52}, "region": {"N": 0.3, "S": 0.7}},
        # population_size=10200,  # optional; default scales shares to sum(active weights)
    )
)
fitted = recipe.prep(min_cell_n=30, max_factor=2.5)
out = collect_weights(fitted, keep_intermediate=True)
print(design_effect(fitted))

boot = bootstrap_weights(recipe, replicates=200, strata="stratum", psu="psu", seed=1)
print(boot_mean(boot, "y"))  # estimate, se, ci_lower, ci_upper
```

## Validation policy

Do **not** loosen recovery tolerances to green CI. Diagnose failures first.
