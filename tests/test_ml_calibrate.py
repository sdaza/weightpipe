"""Tests for forest/GBM model-assisted linear calibration."""

import numpy as np
import pandas as pd
import pytest

from weightpipe.methods.design_matrix import population_totals
from weightpipe.methods.ml_calibrate import ml_linear_calibrate
from weightpipe.recipe import Recipe


def _sample_pop(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pop = pd.DataFrame(
        {
            "region": rng.choice(["N", "S"], size=400),
            "sex": rng.choice(["M", "F"], size=400),
            "age": rng.normal(40, 12, size=400),
        }
    )
    # Draw a simple sample from population rows
    idx = rng.choice(len(pop), size=80, replace=False)
    sample = pop.iloc[idx].reset_index(drop=True).copy()
    sample["pw"] = np.full(len(sample), len(pop) / len(sample))
    return sample, pop


@pytest.mark.parametrize("engine", ["forest", "gbm"])
def test_ml_linear_calibrate_matches_population_size(engine: str) -> None:
    sample, pop = _sample_pop(seed=1 if engine == "forest" else 2)
    w, fac, diag = ml_linear_calibrate(
        sample["pw"],
        sample,
        formula="~ region + sex + age",
        population=pop,
        engine=engine,  # type: ignore[arg-type]
        n_estimators=20,
        max_depth=2,
        seed=1,
        warn=False,
    )
    assert diag["engine"] == engine
    assert diag["converged"] is True or diag["solved"] is True
    assert float(w.sum()) == pytest.approx(float(len(pop)), rel=0.05, abs=1.0)
    assert (fac > 0).all()


@pytest.mark.parametrize("engine", ["forest", "gbm"])
def test_ml_calibrate_in_recipe(engine: str) -> None:
    sample, pop = _sample_pop(seed=3)
    totals = population_totals(pop, "~ region + sex + age")
    fitted = (
        Recipe(sample, base_weight="pw")
        .step_calibrate(
            method="linear",
            engine=engine,
            formula="~ region + sex + age",
            totals=totals,
            population=pop,
            n_estimators=15,
            max_depth=2,
            seed=0,
        )
        .prep(warn=False)
    )
    assert fitted.diagnostics["steps"]["calibrate"]["engine"] == engine
    assert float(fitted.weights.sum()) == pytest.approx(float(totals["(Intercept)"]), rel=0.05, abs=1.0)


def test_ml_calibrate_requires_population() -> None:
    sample, _pop = _sample_pop(seed=4)
    with pytest.raises(ValueError, match="population"):
        Recipe(sample, base_weight="pw").step_calibrate(
            method="linear",
            engine="forest",
            formula="~ region + sex",
            totals={"(Intercept)": 100.0},
        ).prep(warn=False)
