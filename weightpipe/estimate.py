"""Survey estimands with recipe-aware bootstrap or jackknife SE/CI."""

from typing import Literal

import numpy as np
import pandas as pd

from weightpipe.estimands import (
    Estimand,
    assert_proportion_binary,
    weighted_mean,
    weighted_median,
    weighted_ratio,
    weighted_total,
)
from weightpipe.recipe import Recipe
from weightpipe.replicates import (
    boot_mean,
    boot_median,
    boot_proportion,
    boot_ratio,
    boot_total,
    bootstrap_weights,
    jackknife_weights,
)
from weightpipe.replicates.linearization import linearized_estimate
from weightpipe.result import WeightResult

VarianceMethod = Literal["bootstrap", "jackknife", "linearization"]


def estimate(
    recipe: Recipe,
    variable: str,
    *,
    estimand: Estimand = "mean",
    denominator: str | None = None,
    fitted: WeightResult | None = None,
    variance: VarianceMethod = "bootstrap",
    replicates: int = 200,
    level: float = 0.95,
    seed: int | None = None,
    strata: str | None = None,
    psu: str | None = None,
    m: int | None = None,
    p: float = 0.5,
) -> pd.DataFrame:
    """Point estimate, SE, and CI for survey estimands.

    Supported ``estimand`` values:

    - ``mean`` / ``proportion`` / ``total`` / ``median`` — use ``variable``
    - ``ratio`` — ``Σ w * variable / Σ w * denominator`` (``denominator`` required)

    Variance is a recipe-aware Rao–Wu bootstrap or delete-a-PSU jackknife, or
    ``linearization`` (ultimate-cluster Taylor SE that treats weights as fixed).
    """
    if variable not in recipe.data.columns:
        raise KeyError(f"variable not found: {variable}")
    if estimand == "ratio":
        if denominator is None:
            raise ValueError("estimand='ratio' requires denominator=...")
        if denominator not in recipe.data.columns:
            raise KeyError(f"denominator not found: {denominator}")
    elif denominator is not None:
        raise ValueError("denominator is only used with estimand='ratio'")

    design = recipe.design
    st = strata if strata is not None else (design.strata if design is not None else None)
    ps = psu if psu is not None else (design.psu if design is not None else None)

    point = fitted if fitted is not None else recipe.prep()
    if variance == "linearization":
        out = linearized_estimate(
            point.weights,
            recipe.data,
            variable,
            estimand=estimand,
            denominator=denominator,
            strata=st,
            psu=ps,
            level=level,
        )
    elif variance in ("bootstrap", "jackknife"):
        if variance == "bootstrap":
            reps = bootstrap_weights(
                recipe,
                replicates=replicates,
                strata=st,
                psu=ps,
                m=m,
                seed=seed,
                point=point,
            )
        else:
            reps = jackknife_weights(recipe, strata=st, psu=ps, point=point)
        if estimand == "mean":
            out = boot_mean(reps, variable, level=level)
        elif estimand == "proportion":
            out = boot_proportion(reps, variable, level=level)
        elif estimand == "total":
            out = boot_total(reps, variable, level=level)
        elif estimand == "ratio":
            assert denominator is not None
            out = boot_ratio(reps, variable, denominator, level=level)
        elif estimand == "median":
            out = boot_median(reps, variable, level=level, p=p)
        else:
            raise ValueError(f"unknown estimand: {estimand!r}")
    else:
        raise ValueError(f"unknown variance method: {variance!r}")

    out = out.copy()
    out["estimand"] = estimand
    out["variable"] = variable
    if estimand == "ratio":
        out["denominator"] = denominator
    if estimand == "median" and p != 0.5:
        out["p"] = p
    out["variance"] = variance
    if design is not None:
        out["design"] = design.kind
    return out


def point_estimate(
    weights: pd.Series | np.ndarray,
    data: pd.DataFrame,
    variable: str,
    *,
    estimand: Estimand = "mean",
    denominator: str | None = None,
    p: float = 0.5,
) -> float:
    """Point estimate without variance (Hájek mean/proportion/median, total, or ratio)."""
    if variable not in data.columns:
        raise KeyError(f"variable not found: {variable}")
    ww = np.asarray(weights, dtype=float)
    if estimand == "total":
        return weighted_total(ww, data[variable])
    if estimand == "ratio":
        if denominator is None:
            raise ValueError("estimand='ratio' requires denominator=...")
        if denominator not in data.columns:
            raise KeyError(f"denominator not found: {denominator}")
        return weighted_ratio(ww, data[variable], data[denominator])
    if estimand == "median":
        return weighted_median(ww, data[variable], p=p)
    if estimand == "proportion":
        assert_proportion_binary(data[variable])
    if estimand in ("mean", "proportion"):
        return weighted_mean(ww, data[variable])
    raise ValueError(f"unknown estimand: {estimand!r}")
