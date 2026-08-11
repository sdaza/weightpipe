"""Tests for ratio and median estimands."""

import numpy as np
import pandas as pd
import pytest

from weightpipe import Design, Recipe, estimate, point_estimate
from weightpipe.estimands import weighted_median, weighted_ratio


def test_weighted_ratio_and_median_analytical() -> None:
    w = np.array([1.0, 1.0, 2.0, 2.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    x = np.array([1.0, 1.0, 1.0, 1.0])
    assert weighted_ratio(w, y, x) == pytest.approx(170.0 / 6.0)
    # equal weights → ordinary median of sorted values
    assert weighted_median([1, 1, 1, 1], [1, 2, 3, 4]) == pytest.approx(2.0)
    # heavier upper half pulls median up
    assert weighted_median([1, 1, 10], [1, 2, 100]) == pytest.approx(100.0)


def test_estimate_ratio_and_median() -> None:
    df = pd.DataFrame(
        {
            "stratum": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "psu": [1, 1, 2, 2, 3, 3, 4, 4],
            "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
            "x": [2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0],
            "pw": [1.0] * 8,
        }
    )
    design = Design.cluster(df, weight="pw", psu="psu", strata="stratum")
    recipe = Recipe.from_design(design)
    fitted = recipe.prep()

    ratio = estimate(
        recipe,
        "y",
        estimand="ratio",
        denominator="x",
        fitted=fitted,
        variance="jackknife",
    )
    expected = point_estimate(fitted.weights, df, "y", estimand="ratio", denominator="x")
    assert ratio["estimate"].iloc[0] == pytest.approx(expected)
    assert np.isfinite(ratio["se"].iloc[0])
    assert ratio["denominator"].iloc[0] == "x"

    med = estimate(recipe, "y", estimand="median", fitted=fitted, variance="jackknife")
    assert med["estimate"].iloc[0] == pytest.approx(point_estimate(fitted.weights, df, "y", estimand="median"))
    assert np.isfinite(med["se"].iloc[0])


def test_ratio_requires_denominator() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0]})
    recipe = Recipe.from_design(Design.srs(df, N=10))
    with pytest.raises(ValueError, match="denominator"):
        estimate(recipe, "y", estimand="ratio", replicates=10, seed=1)
