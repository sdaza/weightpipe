"""Analytical and composition tests for Iteration 1 steps."""

import numpy as np
import pandas as pd
import pytest

from weightpipe import Recipe, collect_weights, design_effect, weight_factors
from weightpipe.methods.eligibility import drop_ineligible_weights
from weightpipe.methods.nonresponse import weighting_class_nonresponse
from weightpipe.methods.raking import rake


@pytest.fixture
def cascade_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [1, 2, 3, 4, 5, 6],
            "stratum": ["A", "A", "A", "B", "B", "B"],
            "psu": [10, 10, 11, 20, 20, 21],
            "region": ["N", "N", "S", "N", "S", "S"],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "ineligible": [0, 0, 1, 0, 0, 0],
            "responded": [1, 0, 1, 1, 1, 0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "pw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_drop_ineligible_analytical(cascade_df: pd.DataFrame) -> None:
    w, fac, diag = drop_ineligible_weights(cascade_df["pw"], cascade_df, ineligible="ineligible")
    assert diag["n_dropped"] == 1
    assert float(w.iloc[2]) == 0.0
    assert float(fac.iloc[2]) == 0.0
    assert float(w.sum()) == 5.0


def test_weighting_class_analytical() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "N", "S", "S"],
            "responded": [1, 1, 0, 1, 0],
            "pw": [1.0, 1.0, 1.0, 2.0, 2.0],
        }
    )
    w, fac, diag = weighting_class_nonresponse(df["pw"], df, respondent="responded", by=["region"])
    # N: tot=3, resp=2 -> factor 1.5; S: tot=4, resp=2 -> factor 2
    assert fac.iloc[0] == pytest.approx(1.5)
    assert fac.iloc[2] == pytest.approx(0.0)
    assert fac.iloc[3] == pytest.approx(2.0)
    assert float(w.iloc[0]) == pytest.approx(1.5)
    assert float(w.sum()) == pytest.approx(3.0 + 4.0)  # mass preserved within cells
    assert len(diag["cells"]) == 2


def test_rake_two_way_analytical() -> None:
    # 2x2 balanced sample; targets force known margins
    df = pd.DataFrame(
        {
            "sex": ["M", "M", "F", "F"],
            "region": ["N", "S", "N", "S"],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    margins = {"sex": {"M": 60.0, "F": 40.0}, "region": {"N": 30.0, "S": 70.0}}
    w, _, diag = rake(df["pw"], df, margins=margins, max_iter=100, tol=1e-10)
    assert diag["converged"] is True
    for row in diag["targets"]:
        assert row["achieved"] == pytest.approx(row["target"], abs=1e-8)
    assert float(w.sum()) == pytest.approx(100.0, abs=1e-8)


@pytest.mark.recovery
def test_rake_recovers_known_margins() -> None:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "sex": rng.choice(["M", "F"], size=n),
            "region": rng.choice(["N", "S", "E"], size=n),
            "pw": rng.uniform(0.5, 2.0, size=n),
        }
    )
    margins = {
        "sex": {"M": 500.0, "F": 500.0},
        "region": {"N": 300.0, "S": 400.0, "E": 300.0},
    }
    w, _, diag = rake(df["pw"], df, margins=margins, max_iter=100, tol=1e-8)
    assert diag["converged"] is True
    for row in diag["targets"]:
        assert abs(row["achieved"] - row["target"]) < 1e-6


def test_rake_proportions_match_margins() -> None:
    df = pd.DataFrame(
        {
            "sex": ["M", "M", "F", "F"],
            "region": ["N", "S", "N", "S"],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    proportions = {"sex": {"M": 0.6, "F": 0.4}, "region": {"N": 0.3, "S": 0.7}}
    w_prop, _, diag = rake(df["pw"], df, proportions=proportions, population_size=100.0, max_iter=100, tol=1e-12)
    w_marg, _, _ = rake(
        df["pw"],
        df,
        margins={"sex": {"M": 60.0, "F": 40.0}, "region": {"N": 30.0, "S": 70.0}},
        max_iter=100,
        tol=1e-12,
    )
    np.testing.assert_allclose(w_prop.to_numpy(), w_marg.to_numpy(), rtol=1e-10)
    assert diag["total"] == 100.0
    for row in diag["targets"]:
        assert row["achieved_proportion"] == pytest.approx(row["target_proportion"], abs=1e-8)


def test_rake_proportions_default_total_preserves_weight_sum() -> None:
    df = pd.DataFrame(
        {
            "sex": ["M", "M", "F", "F"],
            "region": ["N", "S", "N", "S"],
            "pw": [2.0, 2.0, 2.0, 2.0],
        }
    )
    proportions = {"sex": {"M": 0.5, "F": 0.5}, "region": {"N": 0.25, "S": 0.75}}
    w, _, diag = rake(df["pw"], df, proportions=proportions, max_iter=100, tol=1e-12)
    assert diag["total_source"] == "sum_active_weights"
    assert diag["total"] == pytest.approx(8.0)
    assert float(w.sum()) == pytest.approx(8.0, abs=1e-8)
    for row in diag["targets"]:
        assert row["achieved_proportion"] == pytest.approx(row["target_proportion"], abs=1e-8)


def test_recipe_composition_matches_methods(cascade_df: pd.DataFrame) -> None:
    # Margins match total mass after NR on this toy (respondents keep cell mass).
    margins = {"sex": {"M": 3.0, "F": 2.0}, "region": {"N": 2.0, "S": 3.0}}
    recipe = (
        Recipe(cascade_df, base_weight="pw")
        .step_drop_ineligible(ineligible="ineligible")
        .step_nonresponse(respondent="responded", method="weighting_class", by=["region"])
        .step_calibrate(method="raking", margins=margins, max_iter=100, tol=1e-8)
    )
    fitted = recipe.prep(min_cell_n=1, max_factor=None, warn=False)

    w = cascade_df["pw"].astype(float)
    w, _, _ = drop_ineligible_weights(w, cascade_df, ineligible="ineligible")
    w, _, _ = weighting_class_nonresponse(w, cascade_df, respondent="responded", by=["region"])
    w, _, _ = rake(w, cascade_df, margins=margins, max_iter=100, tol=1e-8, warn=False)

    np.testing.assert_allclose(fitted.weights.to_numpy(), w.to_numpy(), rtol=1e-8)
    out = collect_weights(fitted, keep_intermediate=True)
    assert "weight" in out.columns
    assert ".wt_base" in out.columns
    factors = weight_factors(fitted)
    assert "factor_nonresponse" in factors.columns
    assert design_effect(fitted) >= 1.0
