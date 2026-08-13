"""Analytical and composition tests for cascade parity steps."""

import numpy as np
import pandas as pd
import pytest

from weightpipe import Design, Recipe, estimate, jackknife_weights, population_totals
from weightpipe.methods.eligibility import unknown_eligibility_weights
from weightpipe.methods.linear import linear_calibrate
from weightpipe.methods.nonresponse import logit_propensity_nonresponse
from weightpipe.methods.poststrat import poststratify
from weightpipe.methods.trim import trim_weights


def test_unknown_eligibility_analytical() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "N", "S", "S"],
            "unknown": [0, 0, 1, 0, 1],
            "pw": [1.0, 1.0, 1.0, 2.0, 2.0],
        }
    )
    w, fac, diag = unknown_eligibility_weights(df["pw"], df, unknown="unknown", by=["region"])
    # N: tot=3, known=2 -> factor 1.5; S: tot=4, known=2 -> factor 2
    assert fac.iloc[0] == pytest.approx(1.5)
    assert fac.iloc[2] == pytest.approx(0.0)
    assert fac.iloc[3] == pytest.approx(2.0)
    assert float(w.sum()) == pytest.approx(3.0 + 4.0)
    assert len(diag["cells"]) == 2


def test_trim_value_cap_redistribute() -> None:
    df = pd.DataFrame({"pw": [1.0, 2.0, 10.0]})
    # Cap at 5: excess 5 can be absorbed by the other two units (capacity 4+3).
    w, _, diag = trim_weights(df["pw"], df, max_ratio=5.0, reference="value", redistribute=True)
    assert float(w.max()) == pytest.approx(5.0)
    assert float(w.sum()) == pytest.approx(13.0, abs=1e-8)
    assert diag["n_capped"] >= 1

    # When the cap is too tight to hold the total, mass cannot be fully preserved.
    w_tight, _, _ = trim_weights(df["pw"], df, max_ratio=4.0, reference="value", redistribute=True)
    assert float(w_tight.max()) == pytest.approx(4.0)
    assert float(w_tight.sum()) == pytest.approx(12.0, abs=1e-8)


def test_trim_base_ratio() -> None:
    df = pd.DataFrame({"pw": [1.0, 1.0, 1.0], "w": [1.0, 2.0, 8.0]})
    w, _, diag = trim_weights(
        df["w"],
        df,
        max_ratio=3.0,
        reference="base",
        base_weights=df["pw"],
        redistribute=False,
    )
    assert float(w.iloc[2]) == pytest.approx(3.0)
    assert float(w.iloc[1]) == pytest.approx(2.0)
    assert diag["reference"] == "base"


def test_poststrat_analytical() -> None:
    df = pd.DataFrame({"region": ["N", "N", "S", "S"], "pw": [1.0, 1.0, 1.0, 1.0]})
    w, fac, diag = poststratify(df["pw"], df, margins={"region": {"N": 10.0, "S": 30.0}})
    assert float(w[df["region"] == "N"].sum()) == pytest.approx(10.0)
    assert float(w[df["region"] == "S"].sum()) == pytest.approx(30.0)
    assert fac.iloc[0] == pytest.approx(5.0)
    assert fac.iloc[2] == pytest.approx(15.0)
    assert diag["method"] == "poststratify"


def test_linear_calibrate_matches_totals() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "S", "S"],
            "age": [20.0, 40.0, 30.0, 50.0],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    # Population of 100 with half N, mean age 35 → totals via design matrix
    pop = pd.DataFrame(
        {
            "region": ["N"] * 50 + ["S"] * 50,
            "age": [35.0] * 100,
        }
    )
    totals = population_totals(pop, "~ region + age")
    w, _, diag = linear_calibrate(df["pw"], df, formula="~ region + age", totals=totals, warn=False)
    for row in diag["targets"]:
        assert row["achieved"] == pytest.approx(row["target"], abs=1e-6)


def test_logit_propensity_direct_preserves_structure() -> None:
    rng = np.random.default_rng(0)
    n = 80
    region = rng.choice(["N", "S"], size=n)
    # Higher response in N
    p = np.where(region == "N", 0.8, 0.4)
    responded = (rng.random(n) < p).astype(int)
    # Ensure both outcomes
    responded[0] = 1
    responded[1] = 0
    df = pd.DataFrame({"region": region, "responded": responded, "pw": np.ones(n)})
    w, fac, diag = logit_propensity_nonresponse(
        df["pw"],
        df,
        respondent="responded",
        formula="~ region",
        num_classes=None,
        weight_model=False,
    )
    assert diag["engine"] == "logit"
    assert (w[df["responded"] == 0] == 0).all()
    assert (fac[df["responded"] == 1] >= 1.0 - 1e-8).all()
    assert float(w.sum()) > 0


def test_recipe_cascade_composition() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "N", "S", "S", "S"],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "unknown": [0, 0, 1, 0, 0, 0],
            "ineligible": [0, 0, 0, 0, 0, 0],
            "responded": [1, 1, 1, 1, 0, 1],
            "pw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    recipe = (
        Recipe(df, base_weight="pw")
        .step_unknown_eligibility(unknown="unknown", by=["region"])
        .step_drop_ineligible(ineligible="ineligible")
        .step_nonresponse(respondent="responded", method="weighting_class", by=["region"])
        .step_calibrate(method="poststratify", margins={"sex": {"M": 40.0, "F": 60.0}})
        .step_trim(max_ratio=10.0, reference="median", redistribute=True)
    )
    fitted = recipe.prep(min_cell_n=1, max_factor=None, warn=False)
    assert float(fitted.weights[df["sex"] == "M"].sum()) == pytest.approx(40.0, abs=1e-6)
    assert float(fitted.weights[df["sex"] == "F"].sum()) == pytest.approx(60.0, abs=1e-6)
    assert "unknown_eligibility" in fitted.diagnostics["steps"]
    assert "trim" in fitted.diagnostics["steps"]


def test_jackknife_factors_and_estimate() -> None:
    df = pd.DataFrame(
        {
            "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "psu": [1, 1, 2, 2, 3, 3, 4, 4],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "pw": [2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
        }
    )
    design = Design(df, weight="pw", psu="psu", strata="stratum")
    recipe = Recipe.from_design(design)
    jack = jackknife_weights(recipe)
    # 2 strata × 2 PSUs = 4 replicates
    assert jack.R == 4
    assert jack.scales is not None
    assert jack.scales[0] == pytest.approx(0.5)
    # Deleted PSU has factor 0 on base weights path → replicate weights 0 there before prep
    out = estimate(recipe, "y", estimand="mean", variance="jackknife")
    assert np.isfinite(out["se"].iloc[0])
    assert out["ci_lower"].iloc[0] <= out["estimate"].iloc[0] <= out["ci_upper"].iloc[0]
    assert out["variance"].iloc[0] == "jackknife"


def test_logit_nr_in_recipe() -> None:
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame(
        {
            "region": rng.choice(["N", "S"], size=n),
            "sex": rng.choice(["M", "F"], size=n),
            "responded": rng.integers(0, 2, size=n),
            "pw": np.ones(n),
            "y": rng.normal(size=n),
        }
    )
    df.loc[0, "responded"] = 1
    df.loc[1, "responded"] = 0
    recipe = Recipe(df, base_weight="pw").step_nonresponse(
        respondent="responded",
        method="propensity",
        engine="logit",
        formula="~ region + sex",
        num_classes=3,
        weight_model=False,
    )
    fitted = recipe.prep(min_cell_n=1, max_factor=None, warn=False)
    assert fitted.diagnostics["steps"]["nonresponse"]["engine"] == "logit"
    assert (fitted.weights[df["responded"] == 0] == 0).all()


def test_linear_calibrate_in_recipe() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "S", "S"],
            "age": [20.0, 40.0, 30.0, 50.0],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    totals = population_totals(
        pd.DataFrame({"region": ["N"] * 50 + ["S"] * 50, "age": [35.0] * 100}),
        ["region", "age"],
    )
    recipe = Recipe(df, base_weight="pw").step_calibrate(
        method="linear",
        formula=["region", "age"],
        totals=totals,
    )
    fitted = recipe.prep(warn=False)
    for row in fitted.diagnostics["steps"]["calibrate"]["targets"]:
        assert row["achieved"] == pytest.approx(row["target"], abs=1e-6)
