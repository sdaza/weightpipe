"""Analytical tests for cluster=, bounds=, penalty=, and auto trim."""

import numpy as np
import pandas as pd
import pytest

from weightpipe import Recipe, population_totals
from weightpipe.methods.eligibility import unknown_eligibility_weights
from weightpipe.methods.linear import linear_calibrate
from weightpipe.methods.nonresponse import weighting_class_nonresponse
from weightpipe.methods.trim import potter_threshold, trim_weights_auto, tukey_threshold


def test_unknown_eligibility_cluster_any() -> None:
    # H1 known (mean w=1), H2 unknown (any unknown) → factor 2 on H1 members
    df = pd.DataFrame(
        {
            "hh": ["H1", "H1", "H2", "H2"],
            "unknown": [0, 0, 1, 0],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    w, fac, diag = unknown_eligibility_weights(df["pw"], df, unknown="unknown", cluster="hh")
    assert diag["cluster"] == "hh"
    assert diag["cells"][0]["level"] == "household"
    assert fac.iloc[0] == pytest.approx(2.0)
    assert fac.iloc[1] == pytest.approx(2.0)
    assert fac.iloc[2] == pytest.approx(0.0)
    assert fac.iloc[3] == pytest.approx(0.0)
    assert float(w.sum()) == pytest.approx(4.0)


def test_weighting_class_cluster_all_respond() -> None:
    # H1 all respond; H2 has a nonrespondent → whole H2 is NR
    df = pd.DataFrame(
        {
            "hh": ["H1", "H1", "H2", "H2"],
            "responded": [1, 1, 1, 0],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    w, fac, diag = weighting_class_nonresponse(df["pw"], df, respondent="responded", cluster="hh")
    assert diag["cluster"] == "hh"
    assert fac.iloc[0] == pytest.approx(2.0)
    assert fac.iloc[1] == pytest.approx(2.0)
    assert fac.iloc[2] == pytest.approx(0.0)
    assert fac.iloc[3] == pytest.approx(0.0)
    assert float(w.sum()) == pytest.approx(4.0)


def test_recipe_cluster_on_eligibility_and_nr() -> None:
    df = pd.DataFrame(
        {
            "hh": ["H1", "H1", "H2", "H2", "H3", "H3"],
            "unknown": [0, 0, 1, 0, 0, 0],
            "responded": [1, 1, 1, 1, 1, 0],
            "pw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    fitted = (
        Recipe(df, base_weight="pw")
        .step_unknown_eligibility(unknown="unknown", cluster="hh")
        .step_nonresponse(respondent="responded", method="weighting_class", cluster="hh")
        .prep(min_cell_n=1, max_factor=None, warn=False)
    )
    assert fitted.diagnostics["steps"]["unknown_eligibility"]["cluster"] == "hh"
    assert fitted.diagnostics["steps"]["nonresponse"]["cluster"] == "hh"
    # H2 unknown → dropped; H3 NR → dropped; mass on H1
    assert (fitted.weights[df["hh"] == "H2"] == 0).all()
    assert (fitted.weights[df["hh"] == "H3"] == 0).all()
    assert float(fitted.weights.sum()) == pytest.approx(6.0)


def test_bounded_linear_g_in_bounds() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "S", "S"],
            "age": [20.0, 40.0, 30.0, 50.0],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    pop = pd.DataFrame(
        {
            "region": ["N"] * 3 + ["S"] * 5,
            "age": [25.0, 35.0, 45.0, 30.0, 40.0, 50.0, 20.0, 60.0],
        }
    )
    totals = population_totals(pop, "~ region + age")
    # Unbounded g spans ~1.1–2.9; these bounds contain the solution.
    w, fac, diag = linear_calibrate(
        df["pw"],
        df,
        formula="~ region + age",
        totals=totals,
        bounds=(0.5, 4.0),
        calfun="linear",
        warn=False,
    )
    assert diag["bounds"] == [0.5, 4.0]
    assert diag["converged"] is True
    assert (fac >= 0.5 - 1e-8).all()
    assert (fac <= 4.0 + 1e-8).all()
    for row in diag["targets"]:
        assert row["achieved"] == pytest.approx(row["target"], abs=1e-4)

    # Tight bounds bind: g stays in [L, U] even if totals are imperfect.
    _, fac_t, diag_t = linear_calibrate(
        df["pw"],
        df,
        formula="~ region + age",
        totals=totals,
        bounds=(0.9, 1.5),
        warn=False,
    )
    assert (fac_t >= 0.9 - 1e-8).all()
    assert (fac_t <= 1.5 + 1e-8).all()
    assert float(fac_t.max()) == pytest.approx(1.5, abs=1e-6)


def test_ridge_shrinks_g_factors() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "S", "S"],
            "age": [20.0, 40.0, 30.0, 50.0],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    pop = pd.DataFrame({"region": ["N"] * 50 + ["S"] * 50, "age": [35.0] * 100})
    totals = population_totals(pop, "~ region + age")
    w0, fac0, _ = linear_calibrate(df["pw"], df, formula="~ region + age", totals=totals, warn=False)
    w_r, fac_r, diag = linear_calibrate(
        df["pw"],
        df,
        formula="~ region + age",
        totals=totals,
        penalty=10.0,
        warn=False,
    )
    assert diag["penalty"] == 10.0
    # Ridge pulls g toward 1 relative to unpenalized solution
    assert float(np.mean(np.abs(fac_r - 1.0))) < float(np.mean(np.abs(fac0 - 1.0)))


def test_ridge_rejects_bounds() -> None:
    df = pd.DataFrame({"region": ["N", "S"], "pw": [1.0, 1.0]})
    totals = population_totals(df, "~ region")
    with pytest.raises(ValueError, match="penalty"):
        linear_calibrate(
            df["pw"],
            df,
            formula="~ region",
            totals=totals,
            bounds=(0.5, 2.0),
            penalty=5.0,
            warn=False,
        )


def test_tukey_and_potter_thresholds() -> None:
    w = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 10.0])
    t_tukey = tukey_threshold(w)
    t_potter = potter_threshold(w)
    assert t_tukey == pytest.approx(float(np.quantile(w, 0.75) + 3 * (np.quantile(w, 0.75) - np.quantile(w, 0.25))))
    assert np.isfinite(t_potter)
    assert t_potter >= float(np.median(w))


def test_trim_weights_auto_tukey_proportional() -> None:
    df = pd.DataFrame({"pw": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 12.0]})
    w, _, diag = trim_weights_auto(df["pw"], method="tukey", redistribute="proportional", lower=0.5)
    assert diag["method"] == "tukey"
    assert float(w.max()) == pytest.approx(diag["upper"], abs=1e-8)
    assert float(w.sum()) == pytest.approx(float(df["pw"].sum()), abs=1e-6)
    assert float(w.max()) < 12.0


def test_trim_weights_auto_potter_in_recipe() -> None:
    df = pd.DataFrame({"pw": [1.0, 1.2, 1.1, 1.3, 1.0, 15.0], "y": np.arange(6.0)})
    fitted = (
        Recipe(df, base_weight="pw")
        .step_trim_weights(method="potter", redistribute="proportional", lower=0.5)
        .prep(min_cell_n=1, max_factor=None, warn=False)
    )
    assert "trim_weights" in fitted.diagnostics["steps"]
    assert fitted.diagnostics["steps"]["trim_weights"]["method"] == "potter"
    assert float(fitted.weights.max()) <= fitted.diagnostics["steps"]["trim_weights"]["upper"] + 1e-8


def test_bounded_and_ridge_in_recipe() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "S", "S"],
            "age": [20.0, 40.0, 30.0, 50.0],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    pop = pd.DataFrame(
        {
            "region": ["N"] * 3 + ["S"] * 5,
            "age": [25.0, 35.0, 45.0, 30.0, 40.0, 50.0, 20.0, 60.0],
        }
    )
    totals = population_totals(pop, "~ region + age")
    fitted_b = (
        Recipe(df, base_weight="pw")
        .step_calibrate(
            method="linear",
            formula="~ region + age",
            totals=totals,
            bounds=(0.5, 4.0),
        )
        .prep(min_cell_n=1, max_factor=None, warn=False)
    )
    assert fitted_b.diagnostics["steps"]["calibrate"]["bounds"] == [0.5, 4.0]

    fitted_r = (
        Recipe(df, base_weight="pw")
        .step_calibrate(method="linear", formula="~ region + age", totals=totals, penalty=5.0)
        .prep(min_cell_n=1, max_factor=None, warn=False)
    )
    assert fitted_r.diagnostics["steps"]["calibrate"]["penalty"] == 5.0
