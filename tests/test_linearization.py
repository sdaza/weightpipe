"""Ultimate-cluster linearization SEs."""

import numpy as np
import pandas as pd
import pytest

from weightpipe import Design, Recipe, WeightPipe, estimate
from weightpipe.replicates.linearization import linearized_estimate, linearized_residuals, ultimate_cluster_variance


def test_linearized_mean_hand_calc() -> None:
    y = np.array([1.0, 3.0, 2.0, 4.0])
    w = np.ones(4)
    psu = np.array(["1", "1", "2", "2"])
    st = np.array(["A"] * 4)
    point, z = linearized_residuals(w, y, estimand="mean")
    assert point == pytest.approx(2.5)
    var, n_psu, lonely = ultimate_cluster_variance(z, st, psu)
    assert lonely == ()
    assert n_psu == 2
    # t1 = -0.25, t2 = 0.25 → v = 2 * (0.0625 + 0.0625) = 0.25
    assert var == pytest.approx(0.25)
    df = pd.DataFrame({"y": y, "pw": w, "psu": psu})
    out = linearized_estimate(w, df, "y", estimand="mean", psu="psu")
    assert out["estimate"].iloc[0] == pytest.approx(2.5)
    assert out["se"].iloc[0] == pytest.approx(0.5)


def test_linearized_total_and_ratio() -> None:
    df = pd.DataFrame(
        {
            "y": [10.0, 12.0, 20.0, 22.0],
            "x": [2.0, 2.0, 4.0, 4.0],
            "pw": [1.0, 1.0, 1.0, 1.0],
            "psu": [1, 1, 2, 2],
        }
    )
    tot = linearized_estimate(df["pw"], df, "y", estimand="total", psu="psu")
    assert tot["estimate"].iloc[0] == pytest.approx(64.0)
    assert tot["se"].iloc[0] > 0
    rat = linearized_estimate(df["pw"], df, "y", estimand="ratio", denominator="x", psu="psu")
    assert rat["estimate"].iloc[0] == pytest.approx(64.0 / 12.0)
    assert rat["se"].iloc[0] > 0


def test_linearization_rejects_median() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0], "pw": [1.0, 1.0], "psu": [1, 2]})
    with pytest.raises(ValueError, match="median"):
        linearized_estimate(df["pw"], df, "y", estimand="median", psu="psu")


def test_estimate_linearization_via_pipe() -> None:
    df = pd.DataFrame(
        {
            "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "psu": [1, 1, 2, 2, 3, 3, 4, 4],
            "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
            "pw": [1.0] * 8,
        }
    )
    pipe = WeightPipe(df, weight="pw", psu="psu", strata="stratum")
    out = pipe.estimate("y", estimand="mean", variance="linearization")
    assert out["variance"].iloc[0] == "linearization"
    assert out["estimate"].iloc[0] == pytest.approx(df["y"].mean())
    assert out["se"].iloc[0] > 0


def test_lean_prep_matches_recorded_weights() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "N", "S", "S", "S"],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "responded": [1, 1, 0, 1, 0, 1],
            "pw": [1.0] * 6,
        }
    )
    recipe = (
        Recipe(df, base_weight="pw")
        .step_nonresponse(respondent="responded", by=["region"])
        .step_calibrate(
            method="raking",
            margins={"sex": {"M": 2.0, "F": 4.0}, "region": {"N": 3.0, "S": 3.0}},
            max_iter=200,
            tol=1e-12,
        )
    )
    full = recipe.prep(min_cell_n=1, max_factor=None, warn=False)
    lean = recipe.prep(record=False, min_cell_n=None, max_factor=None, warn=False)
    np.testing.assert_allclose(lean.weights.to_numpy(), full.weights.to_numpy(), rtol=1e-10, atol=1e-10)
    scaled = df["pw"].to_numpy() * 1.5
    lean_s = recipe.prep(record=False, base_weights=scaled, warn=False)
    data = df.copy()
    data["pw"] = scaled
    full_s = (
        Recipe(data, base_weight="pw")
        .step_nonresponse(respondent="responded", by=["region"])
        .step_calibrate(
            method="raking",
            margins={"sex": {"M": 2.0, "F": 4.0}, "region": {"N": 3.0, "S": 3.0}},
            max_iter=200,
            tol=1e-12,
        )
        .prep(min_cell_n=1, max_factor=None, warn=False)
    )
    np.testing.assert_allclose(lean_s.weights.to_numpy(), full_s.weights.to_numpy(), rtol=1e-10, atol=1e-10)


@pytest.mark.gold
def test_linearization_matches_r_survey_live() -> None:
    pytest.importorskip("rpy2")
    from rpy2.robjects.packages import importr

    try:
        importr("survey")
    except Exception:
        pytest.skip("R package 'survey' not installed")

    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    df = pd.DataFrame(
        {
            "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "psu": [1, 1, 2, 2, 3, 3, 4, 4],
            "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
            "x": [2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0],
            "pw": [1.0] * 8,
        }
    )
    design = Design(df, weight="pw", psu="psu", strata="stratum")
    recipe = Recipe.from_design(design)
    ours = estimate(recipe, "y", estimand="mean", variance="linearization")
    ours_tot = estimate(recipe, "y", estimand="total", variance="linearization")
    ours_rat = estimate(recipe, "y", estimand="ratio", denominator="x", variance="linearization")

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
    ro.globalenv["d"] = r_df
    ro.r(
        """
        d$stratum <- factor(d$stratum)
        d$psu <- factor(d$psu)
        des <- survey::svydesign(ids=~psu, strata=~stratum, weights=~pw, data=d, nest=TRUE)
        m <- survey::svymean(~y, des)
        t <- survey::svytotal(~y, des)
        r <- survey::svyratio(~y, ~x, des)
        mean_est <- as.numeric(coef(m))
        mean_se <- as.numeric(SE(m))
        tot_est <- as.numeric(coef(t))
        tot_se <- as.numeric(SE(t))
        rat_est <- as.numeric(coef(r))
        rat_se <- as.numeric(SE(r))
        """
    )
    assert ours["estimate"].iloc[0] == pytest.approx(float(np.array(list(ro.r("mean_est")))[0]), rel=1e-10)
    assert ours["se"].iloc[0] == pytest.approx(float(np.array(list(ro.r("mean_se")))[0]), rel=1e-8)
    assert ours_tot["estimate"].iloc[0] == pytest.approx(float(np.array(list(ro.r("tot_est")))[0]), rel=1e-10)
    assert ours_tot["se"].iloc[0] == pytest.approx(float(np.array(list(ro.r("tot_se")))[0]), rel=1e-8)
    assert ours_rat["estimate"].iloc[0] == pytest.approx(float(np.array(list(ro.r("rat_est")))[0]), rel=1e-10)
    assert ours_rat["se"].iloc[0] == pytest.approx(float(np.array(list(ro.r("rat_se")))[0]), rel=1e-6)
