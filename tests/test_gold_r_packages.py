"""Live and CSV gold checks against R survey / weightflow.

Live tests need optional extra ``r-gold`` (rpy2) plus R packages.
CSV tests run whenever ``tests/gold/*_r_survey.csv`` (and weightflow CSV)
exist — generate with ``Rscript tests/gold/generate_r_gold.R``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from weightpipe.estimands import weighted_median, weighted_ratio
from weightpipe.methods.nonresponse import weighting_class_nonresponse
from weightpipe.methods.poststrat import poststratify
from weightpipe.methods.raking import rake

GOLD = Path(__file__).resolve().parent / "gold"
pytestmark = pytest.mark.gold


def _read_optional(name: str) -> pd.DataFrame | None:
    path = GOLD / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def test_rake_matches_r_survey_csv() -> None:
    g = _read_optional("raking_2x2_r_survey.csv")
    if g is None:
        pytest.skip("missing tests/gold/raking_2x2_r_survey.csv — run generate_r_gold.R")
    margins = {"sex": {"M": 60.0, "F": 40.0}, "region": {"N": 30.0, "S": 70.0}}
    w, _, diag = rake(g["pw"], g, margins=margins, max_iter=200, tol=1e-12, warn=False)
    assert diag["converged"] is True
    np.testing.assert_allclose(w.to_numpy(), g["weight_r_survey"].to_numpy(), rtol=1e-8, atol=1e-8)


def test_poststrat_matches_r_survey_csv() -> None:
    g = _read_optional("poststrat_region_r_survey.csv")
    if g is None:
        pytest.skip("missing tests/gold/poststrat_region_r_survey.csv — run generate_r_gold.R")
    w, _, _ = poststratify(g["pw"], g, margins={"region": {"N": 10.0, "S": 30.0}})
    np.testing.assert_allclose(w.to_numpy(), g["weight_r_survey"].to_numpy(), rtol=1e-10, atol=1e-10)


def test_ratio_median_match_r_survey_csv() -> None:
    g = _read_optional("ratio_median_r_survey.csv")
    if g is None:
        pytest.skip("missing tests/gold/ratio_median_r_survey.csv — run generate_r_gold.R")
    # Reconstruct the microdata used in generate_r_gold.R
    df = pd.DataFrame(
        {
            "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
            "x": [2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0],
            "pw": [1.0] * 8,
        }
    )
    row_ratio = g.loc[g["estimand"] == "ratio"].iloc[0]
    row_med = g.loc[g["estimand"] == "median"].iloc[0]
    assert weighted_ratio(df["pw"], df["y"], df["x"]) == pytest.approx(float(row_ratio["estimate_r_survey"]), rel=1e-10)
    # survey::svyquantile may use a slightly different quantile definition;
    # allow a small absolute tolerance and document source.
    ours = weighted_median(df["pw"], df["y"])
    theirs = float(row_med["estimate_r_survey"])
    assert ours == pytest.approx(theirs, abs=1e-8) or abs(ours - theirs) <= min(np.diff(np.unique(df["y"])))


def test_cascade_matches_weightflow_csv() -> None:
    g = _read_optional("cascade_nr_rake_weightflow.csv")
    if g is None:
        pytest.skip("missing tests/gold/cascade_nr_rake_weightflow.csv — install weightflow and run generate_r_gold.R")
    w = g["pw"].astype(float)
    w, _, _ = weighting_class_nonresponse(w, g, respondent="responded", by=["region"])
    w, _, diag = rake(
        w,
        g,
        margins={"sex": {"M": 2.0, "F": 4.0}, "region": {"N": 3.0, "S": 3.0}},
        max_iter=200,
        tol=1e-12,
        warn=False,
    )
    assert diag["converged"] is True
    np.testing.assert_allclose(w.to_numpy(), g["weight_weightflow"].to_numpy(), rtol=1e-6, atol=1e-6)


def test_rake_matches_r_survey_live() -> None:
    """Live rpy2 check against survey::rake (skips without rpy2/R/survey)."""
    pytest.importorskip("rpy2")
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    try:
        importr("survey")
    except Exception:
        pytest.skip("R package 'survey' not installed")

    df = pd.DataFrame(
        {
            "sex": ["M", "M", "F", "F"],
            "region": ["N", "S", "N", "S"],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    margins = {"sex": {"M": 60.0, "F": 40.0}, "region": {"N": 30.0, "S": 70.0}}
    ours, _, diag = rake(df["pw"], df, margins=margins, max_iter=200, tol=1e-12, warn=False)
    assert diag["converged"] is True

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
    ro.globalenv["d"] = r_df
    ro.r(
        """
        d$sex <- factor(d$sex, levels=c('M','F'))
        d$region <- factor(d$region, levels=c('N','S'))
        des <- survey::svydesign(ids=~1, weights=~pw, data=d)
        pop.sex <- data.frame(sex=factor(c('M','F'), levels=c('M','F')), Freq=c(60,40))
        pop.region <- data.frame(region=factor(c('N','S'), levels=c('N','S')), Freq=c(30,70))
        raked <- survey::rake(des, sample.margins=list(~sex, ~region),
                              population.margins=list(pop.sex, pop.region),
                              control=list(maxit=200, epsilon=1e-12))
        w <- as.numeric(weights(raked))
        """
    )
    theirs = np.array(list(ro.r("w")), dtype=float)
    np.testing.assert_allclose(ours.to_numpy(), theirs, rtol=1e-8, atol=1e-8)


def test_ratio_median_match_r_survey_live() -> None:
    pytest.importorskip("rpy2")
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    try:
        importr("survey")
    except Exception:
        pytest.skip("R package 'survey' not installed")

    df = pd.DataFrame(
        {
            "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
            "x": [2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0],
            "pw": [1.0] * 8,
        }
    )
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
    ro.globalenv["d"] = r_df
    ro.r(
        """
        des <- survey::svydesign(ids=~1, weights=~pw, data=d)
        rat <- survey::svyratio(~y, ~x, design=des)
        med <- survey::svyquantile(~y, design=des, quantiles=0.5, ci=FALSE)
        ratio_est <- as.numeric(coef(rat))
        med_est <- as.numeric(med)
        """
    )
    ratio_r = float(np.array(list(ro.r("ratio_est")), dtype=float)[0])
    med_r = float(np.array(list(ro.r("med_est")), dtype=float).ravel()[0])
    assert weighted_ratio(df["pw"], df["y"], df["x"]) == pytest.approx(ratio_r, rel=1e-10)
    ours_med = weighted_median(df["pw"], df["y"])
    assert ours_med == pytest.approx(med_r, abs=1e-8) or abs(ours_med - med_r) <= min(np.diff(np.unique(df["y"])))


def test_cascade_matches_weightflow_live() -> None:
    pytest.importorskip("rpy2")
    from rpy2.robjects.packages import importr

    try:
        importr("weightflow")
    except Exception:
        pytest.skip("R package 'weightflow' not installed")

    # Prefer committed CSV path when present; live path exercises the generator contract.
    # Re-run the same Python cascade and compare to a one-shot R call.
    from rpy2 import robjects as ro

    ro.r(
        """
        suppressPackageStartupMessages(library(weightflow))
        df <- data.frame(
          unit_id = 1:6,
          region = factor(c('N','N','N','S','S','S'), levels=c('N','S')),
          sex = factor(c('M','F','M','F','M','F'), levels=c('M','F')),
          responded = c(1L,1L,0L,1L,0L,1L),
          pw = rep(1, 6),
          y = c(1,2,3,4,5,6)
        )
        wf <- weighting_spec(df, base_weights = pw) |>
          step_nonresponse(respondent = responded, method = 'weighting_class', by = 'region') |>
          step_calibrate(method = 'raking',
                         margins = list(sex = c(M = 2, F = 4), region = c(N = 3, S = 3))) |>
          prep()
        out <- collect_weights(wf)
        if ('.weight' %in% names(out)) {
          wcol <- '.weight'
        } else if ('final_weight' %in% names(out)) {
          wcol <- 'final_weight'
        } else if ('weight' %in% names(out)) {
          wcol <- 'weight'
        } else {
          wcol <- names(out)[ncol(out)]
        }
        w_full <- rep(0, nrow(df))
        w_full[match(out$unit_id, df$unit_id)] <- as.numeric(out[[wcol]])
        w_wf <- w_full
        """
    )
    theirs = np.array(list(ro.r("w_wf")), dtype=float)
    df = pd.DataFrame(
        {
            "region": ["N", "N", "N", "S", "S", "S"],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "responded": [1, 1, 0, 1, 0, 1],
            "pw": [1.0] * 6,
        }
    )
    w = df["pw"].astype(float)
    w, _, _ = weighting_class_nonresponse(w, df, respondent="responded", by=["region"])
    w, _, _ = rake(
        w,
        df,
        margins={"sex": {"M": 2.0, "F": 4.0}, "region": {"N": 3.0, "S": 3.0}},
        max_iter=200,
        tol=1e-12,
        warn=False,
    )
    np.testing.assert_allclose(w.to_numpy(), theirs, rtol=1e-6, atol=1e-6)
