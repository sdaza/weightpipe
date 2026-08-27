"""Live and CSV gold checks against R survey / weightflow.

Live tests need optional extra ``r-gold`` (rpy2) plus R packages.
CSV tests run whenever ``tests/gold/*_r_survey.csv`` (and weightflow CSV)
exist — generate with ``Rscript tests/gold/generate_r_gold.R``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from weightpipe.estimands import weighted_mean, weighted_median, weighted_ratio, weighted_total
from weightpipe.methods.eligibility import drop_ineligible_weights, unknown_eligibility_weights
from weightpipe.methods.linear import linear_calibrate
from weightpipe.methods.nonresponse import weighting_class_nonresponse
from weightpipe.methods.poststrat import poststratify
from weightpipe.methods.raking import rake
from weightpipe.methods.trim import trim_weights
from weightpipe.pipeline import WeightPipe

GOLD = Path(__file__).resolve().parent / "gold"
pytestmark = pytest.mark.gold


def _read_optional(name: str) -> pd.DataFrame | None:
    path = GOLD / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def _require_r_package(name: str) -> None:
    pytest.importorskip("rpy2")
    from rpy2.robjects.packages import importr

    try:
        importr(name)
    except Exception:
        pytest.skip(f"R package '{name}' not installed")


def _estimand_toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
            "x": [2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0],
            "pw": [1.0] * 8,
        }
    )


def _glm_toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "psu": [1, 1, 2, 2, 3, 3, 4, 4],
            "region": ["N", "N", "S", "S", "N", "N", "S", "S"],
            "age": [20.0, 40.0, 30.0, 50.0, 22.0, 38.0, 28.0, 52.0],
            "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
            "employed": [1, 0, 1, 1, 0, 1, 1, 0],
            "count": [0, 1, 2, 1, 0, 2, 3, 1],
            "pw": [1.0] * 8,
        }
    )


def _linear_toy() -> tuple[pd.DataFrame, dict[str, float]]:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "S", "S"],
            "age": [20.0, 40.0, 30.0, 50.0],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    totals = {"(Intercept)": 100.0, "regionS": 50.0, "age": 3500.0}
    return df, totals


def _assert_median_vs_survey(ours: float, theirs: float, y: pd.Series) -> None:
    # survey::svyquantile may use a slightly different quantile definition.
    assert ours == pytest.approx(theirs, abs=1e-8) or abs(ours - theirs) <= min(np.diff(np.unique(y)))


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


def test_linear_calibrate_matches_r_survey_csv() -> None:
    g = _read_optional("linear_calibrate_r_survey.csv")
    if g is None:
        pytest.skip("missing tests/gold/linear_calibrate_r_survey.csv — run generate_r_gold.R")
    df, totals = _linear_toy()
    w, _, diag = linear_calibrate(g["pw"], g, formula="~ region + age", totals=totals, warn=False)
    assert diag["converged"] is True
    np.testing.assert_allclose(w.to_numpy(), g["weight_r_survey"].to_numpy(), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(df["pw"].to_numpy(), g["pw"].to_numpy())


def test_estimands_match_r_survey_csv() -> None:
    g = _read_optional("estimands_r_survey.csv")
    if g is None:
        g = _read_optional("ratio_median_r_survey.csv")
    if g is None:
        pytest.skip("missing tests/gold/estimands_r_survey.csv — run generate_r_gold.R")
    df = _estimand_toy()
    expected = {str(row["estimand"]): float(row["estimate_r_survey"]) for _, row in g.iterrows()}
    if "mean" in expected:
        assert weighted_mean(df["pw"], df["y"]) == pytest.approx(expected["mean"], rel=1e-10)
    if "total" in expected:
        assert weighted_total(df["pw"], df["y"]) == pytest.approx(expected["total"], rel=1e-10)
    assert weighted_ratio(df["pw"], df["y"], df["x"]) == pytest.approx(expected["ratio"], rel=1e-10)
    _assert_median_vs_survey(weighted_median(df["pw"], df["y"]), expected["median"], df["y"])


def test_glm_matches_r_survey_csv() -> None:
    g = _read_optional("glm_r_survey.csv")
    if g is None:
        pytest.skip("missing tests/gold/glm_r_survey.csv — run generate_r_gold.R")
    pipe = WeightPipe(_glm_toy(), weight="pw", psu="psu", strata="stratum")
    for (family, formula), gold in g.groupby(["family", "formula"], sort=False):
        ours = pipe.estimate.glm(str(formula), family=str(family), variance="linearization")
        merged = ours.merge(gold, on="term", how="inner")
        assert len(merged) == len(gold)
        np.testing.assert_allclose(
            merged["estimate"].to_numpy(),
            merged["estimate_r_survey"].to_numpy(),
            rtol=1e-8,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            merged["se"].to_numpy(),
            merged["se_r_survey"].to_numpy(),
            rtol=1e-6,
            atol=1e-8,
        )


def test_unknown_eligibility_matches_weightflow_csv() -> None:
    g = _read_optional("unknown_eligibility_weightflow.csv")
    if g is None:
        pytest.skip("missing tests/gold/unknown_eligibility_weightflow.csv — run generate_r_gold.R")
    w, _, _ = unknown_eligibility_weights(g["pw"], g, unknown="unknown", by=["region"])
    np.testing.assert_allclose(w.to_numpy(), g["weight_weightflow"].to_numpy(), rtol=1e-6, atol=1e-6)


def test_drop_ineligible_matches_weightflow_csv() -> None:
    g = _read_optional("drop_ineligible_weightflow.csv")
    if g is None:
        pytest.skip("missing tests/gold/drop_ineligible_weightflow.csv — run generate_r_gold.R")
    w, _, _ = drop_ineligible_weights(g["pw"], g, ineligible="ineligible")
    np.testing.assert_allclose(w.to_numpy(), g["weight_weightflow"].to_numpy(), rtol=1e-6, atol=1e-6)


def test_weighting_class_nr_matches_weightflow_csv() -> None:
    g = _read_optional("nr_weighting_class_weightflow.csv")
    if g is None:
        pytest.skip("missing tests/gold/nr_weighting_class_weightflow.csv — run generate_r_gold.R")
    w, _, _ = weighting_class_nonresponse(g["pw"], g, respondent="responded", by=["region"])
    np.testing.assert_allclose(w.to_numpy(), g["weight_weightflow"].to_numpy(), rtol=1e-6, atol=1e-6)


def test_trim_value_noredist_matches_weightflow_csv() -> None:
    g = _read_optional("trim_value_noredist_weightflow.csv")
    if g is None:
        pytest.skip("missing tests/gold/trim_value_noredist_weightflow.csv — run generate_r_gold.R")
    w, _, _ = trim_weights(g["pw"], g, max_ratio=5.0, reference="value", redistribute=False)
    np.testing.assert_allclose(w.to_numpy(), g["weight_weightflow"].to_numpy(), rtol=1e-6, atol=1e-6)


def test_rake_matches_weightflow_csv() -> None:
    g = _read_optional("raking_2x2_weightflow.csv")
    if g is None:
        pytest.skip("missing tests/gold/raking_2x2_weightflow.csv — run generate_r_gold.R")
    margins = {"sex": {"M": 60.0, "F": 40.0}, "region": {"N": 30.0, "S": 70.0}}
    w, _, diag = rake(g["pw"], g, margins=margins, max_iter=200, tol=1e-12, warn=False)
    assert diag["converged"] is True
    np.testing.assert_allclose(w.to_numpy(), g["weight_weightflow"].to_numpy(), rtol=1e-6, atol=1e-6)


def test_poststrat_matches_weightflow_csv() -> None:
    g = _read_optional("poststrat_region_weightflow.csv")
    if g is None:
        pytest.skip("missing tests/gold/poststrat_region_weightflow.csv — run generate_r_gold.R")
    w, _, _ = poststratify(g["pw"], g, margins={"region": {"N": 10.0, "S": 30.0}})
    np.testing.assert_allclose(w.to_numpy(), g["weight_weightflow"].to_numpy(), rtol=1e-6, atol=1e-6)


def test_linear_calibrate_matches_weightflow_csv() -> None:
    g = _read_optional("linear_calibrate_weightflow.csv")
    if g is None:
        pytest.skip("missing tests/gold/linear_calibrate_weightflow.csv — run generate_r_gold.R")
    _, totals = _linear_toy()
    w, _, diag = linear_calibrate(g["pw"], g, formula="~ region + age", totals=totals, warn=False)
    assert diag["converged"] is True
    np.testing.assert_allclose(w.to_numpy(), g["weight_weightflow"].to_numpy(), rtol=1e-6, atol=1e-6)


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


def test_cascade_full_matches_weightflow_csv() -> None:
    g = _read_optional("cascade_full_weightflow.csv")
    if g is None:
        pytest.skip("missing tests/gold/cascade_full_weightflow.csv — run generate_r_gold.R")
    w = g["pw"].astype(float)
    w, _, _ = unknown_eligibility_weights(w, g, unknown="unknown", by=["region"])
    w, _, _ = drop_ineligible_weights(w, g, ineligible="ineligible")
    w, _, _ = weighting_class_nonresponse(w, g, respondent="responded", by=["region"])
    w, _, diag = rake(
        w,
        g,
        margins={"sex": {"M": 3.0, "F": 3.0}, "region": {"N": 3.0, "S": 3.0}},
        max_iter=200,
        tol=1e-12,
        warn=False,
    )
    assert diag["converged"] is True
    w, _, _ = trim_weights(w, g, max_ratio=10.0, reference="median", redistribute=False)
    np.testing.assert_allclose(w.to_numpy(), g["weight_weightflow"].to_numpy(), rtol=1e-6, atol=1e-6)


def test_rake_matches_r_survey_live() -> None:
    """Live rpy2 check against survey::rake (skips without rpy2/R/survey)."""
    _require_r_package("survey")
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

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
    _require_r_package("survey")
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

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
    _require_r_package("weightflow")
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


def test_glm_matches_r_survey_live() -> None:
    _require_r_package("survey")
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    df = _glm_toy()
    pipe = WeightPipe(df, weight="pw", psu="psu", strata="stratum")
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
    ro.globalenv["d"] = r_df
    ro.r(
        """
        d$stratum <- factor(d$stratum)
        d$psu <- factor(d$psu)
        d$region <- factor(d$region, levels=c('N','S'))
        des <- survey::svydesign(ids=~psu, strata=~stratum, weights=~pw, data=d, nest=TRUE)
        g <- survey::svyglm(y ~ region + age, des, family=gaussian())
        b <- survey::svyglm(employed ~ region, des, family=quasibinomial())
        p <- survey::svyglm(count ~ age, des, family=quasipoisson())
        i <- survey::svyglm(y ~ 1, des, family=gaussian())
        g_est <- as.numeric(coef(g)); g_se <- as.numeric(SE(g)); g_nm <- names(coef(g))
        b_est <- as.numeric(coef(b)); b_se <- as.numeric(SE(b)); b_nm <- names(coef(b))
        p_est <- as.numeric(coef(p)); p_se <- as.numeric(SE(p)); p_nm <- names(coef(p))
        i_est <- as.numeric(coef(i)); i_se <- as.numeric(SE(i)); i_nm <- names(coef(i))
        """
    )

    def _r_fit(prefix: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        names = [str(x) for x in ro.r(f"{prefix}_nm")]
        est = np.array(list(ro.r(f"{prefix}_est")), dtype=float)
        se = np.array(list(ro.r(f"{prefix}_se")), dtype=float)
        return names, est, se

    checks = [
        ("y ~ region + age", "gaussian", "g"),
        ("employed ~ region", "binomial", "b"),
        ("count ~ age", "poisson", "p"),
        ("y ~ 1", "gaussian", "i"),
    ]
    for formula, family, prefix in checks:
        ours = pipe.estimate.glm(formula, family=family, variance="linearization")
        names, est, se = _r_fit(prefix)
        assert list(ours["term"]) == names
        np.testing.assert_allclose(ours["estimate"].to_numpy(), est, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(ours["se"].to_numpy(), se, rtol=1e-6, atol=1e-8)
