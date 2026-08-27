"""Design-based GLM (weighted IRLS + Binder / replicate SEs)."""

import numpy as np
import pandas as pd
import pytest

from weightpipe import WeightPipe
from weightpipe.methods.design_matrix import design_matrix, parse_formula
from weightpipe.methods.glm import fit_glm
from weightpipe.replicates.linearization import ultimate_cluster_covariance, ultimate_cluster_variance


def _hh() -> pd.DataFrame:
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


def test_parse_formula_intercept_only() -> None:
    assert parse_formula("~ 1", allow_intercept_only=True) == []
    assert parse_formula("1 + region", allow_intercept_only=True) == ["region"]
    with pytest.raises(ValueError, match="at least one term"):
        parse_formula("~ 1")


def test_design_matrix_intercept_only() -> None:
    x = design_matrix(_hh(), "~ 1", allow_intercept_only=True)
    assert list(x.columns) == ["(Intercept)"]
    assert (x["(Intercept)"] == 1.0).all()


def test_covariance_matches_scalar_variance() -> None:
    z = np.array([1.0, 2.0, -1.0, 0.5])
    st = np.array(["A", "A", "A", "A"])
    psu = np.array(["1", "1", "2", "2"])
    var, n_psu, lonely = ultimate_cluster_variance(z, st, psu)
    cov, n2, lonely2 = ultimate_cluster_covariance(z.reshape(-1, 1), st, psu)
    assert n_psu == n2
    assert lonely == lonely2
    assert cov.shape == (1, 1)
    assert float(cov[0, 0]) == pytest.approx(var)


def test_gaussian_intercept_matches_mean_linearization() -> None:
    pipe = WeightPipe(_hh(), weight="pw", psu="psu", strata="stratum")
    mean = pipe.estimate.mean("y", variance="linearization")
    glm = pipe.estimate.glm("y ~ 1", family="gaussian", variance="linearization")
    assert glm["term"].iloc[0] == "(Intercept)"
    assert glm["estimate"].iloc[0] == pytest.approx(mean["estimate"].iloc[0])
    assert glm["se"].iloc[0] == pytest.approx(mean["se"].iloc[0])


def test_gaussian_intercept_matches_mean_jackknife() -> None:
    pipe = WeightPipe(_hh(), weight="pw", psu="psu", strata="stratum")
    mean = pipe.estimate.mean("y", variance="jackknife")
    glm = pipe.estimate.glm("y ~ 1", family="gaussian", variance="jackknife")
    assert glm["estimate"].iloc[0] == pytest.approx(mean["estimate"].iloc[0])
    assert glm["se"].iloc[0] == pytest.approx(mean["se"].iloc[0])


def test_binomial_intercept_is_logit_of_proportion() -> None:
    pipe = WeightPipe(_hh(), weight="pw", psu="psu", strata="stratum")
    prop = float(pipe.estimate.proportion("employed", variance="linearization")["estimate"].iloc[0])
    glm = pipe.estimate.glm("employed ~ 1", family="binomial", variance="linearization")
    assert glm["estimate"].iloc[0] == pytest.approx(np.log(prop / (1.0 - prop)))
    assert glm["se"].iloc[0] > 0
    assert glm["se"].iloc[0] != pytest.approx(
        float(pipe.estimate.proportion("employed", variance="linearization")["se"].iloc[0])
    )


def test_glm_gaussian_and_poisson_have_terms() -> None:
    pipe = WeightPipe(_hh(), weight="pw", psu="psu", strata="stratum")
    g = pipe.estimate.glm("y ~ region + age", family="gaussian", variance="linearization")
    assert list(g["term"]) == ["(Intercept)", "regionS", "age"]
    assert g["converged"].all()
    assert np.isfinite(g["se"]).all()
    p = pipe.estimate.glm("count ~ age", family="poisson", variance="linearization")
    assert list(p["term"]) == ["(Intercept)", "age"]
    assert np.isfinite(p["se"]).all()


def test_glm_binomial_aliases_and_pipe_method() -> None:
    pipe = WeightPipe(_hh(), weight="pw", psu="psu", strata="stratum")
    a = pipe.estimate.glm("employed ~ region", family="binomial", variance="linearization")
    b = pipe.estimate.glm("employed ~ region", family="logit", variance="linearization")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert a["family"].iloc[0] == "binomial"


def test_glm_jackknife_finite() -> None:
    pipe = WeightPipe(_hh(), weight="pw", psu="psu", strata="stratum")
    out = pipe.estimate.glm("y ~ region", family="gaussian", variance="jackknife")
    assert np.isfinite(out["se"]).all()
    assert (out["ci_upper"] > out["ci_lower"]).all()


def test_glm_unknown_family() -> None:
    pipe = WeightPipe(_hh(), weight="pw", psu="psu", strata="stratum")
    with pytest.raises(ValueError, match="unknown glm family"):
        pipe.estimate.glm("y ~ 1", family="gamma", variance="linearization")


def test_fit_glm_rejects_nonbinary_binomial() -> None:
    df = _hh()
    with pytest.raises(ValueError):
        fit_glm(df["pw"], df, "y ~ 1", family="binomial")
