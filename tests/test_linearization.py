"""Ultimate-cluster linearization SEs."""

import numpy as np
import pandas as pd
import pytest

from weightpipe import Design, Recipe, WeightPipe, estimate
from weightpipe.replicates.linearization import (
    _stratum_psu_codes,
    linearized_estimate,
    linearized_residuals,
    ultimate_cluster_covariance,
    ultimate_cluster_variance,
)


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


def _loop_ultimate_cluster_variance(
    z: np.ndarray,
    strata: np.ndarray,
    psu: np.ndarray,
) -> tuple[float, int, tuple[str, ...]]:
    """Pre-vectorization PSU loop (reference for regression tests)."""
    z = np.asarray(z, dtype=float)
    var = 0.0
    n_psu = 0
    lonely: list[str] = []
    for h in pd.unique(strata):
        idx = np.where(strata == h)[0]
        psus = pd.unique(psu[idx])
        nh = len(psus)
        if nh < 2:
            lonely.append(str(h))
            continue
        totals = np.array([float(z[idx][psu[idx] == p].sum()) for p in psus], dtype=float)
        mean_t = float(totals.mean())
        var += (nh / (nh - 1.0)) * float(np.sum((totals - mean_t) ** 2))
        n_psu += nh
    return var, n_psu, tuple(lonely)


def _loop_ultimate_cluster_covariance(
    z: np.ndarray,
    strata: np.ndarray,
    psu: np.ndarray,
) -> tuple[np.ndarray, int, tuple[str, ...]]:
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n, p = z.shape
    var = np.zeros((p, p), dtype=float)
    n_psu = 0
    lonely: list[str] = []
    for h in pd.unique(strata):
        idx = np.where(strata == h)[0]
        psus = pd.unique(psu[idx])
        nh = len(psus)
        if nh < 2:
            lonely.append(str(h))
            continue
        totals = np.array([z[idx][psu[idx] == q].sum(axis=0) for q in psus], dtype=float)
        mean_t = totals.mean(axis=0)
        centered = totals - mean_t
        var += (nh / (nh - 1.0)) * (centered.T @ centered)
        n_psu += nh
    return var, n_psu, tuple(lonely)


def _small_designs() -> list[tuple[pd.DataFrame, str | None, str | None, np.ndarray | None]]:
    rng = np.random.default_rng(0)
    n = 48
    y = rng.normal(size=n)
    x = rng.normal(loc=2.0, scale=0.5, size=n)
    binary = (y > 0).astype(float)
    w = rng.uniform(0.5, 2.0, size=n)
    psu = np.repeat(np.arange(12), 4)
    strata = np.repeat(["A", "B", "C"], 16)
    lonely = np.array(["L"] + ["M"] * 47)
    mask = rng.random(n) > 0.3
    df = pd.DataFrame({"y": y, "x": x, "bin": binary, "pw": w, "psu": psu, "stratum": strata, "lonely": lonely})
    return [
        (df, None, None, None),
        (df, None, "psu", None),
        (df, "stratum", None, None),
        (df, "stratum", "psu", None),
        (df, "stratum", "psu", mask),
        (df, "lonely", "psu", None),
    ]


def test_vectorized_variance_matches_psu_loop() -> None:
    df, *_ = _small_designs()[3]
    z = (df["pw"].to_numpy() * (df["y"].to_numpy() - df["y"].mean())).astype(float)
    st = df["stratum"].astype(str).to_numpy()
    psu = df["psu"].astype(str).to_numpy()
    got = ultimate_cluster_variance(z, st, psu)
    ref = _loop_ultimate_cluster_variance(z, st, psu)
    assert got[1:] == ref[1:]
    assert got[0] == ref[0]
    z2 = np.column_stack([z, z * 0.5, np.ones_like(z)])
    cov_got = ultimate_cluster_covariance(z2, st, psu)
    cov_ref = _loop_ultimate_cluster_covariance(z2, st, psu)
    assert cov_got[1:] == cov_ref[1:]
    np.testing.assert_array_equal(cov_got[0], cov_ref[0])


@pytest.mark.filterwarnings("ignore:Strata with a single PSU:RuntimeWarning")
@pytest.mark.parametrize(
    "estimand,variable,denominator",
    [("mean", "y", None), ("total", "y", None), ("proportion", "bin", None), ("ratio", "y", "x")],
)
def test_linearized_estimate_matches_loop_se(estimand: str, variable: str, denominator: str | None) -> None:
    for df, strata, psu, mask in _small_designs():
        w = df["pw"].to_numpy(dtype=float).copy()
        if mask is not None:
            w = w.copy()
            w[~mask] = 0.0
        kwargs: dict[str, object] = {"estimand": estimand, "strata": strata, "psu": psu, "mask": mask}
        if denominator is not None:
            kwargs["denominator"] = denominator
        out = linearized_estimate(df["pw"], df, variable, **kwargs)  # type: ignore[arg-type]
        y = df[variable].to_numpy(dtype=float)
        x = None if denominator is None else df[denominator].to_numpy(dtype=float)
        point, z = linearized_residuals(w, y, estimand=estimand, x=x)  # type: ignore[arg-type]
        st, cl = _stratum_psu_codes(df, len(df), strata, psu)
        var, n_psu, _lonely = _loop_ultimate_cluster_variance(z, st, cl)
        se = float(np.sqrt(var))
        assert out["estimate"].iloc[0] == point
        assert out["se"].iloc[0] == se
        assert out["R_used"].iloc[0] == n_psu
        assert list(out.columns) == ["estimate", "se", "ci_lower", "ci_upper", "level", "R_used"]


def test_linearization_n50k_rows_as_psu_is_fast() -> None:
    import time

    rng = np.random.default_rng(1)
    n = 50_000
    z = rng.normal(size=n)
    st = np.array(["1"] * n, dtype=object)
    psu = np.arange(n).astype(str)
    t0 = time.perf_counter()
    var, n_psu, lonely = ultimate_cluster_variance(z, st, psu)
    elapsed = time.perf_counter() - t0
    assert lonely == ()
    assert n_psu == n
    assert var > 0
    assert elapsed < 1.0
    z2 = np.column_stack([z, z + 1.0])
    t1 = time.perf_counter()
    cov, n2, lonely2 = ultimate_cluster_covariance(z2, st, psu)
    elapsed2 = time.perf_counter() - t1
    assert n2 == n
    assert lonely2 == ()
    assert cov.shape == (2, 2)
    assert elapsed2 < 1.0
