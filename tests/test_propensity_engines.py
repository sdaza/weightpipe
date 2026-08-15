"""Analytical tests for GBM/forest propensity and propensity-assisted calibration."""

import numpy as np
import pandas as pd
import pytest

from weightpipe.methods.nonresponse import propensity_nonresponse
from weightpipe.recipe import Recipe


def _nr_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.choice(["N", "S"], size=n)
    sex = rng.choice(["M", "F"], size=n)
    # Nonlinear-ish response: higher in N×M
    p = np.where((region == "N") & (sex == "M"), 0.85, 0.45)
    responded = (rng.random(n) < p).astype(int)
    responded[0] = 1
    responded[1] = 0
    return pd.DataFrame(
        {
            "region": region,
            "sex": sex,
            "responded": responded,
            "pw": np.ones(n),
            "y": rng.normal(size=n),
        }
    )


@pytest.mark.parametrize("engine", ["logit", "gbm", "forest"])
def test_propensity_engines_class_and_direct(engine: str) -> None:
    df = _nr_frame()
    w_cls, fac_cls, diag_cls, extra_cls = propensity_nonresponse(
        df["pw"],
        df,
        respondent="responded",
        formula="~ region + sex",
        engine=engine,  # type: ignore[arg-type]
        num_classes=4,
        weight_model=False,
        seed=1,
    )
    assert diag_cls["engine"] == engine
    assert (w_cls[df["responded"] == 0] == 0).all()
    assert (fac_cls[df["responded"] == 1] >= 1.0 - 1e-8).all()
    assert "propensity" in extra_cls
    assert "propensity_class" in extra_cls
    assert np.isfinite(extra_cls["propensity"][df["responded"] == 1]).all()

    w_dir, fac_dir, diag_dir, _ = propensity_nonresponse(
        df["pw"],
        df,
        respondent="responded",
        formula="~ region + sex",
        engine=engine,  # type: ignore[arg-type]
        num_classes=None,
        weight_model=False,
        seed=1,
    )
    assert diag_dir["detail"] == "direct_1_over_p"
    assert (w_dir[df["responded"] == 0] == 0).all()
    assert (fac_dir[df["responded"] == 1] >= 1.0 - 1e-8).all()


@pytest.mark.parametrize("engine", ["gbm", "forest"])
def test_propensity_engines_in_recipe(engine: str) -> None:
    df = _nr_frame(seed=2)
    fitted = (
        Recipe(df, base_weight="pw")
        .step_nonresponse(
            respondent="responded",
            method="propensity",
            engine=engine,
            formula="~ region + sex",
            num_classes=3,
            weight_model=False,
            seed=2,
        )
        .prep(min_cell_n=1, max_factor=None, warn=False)
    )
    assert fitted.diagnostics["steps"]["nonresponse"]["engine"] == engine
    assert "propensity" in fitted.frame.data.columns
    assert "propensity_class" in fitted.frame.data.columns
    assert (fitted.weights[df["responded"] == 0] == 0).all()


def test_propensity_assisted_raking_preserves_class_mass() -> None:
    df = _nr_frame(n=150, seed=3)
    recipe = (
        Recipe(df, base_weight="pw")
        .step_nonresponse(
            respondent="responded",
            method="propensity",
            engine="gbm",
            formula="~ region + sex",
            num_classes=3,
            weight_model=False,
            seed=3,
        )
        .step_calibrate(
            method="raking",
            margins={"sex": {"M": 70.0, "F": 80.0}},
            assist="propensity_class",
        )
    )
    fitted = recipe.prep(min_cell_n=1, max_factor=None, warn=False)
    assert fitted.diagnostics["steps"]["calibrate"]["assist"] == "propensity_class"
    assert float(fitted.weights[df["sex"] == "M"].sum()) == pytest.approx(70.0, abs=1e-4)
    assert float(fitted.weights[df["sex"] == "F"].sum()) == pytest.approx(80.0, abs=1e-4)

    # Class mass after NR (among positive weights) should match after calibrate.
    mid = (
        Recipe(df, base_weight="pw")
        .step_nonresponse(
            respondent="responded",
            method="propensity",
            engine="gbm",
            formula="~ region + sex",
            num_classes=3,
            weight_model=False,
            seed=3,
        )
        .prep(min_cell_n=1, max_factor=None, warn=False)
    )
    cls = mid.frame.data["propensity_class"]
    w_mid = mid.weights.to_numpy(dtype=float)
    w_fit = fitted.weights.to_numpy(dtype=float)
    for g in pd.unique(cls.dropna()):
        mask = cls == g
        assert float(w_fit[mask].sum()) == pytest.approx(float(w_mid[mask].sum()), rel=1e-5, abs=1e-5)


def test_propensity_assisted_raking_autoconverts_proportions() -> None:
    df = _nr_frame(n=150, seed=5)
    n_target = 150.0
    props = {"sex": {"M": 0.4, "F": 0.6}}
    recipe = (
        Recipe(df, base_weight="pw")
        .step_nonresponse(
            respondent="responded",
            method="propensity",
            engine="logit",
            formula="~ region + sex",
            num_classes=3,
            weight_model=False,
            seed=5,
        )
        .step_calibrate(
            method="raking",
            proportions=props,
            population_size=n_target,
            assist="propensity_class",
            max_iter=100,
        )
    )
    fitted = recipe.prep(min_cell_n=1, max_factor=None, warn=False)
    cal = fitted.diagnostics["steps"]["calibrate"]
    assert cal["assist"] == "propensity_class"
    assert cal["proportions_converted_to_margins"] is True
    assert cal["proportions_scale_total"] == n_target
    assert cal["resolved_margins"]["sex"]["M"] == pytest.approx(0.4 * n_target)
    assert cal["resolved_margins"]["sex"]["F"] == pytest.approx(0.6 * n_target)

    # Same as hand-converting proportions → margins, then assisting.
    manual = (
        Recipe(df, base_weight="pw")
        .step_nonresponse(
            respondent="responded",
            method="propensity",
            engine="logit",
            formula="~ region + sex",
            num_classes=3,
            weight_model=False,
            seed=5,
        )
        .step_calibrate(
            method="raking",
            margins={var: {k: v * n_target for k, v in d.items()} for var, d in props.items()},
            assist="propensity_class",
            max_iter=100,
        )
        .prep(min_cell_n=1, max_factor=None, warn=False)
    )
    np.testing.assert_allclose(fitted.weights.to_numpy(), manual.weights.to_numpy(), rtol=0, atol=1e-10)

    mid = (
        Recipe(df, base_weight="pw")
        .step_nonresponse(
            respondent="responded",
            method="propensity",
            engine="logit",
            formula="~ region + sex",
            num_classes=3,
            weight_model=False,
            seed=5,
        )
        .prep(min_cell_n=1, max_factor=None, warn=False)
    )
    cls = mid.frame.data["propensity_class"]
    w_mid = mid.weights.to_numpy(dtype=float)
    w_fit = fitted.weights.to_numpy(dtype=float)
    for g in pd.unique(cls.dropna()):
        mask = cls == g
        assert float(w_fit[mask].sum()) == pytest.approx(float(w_mid[mask].sum()), rel=1e-5, abs=1e-5)


def test_propensity_assisted_linear() -> None:
    df = _nr_frame(n=100, seed=4)
    pop = pd.DataFrame(
        {
            "region": ["N", "N", "S", "S"],
            "sex": ["M", "F", "M", "F"],
            "pw": [40.0, 35.0, 45.0, 30.0],
        }
    )
    from weightpipe.methods.design_matrix import population_totals

    totals = population_totals(pop, "~ region + sex", weight="pw")
    fitted = (
        Recipe(df, base_weight="pw")
        .step_nonresponse(
            respondent="responded",
            method="propensity",
            engine="forest",
            formula="~ region + sex",
            num_classes=None,
            weight_model=False,
            seed=4,
        )
        .step_calibrate(
            method="linear",
            formula="~ region + sex",
            totals=totals,
            assist="propensity",
            population_size=float(totals["(Intercept)"]),
        )
        .prep(min_cell_n=1, max_factor=None, warn=False)
    )
    assert fitted.diagnostics["steps"]["calibrate"]["assist"] == "propensity"
    assert (fitted.weights[df["responded"] == 0] == 0).all()
    assert float(fitted.weights.sum()) == pytest.approx(float(totals["(Intercept)"]), rel=1e-4, abs=1e-3)
