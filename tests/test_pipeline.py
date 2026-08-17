"""WeightPipe facade: design inputs, chained steps, lazy fit, estimation."""

import numpy as np
import pandas as pd
import pytest

from weightpipe import Design, Recipe, WeightPipe, collect_weights, estimate


@pytest.fixture
def cluster_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "psu": [1, 1, 2, 2, 3, 3, 4, 4],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "pw": [2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
        }
    )


def test_estimate_without_steps(cluster_df: pd.DataFrame) -> None:
    pipe = WeightPipe(cluster_df, weight="pw", psu="psu", strata="stratum")
    assert pipe.steps == []
    out = pipe.estimate("y", estimand="mean", variance="jackknife")
    assert out["variance"].iloc[0] == "jackknife"
    assert out["design"].iloc[0] == "stratified_cluster"
    assert np.isfinite(out["se"].iloc[0])
    # Design weights pass through untouched when no step is recorded.
    np.testing.assert_allclose(pipe.weights.to_numpy(), cluster_df["pw"].to_numpy())


def test_srs_pipe_infers_kind_and_weights() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})
    pipe = WeightPipe(df, N=400)
    assert pipe.kind == "srs"
    np.testing.assert_allclose(pipe.weights.to_numpy(), np.full(4, 100.0))
    assert float(pipe.weights.sum()) == pytest.approx(400.0)


def test_pipe_matches_design_plus_recipe(cluster_df: pd.DataFrame) -> None:
    props = {"sex": {"M": 0.4, "F": 0.6}}
    pipe = (
        WeightPipe(cluster_df, weight="pw", psu="psu", strata="stratum")
        .calibrate(method="raking", proportions=props, max_iter=100, tol=1e-10)
        .trim(max_ratio=5.0, reference="value", redistribute=True)
    )
    assert pipe.steps == ["calibrate", "trim"]

    design = Design(cluster_df, weight="pw", psu="psu", strata="stratum")
    recipe = (
        Recipe.from_design(design)
        .step_calibrate(method="raking", proportions=props, max_iter=100, tol=1e-10)
        .step_trim(max_ratio=5.0, reference="value", redistribute=True)
    )
    fitted = recipe.prep(warn=False)

    np.testing.assert_allclose(pipe.weights.to_numpy(), fitted.weights.to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        pipe.estimate("y", estimand="mean", variance="jackknife")["se"].to_numpy(),
        estimate(recipe, "y", estimand="mean", variance="jackknife", fitted=fitted)["se"].to_numpy(),
        rtol=0,
        atol=1e-12,
    )


def test_steps_return_new_pipe_and_cache_is_per_pipe(cluster_df: pd.DataFrame) -> None:
    base = WeightPipe(cluster_df, weight="pw", psu="psu", strata="stratum")
    calibrated = base.calibrate(method="poststratify", margins={"sex": {"M": 60.0, "F": 40.0}})

    assert base.steps == []
    assert calibrated.steps == ["calibrate"]
    assert float(base.weights.sum()) == pytest.approx(20.0)
    assert float(calibrated.weights.sum()) == pytest.approx(100.0)


def test_table_and_diagnostics(cluster_df: pd.DataFrame) -> None:
    pipe = WeightPipe(cluster_df, weight="pw", psu="psu", strata="stratum").calibrate(
        method="poststratify", margins={"sex": {"M": 60.0, "F": 40.0}}
    )
    tbl = pipe.collect_weights(keep_intermediate=True)
    assert isinstance(tbl, pd.DataFrame)
    assert "weight" in tbl.columns
    np.testing.assert_allclose(
        tbl["weight"].to_numpy(dtype=float),
        collect_weights(pipe.result, keep_intermediate=True)["weight"].to_numpy(dtype=float),
    )
    assert pipe.diagnostics["steps_applied"] == ["calibrate"]
    assert isinstance(pipe.alerts, tuple)


def test_options_forwards_prep_arguments(cluster_df: pd.DataFrame) -> None:
    pipe = WeightPipe(cluster_df, weight="pw", psu="psu", strata="stratum").options(min_cell_n=1, max_factor=None)
    fitted = pipe.calibrate(method="poststratify", margins={"sex": {"M": 60.0, "F": 40.0}}).result
    assert fitted.diagnostics["min_cell_n"] == 1
    assert fitted.diagnostics["max_factor"] is None


def test_pipe_requires_one_design_source(cluster_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        WeightPipe(cluster_df, weight="pw", N=100)
