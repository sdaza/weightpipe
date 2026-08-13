"""Tests for Design helpers and estimate()."""

import numpy as np
import pandas as pd
import pytest

from weightpipe import Design, Recipe, estimate, point_estimate


def test_design_srs_weights() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})
    design = Design(df, N=100)
    assert design.kind == "srs"
    assert design.strata is None and design.psu is None
    assert (design.data[design.weight] == 25.0).all()


def test_design_stratified_weights() -> None:
    df = pd.DataFrame(
        {
            "stratum": ["A", "A", "B", "B", "B"],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    design = Design(df, strata="stratum", N_h={"A": 20, "B": 30})
    assert design.kind == "stratified"
    assert design.strata == "stratum"
    w = design.data[design.weight]
    assert float(w.iloc[0]) == pytest.approx(10.0)  # 20/2
    assert float(w.iloc[2]) == pytest.approx(10.0)  # 30/3


def test_design_cluster_from_weight_and_psu() -> None:
    df = pd.DataFrame({"psu": [1, 1, 2, 2], "pw": [2.0, 2.0, 3.0, 3.0], "y": [1, 0, 1, 0]})
    design = Design(df, weight="pw", psu="psu")
    assert design.kind == "cluster"
    assert design.psu == "psu"


def test_design_infers_kind_from_inputs() -> None:
    df = pd.DataFrame({"region": ["N", "S"], "pw": [1.0, 2.0], "psu": [1, 2]})
    assert Design(df, N=10).kind == "srs"
    assert Design(df, strata="region", N_h={"N": 5, "S": 5}).kind == "stratified"
    assert Design(df, weight="pw").kind == "custom"
    assert Design(df, weight="pw", psu="psu").kind == "cluster"
    assert Design(df, weight="pw", psu="psu", strata="region").kind == "stratified_cluster"


def test_design_rejects_ambiguous_inputs() -> None:
    df = pd.DataFrame({"region": ["N", "S"], "pw": [1.0, 2.0]})
    with pytest.raises(ValueError, match="exactly one"):
        Design(df, N=10, weight="pw")
    with pytest.raises(ValueError, match="N_h= requires strata"):
        Design(df, N_h={"N": 5, "S": 5})


def test_recipe_from_design_and_estimate_mean() -> None:
    df = pd.DataFrame(
        {
            "stratum": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "psu": [1, 1, 2, 2, 3, 3, 4, 4],
            "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
            "employed": [1, 0, 1, 1, 0, 1, 1, 0],
            "pw": [1.0] * 8,
        }
    )
    design = Design(df, weight="pw", psu="psu", strata="stratum")
    recipe = Recipe.from_design(design)
    fitted = recipe.prep()
    out = estimate(recipe, "y", estimand="mean", fitted=fitted, replicates=50, seed=1)
    assert out["estimand"].iloc[0] == "mean"
    assert out["design"].iloc[0] == "stratified_cluster"
    assert out["se"].iloc[0] > 0
    assert out["ci_lower"].iloc[0] <= out["estimate"].iloc[0] <= out["ci_upper"].iloc[0]

    prop = estimate(recipe, "employed", estimand="proportion", fitted=fitted, replicates=50, seed=1)
    assert 0.0 <= prop["estimate"].iloc[0] <= 1.0

    tot = estimate(recipe, "y", estimand="total", fitted=fitted, replicates=50, seed=1)
    expected_total = float(np.sum(fitted.weights.to_numpy() * df["y"].to_numpy()))
    assert tot["estimate"].iloc[0] == pytest.approx(expected_total)


def test_estimate_srs_unit_bootstrap() -> None:
    df = pd.DataFrame({"y": np.linspace(1, 10, 20), "flag": [0, 1] * 10})
    design = Design(df, N=200)
    recipe = Recipe.from_design(design)
    out = estimate(recipe, "y", estimand="mean", replicates=80, seed=2)
    assert out["design"].iloc[0] == "srs"
    assert out["se"].iloc[0] > 0


def test_design_multistage_from_probabilities() -> None:
    df = pd.DataFrame(
        {
            "stratum": ["A", "A", "B", "B"],
            "psu": [1, 1, 2, 2],
            "p1": [0.1, 0.1, 0.2, 0.2],
            "p2": [0.5, 0.5, 0.25, 0.25],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    design = Design(df, probabilities=["p1", "p2"], psu="psu", strata="stratum")
    assert design.kind == "stratified_multistage"
    assert design.psu == "psu"
    assert design.meta["stages"] == 2
    w = design.data[design.weight].to_numpy()
    np.testing.assert_allclose(w, 1.0 / (df["p1"] * df["p2"]), rtol=0, atol=1e-12)


def test_design_multistage_from_stage_weights() -> None:
    df = pd.DataFrame(
        {
            "psu": [1, 1, 2, 2],
            "w1": [10.0, 10.0, 5.0, 5.0],
            "w2": [2.0, 2.0, 4.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    design = Design(df, stage_weights=["w1", "w2"], psu="psu")
    assert design.kind == "multistage"
    np.testing.assert_allclose(
        design.data[design.weight].to_numpy(),
        df["w1"] * df["w2"],
        rtol=0,
        atol=1e-12,
    )


def test_design_multistage_requires_psu_and_valid_probs() -> None:
    df = pd.DataFrame({"p1": [0.1, 0.2], "p2": [0.5, 0.5]})
    with pytest.raises(ValueError, match="require psu"):
        Design(df, probabilities=["p1", "p2"])
    df2 = df.assign(psu=[1, 2], p1=[0.0, 0.2])
    with pytest.raises(ValueError, match="probabilities"):
        Design(df2, probabilities=["p1", "p2"], psu="psu")


def test_design_multistage_estimate_uses_ultimate_psu() -> None:
    df = pd.DataFrame(
        {
            "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "psu": [1, 1, 2, 2, 3, 3, 4, 4],
            "p1": [0.2] * 8,
            "p2": [0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25],
            "y": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
        }
    )
    design = Design(df, probabilities=["p1", "p2"], psu="psu", strata="stratum")
    recipe = Recipe.from_design(design)
    out = estimate(recipe, "y", estimand="mean", variance="jackknife")
    assert out["design"].iloc[0] == "stratified_multistage"
    assert np.isfinite(out["se"].iloc[0])


def test_point_estimate_proportion_rejects_non_binary() -> None:
    df = pd.DataFrame({"y": [0.0, 1.0, 2.0], "w": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="0/1"):
        point_estimate(df["w"], df, "y", estimand="proportion")
