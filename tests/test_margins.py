"""Anytime weighted margins and calibrate margin_table."""

import pandas as pd
import pytest

from weightpipe import WeightPipe, margins


@pytest.fixture
def sex_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sex": ["M", "M", "F", "F", "M", "F"],
            "region": ["N", "S", "N", "S", "N", "S"],
            "pw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def test_margins_anytime_without_targets(sex_df: pd.DataFrame) -> None:
    pipe = WeightPipe(sex_df, weight="pw")
    tbl = pipe.margins("sex")
    assert list(tbl.columns) == ["variable", "category", "achieved", "achieved_proportion", "n"]
    assert set(tbl["category"]) == {"F", "M"}
    assert float(tbl["achieved_proportion"].sum()) == pytest.approx(1.0)
    assert float(tbl.loc[tbl["category"] == "M", "n"].iloc[0]) == 3


def test_calibrate_attaches_margin_table(sex_df: pd.DataFrame) -> None:
    props = {"sex": {"M": 0.4, "F": 0.6}}
    pipe = WeightPipe(sex_df, weight="pw").calibrate(
        method="raking", proportions=props, population_size=100.0, max_iter=100, tol=1e-10
    )
    cal = pipe.diagnostics["steps"]["calibrate"]
    assert "margin_table" in cal
    mt = cal["margin_table"]
    assert isinstance(mt, pd.DataFrame)
    assert {"variable", "category", "target", "achieved", "abs_diff"}.issubset(mt.columns)
    assert (mt["abs_diff"] < 1e-6).all()


def test_margins_vs_proportions_after_rake(sex_df: pd.DataFrame) -> None:
    props = {"sex": {"M": 0.4, "F": 0.6}}
    pipe = WeightPipe(sex_df, weight="pw").calibrate(
        method="raking", proportions=props, population_size=100.0, max_iter=100, tol=1e-10
    )
    tbl = pipe.margins("sex", proportions=props, population_size=100.0)
    assert float(tbl.loc[tbl["category"] == "M", "achieved_proportion"].iloc[0]) == pytest.approx(0.4, abs=1e-6)
    assert float(tbl.loc[tbl["category"] == "F", "target_proportion"].iloc[0]) == pytest.approx(0.6)
    assert (tbl["abs_diff"] < 1e-6).all()


def test_margins_targets_calibrate(sex_df: pd.DataFrame) -> None:
    props = {"sex": {"M": 0.5, "F": 0.5}}
    pipe = WeightPipe(sex_df, weight="pw").calibrate(
        method="raking", proportions=props, population_size=100.0, max_iter=100, tol=1e-10
    )
    tbl = pipe.margins(targets="calibrate")
    assert set(tbl["category"]) == {"F", "M"}
    assert (tbl["abs_diff"] < 1e-6).all()


def test_margins_after_trim_can_drift(sex_df: pd.DataFrame) -> None:
    # Imbalanced design weights so trim can matter after rake.
    df = sex_df.copy()
    df["pw"] = [10.0, 10.0, 1.0, 1.0, 10.0, 1.0]
    props = {"sex": {"M": 0.5, "F": 0.5}}
    raked = WeightPipe(df, weight="pw").calibrate(
        method="raking", proportions=props, population_size=100.0, max_iter=100, tol=1e-10
    )
    trimmed = raked.trim(max_ratio=1.2, reference="value", redistribute=False)
    before = raked.margins(targets="calibrate")
    after = trimmed.margins(targets="calibrate")
    assert (before["abs_diff"] < 1e-6).all()
    # Without redistribute, trim can move margins off target.
    assert float(after["abs_diff"].max()) >= float(before["abs_diff"].max())


def test_top_level_margins_matches_pipe(sex_df: pd.DataFrame) -> None:
    pipe = WeightPipe(sex_df, weight="pw")
    a = pipe.margins(["sex", "region"])
    b = margins(pipe.result, ["sex", "region"])
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_margins_requires_variables_or_targets(sex_df: pd.DataFrame) -> None:
    pipe = WeightPipe(sex_df, weight="pw")
    with pytest.raises(ValueError, match="provide variables"):
        pipe.margins()
