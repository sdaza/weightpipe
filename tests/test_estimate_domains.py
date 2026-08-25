"""Multi-variable estimates and domain splits (by=)."""

import numpy as np
import pandas as pd
import pytest

from weightpipe import WeightPipe, estimate, point_estimate


@pytest.fixture
def hh() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "psu": [1, 1, 2, 2, 3, 3, 4, 4],
            "urban_rural": ["urban", "urban", "rural", "rural", "urban", "urban", "rural", "rural"],
            "income": [10.0, 12.0, 20.0, 22.0, 11.0, 13.0, 21.0, 23.0],
            "food_share": [0.4, 0.5, 0.3, 0.2, 0.45, 0.55, 0.25, 0.15],
            "employed": [1, 0, 1, 1, 0, 1, 1, 0],
            "pw": [1.0] * 8,
        }
    )


def test_estimate_mean_by_matches_subset_point(hh: pd.DataFrame) -> None:
    pipe = WeightPipe(hh, weight="pw", psu="psu", strata="stratum")
    out = pipe.estimate.mean(["income", "food_share"], by="urban_rural", variance="linearization")
    assert set(out["variable"]) == {"income", "food_share"}
    assert set(out["urban_rural"]) == {"rural", "urban"}
    urban = hh["urban_rural"] == "urban"
    income = out[out["variable"] == "income"]
    urban_est = income.loc[income["urban_rural"] == "urban", "estimate"].iloc[0]
    rural_est = income.loc[income["urban_rural"] == "rural", "estimate"].iloc[0]
    assert urban_est == pytest.approx(point_estimate(hh.loc[urban, "pw"], hh.loc[urban], "income"))
    assert rural_est == pytest.approx(point_estimate(hh.loc[~urban, "pw"], hh.loc[~urban], "income"))


def test_estimate_call_and_method_match(hh: pd.DataFrame) -> None:
    pipe = WeightPipe(hh, weight="pw", psu="psu", strata="stratum")
    a = pipe.estimate("income", estimand="mean", by="urban_rural", variance="linearization")
    b = pipe.estimate.mean("income", by="urban_rural", variance="linearization")
    c = pipe.estimation.mean("income", by="urban_rural", variance="linearization")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    pd.testing.assert_frame_equal(b.reset_index(drop=True), c.reset_index(drop=True))


def test_by_string_is_not_iterated(hh: pd.DataFrame) -> None:
    pipe = WeightPipe(hh, weight="pw", psu="psu", strata="stratum")
    out = pipe.estimate.mean("income", by="urban_rural", variance="linearization")
    assert "urban_rural" in out.columns
    assert "u" not in out.columns


def test_ratio_and_proportion_by(hh: pd.DataFrame) -> None:
    pipe = WeightPipe(hh, weight="pw", psu="psu", strata="stratum")
    rat = pipe.estimate.ratio("income", "food_share", by="urban_rural", variance="linearization")
    assert (rat["estimand"] == "ratio").all()
    assert (rat["denominator"] == "food_share").all()
    prop = pipe.estimate.proportion("employed", by="urban_rural", variance="linearization")
    assert (prop["estimate"].between(0.0, 1.0)).all()


def test_ratio_paired_denominators(hh: pd.DataFrame) -> None:
    pipe = WeightPipe(hh, weight="pw", psu="psu", strata="stratum")
    out = pipe.estimate.ratio(
        ["income", "food_share"],
        ["employed", "income"],
        variance="linearization",
    )
    assert list(out["variable"]) == ["income", "food_share"]
    assert list(out["denominator"]) == ["employed", "income"]
    assert out.loc[out["variable"] == "income", "estimate"].iloc[0] == pytest.approx(
        point_estimate(hh["pw"], hh, "income", estimand="ratio", denominator="employed")
    )
    assert out.loc[out["variable"] == "food_share", "estimate"].iloc[0] == pytest.approx(
        point_estimate(hh["pw"], hh, "food_share", estimand="ratio", denominator="income")
    )
    with pytest.raises(ValueError, match="same length"):
        pipe.estimate.ratio(["income", "food_share"], ["employed", "income", "y"], variance="linearization")


def test_top_level_estimate_by(hh: pd.DataFrame) -> None:
    pipe = WeightPipe(hh, weight="pw", psu="psu", strata="stratum")
    via_fn = estimate(pipe.recipe, ["income", "food_share"], by="urban_rural", variance="jackknife", fitted=pipe.result)
    via_pipe = pipe.estimate.mean(["income", "food_share"], by="urban_rural", variance="jackknife")
    pd.testing.assert_frame_equal(via_fn.reset_index(drop=True), via_pipe.reset_index(drop=True))
    assert np.isfinite(via_pipe["se"]).all()


def test_missing_by_column(hh: pd.DataFrame) -> None:
    pipe = WeightPipe(hh, weight="pw")
    with pytest.raises(KeyError, match="by column"):
        pipe.estimate.mean("income", by="missing", variance="linearization")
