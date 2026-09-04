"""Covariate balance diagnostics (SMD before/after)."""

import numpy as np
import pandas as pd
import pytest

from weightpipe import BalanceReport, WeightPipe, balance


@pytest.fixture
def skewed_sample() -> pd.DataFrame:
    # Over-represents young males relative to a 50/50, mean-age-40 target.
    return pd.DataFrame(
        {
            "sex": ["M", "M", "M", "M", "F", "F"],
            "age": [20.0, 22.0, 24.0, 26.0, 50.0, 52.0],
            "region": ["N", "N", "S", "S", "N", "S"],
            "pw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_balance_improves_after_raking(skewed_sample: pd.DataFrame) -> None:
    props = {"sex": {"M": 0.5, "F": 0.5}}
    pipe = WeightPipe(skewed_sample, weight="pw").calibrate(
        method="raking", proportions=props, population_size=100.0, max_iter=100, tol=1e-10
    )
    report = pipe.balance(["sex"], proportions=props, threshold=0.05)
    assert isinstance(report, BalanceReport)
    assert report.summary["n_imbalanced"] == 0
    assert report.summary["max_abs_smd_after"] < report.summary["max_abs_smd_before"]
    sex = report.table[report.table["variable"] == "sex"]
    assert set(sex["level"]) == {"F", "M"}
    assert (sex["abs_smd_after"] < 1e-6).all()


def test_balance_continuous_with_means(skewed_sample: pd.DataFrame) -> None:
    props = {"sex": {"M": 0.5, "F": 0.5}}
    pipe = WeightPipe(skewed_sample, weight="pw").calibrate(
        method="raking", proportions=props, population_size=100.0, max_iter=100, tol=1e-10
    )
    # Age is not calibrated, so imbalance can remain; still a valid continuous row.
    report = pipe.balance(
        ["age", "sex"],
        means={"age": 40.0},
        proportions=props,
        sds={"age": 10.0},
        threshold=0.1,
    )
    age = report.table[report.table["variable"] == "age"].iloc[0]
    assert age["type"] == "continuous"
    assert pd.isna(age["level"])
    assert np.isfinite(age["smd_before"])
    assert np.isfinite(age["smd_after"])
    # Target mean 40 with sd 10 → SMD is (mean - 40) / 10
    assert float(age["smd_before"]) == pytest.approx((float(age["before"]) - 40.0) / 10.0)


def test_balance_from_population_microdata(skewed_sample: pd.DataFrame) -> None:
    pop = pd.DataFrame(
        {
            "sex": ["M", "M", "F", "F"],
            "age": [30.0, 50.0, 30.0, 50.0],
            "region": ["N", "S", "N", "S"],
            "N": [25.0, 25.0, 25.0, 25.0],
        }
    )
    props = {"sex": {"M": 0.5, "F": 0.5}, "region": {"N": 0.5, "S": 0.5}}
    pipe = WeightPipe(skewed_sample, weight="pw").calibrate(
        method="raking", proportions=props, population_size=100.0, max_iter=100, tol=1e-10
    )
    report = pipe.balance(["sex", "age", "region"], target=pop, target_weight="N")
    assert report.summary["n_covariates"] == 3
    assert {"continuous", "categorical"}.issubset(set(report.table["type"]))
    sex = report.table[report.table["variable"] == "sex"]
    assert (sex["abs_smd_after"] < 1e-6).all()


def test_balance_requires_targets(skewed_sample: pd.DataFrame) -> None:
    pipe = WeightPipe(skewed_sample, weight="pw")
    with pytest.raises(ValueError, match="provide target"):
        pipe.balance(["sex"])


def test_top_level_balance_matches_pipe(skewed_sample: pd.DataFrame) -> None:
    props = {"sex": {"M": 0.5, "F": 0.5}}
    pipe = WeightPipe(skewed_sample, weight="pw").calibrate(
        method="raking", proportions=props, population_size=100.0, max_iter=100, tol=1e-10
    )
    a = pipe.balance(["sex"], proportions=props)
    b = balance(pipe.result, ["sex"], proportions=props)
    pd.testing.assert_frame_equal(a.table.reset_index(drop=True), b.table.reset_index(drop=True))
    assert a.summary["max_abs_smd_after"] == pytest.approx(b.summary["max_abs_smd_after"])


def test_balance_before_series_uses_respondent_weights() -> None:
    # Universe includes nonrespondents with base_weight=1; default before="base"
    # would compare the population to itself.
    sample = pd.DataFrame(
        {
            "age": [20.0, 22.0, 24.0, 50.0, 52.0, 54.0],
            "responded": [1, 1, 1, 0, 0, 0],
            "pw": [1.0] * 6,
        }
    )
    pop = pd.DataFrame({"age": [20.0, 22.0, 24.0, 50.0, 52.0, 54.0], "N": [1.0] * 6})
    pipe = WeightPipe(sample, weight="pw")
    respondent_weights = sample["pw"] * sample["responded"]
    report = pipe.balance(["age"], target=pop, target_weight="N", before=respondent_weights)
    assert isinstance(report, BalanceReport)
    assert not report.table.empty
    smd_before = report.table["smd_before"].to_numpy(dtype=float)
    assert np.isfinite(smd_before).any()
    assert not np.allclose(smd_before, 0.0)


def test_balance_ess_in_summary(skewed_sample: pd.DataFrame) -> None:
    props = {"sex": {"M": 0.4, "F": 0.6}}
    pipe = WeightPipe(skewed_sample, weight="pw").calibrate(
        method="raking", proportions=props, population_size=100.0, max_iter=100, tol=1e-10
    )
    report = pipe.balance(["sex"], proportions=props)
    assert report.summary["ess_before"] == pytest.approx(6.0)
    assert report.summary["ess_after"] < report.summary["ess_before"]
