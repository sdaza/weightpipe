"""Cross-checks against svy (optional ``gold`` extra).

Install with ``uv sync --extra gold`` (CI does this). Tests skip if svy
is not installed.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("svy")

from svy import Cat, Design, Sample  # noqa: E402

from weightpipe.methods.design_matrix import population_totals  # noqa: E402
from weightpipe.methods.linear import linear_calibrate  # noqa: E402
from weightpipe.methods.nonresponse import weighting_class_nonresponse  # noqa: E402
from weightpipe.methods.poststrat import poststratify  # noqa: E402
from weightpipe.methods.raking import rake  # noqa: E402

pytestmark = pytest.mark.gold


def _svy_weights(sample: Sample, column: str) -> np.ndarray:
    return sample.data.sort("svy_row_index").select(pl.col(column)).to_series().to_numpy().astype(float)


def _sample(df: pd.DataFrame, *, wgt: str = "pw") -> Sample:
    return Sample(data=pl.from_pandas(df), design=Design(wgt=wgt))


def test_poststrat_matches_svy() -> None:
    df = pd.DataFrame({"region": ["N", "N", "S", "S"], "pw": [1.0, 1.0, 1.0, 1.0]})
    theirs_s = _sample(df).weighting.poststratify(controls={"N": 10.0, "S": 30.0}, by="region", wgt_name="ps_wgt")
    ours, _, _ = poststratify(df["pw"], df, margins={"region": {"N": 10.0, "S": 30.0}})
    np.testing.assert_allclose(ours.to_numpy(), _svy_weights(theirs_s, "ps_wgt"), rtol=0, atol=1e-12)


def test_weighting_class_nr_matches_svy() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "N", "S", "S"],
            "responded": [1, 1, 0, 1, 0],
            "pw": [1.0, 1.0, 1.0, 2.0, 2.0],
            "resp_status": ["rr", "rr", "nr", "rr", "nr"],
        }
    )
    theirs_s = _sample(df).weighting.adjust(
        resp_status="resp_status",
        by="region",
        wgt_name="nr_wgt",
        unknown_to_inelig=False,
        respondents_only=False,
    )
    ours, _, _ = weighting_class_nonresponse(df["pw"], df, respondent="responded", by=["region"])
    np.testing.assert_allclose(ours.to_numpy(), _svy_weights(theirs_s, "nr_wgt"), rtol=0, atol=1e-12)


def test_raking_matches_svy() -> None:
    df = pd.DataFrame(
        {
            "sex": ["M", "M", "F", "F"],
            "region": ["N", "S", "N", "S"],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    control = {"sex": {"M": 60.0, "F": 40.0}, "region": {"N": 30.0, "S": 70.0}}
    theirs_s = _sample(df).weighting.rake(controls=control, wgt_name="rk_wgt", tol=1e-12, max_iter=200)
    ours, _, diag = rake(df["pw"], df, margins=control, max_iter=200, tol=1e-12, warn=False)
    assert diag["converged"] is True
    np.testing.assert_allclose(ours.to_numpy(), _svy_weights(theirs_s, "rk_wgt"), rtol=1e-10, atol=1e-10)


def test_linear_calibrate_matches_svy() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "S", "S"],
            "age": [20.0, 40.0, 30.0, 50.0],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    pop = pd.DataFrame({"region": ["N"] * 50 + ["S"] * 50, "age": [35.0] * 100})
    totals = population_totals(pop, "~ region + age")
    theirs_s = _sample(df).weighting.calibrate(
        controls={Cat("region"): {"N": 50.0, "S": 50.0}, "age": 3500.0},
        wgt_name="cal_wgt",
    )
    ours, _, diag = linear_calibrate(df["pw"], df, formula="~ region + age", totals=totals, warn=False)
    np.testing.assert_allclose(ours.to_numpy(), _svy_weights(theirs_s, "cal_wgt"), rtol=1e-10, atol=1e-10)
    for row in diag["targets"]:
        assert row["achieved"] == pytest.approx(row["target"], abs=1e-8)
