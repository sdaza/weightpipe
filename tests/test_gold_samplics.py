"""Cross-checks against samplics (optional ``gold`` extra).

Install with ``uv sync --extra gold`` (CI does this). Tests skip if samplics
is not installed. samplics is archived upstream; we use it as a frozen Python
reference until a stable ``svy`` gold path is wired.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("samplics")

from samplics.weighting import SampleWeight  # noqa: E402

from weightpipe.methods.design_matrix import design_matrix, population_totals  # noqa: E402
from weightpipe.methods.linear import linear_calibrate  # noqa: E402
from weightpipe.methods.nonresponse import weighting_class_nonresponse  # noqa: E402
from weightpipe.methods.poststrat import poststratify  # noqa: E402
from weightpipe.methods.raking import rake  # noqa: E402

pytestmark = pytest.mark.gold


def test_poststrat_matches_samplics() -> None:
    df = pd.DataFrame({"region": ["N", "N", "S", "S"], "pw": [1.0, 1.0, 1.0, 1.0]})
    control = {"N": 10.0, "S": 30.0}
    theirs = SampleWeight().poststratify(samp_weight=df["pw"], control=control, domain=df["region"])
    ours, _, _ = poststratify(df["pw"], df, margins={"region": control})
    np.testing.assert_allclose(ours.to_numpy(), theirs, rtol=0, atol=1e-12)


def test_weighting_class_nr_matches_samplics() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "N", "S", "S"],
            "responded": [1, 1, 0, 1, 0],
            "pw": [1.0, 1.0, 1.0, 2.0, 2.0],
        }
    )
    status = np.where(df["responded"] == 1, "rr", "nr")
    theirs = SampleWeight().adjust(
        samp_weight=df["pw"].to_numpy(),
        adj_class=df["region"].to_numpy(),
        resp_status=status,
        resp_dict={"rr": "respondent", "nr": "non-respondent", "in": "ineligible", "uk": "unknown"},
        unknown_to_inelig=False,
    )
    ours, _, _ = weighting_class_nonresponse(df["pw"], df, respondent="responded", by=["region"])
    np.testing.assert_allclose(ours.to_numpy(), theirs, rtol=0, atol=1e-12)


def test_raking_matches_samplics() -> None:
    df = pd.DataFrame(
        {
            "sex": ["M", "M", "F", "F"],
            "region": ["N", "S", "N", "S"],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    control = {"sex": {"M": 60.0, "F": 40.0}, "region": {"N": 30.0, "S": 70.0}}
    theirs = SampleWeight().rake(
        samp_weight=df["pw"],
        margins={"sex": df["sex"], "region": df["region"]},
        control=control,
        tol=1e-12,
        ctrl_tol=1e-12,
        max_iter=200,
    )
    ours, _, diag = rake(df["pw"], df, margins=control, max_iter=200, tol=1e-12, warn=False)
    assert diag["converged"] is True
    np.testing.assert_allclose(ours.to_numpy(), theirs, rtol=1e-10, atol=1e-10)


def test_linear_calibrate_matches_samplics() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "S", "S"],
            "age": [20.0, 40.0, 30.0, 50.0],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    pop = pd.DataFrame({"region": ["N"] * 50 + ["S"] * 50, "age": [35.0] * 100})
    totals = population_totals(pop, "~ region + age")
    x = design_matrix(df, "~ region + age")
    control = {c: totals[c] for c in x.columns}
    theirs = SampleWeight().calibrate(
        samp_weight=df["pw"].to_numpy(),
        aux_vars=x.to_numpy(),
        control=control,
        bounded=False,
    )
    ours, _, diag = linear_calibrate(df["pw"], df, formula="~ region + age", totals=totals, warn=False)
    np.testing.assert_allclose(ours.to_numpy(), theirs, rtol=1e-10, atol=1e-10)
    for row in diag["targets"]:
        assert row["achieved"] == pytest.approx(row["target"], abs=1e-8)
