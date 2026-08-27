"""Always-on gold CSV comparisons (frozen svy exports).

Regenerate with: ``uv run --extra gold python tests/gold/generate_svy_gold.py``
Source: svy Sample.weighting on the toy frames embedded in each CSV.
Tolerances: rtol=1e-10 (do not loosen without diagnosing).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from weightpipe.methods.linear import linear_calibrate
from weightpipe.methods.nonresponse import weighting_class_nonresponse
from weightpipe.methods.poststrat import poststratify
from weightpipe.methods.raking import rake

GOLD = Path(__file__).resolve().parent / "gold"
pytestmark = pytest.mark.gold


def test_rake_matches_svy_gold_csv() -> None:
    g = pd.read_csv(GOLD / "raking_2x2_svy.csv")
    margins = {"sex": {"M": 60.0, "F": 40.0}, "region": {"N": 30.0, "S": 70.0}}
    w, _, diag = rake(g["pw"], g, margins=margins, max_iter=200, tol=1e-12, warn=False)
    assert diag["converged"] is True
    np.testing.assert_allclose(w.to_numpy(), g["weight_svy"].to_numpy(), rtol=1e-10, atol=1e-10)


def test_poststrat_matches_svy_gold_csv() -> None:
    g = pd.read_csv(GOLD / "poststrat_region_svy.csv")
    w, _, _ = poststratify(g["pw"], g, margins={"region": {"N": 10.0, "S": 30.0}})
    np.testing.assert_allclose(w.to_numpy(), g["weight_svy"].to_numpy(), rtol=0, atol=1e-12)


def test_nr_matches_svy_gold_csv() -> None:
    g = pd.read_csv(GOLD / "nr_weighting_class_svy.csv")
    w, _, _ = weighting_class_nonresponse(g["pw"], g, respondent="responded", by=["region"])
    np.testing.assert_allclose(w.to_numpy(), g["weight_svy"].to_numpy(), rtol=0, atol=1e-12)


def test_linear_calibrate_matches_svy_gold_csv() -> None:
    g = pd.read_csv(GOLD / "linear_calibrate_svy.csv")
    totals = {"(Intercept)": 100.0, "regionS": 50.0, "age": 3500.0}
    w, _, diag = linear_calibrate(g["pw"], g, formula="~ region + age", totals=totals, warn=False)
    assert diag["converged"] is True
    np.testing.assert_allclose(w.to_numpy(), g["weight_svy"].to_numpy(), rtol=1e-10, atol=1e-10)
