"""Recovery checks for raking.

If a claim fails: diagnose; do not loosen tolerances without understanding why.
"""

import pandas as pd
import pytest

from weightpipe.methods.raking import rake


@pytest.mark.recovery
def test_rake_analytical_toy_totals() -> None:
    df = pd.DataFrame(
        {
            "sex": ["M", "M", "F", "F"],
            "region": ["N", "S", "N", "S"],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    margins = {"sex": {"M": 60.0, "F": 40.0}, "region": {"N": 30.0, "S": 70.0}}
    w, _, diag = rake(df["pw"], df, margins=margins, max_iter=100, tol=1e-12)
    assert diag["converged"] is True
    for row in diag["targets"]:
        assert row["achieved"] == pytest.approx(row["target"], abs=1e-8)
    assert float(w.sum()) == pytest.approx(100.0, abs=1e-8)


def test_proportions_force1_renormalizes_rounding() -> None:
    from weightpipe.methods.raking import proportions_to_margins

    # Same kind of rounded census vector as in the CEP example (sums to 0.998).
    region = {
        "1": 0.015,
        "2": 0.031,
        "3": 0.016,
        "4": 0.039,
        "5": 0.102,
        "6": 0.051,
        "7": 0.059,
        "8": 0.123,
        "9": 0.056,
        "10": 0.046,
        "11": 0.006,
        "12": 0.010,
        "13": 0.408,
        "14": 0.023,
        "15": 0.013,
    }
    assert abs(sum(region.values()) - 1.0) > 1e-6

    with pytest.raises(ValueError, match="must sum to 1"):
        proportions_to_margins({"region": region}, total=100.0, force1=False)

    # Default force1=True renormalizes rounded targets.
    margins = proportions_to_margins({"region": region}, total=100.0)
    assert sum(margins["region"].values()) == pytest.approx(100.0)
    assert sum(v / 100.0 for v in margins["region"].values()) == pytest.approx(1.0)


def test_recipe_calibrate_force1() -> None:
    from weightpipe import Recipe

    df = pd.DataFrame({"sex": ["M", "F", "M", "F"], "pw": [1.0, 1.0, 1.0, 1.0]})
    # deliberately does not sum to 1
    props = {"sex": {"M": 0.49, "F": 0.50}}
    fitted = Recipe(df, base_weight="pw").step_calibrate(method="raking", proportions=props).prep(warn=False)
    assert fitted.diagnostics["steps"]["calibrate"]["force1"] is True
    assert fitted.diagnostics["steps"]["calibrate"]["converged"] is True
