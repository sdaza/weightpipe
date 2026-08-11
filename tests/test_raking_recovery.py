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
