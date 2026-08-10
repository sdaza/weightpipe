"""Shared fixtures for weightpipe tests."""

import pandas as pd
import pytest


@pytest.fixture
def toy_sample() -> pd.DataFrame:
    """Tiny survey-like table for WeightFrame / Recipe smoke tests."""
    return pd.DataFrame(
        {
            "unit_id": [1, 2, 3, 4],
            "stratum": ["A", "A", "B", "B"],
            "psu": [10, 10, 20, 21],
            "age_group": ["young", "old", "young", "old"],
            "design_weight": [1.0, 1.5, 2.0, 1.2],
        }
    )
