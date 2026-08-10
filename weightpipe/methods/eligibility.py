"""Eligibility helpers: drop ineligible units."""

from typing import Any

import numpy as np
import pandas as pd

from weightpipe.frame import WeightFrame
from weightpipe.steps.base import StepResult, as_logical_mask


def drop_ineligible_weights(
    weights: pd.Series,
    data: pd.DataFrame,
    *,
    ineligible: str,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Zero out active units flagged as ineligible (not redistributed)."""
    w = weights.astype(float).to_numpy(copy=True)
    active = w > 0
    inelig = as_logical_mask(data, ineligible).to_numpy()
    drop = active & inelig
    w_before = w.copy()
    w[drop] = 0.0
    factors = np.ones_like(w)
    factors[drop] = 0.0
    # inactive units stay inactive with factor 1 for bookkeeping
    factors[~active] = 1.0
    diag = {
        "n_dropped": int(drop.sum()),
        "weight_dropped": float(w_before[drop].sum()),
        "n_remaining": int((w > 0).sum()),
    }
    return (
        pd.Series(w, index=weights.index, name="weights"),
        pd.Series(factors, index=weights.index, name="factors"),
        diag,
    )


def apply_drop_ineligible(frame: WeightFrame, *, ineligible: str) -> StepResult:
    weights, factors, diag = drop_ineligible_weights(frame.weights, frame.data, ineligible=ineligible)
    return StepResult(weights=weights, factors=factors, diagnostics=diag)
