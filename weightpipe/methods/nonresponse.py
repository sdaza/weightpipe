"""Nonresponse adjustment methods."""

from typing import Any

import numpy as np
import pandas as pd

from weightpipe.frame import WeightFrame
from weightpipe.steps.base import StepResult, as_logical_mask, make_cells


def weighting_class_nonresponse(
    weights: pd.Series,
    data: pd.DataFrame,
    *,
    respondent: str,
    by: list[str] | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Inflate respondents within cells; zero nonrespondents (class adjustment)."""
    w = weights.astype(float).to_numpy(copy=True)
    n = len(w)
    respondent_mask = as_logical_mask(data, respondent).to_numpy()
    eligible = w > 0
    cells = make_cells(data, by, n).to_numpy()
    factors = np.ones(n, dtype=float)
    cell_rows: list[dict[str, Any]] = []

    for g in pd.unique(cells):
        idx = np.where((cells == g) & eligible)[0]
        if idx.size == 0:
            continue
        resp = idx[respondent_mask[idx]]
        nr = idx[~respondent_mask[idx]]
        w_resp = float(w[resp].sum()) if resp.size else 0.0
        w_tot = float(w[idx].sum())
        factor = (w_tot / w_resp) if w_resp > 0 else np.nan
        if np.isfinite(factor):
            factors[resp] = factor
            factors[nr] = 0.0
            w[resp] = w[resp] * factor
            w[nr] = 0.0
        cell_rows.append(
            {
                "cell": str(g),
                "n_respondents": int(resp.size),
                "n_nonresponse": int(nr.size),
                "factor": float(factor) if np.isfinite(factor) else None,
            }
        )

    diag = {"cells": cell_rows, "method": "weighting_class"}
    return (
        pd.Series(w, index=weights.index, name="weights"),
        pd.Series(factors, index=weights.index, name="factors"),
        diag,
    )


def apply_weighting_class_nonresponse(
    frame: WeightFrame,
    *,
    respondent: str,
    by: list[str] | None = None,
) -> StepResult:
    weights, factors, diag = weighting_class_nonresponse(frame.weights, frame.data, respondent=respondent, by=by)
    return StepResult(weights=weights, factors=factors, diagnostics=diag)
