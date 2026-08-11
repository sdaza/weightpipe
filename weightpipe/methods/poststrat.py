"""Post-stratification calibration."""

from typing import Any

import numpy as np
import pandas as pd

from weightpipe.frame import WeightFrame
from weightpipe.methods.raking import MarginDict, resolve_raking_margins
from weightpipe.steps.base import StepResult


def poststratify(
    weights: pd.Series,
    data: pd.DataFrame,
    *,
    margins: MarginDict | None = None,
    proportions: MarginDict | None = None,
    population_size: float | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Post-stratify on a single categorical variable (cell factors ``N_g / hat N_g``).

    ``margins`` / ``proportions`` must contain exactly one variable. For a full
    cross-classification, create an interaction column and post-stratify on it,
    or use ``method="linear"`` with an interaction formula.
    """
    resolved, target_meta = resolve_raking_margins(
        weights,
        margins=margins,
        proportions=proportions,
        population_size=population_size,
    )
    if len(resolved) != 1:
        raise ValueError(
            "poststratify requires exactly one variable in margins/proportions; "
            "for multiple crossed variables use an interaction column or linear calibration"
        )
    var = next(iter(resolved))
    if var not in data.columns:
        raise KeyError(f"post-stratum variable not found: {var}")

    w0 = weights.astype(float).to_numpy(copy=True)
    active = w0 > 0
    new_w = w0.copy()
    factors = np.ones_like(w0)
    f = data[var].astype(str).to_numpy()
    target = resolved[var]
    diag_rows: list[dict[str, Any]] = []

    for lev, tot in target.items():
        idx = np.where((f == str(lev)) & active)[0]
        cur = float(new_w[idx].sum()) if idx.size else 0.0
        if cur <= 0:
            factor = np.nan
        else:
            factor = float(tot) / cur
            factors[idx] = factor
            new_w[idx] = new_w[idx] * factor
        row: dict[str, Any] = {
            "variable": var,
            "category": str(lev),
            "target": float(tot),
            "prev_total": cur,
            "factor": float(factor) if np.isfinite(factor) else None,
            "achieved": float(new_w[idx].sum()) if idx.size else 0.0,
        }
        diag_rows.append(row)

    diag = {
        "method": "poststratify",
        "variable": var,
        "targets": diag_rows,
        **target_meta,
    }
    return (
        pd.Series(new_w, index=weights.index, name="weights"),
        pd.Series(factors, index=weights.index, name="factors"),
        diag,
    )


def apply_poststratify(
    frame: WeightFrame,
    *,
    margins: MarginDict | None = None,
    proportions: MarginDict | None = None,
    population_size: float | None = None,
) -> StepResult:
    weights, factors, diag = poststratify(
        frame.weights,
        frame.data,
        margins=margins,
        proportions=proportions,
        population_size=population_size,
    )
    return StepResult(weights=weights, factors=factors, diagnostics=diag)
