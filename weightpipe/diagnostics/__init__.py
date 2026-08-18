"""Weight / design-effect / balance diagnostics."""

import numpy as np
import pandas as pd

from weightpipe.diagnostics.balance import BalanceReport, balance
from weightpipe.diagnostics.margins import (
    attach_margin_table,
    margin_table_from_targets,
    margins,
    weighted_category_margins,
)
from weightpipe.result import WeightResult

__all__ = [
    "BalanceReport",
    "attach_margin_table",
    "balance",
    "design_effect",
    "ess",
    "margin_table_from_targets",
    "margins",
    "weighted_category_margins",
]


def design_effect(weights: pd.Series | WeightResult) -> float:
    """Kish design effect from unequal weighting: 1 + CV(w)^2 = n * sum(w^2) / sum(w)^2."""
    w = weights.weights if isinstance(weights, WeightResult) else weights
    w = w.astype(float)
    active = w[w > 0]
    if active.empty:
        return float("nan")
    n = float(len(active))
    s = float(active.sum())
    if s <= 0:
        return float("nan")
    return float(n * float(np.sum(np.square(active.to_numpy()))) / (s * s))


def ess(weights: pd.Series | WeightResult) -> float:
    """Effective sample size under Kish: (sum w)^2 / sum(w^2)."""
    w = weights.weights if isinstance(weights, WeightResult) else weights
    w = w.astype(float)
    active = w[w > 0]
    if active.empty:
        return float("nan")
    s = float(active.sum())
    ss = float(np.sum(np.square(active.to_numpy())))
    if ss <= 0:
        return float("nan")
    return float((s * s) / ss)
