"""Trim stub — Iteration 2+."""

import pandas as pd


def trim_weights(
    weights: pd.Series,
    *,
    lower: float | None = None,
    upper: float | None = None,
    preserve_totals: bool = True,
) -> pd.Series:
    raise NotImplementedError("trim_weights() is deferred past Iteration 1. Validate before Recipe.trim().")
