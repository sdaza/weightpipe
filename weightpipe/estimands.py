"""Point estimators for survey estimands (Hájek mean, total, ratio, median)."""

from typing import Literal

import numpy as np
import pandas as pd

Estimand = Literal["mean", "proportion", "total", "ratio", "median"]


def weighted_total(
    weights: np.ndarray | pd.Series,
    values: np.ndarray | pd.Series,
) -> float:
    ww = np.asarray(weights, dtype=float)
    x = np.asarray(values, dtype=float)
    return float(np.nansum(ww * x))


def weighted_mean(
    weights: np.ndarray | pd.Series,
    values: np.ndarray | pd.Series,
) -> float:
    ww = np.asarray(weights, dtype=float)
    x = np.asarray(values, dtype=float)
    ok = np.isfinite(x) & (ww > 0)
    denom = float(ww[ok].sum())
    if denom <= 0:
        return float("nan")
    return float((ww[ok] * x[ok]).sum() / denom)


def weighted_ratio(
    weights: np.ndarray | pd.Series,
    numerator: np.ndarray | pd.Series,
    denominator: np.ndarray | pd.Series,
) -> float:
    """Ratio of weighted totals: ``Σ w y / Σ w x``."""
    ww = np.asarray(weights, dtype=float)
    y = np.asarray(numerator, dtype=float)
    x = np.asarray(denominator, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x) & (ww > 0)
    denom = float((ww[ok] * x[ok]).sum())
    if denom == 0.0:
        return float("nan")
    return float((ww[ok] * y[ok]).sum() / denom)


def weighted_median(
    weights: np.ndarray | pd.Series,
    values: np.ndarray | pd.Series,
    *,
    p: float = 0.5,
) -> float:
    """Weighted quantile (default median) via the Hájek empirical distribution.

    Sort active units by ``values`` and find the smallest value whose cumulative
    weight share is at least ``p``. Ties keep the first crossing point.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    ww = np.asarray(weights, dtype=float)
    x = np.asarray(values, dtype=float)
    ok = np.isfinite(x) & (ww > 0)
    if not ok.any():
        return float("nan")
    xv = x[ok]
    wv = ww[ok]
    order = np.argsort(xv, kind="mergesort")
    xv = xv[order]
    wv = wv[order]
    cdf = np.cumsum(wv) / float(wv.sum())
    idx = int(np.searchsorted(cdf, p, side="left"))
    idx = min(idx, len(xv) - 1)
    return float(xv[idx])


def assert_proportion_binary(values: pd.Series | np.ndarray) -> None:
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_bool_dtype(s):
        return
    vals = pd.to_numeric(s, errors="coerce")
    uniq = set(vals.dropna().unique().tolist())
    if not uniq.issubset({0.0, 1.0}):
        raise ValueError(f"proportion requires a 0/1 variable; got values {sorted(uniq)[:8]}")
