"""Ultimate-cluster / Taylor linearization SEs (weights treated as fixed)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from weightpipe.estimands import Estimand, assert_proportion_binary


def _stratum_psu_codes(
    data: pd.DataFrame,
    n: int,
    strata: str | None,
    psu: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    st = np.array(["1"] * n, dtype=object) if strata is None else data[strata].astype(str).to_numpy()
    cl = np.arange(n).astype(str) if psu is None else data[psu].astype(str).to_numpy()
    return st, cl


def ultimate_cluster_variance(
    z: np.ndarray,
    strata: np.ndarray,
    psu: np.ndarray,
) -> tuple[float, int, tuple[str, ...]]:
    """With-replacement PSU-total variance of linearized residuals ``z``.

    For each stratum *h* with ``n_h >= 2`` PSUs:

    ``v_h = n_h / (n_h - 1) * sum_i (t_{hi} - mean_h)^2``

    Lonely strata (one PSU) contribute 0 and are named in the warning list.
    """
    z = np.asarray(z, dtype=float)
    var = 0.0
    n_psu = 0
    lonely: list[str] = []
    for h in pd.unique(strata):
        idx = np.where(strata == h)[0]
        psus = pd.unique(psu[idx])
        nh = len(psus)
        if nh < 2:
            lonely.append(str(h))
            continue
        totals = np.array([float(z[idx][psu[idx] == p].sum()) for p in psus], dtype=float)
        mean_t = float(totals.mean())
        var += (nh / (nh - 1.0)) * float(np.sum((totals - mean_t) ** 2))
        n_psu += nh
    return var, n_psu, tuple(lonely)


def linearized_residuals(
    weights: np.ndarray,
    y: np.ndarray,
    *,
    estimand: Estimand,
    x: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Point estimate and unit-level linearized residual for a Hájek estimand."""
    w = np.asarray(weights, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(w) & np.isfinite(yy) & (w > 0)
    if estimand == "ratio":
        if x is None:
            raise ValueError("ratio linearization requires a denominator")
        xx = np.asarray(x, dtype=float)
        ok = ok & np.isfinite(xx)
        ww, yv, xv = w[ok], yy[ok], xx[ok]
        x_tot = float(np.sum(ww * xv))
        if x_tot == 0:
            raise ValueError("ratio linearization requires a non-zero weighted denominator total")
        point = float(np.sum(ww * yv) / x_tot)
        z = np.zeros_like(w)
        z[ok] = ww * (yv - point * xv) / x_tot
        return point, z
    ww, yv = w[ok], yy[ok]
    w_tot = float(ww.sum())
    if estimand == "total":
        point = float(np.sum(ww * yv))
        z = np.zeros_like(w)
        z[ok] = ww * yv
        return point, z
    if w_tot <= 0:
        raise ValueError("linearization requires a positive weight total")
    point = float(np.sum(ww * yv) / w_tot)
    z = np.zeros_like(w)
    z[ok] = (ww / w_tot) * (yv - point)
    return point, z


def linearized_estimate(
    weights: pd.Series | np.ndarray,
    data: pd.DataFrame,
    variable: str,
    *,
    estimand: Estimand = "mean",
    denominator: str | None = None,
    strata: str | None = None,
    psu: str | None = None,
    level: float = 0.95,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Hájek point estimate with ultimate-cluster linearization SE (weights fixed)."""
    if estimand == "median":
        raise ValueError("variance='linearization' does not support median; use bootstrap or jackknife")
    if variable not in data.columns:
        raise KeyError(f"variable not found: {variable}")
    if estimand == "proportion":
        assert_proportion_binary(data[variable])
    if estimand == "ratio":
        if denominator is None:
            raise ValueError("estimand='ratio' requires denominator=...")
        if denominator not in data.columns:
            raise KeyError(f"denominator not found: {denominator}")
    elif denominator is not None:
        raise ValueError("denominator is only used with estimand='ratio'")

    w = np.asarray(weights, dtype=float)
    n = len(data)
    if len(w) != n:
        raise ValueError("weights length must match data rows")
    if mask is not None:
        mask_a = np.asarray(mask, dtype=bool)
        if mask_a.shape[0] != n:
            raise ValueError("mask length must match data rows")
        w = w.copy()
        w[~mask_a] = 0.0
    y = data[variable].to_numpy(dtype=float)
    x = None if denominator is None else data[denominator].to_numpy(dtype=float)
    point, z = linearized_residuals(w, y, estimand=estimand, x=x)
    st, cl = _stratum_psu_codes(data, n, strata, psu)
    var, n_psu, lonely = ultimate_cluster_variance(z, st, cl)
    if lonely:
        warnings.warn(
            "Strata with a single PSU contributed no linearization variance: " + ", ".join(sorted(lonely)),
            RuntimeWarning,
            stacklevel=2,
        )
    if n_psu < 2:
        raise ValueError(
            "linearization requires at least one stratum with 2+ PSUs (or omit psu= to treat rows as PSUs)"
        )
    se = float(np.sqrt(var))
    zcrit = float(stats.norm.ppf(1 - (1 - level) / 2))
    return pd.DataFrame(
        {
            "estimate": [point],
            "se": [se],
            "ci_lower": [point - zcrit * se],
            "ci_upper": [point + zcrit * se],
            "level": [level],
            "R_used": [n_psu],
        }
    )
