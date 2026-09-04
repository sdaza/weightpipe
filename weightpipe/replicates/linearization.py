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


def _psu_totals(
    z: np.ndarray,
    strata: np.ndarray,
    psu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum unit-level scores ``z`` by (stratum, PSU).

    When every row is already its own PSU, totals are ``z`` itself.
    Group sums use ``np.bincount`` so they match the previous per-PSU
    ``.sum()`` (row order) rather than pandas ``groupby``.
    """
    z = np.asarray(z, dtype=float)
    st = np.asarray(strata)
    cl = np.asarray(psu)
    one_d = z.ndim == 1
    z2 = z.reshape(-1, 1) if one_d else z
    n, p = z2.shape
    if n == 0 or pd.Index(cl).is_unique:
        return z, st
    st_codes, st_uniques = pd.factorize(st, sort=False)
    psu_codes, _ = pd.factorize(cl, sort=False)
    nlev = np.int64(psu_codes.max()) + 1
    combined = st_codes.astype(np.int64, copy=False) * nlev + psu_codes.astype(np.int64, copy=False)
    pair_codes, pair_uniques = pd.factorize(combined, sort=False)
    n_groups = len(pair_uniques)
    totals = np.empty((n_groups, p), dtype=float)
    for j in range(p):
        totals[:, j] = np.bincount(pair_codes, weights=z2[:, j], minlength=n_groups)
    st_of = np.asarray(pair_uniques, dtype=np.int64) // nlev
    totals_strata = np.asarray(st_uniques)[st_of]
    if one_d:
        return totals[:, 0], totals_strata
    return totals, totals_strata


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
    totals, totals_strata = _psu_totals(z, strata, psu)
    frame = pd.DataFrame({"t": totals, "st": totals_strata})
    grp = frame.groupby("st", sort=False, observed=True)["t"]
    var = 0.0
    n_psu = 0
    lonely: list[str] = []
    for h, t in grp:
        tot = np.asarray(t, dtype=float)
        nh = tot.size
        if nh < 2:
            lonely.append(str(h))
            continue
        mean_t = float(tot.mean())
        var += (nh / (nh - 1.0)) * float(np.sum((tot - mean_t) ** 2))
        n_psu += nh
    return var, n_psu, tuple(lonely)


def ultimate_cluster_covariance(
    z: np.ndarray,
    strata: np.ndarray,
    psu: np.ndarray,
) -> tuple[np.ndarray, int, tuple[str, ...]]:
    """PSU-total covariance of a unit-level score matrix ``z`` (n × p)."""
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n, p = z.shape
    if n != len(strata) or n != len(psu):
        raise ValueError("z, strata, and psu lengths must match")
    totals, totals_strata = _psu_totals(z, strata, psu)
    var = np.zeros((p, p), dtype=float)
    n_psu = 0
    lonely: list[str] = []
    st_frame = pd.DataFrame(totals)
    st_frame["st"] = totals_strata
    value_cols = [c for c in st_frame.columns if c != "st"]
    for h, g in st_frame.groupby("st", sort=False, observed=True):
        arr = g[value_cols].to_numpy(dtype=float)
        n_h = arr.shape[0]
        if n_h < 2:
            lonely.append(str(h))
            continue
        mean_t = arr.mean(axis=0)
        centered = arr - mean_t
        var += (n_h / (n_h - 1.0)) * (centered.T @ centered)
        n_psu += n_h
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
