"""Recipe-aware bootstrap and jackknife replicate weights."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from weightpipe.recipe import Recipe
from weightpipe.result import WeightResult


@dataclass(frozen=True)
class BootstrapResult:
    """Recipe-aware bootstrap replicate weights."""

    replicates: np.ndarray  # (n_units, R)
    weights: pd.Series
    data: pd.DataFrame
    strata: str | None
    psu: str | None
    R: int
    base_weight: str
    method: str = "bootstrap"
    seed: int | None = None
    lonely_strata: tuple[str, ...] = ()


@dataclass(frozen=True)
class JackknifeResult:
    """Recipe-aware delete-a-PSU (JKn) replicate weights."""

    replicates: np.ndarray  # (n_units, R)
    weights: pd.Series
    data: pd.DataFrame
    strata: str | None
    psu: str | None
    R: int
    base_weight: str
    method: str = "jackknife"
    # Per-replicate JKn scale (n_h - 1) / n_h for the deleted PSU's stratum.
    scales: np.ndarray | None = None
    lonely_strata: tuple[str, ...] = ()
    deleted_psu: tuple[tuple[str, str], ...] = ()  # (stratum, psu) per replicate


def _rao_wu_factors(
    strata: np.ndarray,
    psu: np.ndarray,
    *,
    rng: np.random.Generator,
    m: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Draw one Rao–Wu rescaling factor vector."""
    n = len(strata)
    fac = np.ones(n, dtype=float)
    lonely: list[str] = []
    for h in pd.unique(strata):
        idx = np.where(strata == h)[0]
        psus = pd.unique(psu[idx])
        nh = len(psus)
        if nh < 2:
            fac[idx] = 1.0
            lonely.append(str(h))
            continue
        mh = nh - 1 if m is None else min(int(m), nh - 1)
        if mh < 1:
            fac[idx] = 1.0
            lonely.append(str(h))
            continue
        draws = rng.integers(0, nh, size=mh)
        cnt = np.bincount(draws, minlength=nh).astype(float)
        lam = 1.0 - np.sqrt(mh / (nh - 1)) + np.sqrt(mh / (nh - 1)) * (nh / mh) * cnt
        lam_map = dict(zip(psus.astype(str), lam, strict=True))
        fac[idx] = np.array([lam_map[str(p)] for p in psu[idx]], dtype=float)
    return fac, lonely


def _prep_replicate_weights(recipe: Recipe, base_weights: np.ndarray) -> np.ndarray:
    """Re-run the recipe on scaled base weights without copying the microdata frame."""
    return recipe.prep(
        warn=False,
        record=False,
        min_cell_n=None,
        max_factor=None,
        base_weights=base_weights,
    ).weights.to_numpy(dtype=float)


def bootstrap_weights(
    recipe: Recipe,
    *,
    replicates: int = 200,
    strata: str | None = None,
    psu: str | None = None,
    m: int | None = None,
    seed: int | None = None,
    point: WeightResult | None = None,
) -> BootstrapResult:
    """Resample PSUs within strata and re-``prep`` the full recipe per replicate.

    If ``strata`` / ``psu`` are omitted and ``recipe.design`` is set, those fields
    are taken from the design.
    """
    if replicates < 1:
        raise ValueError("replicates must be >= 1")

    design = recipe.design
    if strata is None and design is not None:
        strata = design.strata
    if psu is None and design is not None:
        psu = design.psu

    data = recipe.data
    n = len(data)
    bw_col = recipe.base_weight
    bw0 = data[bw_col].to_numpy(dtype=float)

    st = np.array(["1"] * n, dtype=object) if strata is None else data[strata].astype(str).to_numpy()
    cl = np.arange(n).astype(str) if psu is None else data[psu].astype(str).to_numpy()

    point_fit = point if point is not None else recipe.prep()
    point_w = point_fit.weights.to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    # Pre-draw factors for reproducibility; re-prep is deterministic given factors.
    facs = []
    lonely_all: set[str] = set()
    for _ in range(replicates):
        fac, lonely = _rao_wu_factors(st, cl, rng=rng, m=m)
        facs.append(fac)
        lonely_all.update(lonely)

    reps = np.empty((n, replicates), dtype=float)
    failed = 0
    for b, fac in enumerate(facs):
        try:
            reps[:, b] = _prep_replicate_weights(recipe, bw0 * fac)
        except Exception:
            reps[:, b] = np.nan
            failed += 1

    if lonely_all:
        warnings.warn(
            "Strata with a single PSU were not resampled (no bootstrap variance): " + ", ".join(sorted(lonely_all)),
            RuntimeWarning,
            stacklevel=2,
        )
    if failed:
        warnings.warn(
            f"{failed} replicate(s) failed and were set to NA.",
            RuntimeWarning,
            stacklevel=2,
        )

    return BootstrapResult(
        replicates=reps,
        weights=pd.Series(point_w, index=data.index, name="weight"),
        data=data,
        strata=strata,
        psu=psu,
        R=replicates,
        base_weight=bw_col,
        seed=seed,
        lonely_strata=tuple(sorted(lonely_all)),
    )


def _jkn_factor_plan(
    strata: np.ndarray,
    psu: np.ndarray,
) -> tuple[list[np.ndarray], list[float], list[tuple[str, str]], list[str]]:
    """Build delete-a-PSU factor vectors and JKn variance scales."""
    n = len(strata)
    factors: list[np.ndarray] = []
    scales: list[float] = []
    deleted: list[tuple[str, str]] = []
    lonely: list[str] = []
    for h in pd.unique(strata):
        idx = np.where(strata == h)[0]
        psus = pd.unique(psu[idx])
        nh = len(psus)
        if nh < 2:
            lonely.append(str(h))
            continue
        scale = (nh - 1) / nh
        for p in psus:
            fac = np.ones(n, dtype=float)
            # leave other strata at 1; within h, deleted PSU → 0, others → nh/(nh-1)
            for pp in psus:
                p_idx = idx[psu[idx] == pp]
                if str(pp) == str(p):
                    fac[p_idx] = 0.0
                else:
                    fac[p_idx] = nh / (nh - 1)
            factors.append(fac)
            scales.append(scale)
            deleted.append((str(h), str(p)))
    return factors, scales, deleted, lonely


def jackknife_weights(
    recipe: Recipe,
    *,
    strata: str | None = None,
    psu: str | None = None,
    point: WeightResult | None = None,
) -> JackknifeResult:
    """Delete-a-PSU (JKn) factors; re-``prep`` the full recipe per replicate.

    Lonely strata (single PSU) contribute no jackknife replicates.
    """
    design = recipe.design
    if strata is None and design is not None:
        strata = design.strata
    if psu is None and design is not None:
        psu = design.psu

    data = recipe.data
    n = len(data)
    bw_col = recipe.base_weight
    bw0 = data[bw_col].to_numpy(dtype=float)

    st = np.array(["1"] * n, dtype=object) if strata is None else data[strata].astype(str).to_numpy()
    cl = np.arange(n).astype(str) if psu is None else data[psu].astype(str).to_numpy()

    point_fit = point if point is not None else recipe.prep()
    point_w = point_fit.weights.to_numpy(dtype=float)

    facs, scales, deleted, lonely = _jkn_factor_plan(st, cl)
    if not facs:
        raise ValueError("jackknife requires at least one stratum with 2+ PSUs")

    r = len(facs)
    reps = np.empty((n, r), dtype=float)
    failed = 0
    for b, fac in enumerate(facs):
        try:
            reps[:, b] = _prep_replicate_weights(recipe, bw0 * fac)
        except Exception:
            reps[:, b] = np.nan
            failed += 1

    if lonely:
        warnings.warn(
            "Strata with a single PSU were skipped (no jackknife contribution): " + ", ".join(sorted(lonely)),
            RuntimeWarning,
            stacklevel=2,
        )
    if failed:
        warnings.warn(
            f"{failed} jackknife replicate(s) failed and were set to NA.",
            RuntimeWarning,
            stacklevel=2,
        )

    return JackknifeResult(
        replicates=reps,
        weights=pd.Series(point_w, index=data.index, name="weight"),
        data=data,
        strata=strata,
        psu=psu,
        R=r,
        base_weight=bw_col,
        scales=np.asarray(scales, dtype=float),
        lonely_strata=tuple(sorted(lonely)),
        deleted_psu=tuple(deleted),
    )


def _replicate_estimate(
    result: BootstrapResult | JackknifeResult,
    statistic: Callable[[np.ndarray | pd.Series, pd.DataFrame], float],
    *,
    level: float = 0.95,
) -> pd.DataFrame:
    """Point estimate, SE, and normal CI from bootstrap or jackknife replicates."""
    theta_hat = float(statistic(result.weights, result.data))
    thetas = []
    for b in range(result.R):
        w = result.replicates[:, b]
        if not np.all(np.isfinite(w)):
            thetas.append(np.nan)
            continue
        thetas.append(float(statistic(w, result.data)))
    thetas_a = np.asarray(thetas, dtype=float)
    good = np.isfinite(thetas_a)
    dropped = int((~good).sum())
    if dropped:
        warnings.warn(
            f"{dropped} non-finite replicate(s) dropped from SE.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not good.any():
        se = float("nan")
    elif isinstance(result, JackknifeResult) and result.scales is not None:
        scales = np.asarray(result.scales, dtype=float)[good]
        se = float(np.sqrt(np.sum(scales * (thetas_a[good] - theta_hat) ** 2)))
    else:
        se = float(np.sqrt(np.mean((thetas_a[good] - theta_hat) ** 2)))
    z = float(stats.norm.ppf(1 - (1 - level) / 2))
    return pd.DataFrame(
        {
            "estimate": [theta_hat],
            "se": [se],
            "ci_lower": [theta_hat - z * se],
            "ci_upper": [theta_hat + z * se],
            "level": [level],
            "R_used": [int(good.sum())],
        }
    )


def bootstrap_estimate(
    boot: BootstrapResult,
    statistic: Callable[[np.ndarray | pd.Series, pd.DataFrame], float],
    *,
    level: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap estimate, SE, and normal CI for a scalar statistic."""
    return _replicate_estimate(boot, statistic, level=level)


def jackknife_estimate(
    jack: JackknifeResult,
    statistic: Callable[[np.ndarray | pd.Series, pd.DataFrame], float],
    *,
    level: float = 0.95,
) -> pd.DataFrame:
    """JKn estimate, SE, and normal CI for a scalar statistic."""
    return _replicate_estimate(jack, statistic, level=level)


def boot_total(
    boot: BootstrapResult | JackknifeResult,
    variable: str,
    *,
    level: float = 0.95,
) -> pd.DataFrame:
    def _stat(w: Any, d: pd.DataFrame) -> float:
        ww = np.asarray(w, dtype=float)
        return float(np.nansum(ww * d[variable].to_numpy(dtype=float)))

    return _replicate_estimate(boot, _stat, level=level)


def boot_mean(
    boot: BootstrapResult | JackknifeResult,
    variable: str,
    *,
    level: float = 0.95,
) -> pd.DataFrame:
    def _stat(w: Any, d: pd.DataFrame) -> float:
        ww = np.asarray(w, dtype=float)
        x = d[variable].to_numpy(dtype=float)
        ok = np.isfinite(x) & (ww > 0)
        denom = float(ww[ok].sum())
        if denom <= 0:
            return float("nan")
        return float((ww[ok] * x[ok]).sum() / denom)

    return _replicate_estimate(boot, _stat, level=level)


def boot_proportion(
    boot: BootstrapResult | JackknifeResult,
    variable: str,
    *,
    level: float = 0.95,
) -> pd.DataFrame:
    """Hájek proportion for a 0/1 (or boolean) variable."""
    from weightpipe.estimands import assert_proportion_binary, weighted_mean

    assert_proportion_binary(boot.data[variable])

    def _stat(w: Any, d: pd.DataFrame) -> float:
        xv = d[variable]
        if pd.api.types.is_bool_dtype(xv):
            xnum = xv.astype(float)
        else:
            xnum = pd.to_numeric(xv, errors="coerce")
        return weighted_mean(w, xnum)

    return _replicate_estimate(boot, _stat, level=level)


def boot_median(
    boot: BootstrapResult | JackknifeResult,
    variable: str,
    *,
    level: float = 0.95,
    p: float = 0.5,
) -> pd.DataFrame:
    """Weighted quantile (default median) with replicate SE/CI."""
    from weightpipe.estimands import weighted_median

    if variable not in boot.data.columns:
        raise KeyError(f"variable not found: {variable}")

    def _stat(w: Any, d: pd.DataFrame) -> float:
        return weighted_median(w, d[variable], p=p)

    return _replicate_estimate(boot, _stat, level=level)


def boot_ratio(
    boot: BootstrapResult | JackknifeResult,
    numerator: str,
    denominator: str,
    *,
    level: float = 0.95,
) -> pd.DataFrame:
    """Ratio of weighted totals ``Σ w * numerator / Σ w * denominator``."""
    from weightpipe.estimands import weighted_ratio

    if numerator not in boot.data.columns:
        raise KeyError(f"numerator variable not found: {numerator}")
    if denominator not in boot.data.columns:
        raise KeyError(f"denominator variable not found: {denominator}")

    def _stat(w: Any, d: pd.DataFrame) -> float:
        return weighted_ratio(w, d[numerator], d[denominator])

    return _replicate_estimate(boot, _stat, level=level)


jack_total = boot_total
jack_mean = boot_mean
jack_proportion = boot_proportion
jack_median = boot_median
jack_ratio = boot_ratio
