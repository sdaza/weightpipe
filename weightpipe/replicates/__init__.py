"""Recipe-aware Rao–Wu bootstrap replicate weights."""

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
    """Resample PSUs within strata and re-``prep`` the full recipe per replicate."""
    if replicates < 1:
        raise ValueError("replicates must be >= 1")

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
        scaled = data.copy()
        scaled[bw_col] = bw0 * fac
        try:
            rep_recipe = Recipe(
                data=scaled,
                base_weight=bw_col,
                unit_id=recipe.unit_id,
                steps=recipe.steps,
                meta=dict(recipe.meta),
            )
            reps[:, b] = rep_recipe.prep(warn=False).weights.to_numpy(dtype=float)
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


def bootstrap_estimate(
    boot: BootstrapResult,
    statistic: Callable[[np.ndarray | pd.Series, pd.DataFrame], float],
    *,
    level: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap estimate, SE, and normal CI for a scalar statistic."""
    theta_hat = float(statistic(boot.weights, boot.data))
    thetas = []
    for b in range(boot.R):
        w = boot.replicates[:, b]
        if not np.all(np.isfinite(w)):
            thetas.append(np.nan)
            continue
        thetas.append(float(statistic(w, boot.data)))
    thetas_a = np.asarray(thetas, dtype=float)
    good = np.isfinite(thetas_a)
    if not good.any():
        se = float("nan")
    else:
        se = float(np.sqrt(np.mean((thetas_a[good] - theta_hat) ** 2)))
    dropped = int((~good).sum())
    if dropped:
        warnings.warn(
            f"{dropped} non-finite replicate(s) dropped from SE.",
            RuntimeWarning,
            stacklevel=2,
        )
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


def boot_total(boot: BootstrapResult, variable: str, *, level: float = 0.95) -> pd.DataFrame:
    def _stat(w: Any, d: pd.DataFrame) -> float:
        ww = np.asarray(w, dtype=float)
        return float(np.nansum(ww * d[variable].to_numpy(dtype=float)))

    return bootstrap_estimate(boot, _stat, level=level)


def boot_mean(boot: BootstrapResult, variable: str, *, level: float = 0.95) -> pd.DataFrame:
    def _stat(w: Any, d: pd.DataFrame) -> float:
        ww = np.asarray(w, dtype=float)
        x = d[variable].to_numpy(dtype=float)
        ok = np.isfinite(x) & (ww > 0)
        denom = float(ww[ok].sum())
        if denom <= 0:
            return float("nan")
        return float((ww[ok] * x[ok]).sum() / denom)

    return bootstrap_estimate(boot, _stat, level=level)
