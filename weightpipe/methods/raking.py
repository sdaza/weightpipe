"""Raking / IPF calibration."""

import warnings
from typing import Any

import numpy as np
import pandas as pd

from weightpipe.frame import WeightFrame
from weightpipe.steps.base import StepResult

MarginDict = dict[str, dict[str, float]]


def proportions_to_margins(
    proportions: MarginDict,
    *,
    total: float,
    atol: float = 1e-6,
    force1: bool = True,
) -> MarginDict:
    """Convert per-variable category proportions into absolute margin totals.

    Each variable's proportions must be non-negative. By default (``force1=True``),
each distribution is renormalized to sum to 1 — useful for rounded census
targets. Set ``force1=False`` to require an exact sum of 1 (within ``atol``).
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    margins: MarginDict = {}
    for var, dist in proportions.items():
        if not dist:
            raise ValueError(f"proportions[{var!r}] is empty")
        vals = {str(k): float(v) for k, v in dist.items()}
        if any(v < 0 for v in vals.values()):
            raise ValueError(f"proportions[{var!r}] must be non-negative")
        s = sum(vals.values())
        if force1:
            if s <= 0:
                raise ValueError(f"proportions[{var!r}] must have a positive sum to force1")
            vals = {k: v / s for k, v in vals.items()}
        elif abs(s - 1.0) > atol:
            raise ValueError(
                f"proportions[{var!r}] must sum to 1 (got {s}); "
                "pass shares that form a distribution, or set force1=True"
            )
        margins[var] = {k: v * float(total) for k, v in vals.items()}
    return margins


def resolve_raking_margins(
    weights: pd.Series,
    *,
    margins: MarginDict | None = None,
    proportions: MarginDict | None = None,
    population_size: float | None = None,
    force1: bool = True,
) -> tuple[MarginDict, dict[str, Any]]:
    """Resolve absolute margins from either counts or proportions.

    If ``proportions`` is given, they are scaled by ``population_size`` when
    provided, otherwise by the sum of strictly positive incoming weights
    (preserves the current weighted total).
    """
    if margins is not None and proportions is not None:
        raise ValueError("Pass only one of margins or proportions")
    if margins is None and proportions is None:
        raise ValueError("raking requires margins=... or proportions=...")

    meta: dict[str, Any] = {}
    if proportions is not None:
        w = weights.astype(float).to_numpy()
        total = float(population_size) if population_size is not None else float(w[w > 0].sum())
        if population_size is None:
            meta["total_source"] = "sum_active_weights"
        else:
            meta["total_source"] = "population_size"
        meta["total"] = total
        meta["force1"] = bool(force1)
        resolved = proportions_to_margins(proportions, total=total, force1=force1)
        # Store the (possibly renormalized) proportions used as targets.
        meta["proportions"] = {
            var: {lev: float(tot) / total for lev, tot in dist.items()} for var, dist in resolved.items()
        }
        meta["resolved_margins"] = resolved
        return resolved, meta

    assert margins is not None
    meta["total_source"] = "margins"
    meta["resolved_margins"] = margins
    return margins, meta


def rake(
    weights: pd.Series,
    data: pd.DataFrame,
    *,
    margins: MarginDict | None = None,
    proportions: MarginDict | None = None,
    population_size: float | None = None,
    force1: bool = True,
    max_iter: int = 50,
    tol: float = 1e-6,
    warn: bool = True,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Iterative proportional fitting to margins or proportion distributions."""
    resolved, target_meta = resolve_raking_margins(
        weights,
        margins=margins,
        proportions=proportions,
        population_size=population_size,
        force1=force1,
    )

    w0 = weights.astype(float).to_numpy(copy=True)
    active = w0 > 0
    new_w = w0.copy()
    it = 0
    maxdiff = np.inf

    while it < max_iter and maxdiff >= tol:
        it += 1
        maxdiff = 0.0
        for var, target in resolved.items():
            if var not in data.columns:
                raise KeyError(f"Margin variable not found: {var}")
            f = data[var].astype(str).to_numpy()
            for lev, tot in target.items():
                idx = np.where((f == str(lev)) & active)[0]
                cur = float(new_w[idx].sum()) if idx.size else 0.0
                if cur > 0:
                    adj = float(tot) / cur
                    new_w[idx] *= adj
                    maxdiff = max(maxdiff, abs(adj - 1.0))

    converged = maxdiff < tol
    if not converged and warn:
        warnings.warn(
            f"Raking did not converge after {it} iterations (max relative change = {maxdiff:.2e}, tol = {tol:.2e}).",
            RuntimeWarning,
            stacklevel=2,
        )

    factors = np.ones_like(new_w)
    factors[active] = np.where(w0[active] > 0, new_w[active] / w0[active], 1.0)

    diag_rows: list[dict[str, Any]] = []
    for var, target in resolved.items():
        f = data[var].astype(str).to_numpy()
        for lev, tot in target.items():
            idx = np.where((f == str(lev)) & active)[0]
            achieved = float(new_w[idx].sum()) if idx.size else 0.0
            row: dict[str, Any] = {
                "variable": var,
                "category": str(lev),
                "target": float(tot),
                "achieved": achieved,
                "abs_diff": abs(achieved - float(tot)),
            }
            if "proportions" in target_meta:
                row["target_proportion"] = float(target_meta["proportions"][var][str(lev)])
                row["achieved_proportion"] = achieved / float(target_meta["total"]) if target_meta["total"] else np.nan
            diag_rows.append(row)

    diag = {
        "method": "raking",
        "iterations": it,
        "converged": converged,
        "max_rel_change": float(maxdiff if np.isfinite(maxdiff) else np.nan),
        "targets": diag_rows,
        **target_meta,
    }
    return (
        pd.Series(new_w, index=weights.index, name="weights"),
        pd.Series(factors, index=weights.index, name="factors"),
        diag,
    )


def apply_raking(
    frame: WeightFrame,
    *,
    margins: MarginDict | None = None,
    proportions: MarginDict | None = None,
    population_size: float | None = None,
    force1: bool = True,
    max_iter: int = 50,
    tol: float = 1e-6,
    warn: bool = True,
) -> StepResult:
    weights, factors, diag = rake(
        frame.weights,
        frame.data,
        margins=margins,
        proportions=proportions,
        population_size=population_size,
        force1=force1,
        max_iter=max_iter,
        tol=tol,
        warn=warn,
    )
    return StepResult(weights=weights, factors=factors, diagnostics=diag)
