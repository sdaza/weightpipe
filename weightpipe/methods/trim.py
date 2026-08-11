"""Weight trimming: ratio caps and automatic Tukey / Potter rules."""

from typing import Any, Literal

import numpy as np
import pandas as pd

from weightpipe.frame import WeightFrame
from weightpipe.steps.base import StepResult, make_cells

Reference = Literal["base", "median", "value"]
AutoMethod = Literal["tukey", "potter"]
RedistributeMode = Literal["proportional", "uniform", True, False]


def potter_threshold(weights: np.ndarray, *, ngrid: int = 100) -> float:
    """Potter MSE-optimal upper cutoff (weightflow ``.potter_threshold``)."""
    wv = np.asarray(weights, dtype=float)
    wv = wv[np.isfinite(wv) & (wv > 0)]
    if wv.size == 0:
        return float("nan")
    q_lo, q_hi = np.quantile(wv, [0.5, 0.999])
    if q_hi <= q_lo:
        return float(q_hi)
    grid = np.linspace(float(q_lo), float(q_hi), int(ngrid))
    best_t = float(grid[0])
    best_mse = np.inf
    for t in grid:
        capped = np.minimum(wv, t)
        bias = float(np.sum(wv[wv > t] - t))
        varc = float(np.sum(capped**2))
        mse = bias**2 + varc
        if mse < best_mse:
            best_mse = mse
            best_t = float(t)
    return best_t


def tukey_threshold(weights: np.ndarray) -> float:
    """Tukey far-out fence: ``Q3 + 3 * IQR``."""
    wv = np.asarray(weights, dtype=float)
    wv = wv[np.isfinite(wv) & (wv > 0)]
    if wv.size == 0:
        return float("nan")
    q1, q3 = np.quantile(wv, [0.25, 0.75])
    return float(q3 + 3.0 * (q3 - q1))


def trim_weights(
    weights: pd.Series,
    data: pd.DataFrame | None = None,
    *,
    max_ratio: float,
    min_ratio: float | None = None,
    reference: Reference = "median",
    base_weights: pd.Series | np.ndarray | None = None,
    redistribute: bool = True,
    by: list[str] | None = None,
    max_iter: int = 50,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Cap extreme weights and optionally redistribute excess to preserve totals."""
    if max_ratio <= 0:
        raise ValueError("max_ratio must be positive")
    if min_ratio is not None and min_ratio < 0:
        raise ValueError("min_ratio must be non-negative")
    if reference not in ("base", "median", "value"):
        raise ValueError(f"unknown reference: {reference!r}")
    if reference == "base" and base_weights is None:
        raise ValueError("reference='base' requires base_weights")

    w0 = weights.astype(float).to_numpy(copy=True)
    n = len(w0)
    active = w0 > 0
    frame = data if data is not None else pd.DataFrame(index=weights.index)
    cells = make_cells(frame, by, n).to_numpy()
    base = None if base_weights is None else np.asarray(base_weights, dtype=float)
    if base is not None and len(base) != n:
        raise ValueError("base_weights length must match weights")

    new_w = w0.copy()
    n_capped = 0
    iterations = 0
    for i in range(1, max_iter + 1):
        iterations = i
        upper = np.full(n, np.inf, dtype=float)
        lower = np.zeros(n, dtype=float)
        for g in pd.unique(cells):
            idx = np.where((cells == g) & active)[0]
            if idx.size == 0:
                continue
            if reference == "value":
                upper[idx] = float(max_ratio)
                lower[idx] = float(min_ratio) if min_ratio is not None else 0.0
            elif reference == "median":
                med = float(np.median(new_w[idx]))
                if med <= 0:
                    continue
                upper[idx] = max_ratio * med
                lower[idx] = (min_ratio * med) if min_ratio is not None else 0.0
            else:
                assert base is not None
                upper[idx] = max_ratio * base[idx]
                if min_ratio is not None:
                    lower[idx] = min_ratio * base[idx]

        before = new_w.copy()
        over = active & (new_w > upper)
        under = active & (new_w < lower) & (lower > 0)
        if not over.any() and not under.any():
            break
        n_capped += int(over.sum() + under.sum())
        new_w[over] = upper[over]
        new_w[under] = lower[under]

        if not redistribute:
            break

        for g in pd.unique(cells):
            idx = np.where((cells == g) & active)[0]
            if idx.size == 0:
                continue
            target = float(before[idx].sum())
            cur = float(new_w[idx].sum())
            residual = target - cur
            if abs(residual) < 1e-12:
                continue
            if residual > 0:
                room = upper[idx] - new_w[idx]
                free = idx[room > 1e-12]
                if free.size == 0:
                    continue
                capacity = upper[free] - new_w[free]
                total_cap = float(capacity.sum())
                if total_cap <= 0:
                    continue
                take = min(residual, total_cap)
                new_w[free] = new_w[free] + take * (capacity / total_cap)
            else:
                room = new_w[idx] - lower[idx]
                free = idx[room > 1e-12]
                if free.size == 0:
                    continue
                capacity = new_w[free] - lower[free]
                total_cap = float(capacity.sum())
                if total_cap <= 0:
                    continue
                take = min(-residual, total_cap)
                new_w[free] = new_w[free] - take * (capacity / total_cap)

        if np.allclose(before[active], new_w[active], rtol=0, atol=1e-10):
            break

    factors = np.ones(n, dtype=float)
    factors[active] = np.divide(
        new_w[active],
        w0[active],
        out=np.ones(int(active.sum())),
        where=w0[active] > 0,
    )
    diag = {
        "method": "trim",
        "reference": reference,
        "max_ratio": float(max_ratio),
        "min_ratio": None if min_ratio is None else float(min_ratio),
        "redistribute": redistribute,
        "iterations": iterations,
        "n_capped": int(n_capped),
        "sum_before": float(w0[active].sum()),
        "sum_after": float(new_w[active].sum()),
    }
    return (
        pd.Series(new_w, index=weights.index, name="weights"),
        pd.Series(factors, index=weights.index, name="factors"),
        diag,
    )


def trim_weights_auto(
    weights: pd.Series,
    *,
    lower: float = 1.0,
    upper: float | None = None,
    method: AutoMethod = "tukey",
    redistribute: Literal["proportional", "uniform"] = "proportional",
    strict: bool = True,
    max_iter: int = 50,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Automatic trimming (weightflow ``step_trim_weights`` / survey-style).

    If ``upper`` is None, choose Tukey (``Q3 + 3 IQR``) or Potter MSE cutoff.
    """
    if method not in ("tukey", "potter"):
        raise ValueError(f"unknown auto-trim method: {method!r}")
    if redistribute not in ("proportional", "uniform"):
        raise ValueError(f"unknown redistribute mode: {redistribute!r}")

    w0 = weights.astype(float).to_numpy(copy=True)
    active = w0 != 0
    wv = w0[active].copy()
    if wv.size == 0:
        fac = pd.Series(np.ones(len(w0)), index=weights.index, name="factors")
        return weights.astype(float), fac, {"method": method, "n_capped": 0}

    if upper is None:
        up = potter_threshold(wv) if method == "potter" else tukey_threshold(wv)
    else:
        up = float(upper)
    lo = float(lower)

    it = 0
    if redistribute == "uniform":
        has_trimmed = np.zeros(wv.size, dtype=bool)
        while True:
            it += 1
            outside = (wv < lo) | (wv > up)
            if not outside.any() or it > max_iter:
                break
            wvnew = np.clip(wv, lo, up)
            trimmings = wv - wvnew
            can_trim = (~outside) & (~has_trimmed)
            if can_trim.any():
                wvnew[can_trim] = wvnew[can_trim] + float(trimmings.sum()) / float(can_trim.sum())
            has_trimmed = outside | has_trimmed
            wv = wvnew
            if not strict:
                break
    else:
        while True:
            it += 1
            over = wv > up
            under = wv < lo
            if not over.any() and not under.any():
                break
            if it > max_iter:
                break
            net = float(np.sum(wv[over] - up) - np.sum(lo - wv[under]))
            wv[over] = up
            wv[under] = lo
            free = (wv < up) & (wv > lo)
            if abs(net) > 1e-12 and free.any():
                wv[free] = wv[free] + net * (wv[free] / float(wv[free].sum()))
            if not strict:
                break

    new_w = w0.copy()
    new_w[active] = wv
    factors = np.ones_like(w0)
    factors[active] = np.divide(
        new_w[active],
        w0[active],
        out=np.ones(int(active.sum())),
        where=w0[active] != 0,
    )
    diag = {
        "method": method,
        "lower": lo,
        "upper": float(up),
        "redistribute": redistribute,
        "strict": strict,
        "iterations": it,
        "n_capped": int(np.sum(w0[active] > up)),
        "n_raised": int(np.sum(w0[active] < lo)),
        "sum_before": float(w0[active].sum()),
        "sum_after": float(new_w[active].sum()),
    }
    return (
        pd.Series(new_w, index=weights.index, name="weights"),
        pd.Series(factors, index=weights.index, name="factors"),
        diag,
    )


def apply_trim(
    frame: WeightFrame,
    *,
    max_ratio: float,
    min_ratio: float | None = None,
    reference: Reference = "median",
    redistribute: bool = True,
    by: list[str] | None = None,
    max_iter: int = 50,
    base_weight: str | None = None,
) -> StepResult:
    base = None
    if reference == "base":
        col = "base_weight" if base_weight is None else base_weight
        if col not in frame.data.columns:
            raise KeyError(f"base weight column not found: {col}")
        base = frame.data[col]
    weights, factors, diag = trim_weights(
        frame.weights,
        frame.data,
        max_ratio=max_ratio,
        min_ratio=min_ratio,
        reference=reference,
        base_weights=base,
        redistribute=redistribute,
        by=by,
        max_iter=max_iter,
    )
    return StepResult(weights=weights, factors=factors, diagnostics=diag)


def apply_trim_weights(
    frame: WeightFrame,
    *,
    lower: float = 1.0,
    upper: float | None = None,
    method: AutoMethod = "tukey",
    redistribute: Literal["proportional", "uniform"] = "proportional",
    strict: bool = True,
    max_iter: int = 50,
) -> StepResult:
    weights, factors, diag = trim_weights_auto(
        frame.weights,
        lower=lower,
        upper=upper,
        method=method,
        redistribute=redistribute,
        strict=strict,
        max_iter=max_iter,
    )
    return StepResult(weights=weights, factors=factors, diagnostics=diag)
