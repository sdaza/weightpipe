"""Linear / GREG calibration (unbounded, ridge, and bounded)."""

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from weightpipe.frame import WeightFrame
from weightpipe.methods.design_matrix import align_totals, design_matrix
from weightpipe.steps.base import StepResult

CalFun = Literal["linear", "logit", "raking"]


def _ridge_diag(penalty: float | dict[str, float], columns: list[str], a: np.ndarray) -> np.ndarray:
    """Weightflow-compatible ridge diagonal: ``diag(mean(diag(A)) / costs)``."""
    s = float(np.mean(np.diag(a)))
    if isinstance(penalty, (int, float)):
        costs = np.full(len(columns), float(penalty))
    else:
        costs = np.array([float(penalty[c]) for c in columns], dtype=float)
    if np.any(costs <= 0):
        raise ValueError("penalty costs must be positive")
    return np.diag(s / costs)


def _solve_calib(a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:
        lam, *_ = np.linalg.lstsq(a, rhs, rcond=None)
        return lam


def _calib_ds(
    x: np.ndarray,
    d: np.ndarray,
    t_vec: np.ndarray,
    *,
    calfun: CalFun,
    bounds: tuple[float, float] | None,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> tuple[np.ndarray, bool]:
    """Deville–Särndal Newton solver (weightflow ``.calib_ds``)."""
    if bounds is None:
        lo, up = -np.inf, np.inf
    else:
        lo, up = float(bounds[0]), float(bounds[1])
        if not (lo < 1.0 < up):
            raise ValueError(f"bounds must satisfy L < 1 < U, got {(lo, up)}")

    clz = 500.0
    if calfun == "logit":
        if bounds is None:
            raise ValueError("calfun='logit' requires bounds")
        a_coef = (up - lo) / ((1.0 - lo) * (up - 1.0))

        def ffun(u: np.ndarray) -> np.ndarray:
            e = np.exp(np.clip(a_coef * u, -clz, clz))
            return (lo * (up - 1.0) + up * (1.0 - lo) * e) / ((up - 1.0) + (1.0 - lo) * e)

        def fp(u: np.ndarray) -> np.ndarray:
            g = ffun(u)
            return a_coef * (g - lo) * (up - g) / (up - lo)

    elif calfun == "raking":

        def ffun(u: np.ndarray) -> np.ndarray:
            return np.clip(np.exp(np.clip(u, -clz, clz)), lo, up)

        def fp(u: np.ndarray) -> np.ndarray:
            g = np.exp(np.clip(u, -clz, clz))
            return np.where((g > lo) & (g < up), g, 0.0)

    else:

        def ffun(u: np.ndarray) -> np.ndarray:
            return np.clip(1.0 + u, lo, up)

        def fp(u: np.ndarray) -> np.ndarray:
            g = 1.0 + u
            return np.where((g > lo) & (g < up), 1.0, 0.0)

    # column scale for numerical stability
    s = np.sqrt(np.mean(x**2, axis=0))
    s = np.where(s == 0, 1.0, s)
    xs = x / s
    ts = t_vec / s
    lam = np.zeros(xs.shape[1], dtype=float)

    def resid_norm(lam_vec: np.ndarray) -> float:
        ach = (d[:, None] * ffun(xs @ lam_vec)[:, None] * xs).sum(axis=0)
        return float(np.max(np.abs(ach - ts) / (np.abs(ts) + 1.0)))

    cur = resid_norm(lam)
    ok = False
    for _ in range(max_iter):
        if cur < tol:
            ok = True
            break
        u = xs @ lam
        j = (xs * (d * fp(u))[:, None]).T @ xs
        rhs = ts - (d[:, None] * ffun(u)[:, None] * xs).sum(axis=0)
        ridge = 1e-7 * (float(np.mean(np.diag(j))) + np.finfo(float).eps)
        dl = _solve_calib(j + np.eye(j.shape[0]) * ridge, rhs)
        stepf = 1.0
        improved = False
        for _h in range(20):
            nr = resid_norm(lam + stepf * dl)
            if np.isfinite(nr) and nr <= cur:
                lam = lam + stepf * dl
                cur = nr
                improved = True
                break
            stepf *= 0.5
        if not improved:
            break

    return ffun(xs @ lam), ok


def linear_calibrate(
    weights: pd.Series,
    data: pd.DataFrame,
    *,
    formula: str | list[str] | tuple[str, ...],
    totals: dict[str, Any] | pd.Series,
    bounds: tuple[float, float] | list[float] | None = None,
    penalty: float | dict[str, float] | None = None,
    calfun: CalFun = "linear",
    max_iter: int = 100,
    warn: bool = True,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Linear / GREG calibration with optional bounds and ridge penalty.

    - Unbounded ``calfun="linear"``: ``g = 1 + x'λ`` (may be negative).
    - ``penalty``: ridge; only for unbounded linear (weightflow rule).
    - ``bounds=(L, U)`` or ``calfun="logit"|"raking"``: Deville–Särndal solver.
    """
    if bounds is not None:
        bounds_t = (float(bounds[0]), float(bounds[1]))
    else:
        bounds_t = None
    if penalty is not None and (bounds_t is not None or calfun != "linear"):
        raise ValueError("penalty (ridge) is only available for unbounded linear calibration")

    x_df = design_matrix(data, formula)
    cols = list(x_df.columns)
    t_map = align_totals(cols, totals)
    t_vec = np.array([t_map[c] for c in cols], dtype=float)

    w0 = weights.astype(float).to_numpy(copy=True)
    active = w0 > 0
    if not active.any():
        raise ValueError("no active units for linear calibration")

    x = x_df.to_numpy(dtype=float)
    xa = x[active]
    wa = w0[active]
    use_ds = calfun != "linear" or bounds_t is not None
    converged = True

    if not use_ds:
        achieved0 = (wa[:, None] * xa).sum(axis=0)
        diff = t_vec - achieved0
        a = (xa * wa[:, None]).T @ xa
        if penalty is not None:
            a = a + _ridge_diag(penalty, cols, a)
        try:
            lam = np.linalg.solve(a, diff)
            solved = True
        except np.linalg.LinAlgError:
            lam = _solve_calib(a, diff)
            solved = False
            if warn:
                warnings.warn(
                    "linear calibration design matrix was singular; used least-squares solution.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        g_active = 1.0 + xa @ lam
    else:
        g_active, converged = _calib_ds(xa, wa, t_vec, calfun=calfun, bounds=bounds_t, max_iter=max_iter)
        lam = np.full(len(cols), np.nan)
        solved = converged
        if warn and not converged:
            warnings.warn(
                "Bounded calibration did not fully converge (bounds may be infeasible).",
                RuntimeWarning,
                stacklevel=2,
            )

    g = np.ones_like(w0)
    g[active] = g_active
    new_w = w0 * g
    if warn and not use_ds and (new_w[active] < 0).any():
        warnings.warn(
            f"linear calibration produced {(new_w[active] < 0).sum()} negative weight(s).",
            RuntimeWarning,
            stacklevel=2,
        )

    factors = np.ones_like(w0)
    factors[active] = g[active]
    achieved = (new_w[active, None] * xa).sum(axis=0)
    diag_rows = [
        {
            "variable": c,
            "target": float(t_vec[i]),
            "achieved": float(achieved[i]),
            "abs_diff": abs(float(achieved[i] - t_vec[i])),
        }
        for i, c in enumerate(cols)
    ]
    diag = {
        "method": "linear",
        "calfun": calfun,
        "bounds": None if bounds_t is None else [bounds_t[0], bounds_t[1]],
        "penalty": penalty if not isinstance(penalty, dict) else dict(penalty),
        "formula": formula if isinstance(formula, str) else list(formula),
        "columns": cols,
        "lambda": {c: float(lam[i]) for i, c in enumerate(cols)},
        "solved": solved,
        "converged": converged,
        "n_negative": int((new_w[active] < 0).sum()),
        "targets": diag_rows,
    }
    return (
        pd.Series(new_w, index=weights.index, name="weights"),
        pd.Series(factors, index=weights.index, name="factors"),
        diag,
    )


def apply_linear_calibrate(
    frame: WeightFrame,
    *,
    formula: str | list[str] | tuple[str, ...],
    totals: dict[str, Any] | pd.Series,
    bounds: tuple[float, float] | list[float] | None = None,
    penalty: float | dict[str, float] | None = None,
    calfun: CalFun = "linear",
    max_iter: int = 100,
    warn: bool = True,
) -> StepResult:
    weights, factors, diag = linear_calibrate(
        frame.weights,
        frame.data,
        formula=formula,
        totals=totals,
        bounds=bounds,
        penalty=penalty,
        calfun=calfun,
        max_iter=max_iter,
        warn=warn,
    )
    return StepResult(weights=weights, factors=factors, diagnostics=diag)
