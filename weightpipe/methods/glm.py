"""Design-based GLM (weighted IRLS + Binder / replicate SEs)."""

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from weightpipe.estimands import assert_proportion_binary
from weightpipe.methods.design_matrix import design_matrix
from weightpipe.replicates.linearization import _stratum_psu_codes, ultimate_cluster_covariance

GlmFamily = Literal["gaussian", "binomial", "poisson"]

_FAMILY_ALIASES = {
    "gaussian": "gaussian",
    "normal": "gaussian",
    "identity": "gaussian",
    "binomial": "binomial",
    "logit": "binomial",
    "logistic": "binomial",
    "quasibinomial": "binomial",
    "poisson": "poisson",
    "log": "poisson",
    "quasipoisson": "poisson",
}


def split_glm_formula(formula: str) -> tuple[str, str]:
    """Split ``y ~ x1 + x2`` into response name and RHS (``~ x1 + x2`` or ``1``)."""
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("glm formula must look like 'y ~ x1 + x2'")
    lhs, rhs = formula.split("~", 1)
    y = lhs.strip()
    rhs = rhs.strip()
    if not y:
        raise ValueError("glm formula needs a response on the left of ~")
    if not rhs:
        rhs = "1"
    return y, rhs


def resolve_family(family: str) -> GlmFamily:
    key = str(family).strip().lower()
    if key not in _FAMILY_ALIASES:
        raise ValueError(f"unknown glm family: {family!r} (use gaussian, binomial, or poisson)")
    return _FAMILY_ALIASES[key]  # type: ignore[return-value]


def _expit(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


@dataclass
class GlmFit:
    coef: np.ndarray
    names: list[str]
    mu: np.ndarray
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    ok: np.ndarray
    family: GlmFamily
    outcome: str
    formula: str
    converged: bool


def _solve_wls(X: np.ndarray, z: np.ndarray, ww: np.ndarray) -> np.ndarray:
    xtw = X.T * ww
    a = xtw @ X
    try:
        return np.linalg.solve(a, xtw @ z)
    except np.linalg.LinAlgError as exc:
        raise ValueError("glm design matrix is rank-deficient") from exc


def fit_glm(
    weights: pd.Series | np.ndarray,
    data: pd.DataFrame,
    formula: str,
    *,
    family: str = "gaussian",
    mask: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> GlmFit:
    """Weighted IRLS coefficients (survey weights treated as known)."""
    fam = resolve_family(family)
    outcome, rhs = split_glm_formula(formula)
    if outcome not in data.columns:
        raise KeyError(f"response not found: {outcome}")
    x_df = design_matrix(data, rhs, allow_intercept_only=True)
    X_all = x_df.to_numpy(dtype=float)
    y_all = data[outcome].to_numpy(dtype=float)
    w_all = np.asarray(weights, dtype=float).copy()
    if len(w_all) != len(data):
        raise ValueError("weights length must match data rows")
    if mask is not None:
        mask_a = np.asarray(mask, dtype=bool)
        if mask_a.shape[0] != len(data):
            raise ValueError("mask length must match data rows")
        w_all[~mask_a] = 0.0

    if fam == "binomial":
        assert_proportion_binary(data[outcome])

    ok = np.isfinite(w_all) & (w_all > 0) & np.isfinite(y_all) & np.isfinite(X_all).all(axis=1)
    if fam == "poisson":
        ok = ok & (y_all >= 0)
    if not ok.any():
        raise ValueError("glm has no rows with positive weight and finite data")

    X = X_all[ok]
    y = y_all[ok]
    w = w_all[ok]
    names = list(x_df.columns)
    p = X.shape[1]
    mu_all = np.full(len(data), np.nan)

    if fam == "gaussian":
        beta = _solve_wls(X, y, w)
        mu = X @ beta
        converged = True
    else:
        beta = np.zeros(p, dtype=float)
        if names and names[0] == "(Intercept)":
            if fam == "binomial":
                p0 = float(np.clip(np.average(y, weights=w), 1e-6, 1.0 - 1e-6))
                beta[0] = float(np.log(p0 / (1.0 - p0)))
            else:
                m0 = float(max(np.average(y, weights=w), 1e-8))
                beta[0] = float(np.log(m0))
        converged = False
        for _ in range(max_iter):
            eta = X @ beta
            if fam == "binomial":
                mu = np.clip(_expit(eta), 1e-12, 1.0 - 1e-12)
                varmu = mu * (1.0 - mu)
                z = eta + (y - mu) / varmu
                ww = w * varmu
            else:
                mu = np.exp(np.clip(eta, -20.0, 20.0))
                mu = np.maximum(mu, 1e-12)
                z = eta + (y - mu) / mu
                ww = w * mu
            beta_new = _solve_wls(X, z, ww)
            if float(np.max(np.abs(beta_new - beta))) < tol:
                beta = beta_new
                converged = True
                break
            beta = beta_new
        eta = X @ beta
        if fam == "binomial":
            mu = np.clip(_expit(eta), 1e-12, 1.0 - 1e-12)
        else:
            mu = np.exp(np.clip(eta, -20.0, 20.0))

    mu_all[ok] = mu
    return GlmFit(
        coef=np.asarray(beta, dtype=float),
        names=names,
        mu=mu_all,
        X=X_all,
        y=y_all,
        w=w_all,
        ok=ok,
        family=fam,
        outcome=outcome,
        formula=formula,
        converged=converged,
    )


def glm_information(fit: GlmFit) -> np.ndarray:
    """Observed information A = -∂U/∂β from the weighted GLM estimating equations."""
    X = fit.X[fit.ok]
    w = fit.w[fit.ok]
    mu = fit.mu[fit.ok]
    if fit.family == "gaussian":
        ww = w
    elif fit.family == "binomial":
        ww = w * mu * (1.0 - mu)
    else:
        ww = w * mu
    return (X.T * ww) @ X


def glm_scores(fit: GlmFit) -> np.ndarray:
    """Unit-level scores U_i = w_i (y_i - μ_i) x_i (canonical links)."""
    z = np.zeros((len(fit.w), len(fit.coef)), dtype=float)
    resid = np.zeros(len(fit.w), dtype=float)
    resid[fit.ok] = fit.y[fit.ok] - fit.mu[fit.ok]
    z[fit.ok] = (fit.w[fit.ok] * resid[fit.ok])[:, None] * fit.X[fit.ok]
    return z


def glm_linearized_vcov(
    fit: GlmFit,
    data: pd.DataFrame,
    *,
    strata: str | None = None,
    psu: str | None = None,
) -> tuple[np.ndarray, int, tuple[str, ...]]:
    """Binder sandwich: A^{-1} Var_design(U) A^{-1} with ultimate-cluster Var(U)."""
    a = glm_information(fit)
    try:
        ainv = np.linalg.inv(a)
    except np.linalg.LinAlgError as exc:
        raise ValueError("glm information matrix is singular") from exc
    scores = glm_scores(fit)
    st, cl = _stratum_psu_codes(data, len(fit.w), strata, psu)
    vu, n_psu, lonely = ultimate_cluster_covariance(scores, st, cl)
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
    vcov = ainv @ vu @ ainv
    return vcov, n_psu, lonely
