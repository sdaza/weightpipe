"""Survey estimands with recipe-aware bootstrap or jackknife SE/CI."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from weightpipe.estimands import (
    Estimand,
    assert_proportion_binary,
    weighted_mean,
    weighted_median,
    weighted_ratio,
    weighted_total,
)
from weightpipe.methods.glm import GlmFit, fit_glm, glm_linearized_vcov
from weightpipe.recipe import Recipe
from weightpipe.replicates import (
    BootstrapResult,
    JackknifeResult,
    boot_mean,
    boot_median,
    boot_proportion,
    boot_ratio,
    boot_total,
    bootstrap_weights,
    jackknife_weights,
)
from weightpipe.replicates.linearization import linearized_estimate
from weightpipe.result import WeightResult

VarianceMethod = Literal["bootstrap", "jackknife", "linearization"]
VariableSpec = str | Sequence[str]


def _as_names(value: VariableSpec, *, what: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    names = [str(v) for v in value]
    if not names:
        raise ValueError(f"{what} must be non-empty")
    return names


def _ratio_pairs(variables: list[str], denominator: VariableSpec) -> list[tuple[str, str]]:
    dens = _as_names(denominator, what="denominator")
    if len(dens) == 1:
        return [(name, dens[0]) for name in variables]
    if len(variables) == 1:
        return [(variables[0], den) for den in dens]
    if len(variables) != len(dens):
        raise ValueError("ratio numerators and denominators must be one value or the same length")
    return list(zip(variables, dens, strict=True))


def _domain_groups(data: pd.DataFrame, by: VariableSpec | None) -> list[tuple[dict[str, Any], np.ndarray]]:
    if by is None:
        return [({}, np.ones(len(data), dtype=bool))]
    cols = _as_names(by, what="by")
    for col in cols:
        if col not in data.columns:
            raise KeyError(f"by column not found: {col}")
    idx = np.arange(len(data))
    work = data.loc[:, cols].assign(_i=idx).dropna(subset=cols)
    if work.empty:
        return []
    groups = work.groupby(cols, sort=True, observed=True, dropna=True)
    out: list[tuple[dict[str, Any], np.ndarray]] = []
    for keys, g in groups:
        key_t = (keys,) if len(cols) == 1 else tuple(keys)
        mask = np.zeros(len(data), dtype=bool)
        mask[g["_i"].to_numpy()] = True
        out.append((dict(zip(cols, key_t, strict=True)), mask))
    return out


def _empty_row(level: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "estimate": [np.nan],
            "se": [np.nan],
            "ci_lower": [np.nan],
            "ci_upper": [np.nan],
            "level": [level],
            "R_used": [0],
        }
    )


def _annotate(
    out: pd.DataFrame,
    *,
    estimand: Estimand,
    variable: str,
    variance: VarianceMethod,
    design: str | None,
    denominator: str | None,
    p: float,
    by_labels: dict[str, Any],
) -> pd.DataFrame:
    row = out.copy()
    row["estimand"] = estimand
    row["variable"] = variable
    for name, value in by_labels.items():
        row[name] = value
    est = float(row["estimate"].iloc[0])
    se = float(row["se"].iloc[0])
    row["cv"] = abs(se / est) if np.isfinite(est) and est != 0 and np.isfinite(se) else np.nan
    if estimand == "ratio":
        row["denominator"] = denominator
    if estimand == "median" and p != 0.5:
        row["p"] = p
    row["variance"] = variance
    if design is not None:
        row["design"] = design
    front = ["estimand", "variable", *by_labels, "estimate", "se", "cv", "ci_lower", "ci_upper"]
    rest = [c for c in row.columns if c not in front]
    return row.loc[:, front + rest]


def _glm_coef_table(
    *,
    fit: GlmFit,
    se: np.ndarray,
    variance: VarianceMethod,
    design: str | None,
    r_used: int,
    level: float,
) -> pd.DataFrame:
    z = float(stats.norm.ppf(1 - (1 - level) / 2))
    rows = []
    for i, name in enumerate(fit.names):
        est = float(fit.coef[i])
        sei = float(se[i])
        cv = abs(sei / est) if est != 0 and np.isfinite(est) and np.isfinite(sei) else np.nan
        rows.append(
            {
                "estimand": "glm",
                "variable": fit.outcome,
                "term": name,
                "estimate": est,
                "se": sei,
                "cv": cv,
                "ci_lower": est - z * sei,
                "ci_upper": est + z * sei,
                "family": fit.family,
                "formula": fit.formula,
                "variance": variance,
                "design": design,
                "level": level,
                "R_used": r_used,
                "converged": fit.converged,
            }
        )
    return pd.DataFrame(rows)


def _glm_replicate_se(
    recipe: Recipe,
    *,
    formula: str,
    family: str,
    variance: VarianceMethod,
    point: WeightResult,
    replicates: int,
    seed: int | None,
    level: float,
    strata: str | None,
    psu: str | None,
    m: int | None,
) -> pd.DataFrame:
    fit0 = fit_glm(point.weights, recipe.data, formula, family=family)
    if variance == "bootstrap":
        reps = bootstrap_weights(
            recipe,
            replicates=replicates,
            strata=strata,
            psu=psu,
            m=m,
            seed=seed,
            point=point,
        )
    elif variance == "jackknife":
        reps = jackknife_weights(recipe, strata=strata, psu=psu, point=point)
    else:
        raise ValueError(f"unknown variance method: {variance!r}")
    coefs = []
    scales = []
    for r in range(reps.R):
        w = reps.replicates[:, r]
        if not np.all(np.isfinite(w)):
            continue
        try:
            fit_r = fit_glm(w, recipe.data, formula, family=family)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if len(fit_r.coef) != len(fit0.coef):
            continue
        coefs.append(np.asarray(fit_r.coef, dtype=float))
        if isinstance(reps, JackknifeResult) and reps.scales is not None:
            scales.append(float(reps.scales[r]))
        else:
            scales.append(1.0)
    if len(coefs) < 2:
        raise ValueError("glm replicate variance needs at least 2 successful replicates")
    arr = np.vstack(coefs)
    p = arr.shape[1]
    if isinstance(reps, JackknifeResult):
        vcov = np.zeros((p, p), dtype=float)
        for coef_r, scale in zip(arr, scales, strict=True):
            d = coef_r - fit0.coef
            vcov += scale * np.outer(d, d)
    else:
        delta = arr - fit0.coef
        vcov = (delta.T @ delta) / float(delta.shape[0])
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
    design_kind = recipe.design.kind if recipe.design is not None else None
    return _glm_coef_table(
        fit=fit0,
        se=se,
        variance=variance,
        design=design_kind,
        r_used=int(arr.shape[0]),
        level=level,
    )


def estimate_glm(
    recipe: Recipe,
    formula: str,
    *,
    family: str = "gaussian",
    variance: VarianceMethod = "bootstrap",
    fitted: WeightResult | None = None,
    replicates: int = 200,
    level: float = 0.95,
    seed: int | None = None,
    strata: str | None = None,
    psu: str | None = None,
    m: int | None = None,
) -> pd.DataFrame:
    """Design-based GLM: weighted IRLS coefficients with linearized or replicate SEs.

    Families: ``gaussian`` (identity), ``binomial`` (logit), ``poisson`` (log).
    Point estimates treat survey weights as known; SEs follow ``variance=``
    (ultimate-cluster Binder sandwich, or recipe-aware bootstrap / jackknife of β).
    """
    if variance not in ("bootstrap", "jackknife", "linearization"):
        raise ValueError(f"unknown variance method: {variance!r}")
    design = recipe.design
    st = strata if strata is not None else (design.strata if design is not None else None)
    ps = psu if psu is not None else (design.psu if design is not None else None)
    point = fitted if fitted is not None else recipe.prep()
    if variance == "linearization":
        fit = fit_glm(point.weights, recipe.data, formula, family=family)
        vcov, n_psu, _lonely = glm_linearized_vcov(fit, recipe.data, strata=st, psu=ps)
        se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
        design_kind = design.kind if design is not None else None
        return _glm_coef_table(
            fit=fit,
            se=se,
            variance=variance,
            design=design_kind,
            r_used=n_psu,
            level=level,
        )
    return _glm_replicate_se(
        recipe,
        formula=formula,
        family=family,
        variance=variance,
        point=point,
        replicates=replicates,
        seed=seed,
        level=level,
        strata=st,
        psu=ps,
        m=m,
    )


def _estimate_one(
    recipe: Recipe,
    variable: str,
    *,
    estimand: Estimand,
    denominator: str | None,
    point: WeightResult,
    variance: VarianceMethod,
    reps: BootstrapResult | JackknifeResult | None,
    level: float,
    p: float,
    mask: np.ndarray | None,
    strata: str | None,
    psu: str | None,
) -> pd.DataFrame:
    if mask is not None and not mask.any():
        return _empty_row(level)
    if variance == "linearization":
        try:
            return linearized_estimate(
                point.weights,
                recipe.data,
                variable,
                estimand=estimand,
                denominator=denominator,
                strata=strata,
                psu=psu,
                level=level,
                mask=mask,
            )
        except ValueError:
            return _empty_row(level)
    if reps is None:
        raise RuntimeError("replicate weights are required for bootstrap/jackknife")
    if estimand == "mean":
        return boot_mean(reps, variable, level=level, mask=mask)
    if estimand == "proportion":
        return boot_proportion(reps, variable, level=level, mask=mask)
    if estimand == "total":
        return boot_total(reps, variable, level=level, mask=mask)
    if estimand == "ratio":
        assert denominator is not None
        return boot_ratio(reps, variable, denominator, level=level, mask=mask)
    if estimand == "median":
        return boot_median(reps, variable, level=level, p=p, mask=mask)
    raise ValueError(f"unknown estimand: {estimand!r}")


def estimate(
    recipe: Recipe,
    variable: VariableSpec,
    *,
    estimand: Estimand = "mean",
    denominator: VariableSpec | None = None,
    by: VariableSpec | None = None,
    fitted: WeightResult | None = None,
    variance: VarianceMethod = "bootstrap",
    replicates: int = 200,
    level: float = 0.95,
    seed: int | None = None,
    strata: str | None = None,
    psu: str | None = None,
    m: int | None = None,
    p: float = 0.5,
) -> pd.DataFrame:
    """Point estimate, SE, and CI for survey estimands.

    ``variable`` may be one name or a list. ``by=`` splits into domains
    (like ``svyby`` / ``sample.estimation.mean(..., by=...)``). Replicate
    weights are built once and reused across variables and domains.

    Supported ``estimand`` values:

    - ``mean`` / ``proportion`` / ``total`` / ``median`` — use ``variable``
    - ``ratio`` — ``Σ w * variable / Σ w * denominator`` (``denominator`` required;
      a list of denominators is paired with ``variable``)

    Variance is a recipe-aware Rao–Wu bootstrap or delete-a-PSU jackknife, or
    ``linearization`` (ultimate-cluster Taylor SE that treats weights as fixed).
    """
    variables = _as_names(variable, what="variable")
    for name in variables:
        if name not in recipe.data.columns:
            raise KeyError(f"variable not found: {name}")
    jobs: list[tuple[str, str | None]]
    if estimand == "ratio":
        if denominator is None:
            raise ValueError("estimand='ratio' requires denominator=...")
        jobs = list(_ratio_pairs(variables, denominator))
        for _, den in jobs:
            if den not in recipe.data.columns:
                raise KeyError(f"denominator not found: {den}")
    elif denominator is not None:
        raise ValueError("denominator is only used with estimand='ratio'")
    else:
        jobs = [(name, None) for name in variables]
    if variance not in ("bootstrap", "jackknife", "linearization"):
        raise ValueError(f"unknown variance method: {variance!r}")

    design = recipe.design
    st = strata if strata is not None else (design.strata if design is not None else None)
    ps = psu if psu is not None else (design.psu if design is not None else None)
    design_kind = design.kind if design is not None else None
    point = fitted if fitted is not None else recipe.prep()
    domains = _domain_groups(recipe.data, by)
    if not domains:
        raise ValueError("by= produced no non-missing domains")

    reps: BootstrapResult | JackknifeResult | None = None
    if variance == "bootstrap":
        reps = bootstrap_weights(
            recipe,
            replicates=replicates,
            strata=st,
            psu=ps,
            m=m,
            seed=seed,
            point=point,
        )
    elif variance == "jackknife":
        reps = jackknife_weights(recipe, strata=st, psu=ps, point=point)

    rows: list[pd.DataFrame] = []
    for name, den in jobs:
        for labels, mask in domains:
            one = _estimate_one(
                recipe,
                name,
                estimand=estimand,
                denominator=den,
                point=point,
                variance=variance,
                reps=reps,
                level=level,
                p=p,
                mask=None if by is None else mask,
                strata=st,
                psu=ps,
            )
            rows.append(
                _annotate(
                    one,
                    estimand=estimand,
                    variable=name,
                    variance=variance,
                    design=design_kind,
                    denominator=den,
                    p=p,
                    by_labels=labels,
                )
            )
    return pd.concat(rows, ignore_index=True)


class Estimation:
    """Estimands with optional domain splits.

    Callable like ``estimate(variable, estimand=...)`` and also
    ``.mean()``, ``.total()``, ``.proportion()``, ``.ratio()``, ``.median()``,
    ``.glm()``.
    """

    def __init__(
        self,
        recipe: Recipe,
        *,
        fitted: WeightResult | None = None,
        fitted_fn: Callable[[], WeightResult] | None = None,
    ) -> None:
        self._recipe = recipe
        self._fitted = fitted
        self._fitted_fn = fitted_fn

    def _point(self) -> WeightResult | None:
        if self._fitted is not None:
            return self._fitted
        if self._fitted_fn is not None:
            return self._fitted_fn()
        return None

    def __call__(self, variable: VariableSpec, **kwargs: Any) -> pd.DataFrame:
        kwargs.setdefault("fitted", self._point())
        return estimate(self._recipe, variable, **kwargs)

    def mean(self, variable: VariableSpec, *, by: VariableSpec | None = None, **kwargs: Any) -> pd.DataFrame:
        return self(variable, estimand="mean", by=by, **kwargs)

    def total(self, variable: VariableSpec, *, by: VariableSpec | None = None, **kwargs: Any) -> pd.DataFrame:
        return self(variable, estimand="total", by=by, **kwargs)

    def proportion(self, variable: VariableSpec, *, by: VariableSpec | None = None, **kwargs: Any) -> pd.DataFrame:
        return self(variable, estimand="proportion", by=by, **kwargs)

    def median(self, variable: VariableSpec, *, by: VariableSpec | None = None, **kwargs: Any) -> pd.DataFrame:
        return self(variable, estimand="median", by=by, **kwargs)

    def ratio(
        self,
        variable: VariableSpec,
        denominator: VariableSpec,
        *,
        by: VariableSpec | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return self(variable, estimand="ratio", denominator=denominator, by=by, **kwargs)

    def glm(self, formula: str, *, family: str = "gaussian", **kwargs: Any) -> pd.DataFrame:
        """Design-based GLM (``y ~ x1 + x2``). One row per coefficient."""
        kwargs.setdefault("fitted", self._point())
        return estimate_glm(self._recipe, formula, family=family, **kwargs)


def point_estimate(
    weights: pd.Series | np.ndarray,
    data: pd.DataFrame,
    variable: str,
    *,
    estimand: Estimand = "mean",
    denominator: str | None = None,
    p: float = 0.5,
) -> float:
    """Point estimate without variance (Hájek mean/proportion/median, total, or ratio)."""
    if variable not in data.columns:
        raise KeyError(f"variable not found: {variable}")
    ww = np.asarray(weights, dtype=float)
    if estimand == "total":
        return weighted_total(ww, data[variable])
    if estimand == "ratio":
        if denominator is None:
            raise ValueError("estimand='ratio' requires denominator=...")
        if denominator not in data.columns:
            raise KeyError(f"denominator not found: {denominator}")
        return weighted_ratio(ww, data[variable], data[denominator])
    if estimand == "median":
        return weighted_median(ww, data[variable], p=p)
    if estimand == "proportion":
        assert_proportion_binary(data[variable])
    if estimand in ("mean", "proportion"):
        return weighted_mean(ww, data[variable])
    raise ValueError(f"unknown estimand: {estimand!r}")
