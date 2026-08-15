"""Nonresponse adjustment methods."""

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from weightpipe.frame import WeightFrame
from weightpipe.methods.cluster_utils import assert_cluster_column, cluster_table
from weightpipe.methods.design_matrix import design_matrix, parse_formula
from weightpipe.steps.base import StepResult, as_logical_mask, make_cells

PropensityEngine = Literal["logit", "gbm", "forest"]

_PROPENSITY_ENGINES = frozenset({"logit", "gbm", "forest"})


def weighting_class_nonresponse(
    weights: pd.Series,
    data: pd.DataFrame,
    *,
    respondent: str,
    by: list[str] | None = None,
    cluster: str | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Inflate respondents within cells; zero nonrespondents (class adjustment).

    With ``cluster``, each cluster counts once (mean weight). A cluster responds
    only if all active members respond; the factor is applied to every member.
    """
    w0 = weights.astype(float).to_numpy(copy=True)
    n = len(w0)
    respondent_mask = as_logical_mask(data, respondent).to_numpy()
    eligible = w0 > 0
    cells = make_cells(data, by, n).to_numpy()
    new_w = w0.copy()
    factors = np.ones(n, dtype=float)
    cell_rows: list[dict[str, Any]] = []

    if cluster is None:
        for g in pd.unique(cells):
            idx = np.where((cells == g) & eligible)[0]
            if idx.size == 0:
                continue
            resp = idx[respondent_mask[idx]]
            nr = idx[~respondent_mask[idx]]
            w_resp = float(w0[resp].sum()) if resp.size else 0.0
            w_tot = float(w0[idx].sum())
            factor = (w_tot / w_resp) if w_resp > 0 else np.nan
            if np.isfinite(factor):
                factors[resp] = factor
                factors[nr] = 0.0
                new_w[resp] = w0[resp] * factor
                new_w[nr] = 0.0
            cell_rows.append(
                {
                    "cell": str(g),
                    "level": "person",
                    "n_respondents": int(resp.size),
                    "n_nonresponse": int(nr.size),
                    "factor": float(factor) if np.isfinite(factor) else None,
                }
            )
    else:
        assert_cluster_column(data, cluster)
        tbl, idx_el = cluster_table(
            w0,
            data,
            cluster=cluster,
            eligible=eligible,
            flag=respondent_mask,
            flag_reduce="all",
            cells=cells,
        )
        cl = data[cluster].astype(str).to_numpy()
        factor_h: dict[str, float] = {}
        for g in tbl["cell"].unique():
            sel = tbl["cell"] == g
            wh = tbl.loc[sel, "weight"].to_numpy(dtype=float)
            resp_h = tbl.loc[sel, "flag"].to_numpy(dtype=bool)
            w_tot = float(wh.sum())
            w_resp = float(wh[resp_h].sum())
            factor = (w_tot / w_resp) if w_resp > 0 else np.nan
            for h, is_resp in zip(tbl.loc[sel, "cluster"].astype(str), resp_h, strict=True):
                factor_h[h] = float(factor) if (is_resp and np.isfinite(factor)) else 0.0
            cell_rows.append(
                {
                    "cell": str(g),
                    "level": "household",
                    "n_respondents": int(resp_h.sum()),
                    "n_nonresponse": int((~resp_h).sum()),
                    "factor": float(factor) if np.isfinite(factor) else None,
                }
            )
        for i in idx_el:
            f = factor_h[str(cl[i])]
            factors[i] = f
            new_w[i] = w0[i] * f

    diag = {"cells": cell_rows, "method": "weighting_class", "cluster": cluster}
    return (
        pd.Series(new_w, index=weights.index, name="weights"),
        pd.Series(factors, index=weights.index, name="factors"),
        diag,
    )


def _propensity_classes(p: np.ndarray, num_classes: int) -> np.ndarray:
    """Assign quantile propensity classes; collapse to one class if constant."""
    finite = np.isfinite(p)
    out = np.full(len(p), -1, dtype=int)
    if not finite.any():
        return out
    vals = p[finite]
    if np.nanmax(vals) - np.nanmin(vals) < 1e-12:
        out[finite] = 0
        return out
    try:
        cats = pd.qcut(vals, q=num_classes, labels=False, duplicates="drop")
    except ValueError:
        out[finite] = 0
        return out
    out[finite] = np.asarray(cats, dtype=int)
    return out


def _feature_matrix(
    data: pd.DataFrame,
    formula: str | list[str] | tuple[str, ...],
    *,
    engine: PropensityEngine,
) -> np.ndarray:
    x = design_matrix(data, formula)
    if engine == "logit":
        return x.to_numpy(dtype=float)
    # Trees do not need an intercept column.
    cols = [c for c in x.columns if c != "(Intercept)"]
    if not cols:
        raise ValueError(f"{engine} propensity needs at least one non-intercept term")
    return x.loc[:, cols].to_numpy(dtype=float)


def _fit_propensity(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    engine: PropensityEngine,
    seed: int | None,
) -> np.ndarray:
    """Return clipped P(respond=1) for rows in ``x``."""
    if len(np.unique(y)) < 2:
        raise ValueError("propensity model needs both respondents and nonrespondents among active units")
    rng = 0 if seed is None else int(seed)
    if engine == "logit":
        model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=200, fit_intercept=False)
        model.fit(x, y, sample_weight=sample_weight)
    elif engine == "gbm":
        model = GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.08,
            max_depth=2,
            random_state=rng,
        )
        model.fit(x, y, sample_weight=sample_weight)
    elif engine == "forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=5,
            random_state=rng,
        )
        model.fit(x, y, sample_weight=sample_weight)
    else:
        raise ValueError(f"unknown propensity engine: {engine!r}")
    return np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)


def propensity_nonresponse(
    weights: pd.Series,
    data: pd.DataFrame,
    *,
    respondent: str,
    formula: str | list[str] | tuple[str, ...],
    engine: PropensityEngine = "logit",
    num_classes: int | None = 5,
    weight_model: bool = True,
    cluster: str | None = None,
    store_propensity: bool = True,
    seed: int | None = 0,
) -> tuple[pd.Series, pd.Series, dict[str, Any], dict[str, pd.Series]]:
    """Response propensity NR; class adjustment or direct ``1/p`` factors.

    ``engine`` is ``logit``, ``gbm`` (gradient boosting), or ``forest``.
    With ``cluster``, the model is fit on one row per cluster (mean weight,
    response = all members responded) and the factor is broadcast to members.

    When ``store_propensity`` is True, returns extra columns ``propensity`` and
    (if ``num_classes`` is not None) ``propensity_class`` for later calibration.
    """
    if engine not in _PROPENSITY_ENGINES:
        raise ValueError(f"unknown propensity engine: {engine!r} (supports logit, gbm, forest)")

    terms = parse_formula(formula)
    missing = [t for t in terms if t not in data.columns]
    if missing:
        raise KeyError(f"propensity formula columns not found: {missing}")

    w0 = weights.astype(float).to_numpy(copy=True)
    n = len(w0)
    respondent_mask = as_logical_mask(data, respondent).to_numpy()
    eligible = w0 > 0
    cells = make_cells(data, None, n).to_numpy()
    new_w = w0.copy()
    factors = np.ones(n, dtype=float)
    cell_rows: list[dict[str, Any]] = []
    single_class_alert = False
    p_full = np.full(n, np.nan, dtype=float)
    class_full = np.full(n, np.nan, dtype=float)

    if cluster is None:
        idx = np.where(eligible)[0]
        if idx.size == 0:
            raise ValueError("no active units for propensity nonresponse")
        x = _feature_matrix(data.iloc[idx], formula, engine=engine)
        y = respondent_mask[idx].astype(int)
        sw = w0[idx] if weight_model else np.ones(idx.size, dtype=float)
        p_hat = _fit_propensity(x, y, sw, engine=engine, seed=seed)
        p_full[idx] = p_hat

        if num_classes is None:
            resp = idx[respondent_mask[idx]]
            nr = idx[~respondent_mask[idx]]
            factors[resp] = 1.0 / p_full[resp]
            factors[nr] = 0.0
            new_w[resp] = w0[resp] * factors[resp]
            new_w[nr] = 0.0
            method_detail = "direct_1_over_p"
        else:
            if num_classes < 1:
                raise ValueError("num_classes must be >= 1 or None")
            classes = _propensity_classes(p_hat, num_classes)
            if len(np.unique(classes[classes >= 0])) <= 1:
                single_class_alert = True
            class_full[idx] = classes.astype(float)
            for g in sorted(set(classes.tolist())):
                if g < 0:
                    continue
                g_idx = np.where((class_full == g) & eligible)[0]
                resp = g_idx[respondent_mask[g_idx]]
                nr = g_idx[~respondent_mask[g_idx]]
                w_resp = float(w0[resp].sum()) if resp.size else 0.0
                w_tot = float(w0[g_idx].sum())
                factor = (w_tot / w_resp) if w_resp > 0 else np.nan
                if np.isfinite(factor):
                    factors[resp] = factor
                    factors[nr] = 0.0
                    new_w[resp] = w0[resp] * factor
                    new_w[nr] = 0.0
                cell_rows.append(
                    {
                        "cell": f"pclass_{g}",
                        "n_respondents": int(resp.size),
                        "n_nonresponse": int(nr.size),
                        "factor": float(factor) if np.isfinite(factor) else None,
                        "mean_p": float(np.mean(p_full[g_idx])),
                    }
                )
            method_detail = f"propensity_classes_{num_classes}"
        mean_p = float(np.nanmean(p_full))
    else:
        assert_cluster_column(data, cluster)
        tbl, idx_el = cluster_table(
            w0,
            data,
            cluster=cluster,
            eligible=eligible,
            flag=respondent_mask,
            flag_reduce="all",
            cells=cells,
        )
        if tbl.empty:
            raise ValueError("no active clusters for propensity nonresponse")
        cl = data[cluster].astype(str).to_numpy()
        rep_pos = []
        for h in tbl["cluster"].astype(str):
            mem = idx_el[cl[idx_el] == h]
            rep_pos.append(int(mem[0]))
        ddh = data.iloc[rep_pos].copy()
        y = tbl["flag"].astype(int).to_numpy()
        sw = tbl["weight"].to_numpy(dtype=float) if weight_model else np.ones(len(tbl))
        x = _feature_matrix(ddh, formula, engine=engine)
        p_hat = _fit_propensity(x, y, sw, engine=engine, seed=seed)
        factor_h: dict[str, float] = {}
        class_h: dict[str, float] = {}
        if num_classes is None:
            for h, resp, p in zip(tbl["cluster"].astype(str), tbl["flag"], p_hat, strict=True):
                factor_h[h] = (1.0 / float(p)) if bool(resp) else 0.0
            method_detail = "direct_1_over_p_household"
        else:
            classes = _propensity_classes(p_hat, num_classes)
            if len(np.unique(classes[classes >= 0])) <= 1:
                single_class_alert = True
            wh = tbl["weight"].to_numpy(dtype=float)
            resp_h = tbl["flag"].to_numpy(dtype=bool)
            hs = tbl["cluster"].astype(str).to_numpy()
            for h, g in zip(hs, classes, strict=True):
                class_h[h] = float(g)
            for g in sorted(set(classes.tolist())):
                if g < 0:
                    continue
                sel = classes == g
                w_tot = float(wh[sel].sum())
                w_resp = float(wh[sel][resp_h[sel]].sum())
                factor = (w_tot / w_resp) if w_resp > 0 else np.nan
                for h, is_resp in zip(hs[sel], resp_h[sel], strict=True):
                    factor_h[h] = float(factor) if (is_resp and np.isfinite(factor)) else 0.0
                cell_rows.append(
                    {
                        "cell": f"pclass_{g}",
                        "level": "household",
                        "n_respondents": int(resp_h[sel].sum()),
                        "n_nonresponse": int((~resp_h[sel]).sum()),
                        "factor": float(factor) if np.isfinite(factor) else None,
                        "mean_p": float(np.mean(p_hat[sel])),
                    }
                )
            method_detail = f"propensity_classes_{num_classes}_household"
        p_map = dict(zip(tbl["cluster"].astype(str), p_hat, strict=True))
        for i in idx_el:
            h = str(cl[i])
            factors[i] = factor_h[h]
            new_w[i] = w0[i] * factor_h[h]
            p_full[i] = float(p_map[h])
            if h in class_h:
                class_full[i] = class_h[h]
        mean_p = float(np.mean(p_hat))

    extra: dict[str, pd.Series] = {}
    if store_propensity:
        extra["propensity"] = pd.Series(p_full, index=weights.index, name="propensity")
        if num_classes is not None:
            # store as nullable Int64-friendly float with nan for inactive
            extra["propensity_class"] = pd.Series(class_full, index=weights.index, name="propensity_class")

    diag: dict[str, Any] = {
        "method": "propensity",
        "engine": engine,
        "formula": formula if isinstance(formula, str) else list(formula),
        "num_classes": num_classes,
        "weight_model": weight_model,
        "detail": method_detail,
        "mean_propensity": mean_p,
        "cells": cell_rows,
        "single_class": single_class_alert,
        "cluster": cluster,
        "store_propensity": store_propensity,
        "seed": seed,
    }
    return (
        pd.Series(new_w, index=weights.index, name="weights"),
        pd.Series(factors, index=weights.index, name="factors"),
        diag,
        extra,
    )


def logit_propensity_nonresponse(
    weights: pd.Series,
    data: pd.DataFrame,
    *,
    respondent: str,
    formula: str | list[str] | tuple[str, ...],
    num_classes: int | None = 5,
    weight_model: bool = True,
    cluster: str | None = None,
    store_propensity: bool = True,
    seed: int | None = 0,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Backward-compatible logit wrapper (no extra column dict in return)."""
    w, f, diag, _extra = propensity_nonresponse(
        weights,
        data,
        respondent=respondent,
        formula=formula,
        engine="logit",
        num_classes=num_classes,
        weight_model=weight_model,
        cluster=cluster,
        store_propensity=store_propensity,
        seed=seed,
    )
    return w, f, diag


def apply_weighting_class_nonresponse(
    frame: WeightFrame,
    *,
    respondent: str,
    by: list[str] | None = None,
    cluster: str | None = None,
) -> StepResult:
    weights, factors, diag = weighting_class_nonresponse(
        frame.weights, frame.data, respondent=respondent, by=by, cluster=cluster
    )
    return StepResult(weights=weights, factors=factors, diagnostics=diag)


def apply_propensity_nonresponse(
    frame: WeightFrame,
    *,
    respondent: str,
    formula: str | list[str] | tuple[str, ...],
    engine: PropensityEngine = "logit",
    num_classes: int | None = 5,
    weight_model: bool = True,
    cluster: str | None = None,
    store_propensity: bool = True,
    seed: int | None = 0,
) -> StepResult:
    weights, factors, diag, extra = propensity_nonresponse(
        frame.weights,
        frame.data,
        respondent=respondent,
        formula=formula,
        engine=engine,
        num_classes=num_classes,
        weight_model=weight_model,
        cluster=cluster,
        store_propensity=store_propensity,
        seed=seed,
    )
    return StepResult(weights=weights, factors=factors, diagnostics=diag, columns=extra)


def apply_logit_propensity_nonresponse(
    frame: WeightFrame,
    *,
    respondent: str,
    formula: str | list[str] | tuple[str, ...],
    num_classes: int | None = 5,
    weight_model: bool = True,
    cluster: str | None = None,
    store_propensity: bool = True,
    seed: int | None = 0,
) -> StepResult:
    return apply_propensity_nonresponse(
        frame,
        respondent=respondent,
        formula=formula,
        engine="logit",
        num_classes=num_classes,
        weight_model=weight_model,
        cluster=cluster,
        store_propensity=store_propensity,
        seed=seed,
    )
