"""Covariate balance before vs after weighting (SMD / ASMD)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from weightpipe.result import WeightResult

Reference = Literal["base"]


def _ess(weights: pd.Series) -> float:
    """Kish effective sample size: (sum w)^2 / sum(w^2)."""
    w = weights.astype(float)
    active = w[w > 0]
    if active.empty:
        return float("nan")
    s = float(active.sum())
    ss = float(np.sum(np.square(active.to_numpy())))
    if ss <= 0:
        return float("nan")
    return float((s * s) / ss)


def _as_list(covariates: str | Sequence[str]) -> list[str]:
    if isinstance(covariates, str):
        return [covariates]
    out = [str(c) for c in covariates]
    if not out:
        raise ValueError("covariates must be non-empty")
    return out


def _weights_and_data(
    result: WeightResult | pd.Series,
    data: pd.DataFrame | None,
) -> tuple[pd.Series, pd.DataFrame]:
    if isinstance(result, WeightResult):
        return result.weights.astype(float), result.data
    if data is None:
        raise ValueError("data= is required when passing weights without a WeightResult")
    return result.astype(float), data


def _resolve_before(
    result: WeightResult | pd.Series,
    frame: pd.DataFrame,
    before: pd.Series | Reference | None,
) -> pd.Series:
    if before is None or before == "base":
        if "base_weight" not in frame.columns:
            raise ValueError("reference='base' requires a base_weight column")
        return frame["base_weight"].astype(float)
    if isinstance(before, pd.Series):
        if len(before) != len(frame):
            raise ValueError("before weights length must match data rows")
        return before.astype(float)
    raise TypeError("before must be None, 'base', or a weight Series")


def _is_continuous(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        # Integer codes with few unique values are treated as categorical.
        nunique = int(series.dropna().nunique())
        if nunique <= 1:
            return True
        if pd.api.types.is_integer_dtype(series) and nunique <= min(10, max(3, len(series) // 20)):
            return False
        return True
    return False


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float("nan")
    ww = w[mask]
    xx = x[mask]
    s = float(ww.sum())
    if s <= 0:
        return float("nan")
    return float(np.dot(ww, xx) / s)


def _weighted_var(x: np.ndarray, w: np.ndarray, mean: float) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not mask.any() or not np.isfinite(mean):
        return float("nan")
    ww = w[mask]
    xx = x[mask]
    s = float(ww.sum())
    if s <= 0:
        return float("nan")
    return float(np.dot(ww, (xx - mean) ** 2) / s)


def _smd(diff: float, sd: float) -> float:
    if not np.isfinite(diff):
        return float("nan")
    if not np.isfinite(sd) or sd <= 0:
        return 0.0 if abs(diff) < 1e-15 else float("nan")
    return float(diff / sd)


def _continuous_row(
    name: str,
    x: np.ndarray,
    w_before: np.ndarray,
    w_after: np.ndarray,
    target_mean: float,
    target_sd: float,
    threshold: float,
) -> dict[str, Any]:
    m_b = _weighted_mean(x, w_before)
    m_a = _weighted_mean(x, w_after)
    smd_b = _smd(m_b - target_mean, target_sd)
    smd_a = _smd(m_a - target_mean, target_sd)
    return {
        "variable": name,
        "type": "continuous",
        "level": None,
        "before": m_b,
        "after": m_a,
        "target": target_mean,
        "smd_before": smd_b,
        "smd_after": smd_a,
        "abs_smd_before": abs(smd_b) if np.isfinite(smd_b) else float("nan"),
        "abs_smd_after": abs(smd_a) if np.isfinite(smd_a) else float("nan"),
        "balanced": bool(np.isfinite(smd_a) and abs(smd_a) < threshold),
    }


def _categorical_rows(
    name: str,
    labels: np.ndarray,
    w_before: np.ndarray,
    w_after: np.ndarray,
    target_props: Mapping[str, float],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    levels = sorted({str(v) for v in labels if pd.notna(v)} | {str(k) for k in target_props})
    for level in levels:
        ind = (pd.Series(labels).astype(str) == level).to_numpy(dtype=float)
        # Missing labels → 0 contribution (not in that category).
        miss = pd.isna(labels)
        ind = np.where(miss, np.nan, ind)
        p_b = _weighted_mean(ind, w_before)
        p_a = _weighted_mean(ind, w_after)
        p_t = float(target_props.get(level, 0.0))
        sd = float(np.sqrt(max(p_t * (1.0 - p_t), 0.0)))
        if sd <= 0:
            # Rare/empty target level: fall back to pooled before/target variance.
            pooled = 0.5 * (p_b * (1.0 - p_b) + p_t * (1.0 - p_t)) if np.isfinite(p_b) else p_t * (1.0 - p_t)
            sd = float(np.sqrt(max(pooled, 0.0)))
        smd_b = _smd(p_b - p_t, sd)
        smd_a = _smd(p_a - p_t, sd)
        rows.append(
            {
                "variable": name,
                "type": "categorical",
                "level": level,
                "before": p_b,
                "after": p_a,
                "target": p_t,
                "smd_before": smd_b,
                "smd_after": smd_a,
                "abs_smd_before": abs(smd_b) if np.isfinite(smd_b) else float("nan"),
                "abs_smd_after": abs(smd_a) if np.isfinite(smd_a) else float("nan"),
                "balanced": bool(np.isfinite(smd_a) and abs(smd_a) < threshold),
            }
        )
    return rows


def _target_from_population(
    target: pd.DataFrame,
    covariates: list[str],
    *,
    target_weight: str | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    """Return continuous means, continuous SDs, and categorical proportions."""
    if target_weight is None:
        w = np.ones(len(target), dtype=float)
    else:
        if target_weight not in target.columns:
            raise KeyError(f"target_weight column not found: {target_weight}")
        w = target[target_weight].to_numpy(dtype=float)

    means: dict[str, float] = {}
    sds: dict[str, float] = {}
    props: dict[str, dict[str, float]] = {}
    for name in covariates:
        if name not in target.columns:
            raise KeyError(f"covariate not found in target: {name}")
        series = target[name]
        if _is_continuous(series):
            x = series.to_numpy(dtype=float)
            m = _weighted_mean(x, w)
            means[name] = m
            sds[name] = float(np.sqrt(max(_weighted_var(x, w, m), 0.0)))
        else:
            labels = series.astype(str).to_numpy()
            ok = np.isfinite(w) & (w > 0) & pd.notna(series).to_numpy()
            total = float(w[ok].sum()) if ok.any() else 0.0
            levels = sorted(set(labels[ok].tolist())) if ok.any() else []
            dist: dict[str, float] = {}
            for level in levels:
                idx = ok & (labels == level)
                dist[level] = (float(w[idx].sum()) / total) if total > 0 else float("nan")
            props[name] = dist
    return means, sds, props


@dataclass(frozen=True)
class BalanceReport:
    """Covariate balance before/after weighting.

    ``table`` has one row per continuous covariate or categorical level.
    ``summary`` aggregates max |SMD| and how many rows fail ``threshold``.
    """

    table: pd.DataFrame
    summary: dict[str, Any]
    threshold: float = 0.1

    def __repr__(self) -> str:
        s = self.summary
        return (
            f"BalanceReport(n_rows={len(self.table)}, "
            f"max_abs_smd_after={s.get('max_abs_smd_after')}, "
            f"n_imbalanced={s.get('n_imbalanced')}, "
            f"threshold={self.threshold})"
        )


def balance(
    result: WeightResult | pd.Series,
    covariates: str | Sequence[str],
    *,
    data: pd.DataFrame | None = None,
    target: pd.DataFrame | None = None,
    target_weight: str | None = None,
    means: Mapping[str, float] | None = None,
    proportions: Mapping[str, Mapping[str, float]] | None = None,
    sds: Mapping[str, float] | None = None,
    before: pd.Series | Reference | None = "base",
    threshold: float = 0.1,
) -> BalanceReport:
    """Compare covariate moments before vs after weighting to a target.

    For each continuous covariate, reports weighted means and standardized mean
    differences (SMD). For each categorical covariate, reports weighted
    proportions and SMDs for every level.

    Parameters
    ----------
    result:
        Fitted ``WeightResult`` (preferred) or final weights with ``data=``.
    covariates:
        Column names to assess (continuous and/or categorical).
    target:
        Optional population microdata. Moments are computed from it (with
        optional ``target_weight``). Alternatively pass ``means`` /
        ``proportions``.
    means / proportions:
        Explicit population targets when you do not have microdata.
    sds:
        Optional population SDs for continuous covariates (used as the SMD
        denominator). If omitted, the base-weighted sample SD is used.
    before:
        Pre-adjustment weights. Default ``'base'`` uses ``base_weight``.
    threshold:
        Absolute SMD below which a row is marked ``balanced`` (default 0.1).
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if target is None and means is None and proportions is None:
        raise ValueError("provide target= population microdata or means=/proportions=")
    if target is not None and (means is not None or proportions is not None):
        raise ValueError("pass target= or means=/proportions=, not both")

    covs = _as_list(covariates)
    w_after, frame = _weights_and_data(result, data)
    for name in covs:
        if name not in frame.columns:
            raise KeyError(f"covariate not found: {name}")

    w_before = _resolve_before(result, frame, before)
    w_b = w_before.to_numpy(dtype=float)
    w_a = w_after.to_numpy(dtype=float)

    target_means: dict[str, float] = dict(means or {})
    target_sds: dict[str, float] = dict(sds or {})
    target_props: dict[str, dict[str, float]] = {
        k: {str(ck): float(cv) for ck, cv in v.items()} for k, v in (proportions or {}).items()
    }

    if target is not None:
        target_means, pop_sds, target_props = _target_from_population(
            target, covs, target_weight=target_weight
        )
        for k, v in pop_sds.items():
            target_sds.setdefault(k, v)

    rows: list[dict[str, Any]] = []
    for name in covs:
        series = frame[name]
        if _is_continuous(series) and name not in target_props:
            if name not in target_means:
                raise ValueError(f"no target mean for continuous covariate {name!r}")
            x = series.to_numpy(dtype=float)
            if name in target_sds and np.isfinite(target_sds[name]) and target_sds[name] > 0:
                sd = float(target_sds[name])
            else:
                m_ref = _weighted_mean(x, w_b)
                sd = float(np.sqrt(max(_weighted_var(x, w_b, m_ref), 0.0)))
            rows.append(
                _continuous_row(name, x, w_b, w_a, float(target_means[name]), sd, threshold)
            )
        else:
            if name not in target_props:
                # Infer categorical targets only when proportions were supplied
                # for this variable, or when population microdata defined them.
                raise ValueError(f"no target proportions for categorical covariate {name!r}")
            labels = series.to_numpy()
            rows.extend(_categorical_rows(name, labels, w_b, w_a, target_props[name], threshold))

    table = pd.DataFrame(rows)
    if not table.empty:
        var_order = {v: i for i, v in enumerate(covs)}
        table["_vo"] = table["variable"].map(var_order)
        table = (
            table.sort_values(["_vo", "type", "level"], kind="mergesort")
            .drop(columns="_vo")
            .reset_index(drop=True)
        )

    abs_after = table["abs_smd_after"] if not table.empty else pd.Series(dtype=float)
    abs_before = table["abs_smd_before"] if not table.empty else pd.Series(dtype=float)
    finite_after = abs_after[np.isfinite(abs_after)]
    finite_before = abs_before[np.isfinite(abs_before)]
    summary = {
        "n_rows": int(len(table)),
        "n_covariates": len(covs),
        "threshold": float(threshold),
        "max_abs_smd_before": float(finite_before.max()) if len(finite_before) else float("nan"),
        "max_abs_smd_after": float(finite_after.max()) if len(finite_after) else float("nan"),
        "mean_abs_smd_before": float(finite_before.mean()) if len(finite_before) else float("nan"),
        "mean_abs_smd_after": float(finite_after.mean()) if len(finite_after) else float("nan"),
        "n_imbalanced": int((~table["balanced"]).sum()) if not table.empty else 0,
        "ess_before": _ess(w_before),
        "ess_after": _ess(w_after),
    }
    return BalanceReport(table=table, summary=summary, threshold=float(threshold))
