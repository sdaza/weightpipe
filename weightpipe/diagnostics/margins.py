"""Weighted category margins: anytime check and calibrate margin tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd

from weightpipe.result import WeightResult

TargetsSource = Literal["calibrate"]
MarginDict = dict[str, dict[str, float]]


def _as_variable_list(variables: str | Sequence[str] | None) -> list[str] | None:
    if variables is None:
        return None
    if isinstance(variables, str):
        return [variables]
    out = [str(v) for v in variables]
    if not out:
        raise ValueError("variables must be non-empty")
    return out


def _weights_and_data(result: WeightResult | pd.Series, data: pd.DataFrame | None) -> tuple[pd.Series, pd.DataFrame]:
    if isinstance(result, WeightResult):
        return result.weights.astype(float), result.data
    if data is None:
        raise ValueError("data= is required when passing weights without a WeightResult")
    return result.astype(float), data


def _achieved_for_variable(weights: pd.Series, data: pd.DataFrame, variable: str) -> pd.DataFrame:
    if variable not in data.columns:
        raise KeyError(f"variable not found: {variable}")
    w = weights.to_numpy(dtype=float)
    active = w > 0
    labels = data[variable].astype(str).to_numpy()
    ok = active & pd.notna(data[variable]).to_numpy()
    empty_cols = ["variable", "category", "achieved", "achieved_proportion", "n"]
    if not ok.any():
        return pd.DataFrame(columns=empty_cols)

    total_w = float(w[ok].sum())
    cats = sorted(set(labels[ok].tolist()))
    rows: list[dict[str, Any]] = []
    for cat in cats:
        idx = ok & (labels == cat)
        achieved = float(w[idx].sum())
        rows.append(
            {
                "variable": variable,
                "category": cat,
                "achieved": achieved,
                "achieved_proportion": (achieved / total_w) if total_w > 0 else float("nan"),
                "n": int(idx.sum()),
            }
        )
    return pd.DataFrame(rows)


def weighted_category_margins(
    weights: pd.Series | WeightResult,
    variables: str | Sequence[str],
    *,
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Current weighted totals and proportions for one or more categorical variables."""
    w, frame = _weights_and_data(weights, data)
    vars_ = _as_variable_list(variables)
    assert vars_ is not None
    tables = [_achieved_for_variable(w, frame, v) for v in vars_]
    if not tables:
        return pd.DataFrame(columns=["variable", "category", "achieved", "achieved_proportion", "n"])
    return pd.concat(tables, ignore_index=True)


def margin_table_from_targets(targets: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Tidy DataFrame from calibrate ``targets`` list-of-dicts diagnostics."""
    if not targets:
        return pd.DataFrame(
            columns=[
                "variable",
                "category",
                "target",
                "achieved",
                "abs_diff",
                "target_proportion",
                "achieved_proportion",
            ]
        )
    rows = []
    for row in targets:
        out = dict(row)
        if "abs_diff" not in out and "target" in out and "achieved" in out:
            out["abs_diff"] = abs(float(out["achieved"]) - float(out["target"]))
        if "category" not in out:
            out["category"] = None
        rows.append(out)
    return pd.DataFrame(rows)


def _calibrate_step_diag(result: WeightResult) -> dict[str, Any] | None:
    steps = result.diagnostics.get("steps") or {}
    if not isinstance(steps, dict):
        return None
    cal = steps.get("calibrate")
    return cal if isinstance(cal, dict) else None


def _targets_from_calibrate(
    result: WeightResult,
) -> tuple[MarginDict | None, MarginDict | None, float | None]:
    """Recover absolute margins from the last calibrate diagnostics."""
    cal = _calibrate_step_diag(result)
    if cal is None:
        raise ValueError("targets='calibrate' requires a calibrate step in diagnostics")

    resolved = cal.get("resolved_margins")
    if isinstance(resolved, dict) and resolved:
        total = cal.get("total")
        pop = float(total) if total is not None else None
        return (
            {str(v): {str(k): float(t) for k, t in dist.items()} for v, dist in resolved.items()},
            None,
            pop,
        )

    rows = cal.get("targets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("calibrate diagnostics have no resolved_margins or categorical targets")
    margins: MarginDict = {}
    for row in rows:
        if "category" not in row or row.get("category") is None:
            raise ValueError(
                "targets='calibrate' only supports categorical calibrate targets "
                "(raking / poststratify); linear formula totals are not category margins"
            )
        var = str(row["variable"])
        margins.setdefault(var, {})[str(row["category"])] = float(row["target"])
    total = cal.get("total")
    pop = float(total) if total is not None else None
    return margins, None, pop


def margins(
    result: WeightResult | pd.Series,
    variables: str | Sequence[str] | None = None,
    *,
    data: pd.DataFrame | None = None,
    margins: MarginDict | None = None,
    proportions: MarginDict | None = None,
    population_size: float | None = None,
    targets: TargetsSource | None = None,
    force1: bool = True,
) -> pd.DataFrame:
    """Weighted category margins from current weights, optionally vs targets.

    Parameters
    ----------
    result:
        Fitted ``WeightResult`` (preferred) or a weight series with ``data=``.
    variables:
        Categorical column name(s). Required unless targets imply them
        (``targets='calibrate'`` or ``margins`` / ``proportions`` keys).
    margins / proportions:
        Optional population targets for a fit check (same shapes as raking).
    targets:
        ``'calibrate'`` reuses absolute targets from the last calibrate step.
    """
    if targets is not None and targets != "calibrate":
        raise ValueError("targets must be None or 'calibrate'")
    if targets == "calibrate":
        if not isinstance(result, WeightResult):
            raise ValueError("targets='calibrate' requires a WeightResult")
        if margins is not None or proportions is not None:
            raise ValueError("pass targets='calibrate' or margins=/proportions=, not both")
        margins, proportions, recovered_pop = _targets_from_calibrate(result)
        if population_size is None:
            population_size = recovered_pop

    w, frame = _weights_and_data(result, data)

    target_margins: MarginDict | None = None
    target_meta: dict[str, Any] = {}
    if margins is not None or proportions is not None:
        from weightpipe.methods.raking import resolve_raking_margins

        target_margins, target_meta = resolve_raking_margins(
            w,
            margins=margins,
            proportions=proportions,
            population_size=population_size,
            force1=force1,
        )

    vars_ = _as_variable_list(variables)
    if vars_ is None:
        if target_margins is not None:
            vars_ = list(target_margins.keys())
        else:
            raise ValueError("provide variables=... or targets / margins / proportions")

    table = weighted_category_margins(w, vars_, data=frame)

    if target_margins is None:
        return table

    total_scale = float(target_meta["total"]) if target_meta.get("total") is not None else float(w[w > 0].sum())
    prop_map = target_meta.get("proportions")
    rows: list[dict[str, Any]] = []
    achieved_lookup = {(r.variable, r.category): r for r in table.itertuples(index=False)}
    for var, dist in target_margins.items():
        for cat, tot in dist.items():
            key = (var, str(cat))
            hit = achieved_lookup.get(key)
            achieved = float(hit.achieved) if hit is not None else 0.0
            n = int(hit.n) if hit is not None else 0
            row: dict[str, Any] = {
                "variable": var,
                "category": str(cat),
                "achieved": achieved,
                "achieved_proportion": (achieved / total_scale) if total_scale > 0 else float("nan"),
                "n": n,
                "target": float(tot),
                "abs_diff": abs(achieved - float(tot)),
            }
            if isinstance(prop_map, dict) and var in prop_map and str(cat) in prop_map[var]:
                row["target_proportion"] = float(prop_map[var][str(cat)])
            elif total_scale > 0:
                row["target_proportion"] = float(tot) / total_scale
            rows.append(row)

        for (v, c), hit in achieved_lookup.items():
            if v != var or c in {str(k) for k in dist}:
                continue
            rows.append(
                {
                    "variable": v,
                    "category": c,
                    "achieved": float(hit.achieved),
                    "achieved_proportion": float(hit.achieved_proportion),
                    "n": int(hit.n),
                    "target": float("nan"),
                    "abs_diff": float("nan"),
                    "target_proportion": float("nan"),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        var_order = {v: i for i, v in enumerate(vars_)}
        out["_vo"] = out["variable"].map(var_order)
        out = out.sort_values(["_vo", "category"], kind="mergesort").drop(columns="_vo").reset_index(drop=True)
    return out


def attach_margin_table(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Add a tidy ``margin_table`` DataFrame from ``targets`` diagnostics rows."""
    targets = diagnostics.get("targets")
    if not isinstance(targets, list) or not targets:
        return diagnostics
    out = dict(diagnostics)
    out["margin_table"] = margin_table_from_targets(targets)
    return out
