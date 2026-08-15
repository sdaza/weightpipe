"""Helpers to fold propensity columns into calibration targets."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from weightpipe.frame import WeightFrame
from weightpipe.methods.design_matrix import design_matrix, parse_formula

AssistMode = Literal["propensity", "propensity_class"]


def _append_formula_term(formula: str | tuple[str, ...] | None, term: str) -> str:
    if formula is None:
        return f"~ {term}"
    if isinstance(formula, tuple):
        terms = list(formula)
        if term not in terms:
            terms.append(term)
        return "~ " + " + ".join(terms)
    s = str(formula).strip()
    if s.startswith("~"):
        body = s[1:].strip()
    else:
        body = s
    parts = [t.strip() for t in body.split("+") if t.strip()]
    if term not in parts:
        parts.append(term)
    return "~ " + " + ".join(parts)


def apply_calibrate_assist(
    frame: WeightFrame,
    *,
    method: str,
    assist: AssistMode,
    margins: dict[str, dict[str, float]] | None,
    proportions: dict[str, dict[str, float]] | None,
    formula: str | tuple[str, ...] | None,
    totals: dict[str, float] | None,
    population_size: float | None,
) -> dict[str, Any]:
    """Return updated calibrate kwargs with propensity assist applied.

    - ``propensity_class``: keep current weighted class totals while matching
      other margins (raking / poststratify / linear).
    - ``propensity``: add continuous ``propensity`` to linear calibration;
      total defaults to ``population_size * mean(p)`` among active units, or
      the current weighted sum of ``propensity`` if ``population_size`` is None.
    """
    data = frame.data
    w = frame.weights.to_numpy(dtype=float)
    active = w > 0

    if assist == "propensity_class":
        if "propensity_class" not in data.columns:
            raise ValueError(
                "assist='propensity_class' requires a prior propensity nonresponse "
                "step with num_classes set (column 'propensity_class' missing)"
            )
        cls = data["propensity_class"]
        # Preserve current weighted class mass among active respondents.
        class_totals: dict[str, float] = {}
        for g in sorted(pd.unique(cls[active].dropna())):
            mask = active & (cls == g)
            class_totals[str(int(g) if float(g).is_integer() else g)] = float(w[mask].sum())

        if method in ("raking", "poststratify"):
            if proportions is not None:
                raise ValueError("assist='propensity_class' with proportions= is not supported; use margins=")
            new_margins = {} if margins is None else {k: dict(v) for k, v in margins.items()}
            # Use string keys matching how raking reads the column as categories.
            # Ensure propensity_class is stored as string-friendly categories on a working copy?
            # Raking uses data column levels as-is; cast frame data externally in step.
            new_margins["propensity_class"] = class_totals
            return {
                "margins": new_margins,
                "proportions": None,
                "formula": formula,
                "totals": totals,
                "population_size": population_size,
                "assist_detail": {"propensity_class_totals": class_totals},
                "cast_propensity_class": True,
            }

        if method == "linear":
            new_formula = _append_formula_term(formula, "propensity_class")
            # Build dummy totals from design matrix of propensity_class alone + existing totals.
            work = data.copy()
            work["propensity_class"] = cls.map(
                lambda v: str(int(v)) if pd.notna(v) and float(v).is_integer() else (str(v) if pd.notna(v) else v)
            )
            x_cls = design_matrix(work.loc[active], "~ propensity_class")
            new_totals = {} if totals is None else dict(totals)
            for col in x_cls.columns:
                if col == "(Intercept)":
                    continue
                if col not in new_totals:
                    new_totals[col] = float((w[active] * x_cls[col].to_numpy(dtype=float)).sum())
            # Intercept: keep user total or current weight sum
            if "(Intercept)" not in new_totals:
                if population_size is not None:
                    new_totals["(Intercept)"] = float(population_size)
                else:
                    new_totals["(Intercept)"] = float(w[active].sum())
            return {
                "margins": margins,
                "proportions": proportions,
                "formula": new_formula,
                "totals": new_totals,
                "population_size": population_size,
                "assist_detail": {"propensity_class_totals": class_totals},
                "cast_propensity_class": True,
            }

        raise ValueError(f"assist not supported for calibrate method={method!r}")

    # assist == "propensity"
    if "propensity" not in data.columns:
        raise ValueError(
            "assist='propensity' requires a prior propensity nonresponse step (column 'propensity' missing)"
        )
    if method != "linear":
        raise ValueError("assist='propensity' is only supported with method='linear'")

    p = data["propensity"].to_numpy(dtype=float)
    new_formula = _append_formula_term(formula, "propensity")
    new_totals = {} if totals is None else dict(totals)
    if "propensity" not in new_totals:
        p_active = p[active]
        w_active = w[active]
        finite = np.isfinite(p_active)
        if not finite.any():
            raise ValueError("no finite propensity values for assist='propensity'")
        mean_p = float(np.average(p_active[finite], weights=w_active[finite]))
        if population_size is not None:
            new_totals["propensity"] = float(population_size) * mean_p
        else:
            new_totals["propensity"] = float((w_active[finite] * p_active[finite]).sum())
    if "(Intercept)" not in new_totals:
        # Ensure intercept present when design_matrix adds it
        terms = parse_formula(new_formula)
        _ = terms
        if population_size is not None:
            new_totals["(Intercept)"] = float(population_size)
        else:
            new_totals["(Intercept)"] = float(w[active].sum())

    return {
        "margins": margins,
        "proportions": proportions,
        "formula": new_formula,
        "totals": new_totals,
        "population_size": population_size,
        "assist_detail": {"propensity_total": new_totals.get("propensity")},
        "cast_propensity_class": False,
    }
