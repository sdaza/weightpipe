"""Build calibration / propensity design matrices from a formula."""

from typing import Any

import pandas as pd


def parse_formula(formula: str | list[str] | tuple[str, ...]) -> list[str]:
    """Parse ``~ a + b`` or a list of term names into term strings."""
    if isinstance(formula, (list, tuple)):
        terms = [str(t).strip() for t in formula if str(t).strip()]
    else:
        s = str(formula).strip()
        if s.startswith("~"):
            s = s[1:].strip()
        terms = [t.strip() for t in s.split("+") if t.strip()]
    if not terms:
        raise ValueError("formula must include at least one term")
    return terms


def design_matrix(
    data: pd.DataFrame,
    formula: str | list[str] | tuple[str, ...],
    *,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Intercept + numeric columns + dummy-coded categoricals.

    Categorical dummies use ``prefixsep`` empty (``regionSouth``). With
    ``drop_first=True`` (default), the first sorted level is the reference
    so the matrix has full column rank with the intercept.
    """
    terms = parse_formula(formula)
    missing = [t for t in terms if t not in data.columns]
    if missing:
        raise KeyError(f"formula term(s) not found: {missing}")

    parts: list[pd.Series | pd.DataFrame] = [
        pd.Series(1.0, index=data.index, name="(Intercept)"),
    ]
    for term in terms:
        s = data[term]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            parts.append(s.astype(float).rename(term))
            continue
        dummies = pd.get_dummies(s.astype(str), prefix=term, prefix_sep="")
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        if drop_first and dummies.shape[1] > 0:
            dummies = dummies.iloc[:, 1:]
        for col in dummies.columns:
            parts.append(dummies[col].astype(float))

    return pd.concat(parts, axis=1)


def population_totals(
    population: pd.DataFrame,
    formula: str | list[str] | tuple[str, ...],
    *,
    weight: str | None = None,
) -> dict[str, float]:
    """Column totals of ``design_matrix(population, formula)`` (optionally weighted)."""
    x = design_matrix(population, formula)
    if weight is None:
        return {c: float(x[c].sum()) for c in x.columns}
    if weight not in population.columns:
        raise KeyError(f"weight column not found: {weight}")
    w = population[weight].astype(float).to_numpy()
    return {c: float((w * x[c].to_numpy()).sum()) for c in x.columns}


def align_totals(columns: list[str], totals: dict[str, Any] | pd.Series) -> dict[str, float]:
    """Map a totals dict/Series onto design-matrix column names."""
    src = dict(totals) if not isinstance(totals, pd.Series) else totals.to_dict()
    src = {str(k): float(v) for k, v in src.items()}
    missing = [c for c in columns if c not in src]
    if missing:
        raise KeyError(
            "totals missing design-matrix columns: "
            f"{missing}. Available keys: {sorted(src)}. "
            "Use weightpipe.methods.population_totals(pop, formula) to build them."
        )
    return {c: float(src[c]) for c in columns}
