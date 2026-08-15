"""Shared step protocol for recipe stages."""

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from weightpipe.frame import WeightFrame


@dataclass(frozen=True)
class StepResult:
    weights: pd.Series
    factors: pd.Series
    diagnostics: dict[str, Any]
    columns: dict[str, pd.Series] | None = None


class Step(Protocol):
    """A recorded, inert recipe step applied during ``prep``."""

    name: str

    def validate(self, frame: WeightFrame) -> None: ...

    def apply(self, frame: WeightFrame) -> StepResult: ...

    def to_dict(self) -> dict[str, Any]: ...


def make_cells(data: pd.DataFrame, by: list[str] | None, n: int) -> pd.Series:
    """Build a grouping factor from ``by`` columns."""
    if not by:
        return pd.Series(["(all)"] * n, index=data.index, dtype="object")
    missing = [c for c in by if c not in data.columns]
    if missing:
        raise KeyError(f"Cell variable(s) not found: {missing}")
    parts = [data[c].astype(str) for c in by]
    out = parts[0]
    for p in parts[1:]:
        out = out + " | " + p
    return out


def as_logical_mask(data: pd.DataFrame, column: str) -> pd.Series:
    """Interpret a column as a boolean mask (logical or 0/1)."""
    if column not in data.columns:
        raise KeyError(f"Column not found: {column}")
    s = data[column]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    vals = pd.to_numeric(s, errors="coerce")
    if not vals.dropna().isin([0, 1]).all():
        raise ValueError(f"Expected logical or 0/1 column for '{column}'")
    return (vals == 1).fillna(False)
