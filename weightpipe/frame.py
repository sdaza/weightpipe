"""Unit-level weight tracking across recipe stages."""

from dataclasses import dataclass, field
from typing import Any, Self

import pandas as pd


@dataclass(frozen=True)
class WeightFrame:
    """Immutable tabular record of base weights, stage weights, and factors.

    Columns convention (when present):
    - ``unit_id``: row identifier
    - ``base_weight``: design / starting weight
    - ``weight_<step>``: weight after named step
    - ``factor_<step>``: multiplicative adjustment for named step
    - ``final_weight``: last stage weight
    - design columns: ``stratum``, ``psu``, disposition fields, etc.
    """

    data: pd.DataFrame
    step_names: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "base_weight" not in self.data.columns:
            raise ValueError("WeightFrame requires a 'base_weight' column")
        # Zero is allowed (dropped / bootstrap non-selected PSUs); negatives are not.
        if (self.data["base_weight"] < 0).any():
            raise ValueError("base_weight must be non-negative")
        if "final_weight" not in self.data.columns:
            object.__setattr__(
                self,
                "data",
                self.data.assign(final_weight=self.data["base_weight"]),
            )

    @classmethod
    def from_base(
        cls,
        data: pd.DataFrame,
        *,
        base_weight: str = "base_weight",
        unit_id: str | None = None,
        copy: bool = True,
    ) -> Self:
        """Build a frame from a survey microdata table and a base-weight column."""
        frame = data.copy() if copy else data
        if base_weight != "base_weight":
            if base_weight not in frame.columns:
                raise KeyError(f"base weight column not found: {base_weight}")
            frame = frame.rename(columns={base_weight: "base_weight"})
        if unit_id is not None and unit_id != "unit_id":
            if unit_id not in frame.columns:
                raise KeyError(f"unit id column not found: {unit_id}")
            frame = frame.rename(columns={unit_id: "unit_id"})
        if "final_weight" not in frame.columns:
            frame = frame.assign(final_weight=frame["base_weight"])
        return cls(data=frame, step_names=(), meta={"base_weight_source": base_weight})

    @property
    def weights(self) -> pd.Series:
        return self.data["final_weight"]

    def with_step(
        self,
        step_name: str,
        *,
        weights: pd.Series,
        factors: pd.Series,
        meta: dict[str, Any] | None = None,
        columns: dict[str, pd.Series] | None = None,
    ) -> Self:
        """Return a new frame after appending one adjustment stage."""
        if step_name in self.step_names:
            raise ValueError(f"step already applied: {step_name}")
        if len(weights) != len(self.data) or len(factors) != len(self.data):
            raise ValueError("weights/factors length must match frame rows")

        out = self.data.copy()
        out[f"weight_{step_name}"] = weights.to_numpy()
        out[f"factor_{step_name}"] = factors.to_numpy()
        out["final_weight"] = weights.to_numpy()
        if columns:
            for name, series in columns.items():
                if len(series) != len(out):
                    raise ValueError(f"column {name!r} length must match frame rows")
                out[name] = series.to_numpy()
        new_meta = dict(self.meta)
        if meta:
            new_meta[step_name] = meta
        return WeightFrame(
            data=out,
            step_names=(*self.step_names, step_name),
            meta=new_meta,
        )
