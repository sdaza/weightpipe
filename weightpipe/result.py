"""Fit result helpers: collect weights, factors, alerts."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from weightpipe.frame import WeightFrame


@dataclass(frozen=True)
class WeightResult:
    """Public result of ``Recipe.prep()`` / ``fit()``."""

    frame: WeightFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)
    recipe: dict[str, Any] = field(default_factory=dict)
    alerts: tuple[str, ...] = ()
    history: dict[str, pd.Series] = field(default_factory=dict)

    @property
    def weights(self) -> pd.Series:
        return self.frame.weights

    @property
    def data(self) -> pd.DataFrame:
        return self.frame.data

    @property
    def final_weight(self) -> pd.Series:
        return self.frame.weights


def collect_weights(
    result: WeightResult,
    *,
    weight_name: str = "weight",
    keep_intermediate: bool = False,
    drop_zero: bool = False,
) -> pd.DataFrame:
    """Extract analysis data with computed weights."""
    out = result.data.copy()
    out[weight_name] = result.weights.to_numpy()
    if keep_intermediate:
        for name, series in result.history.items():
            out[f".wt_{name}"] = series.to_numpy()
    if drop_zero:
        out = out.loc[out[weight_name] > 0].copy()
    return out


def weight_factors(result: WeightResult) -> pd.DataFrame:
    """Per-unit adjustment factors for each applied step."""
    cols = {}
    data = result.data
    if "base_weight" in data.columns:
        cols["base_weight"] = data["base_weight"]
    for step in result.frame.step_names:
        fac = f"factor_{step}"
        if fac in data.columns:
            cols[fac] = data[fac]
    cols["final_weight"] = result.weights
    return pd.DataFrame(cols, index=data.index)
