"""Immutable, lazy weighting recipe."""

from dataclasses import dataclass, field, replace
from typing import Any, Self

import numpy as np
import pandas as pd

from weightpipe.frame import WeightFrame
from weightpipe.result import WeightResult
from weightpipe.steps import CalibrateStep, DropIneligibleStep, NonresponseStep


@dataclass(frozen=True)
class Recipe:
    """Declarative weighting pipeline. Defining a recipe does not compute weights."""

    data: pd.DataFrame
    base_weight: str = "base_weight"
    unit_id: str | None = None
    steps: tuple[Any, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def _with_step(self, step: Any) -> Self:
        return replace(self, steps=(*self.steps, step))

    def step_drop_ineligible(self, *, ineligible: str) -> Self:
        return self._with_step(DropIneligibleStep(ineligible=ineligible))

    def step_nonresponse(
        self,
        *,
        respondent: str,
        method: str = "weighting_class",
        by: list[str] | tuple[str, ...] | str | None = None,
        **kwargs: Any,
    ) -> Self:
        if kwargs:
            raise TypeError(f"Unsupported nonresponse kwargs in Iteration 1: {sorted(kwargs)}")
        by_t: tuple[str, ...] | None
        if by is None:
            by_t = None
        elif isinstance(by, str):
            by_t = (by,)
        else:
            by_t = tuple(by)
        return self._with_step(NonresponseStep(respondent=respondent, method=method, by=by_t))

    def step_calibrate(
        self,
        *,
        method: str = "raking",
        margins: dict[str, dict[str, float]] | None = None,
        proportions: dict[str, dict[str, float]] | None = None,
        population_size: float | None = None,
        max_iter: int = 50,
        tol: float = 1e-6,
        **kwargs: Any,
    ) -> Self:
        if kwargs:
            raise TypeError(f"Unsupported calibrate kwargs in Iteration 1: {sorted(kwargs)}")
        return self._with_step(
            CalibrateStep(
                method=method,
                margins=margins,
                proportions=proportions,
                population_size=population_size,
                max_iter=max_iter,
                tol=tol,
            )
        )

    # Back-compat aliases (gated soft names from scaffold)
    def calibrate(self, **kwargs: Any) -> Self:
        return self.step_calibrate(**kwargs)

    def nonresponse(self, **kwargs: Any) -> Self:
        return self.step_nonresponse(**kwargs)

    def trim(self, **kwargs: Any) -> Self:
        raise NotImplementedError("step_trim is deferred past Iteration 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_weight": self.base_weight,
            "unit_id": self.unit_id,
            "steps": [s.to_dict() for s in self.steps],
            "meta": dict(self.meta),
        }

    def prep(
        self,
        *,
        min_cell_n: int | None = 30,
        max_factor: float | None = 2.5,
        warn: bool = False,
    ) -> WeightResult:
        """Estimate the weighting cascade (apply recorded steps in order)."""
        frame = WeightFrame.from_base(
            self.data,
            base_weight=self.base_weight,
            unit_id=self.unit_id,
        )
        history: dict[str, pd.Series] = {"base": frame.weights.copy()}
        step_diags: dict[str, Any] = {}
        alerts: list[str] = []

        for step in self.steps:
            step.validate(frame)
            w_before = frame.weights
            res = step.apply(frame, warn=warn) if step.name == "calibrate" else step.apply(frame)
            frame = frame.with_step(
                step.name,
                weights=res.weights,
                factors=res.factors,
                meta=res.diagnostics,
            )
            history[f"stage_{step.name}"] = res.weights.copy()
            step_diags[step.name] = res.diagnostics

            step_alerts = _step_alerts(
                w_before,
                res.weights,
                res.diagnostics,
                step_name=step.name,
                min_cell_n=min_cell_n,
                max_factor=max_factor,
            )
            alerts.extend(step_alerts)
            if warn:
                for a in step_alerts:
                    import warnings

                    warnings.warn(a, RuntimeWarning, stacklevel=2)

        return WeightResult(
            frame=frame,
            diagnostics={
                "n": len(frame.data),
                "steps_applied": list(frame.step_names),
                "steps": step_diags,
                "min_cell_n": min_cell_n,
                "max_factor": max_factor,
            },
            recipe=self.to_dict(),
            alerts=tuple(alerts),
            history=history,
        )

    def fit(self, **kwargs: Any) -> WeightResult:
        """Alias for ``prep``."""
        return self.prep(**kwargs)


def _step_alerts(
    w_before: pd.Series,
    w_after: pd.Series,
    diagnostics: dict[str, Any],
    *,
    step_name: str,
    min_cell_n: int | None,
    max_factor: float | None,
) -> list[str]:
    alerts: list[str] = []
    before = w_before.to_numpy(dtype=float)
    after = w_after.to_numpy(dtype=float)
    active = before > 0
    fac = np.ones_like(after)
    fac[active] = np.divide(
        after[active],
        before[active],
        out=np.ones_like(after[active]),
        where=before[active] > 0,
    )
    if max_factor is not None:
        excessive = active & (fac > max_factor) & (after > 0)
        if excessive.any():
            alerts.append(f"[{step_name}] {int(excessive.sum())} unit(s) have adjustment factor > {max_factor}")
    if min_cell_n is not None and "cells" in diagnostics:
        for row in diagnostics["cells"]:
            n_resp = int(row.get("n_respondents") or 0)
            if 0 < n_resp < min_cell_n:
                alerts.append(
                    f"[{step_name}] cell {row.get('cell')!r} has n_respondents={n_resp} < min_cell_n={min_cell_n}"
                )
    return alerts
