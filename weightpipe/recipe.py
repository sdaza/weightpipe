"""Immutable, lazy weighting recipe."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pandas as pd

from weightpipe.frame import WeightFrame
from weightpipe.result import WeightResult
from weightpipe.steps import (
    CalibrateStep,
    DropIneligibleStep,
    NonresponseStep,
    TrimStep,
    TrimWeightsStep,
    UnknownEligibilityStep,
)

if TYPE_CHECKING:
    from weightpipe.design import Design


@dataclass(frozen=True)
class Recipe:
    """Declarative weighting pipeline. Defining a recipe does not compute weights."""

    data: pd.DataFrame
    base_weight: str = "base_weight"
    unit_id: str | None = None
    steps: tuple[Any, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    design: Any | None = None  # Design | None; Any avoids runtime circular import

    @classmethod
    def from_design(cls, design: "Design", *, unit_id: str | None = None) -> Self:
        """Start a recipe from a sampling ``Design`` (base weights + strata/PSU)."""
        return cls(
            data=design.data,
            base_weight=design.weight,
            unit_id=unit_id,
            design=design,
            meta={"design": design.to_dict()},
        )

    def _with_step(self, step: Any) -> Self:
        return replace(self, steps=(*self.steps, step))

    def step_unknown_eligibility(
        self,
        *,
        unknown: str,
        by: list[str] | tuple[str, ...] | str | None = None,
        cluster: str | None = None,
    ) -> Self:
        by_t: tuple[str, ...] | None
        if by is None:
            by_t = None
        elif isinstance(by, str):
            by_t = (by,)
        else:
            by_t = tuple(by)
        return self._with_step(UnknownEligibilityStep(unknown=unknown, by=by_t, cluster=cluster))

    def step_drop_ineligible(self, *, ineligible: str) -> Self:
        return self._with_step(DropIneligibleStep(ineligible=ineligible))

    def step_nonresponse(
        self,
        *,
        respondent: str,
        method: str = "weighting_class",
        by: list[str] | tuple[str, ...] | str | None = None,
        formula: str | list[str] | tuple[str, ...] | None = None,
        engine: str = "logit",
        num_classes: int | None = 5,
        weight_model: bool = True,
        cluster: str | None = None,
        store_propensity: bool = True,
        seed: int | None = 0,
        **kwargs: Any,
    ) -> Self:
        if kwargs:
            raise TypeError(f"Unsupported nonresponse kwargs: {sorted(kwargs)}")
        by_t: tuple[str, ...] | None
        if by is None:
            by_t = None
        elif isinstance(by, str):
            by_t = (by,)
        else:
            by_t = tuple(by)
        formula_t: str | tuple[str, ...] | None
        if formula is None:
            formula_t = None
        elif isinstance(formula, str):
            formula_t = formula
        else:
            formula_t = tuple(formula)
        return self._with_step(
            NonresponseStep(
                respondent=respondent,
                method=method,
                by=by_t,
                formula=formula_t,
                engine=engine,
                num_classes=num_classes,
                weight_model=weight_model,
                cluster=cluster,
                store_propensity=store_propensity,
                seed=seed,
            )
        )

    def step_calibrate(
        self,
        *,
        method: str = "raking",
        margins: dict[str, dict[str, float]] | None = None,
        proportions: dict[str, dict[str, float]] | None = None,
        population_size: float | None = None,
        formula: str | list[str] | tuple[str, ...] | None = None,
        totals: dict[str, float] | None = None,
        bounds: tuple[float, float] | list[float] | None = None,
        penalty: float | dict[str, float] | None = None,
        calfun: str = "linear",
        max_iter: int = 50,
        tol: float = 1e-6,
        force1: bool = True,
        assist: str | None = None,
        engine: str | None = None,
        population: Any = None,
        include_linear: bool = True,
        n_estimators: int = 40,
        max_depth: int = 3,
        seed: int | None = 0,
        **kwargs: Any,
    ) -> Self:
        if kwargs:
            raise TypeError(f"Unsupported calibrate kwargs: {sorted(kwargs)}")
        formula_t: str | tuple[str, ...] | None
        if formula is None:
            formula_t = None
        elif isinstance(formula, str):
            formula_t = formula
        else:
            formula_t = tuple(formula)
        bounds_t = None if bounds is None else (float(bounds[0]), float(bounds[1]))
        return self._with_step(
            CalibrateStep(
                method=method,
                margins=margins,
                proportions=proportions,
                population_size=population_size,
                formula=formula_t,
                totals=totals,
                bounds=bounds_t,
                penalty=penalty,
                calfun=calfun,
                max_iter=max_iter,
                tol=tol,
                force1=force1,
                assist=assist,
                engine=engine,
                population=population,
                include_linear=include_linear,
                n_estimators=n_estimators,
                max_depth=max_depth,
                seed=seed,
            )
        )

    def step_trim(
        self,
        *,
        max_ratio: float,
        min_ratio: float | None = None,
        reference: str = "median",
        redistribute: bool = True,
        by: list[str] | tuple[str, ...] | str | None = None,
        max_iter: int = 50,
    ) -> Self:
        by_t: tuple[str, ...] | None
        if by is None:
            by_t = None
        elif isinstance(by, str):
            by_t = (by,)
        else:
            by_t = tuple(by)
        return self._with_step(
            TrimStep(
                max_ratio=max_ratio,
                min_ratio=min_ratio,
                reference=reference,  # type: ignore[arg-type]
                redistribute=redistribute,
                by=by_t,
                max_iter=max_iter,
            )
        )

    def step_trim_weights(
        self,
        *,
        lower: float = 1.0,
        upper: float | None = None,
        method: str = "tukey",
        redistribute: str = "proportional",
        strict: bool = True,
        max_iter: int = 50,
    ) -> Self:
        """Automatic Tukey / Potter trimming (weightflow ``step_trim_weights``)."""
        return self._with_step(
            TrimWeightsStep(
                lower=lower,
                upper=upper,
                method=method,  # type: ignore[arg-type]
                redistribute=redistribute,
                strict=strict,
                max_iter=max_iter,
            )
        )

    # Back-compat aliases
    def calibrate(self, **kwargs: Any) -> Self:
        return self.step_calibrate(**kwargs)

    def nonresponse(self, **kwargs: Any) -> Self:
        return self.step_nonresponse(**kwargs)

    def trim(self, **kwargs: Any) -> Self:
        return self.step_trim(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "base_weight": self.base_weight,
            "unit_id": self.unit_id,
            "steps": [s.to_dict() for s in self.steps],
            "meta": dict(self.meta),
        }
        if self.design is not None:
            out["design"] = self.design.to_dict()
        return out

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
            if step.name == "calibrate":
                res = step.apply(frame, warn=warn)
            else:
                res = step.apply(frame)
            frame = frame.with_step(
                step.name,
                weights=res.weights,
                factors=res.factors,
                meta=res.diagnostics,
                columns=res.columns,
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
            n_resp = int(row.get("n_respondents") or row.get("n_known") or 0)
            if 0 < n_resp < min_cell_n:
                alerts.append(f"[{step_name}] cell {row.get('cell')!r} has n={n_resp} < min_cell_n={min_cell_n}")
    if diagnostics.get("single_class"):
        alerts.append(f"[{step_name}] propensity classes collapsed to a single class (near-constant p)")
    return alerts
