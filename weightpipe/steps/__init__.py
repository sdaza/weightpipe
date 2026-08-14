"""Concrete recipe steps."""

from dataclasses import dataclass
from typing import Any

from weightpipe.frame import WeightFrame
from weightpipe.methods.eligibility import apply_drop_ineligible, apply_unknown_eligibility
from weightpipe.methods.linear import apply_linear_calibrate
from weightpipe.methods.nonresponse import (
    apply_logit_propensity_nonresponse,
    apply_weighting_class_nonresponse,
)
from weightpipe.methods.poststrat import apply_poststratify
from weightpipe.methods.raking import MarginDict, apply_raking
from weightpipe.methods.trim import AutoMethod, Reference, apply_trim, apply_trim_weights
from weightpipe.steps.base import StepResult


@dataclass(frozen=True)
class DropIneligibleStep:
    ineligible: str
    name: str = "drop_ineligible"

    def validate(self, frame: WeightFrame) -> None:
        if self.ineligible not in frame.data.columns:
            raise KeyError(f"ineligible column not found: {self.ineligible}")

    def apply(self, frame: WeightFrame) -> StepResult:
        return apply_drop_ineligible(frame, ineligible=self.ineligible)

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.name, "ineligible": self.ineligible}


@dataclass(frozen=True)
class UnknownEligibilityStep:
    unknown: str
    by: tuple[str, ...] | None = None
    cluster: str | None = None
    name: str = "unknown_eligibility"

    def validate(self, frame: WeightFrame) -> None:
        if self.unknown not in frame.data.columns:
            raise KeyError(f"unknown column not found: {self.unknown}")
        if self.by:
            missing = [c for c in self.by if c not in frame.data.columns]
            if missing:
                raise KeyError(f"unknown_eligibility by columns not found: {missing}")
        if self.cluster is not None and self.cluster not in frame.data.columns:
            raise KeyError(f"cluster column not found: {self.cluster}")

    def apply(self, frame: WeightFrame) -> StepResult:
        return apply_unknown_eligibility(
            frame,
            unknown=self.unknown,
            by=list(self.by) if self.by else None,
            cluster=self.cluster,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "unknown": self.unknown,
            "by": list(self.by) if self.by else None,
            "cluster": self.cluster,
        }


@dataclass(frozen=True)
class NonresponseStep:
    respondent: str
    method: str = "weighting_class"
    by: tuple[str, ...] | None = None
    formula: str | tuple[str, ...] | None = None
    engine: str = "logit"
    num_classes: int | None = 5
    weight_model: bool = True
    cluster: str | None = None
    name: str = "nonresponse"

    def validate(self, frame: WeightFrame) -> None:
        if self.respondent not in frame.data.columns:
            raise KeyError(f"respondent column not found: {self.respondent}")
        if self.cluster is not None and self.cluster not in frame.data.columns:
            raise KeyError(f"cluster column not found: {self.cluster}")
        if self.method == "weighting_class":
            if self.by:
                missing = [c for c in self.by if c not in frame.data.columns]
                if missing:
                    raise KeyError(f"nonresponse by columns not found: {missing}")
            return
        if self.method == "propensity":
            if self.engine != "logit":
                raise NotImplementedError(f"propensity engine={self.engine!r} is not implemented yet (logit only).")
            if self.formula is None:
                raise ValueError("propensity nonresponse requires formula=...")
            return
        raise NotImplementedError(
            f"nonresponse method={self.method!r} is not implemented yet (supports weighting_class and propensity)."
        )

    def apply(self, frame: WeightFrame) -> StepResult:
        if self.method == "weighting_class":
            return apply_weighting_class_nonresponse(
                frame,
                respondent=self.respondent,
                by=list(self.by) if self.by else None,
                cluster=self.cluster,
            )
        assert self.formula is not None
        return apply_logit_propensity_nonresponse(
            frame,
            respondent=self.respondent,
            formula=self.formula,
            num_classes=self.num_classes,
            weight_model=self.weight_model,
            cluster=self.cluster,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "respondent": self.respondent,
            "method": self.method,
            "by": list(self.by) if self.by else None,
            "formula": self.formula
            if isinstance(self.formula, str)
            else (list(self.formula) if self.formula is not None else None),
            "engine": self.engine,
            "num_classes": self.num_classes,
            "weight_model": self.weight_model,
            "cluster": self.cluster,
        }


@dataclass(frozen=True)
class CalibrateStep:
    method: str = "raking"
    margins: MarginDict | None = None
    proportions: MarginDict | None = None
    population_size: float | None = None
    formula: str | tuple[str, ...] | None = None
    totals: dict[str, float] | None = None
    bounds: tuple[float, float] | None = None
    penalty: float | dict[str, float] | None = None
    calfun: str = "linear"
    max_iter: int = 50
    tol: float = 1e-6
    force1: bool = True
    name: str = "calibrate"

    def validate(self, frame: WeightFrame) -> None:
        if self.method == "raking":
            if (self.margins is None) == (self.proportions is None):
                raise ValueError("raking requires exactly one of margins=... or proportions=...")
            vars_ = self.margins if self.margins is not None else self.proportions
            assert vars_ is not None
            for var in vars_:
                if var not in frame.data.columns:
                    raise KeyError(f"margin variable not found: {var}")
            return
        if self.method == "poststratify":
            if (self.margins is None) == (self.proportions is None):
                raise ValueError("poststratify requires exactly one of margins=... or proportions=...")
            vars_ = self.margins if self.margins is not None else self.proportions
            assert vars_ is not None
            if len(vars_) != 1:
                raise ValueError("poststratify requires exactly one variable in margins/proportions")
            var = next(iter(vars_))
            if var not in frame.data.columns:
                raise KeyError(f"post-stratum variable not found: {var}")
            return
        if self.method == "linear":
            if self.formula is None or self.totals is None:
                raise ValueError("linear calibration requires formula=... and totals=...")
            if self.calfun not in ("linear", "logit", "raking"):
                raise ValueError(f"unknown calfun: {self.calfun!r}")
            return
        raise NotImplementedError(
            f"calibrate method={self.method!r} is not implemented yet (supports raking, poststratify, linear)."
        )

    def apply(self, frame: WeightFrame, *, warn: bool = True) -> StepResult:
        if self.method == "raking":
            return apply_raking(
                frame,
                margins=self.margins,
                proportions=self.proportions,
                population_size=self.population_size,
                force1=self.force1,
                max_iter=self.max_iter,
                tol=self.tol,
                warn=warn,
            )
        if self.method == "poststratify":
            return apply_poststratify(
                frame,
                margins=self.margins,
                proportions=self.proportions,
                population_size=self.population_size,
                force1=self.force1,
            )
        assert self.formula is not None and self.totals is not None
        return apply_linear_calibrate(
            frame,
            formula=self.formula,
            totals=self.totals,
            bounds=self.bounds,
            penalty=self.penalty,
            calfun=self.calfun,  # type: ignore[arg-type]
            max_iter=max(self.max_iter, 100),
            warn=warn,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "method": self.method,
            "margins": self.margins,
            "proportions": self.proportions,
            "population_size": self.population_size,
            "force1": self.force1,
            "formula": self.formula
            if isinstance(self.formula, str)
            else (list(self.formula) if self.formula is not None else None),
            "totals": self.totals,
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "penalty": self.penalty,
            "calfun": self.calfun,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }


@dataclass(frozen=True)
class TrimStep:
    max_ratio: float
    min_ratio: float | None = None
    reference: Reference = "median"
    redistribute: bool = True
    by: tuple[str, ...] | None = None
    max_iter: int = 50
    name: str = "trim"

    def validate(self, frame: WeightFrame) -> None:
        if self.max_ratio <= 0:
            raise ValueError("max_ratio must be positive")
        if self.by:
            missing = [c for c in self.by if c not in frame.data.columns]
            if missing:
                raise KeyError(f"trim by columns not found: {missing}")
        if self.reference == "base" and "base_weight" not in frame.data.columns:
            raise KeyError("reference='base' requires base_weight on the frame")

    def apply(self, frame: WeightFrame) -> StepResult:
        return apply_trim(
            frame,
            max_ratio=self.max_ratio,
            min_ratio=self.min_ratio,
            reference=self.reference,
            redistribute=self.redistribute,
            by=list(self.by) if self.by else None,
            max_iter=self.max_iter,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "max_ratio": self.max_ratio,
            "min_ratio": self.min_ratio,
            "reference": self.reference,
            "redistribute": self.redistribute,
            "by": list(self.by) if self.by else None,
            "max_iter": self.max_iter,
        }


@dataclass(frozen=True)
class TrimWeightsStep:
    lower: float = 1.0
    upper: float | None = None
    method: AutoMethod = "tukey"
    redistribute: str = "proportional"
    strict: bool = True
    max_iter: int = 50
    name: str = "trim_weights"

    def validate(self, frame: WeightFrame) -> None:
        if self.method not in ("tukey", "potter"):
            raise ValueError(f"unknown auto-trim method: {self.method!r}")
        if self.redistribute not in ("proportional", "uniform"):
            raise ValueError(f"unknown redistribute: {self.redistribute!r}")

    def apply(self, frame: WeightFrame) -> StepResult:
        return apply_trim_weights(
            frame,
            lower=self.lower,
            upper=self.upper,
            method=self.method,  # type: ignore[arg-type]
            redistribute=self.redistribute,  # type: ignore[arg-type]
            strict=self.strict,
            max_iter=self.max_iter,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "method": self.method,
            "redistribute": self.redistribute,
            "strict": self.strict,
            "max_iter": self.max_iter,
        }
