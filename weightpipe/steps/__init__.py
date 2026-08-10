"""Concrete recipe steps (Iteration 1)."""

from dataclasses import dataclass
from typing import Any

from weightpipe.frame import WeightFrame
from weightpipe.methods.eligibility import apply_drop_ineligible
from weightpipe.methods.nonresponse import apply_weighting_class_nonresponse
from weightpipe.methods.raking import MarginDict, apply_raking
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
class NonresponseStep:
    respondent: str
    method: str = "weighting_class"
    by: tuple[str, ...] | None = None
    name: str = "nonresponse"

    def validate(self, frame: WeightFrame) -> None:
        if self.respondent not in frame.data.columns:
            raise KeyError(f"respondent column not found: {self.respondent}")
        if self.method != "weighting_class":
            raise NotImplementedError(
                f"nonresponse method={self.method!r} is not implemented yet "
                "(Iteration 1 supports weighting_class only)."
            )
        if self.by:
            missing = [c for c in self.by if c not in frame.data.columns]
            if missing:
                raise KeyError(f"nonresponse by columns not found: {missing}")

    def apply(self, frame: WeightFrame) -> StepResult:
        return apply_weighting_class_nonresponse(
            frame,
            respondent=self.respondent,
            by=list(self.by) if self.by else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "respondent": self.respondent,
            "method": self.method,
            "by": list(self.by) if self.by else None,
        }


@dataclass(frozen=True)
class CalibrateStep:
    method: str = "raking"
    margins: MarginDict | None = None
    proportions: MarginDict | None = None
    population_size: float | None = None
    max_iter: int = 50
    tol: float = 1e-6
    name: str = "calibrate"

    def validate(self, frame: WeightFrame) -> None:
        if self.method != "raking":
            raise NotImplementedError(
                f"calibrate method={self.method!r} is not implemented yet (Iteration 1 supports raking only)."
            )
        if (self.margins is None) == (self.proportions is None):
            raise ValueError("raking requires exactly one of margins=... or proportions=...")
        vars_ = self.margins if self.margins is not None else self.proportions
        assert vars_ is not None
        for var in vars_:
            if var not in frame.data.columns:
                raise KeyError(f"margin variable not found: {var}")

    def apply(self, frame: WeightFrame, *, warn: bool = True) -> StepResult:
        return apply_raking(
            frame,
            margins=self.margins,
            proportions=self.proportions,
            population_size=self.population_size,
            max_iter=self.max_iter,
            tol=self.tol,
            warn=warn,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "method": self.method,
            "margins": self.margins,
            "proportions": self.proportions,
            "population_size": self.population_size,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }
