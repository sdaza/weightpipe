"""Pure weighting algorithms.

Implement and validate here before wiring into ``Recipe.step_*``.
Do not loosen recovery tolerances to make CI green — diagnose first.
"""

from importlib import import_module
from typing import Any

from weightpipe.methods.design_matrix import design_matrix, population_totals

__all__ = [
    "design_matrix",
    "drop_ineligible_weights",
    "linear_calibrate",
    "logit_propensity_nonresponse",
    "population_totals",
    "poststratify",
    "potter_threshold",
    "propensity_nonresponse",
    "proportions_to_margins",
    "rake",
    "trim_weights",
    "trim_weights_auto",
    "tukey_threshold",
    "unknown_eligibility_weights",
    "weighting_class_nonresponse",
]

_LAZY: dict[str, tuple[str, str]] = {
    "drop_ineligible_weights": ("weightpipe.methods.eligibility", "drop_ineligible_weights"),
    "linear_calibrate": ("weightpipe.methods.linear", "linear_calibrate"),
    "logit_propensity_nonresponse": ("weightpipe.methods.nonresponse", "logit_propensity_nonresponse"),
    "poststratify": ("weightpipe.methods.poststrat", "poststratify"),
    "potter_threshold": ("weightpipe.methods.trim", "potter_threshold"),
    "propensity_nonresponse": ("weightpipe.methods.nonresponse", "propensity_nonresponse"),
    "proportions_to_margins": ("weightpipe.methods.raking", "proportions_to_margins"),
    "rake": ("weightpipe.methods.raking", "rake"),
    "trim_weights": ("weightpipe.methods.trim", "trim_weights"),
    "trim_weights_auto": ("weightpipe.methods.trim", "trim_weights_auto"),
    "tukey_threshold": ("weightpipe.methods.trim", "tukey_threshold"),
    "unknown_eligibility_weights": ("weightpipe.methods.eligibility", "unknown_eligibility_weights"),
    "weighting_class_nonresponse": ("weightpipe.methods.nonresponse", "weighting_class_nonresponse"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(import_module(mod), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
