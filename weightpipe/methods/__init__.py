"""Pure weighting algorithms.

Implement and validate here before wiring into ``Recipe.step_*``.
Do not loosen recovery tolerances to make CI green — diagnose first.
"""

from weightpipe.methods.eligibility import drop_ineligible_weights
from weightpipe.methods.nonresponse import weighting_class_nonresponse
from weightpipe.methods.raking import proportions_to_margins, rake

__all__ = [
    "drop_ineligible_weights",
    "proportions_to_margins",
    "rake",
    "weighting_class_nonresponse",
]
