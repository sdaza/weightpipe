"""Pure weighting algorithms.

Implement and validate here before wiring into ``Recipe.step_*``.
Do not loosen recovery tolerances to make CI green — diagnose first.
"""

from weightpipe.methods.design_matrix import design_matrix, population_totals
from weightpipe.methods.eligibility import drop_ineligible_weights, unknown_eligibility_weights
from weightpipe.methods.linear import linear_calibrate
from weightpipe.methods.nonresponse import logit_propensity_nonresponse, weighting_class_nonresponse
from weightpipe.methods.poststrat import poststratify
from weightpipe.methods.raking import proportions_to_margins, rake
from weightpipe.methods.trim import potter_threshold, trim_weights, trim_weights_auto, tukey_threshold

__all__ = [
    "design_matrix",
    "drop_ineligible_weights",
    "linear_calibrate",
    "logit_propensity_nonresponse",
    "population_totals",
    "poststratify",
    "potter_threshold",
    "proportions_to_margins",
    "rake",
    "trim_weights",
    "trim_weights_auto",
    "tukey_threshold",
    "unknown_eligibility_weights",
    "weighting_class_nonresponse",
]
