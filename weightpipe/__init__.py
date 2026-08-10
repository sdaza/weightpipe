"""weightpipe: declarative survey weighting with recipe-aware replicates."""

from weightpipe.diagnostics import design_effect, ess
from weightpipe.frame import WeightFrame
from weightpipe.recipe import Recipe
from weightpipe.replicates import (
    BootstrapResult,
    boot_mean,
    boot_total,
    bootstrap_estimate,
    bootstrap_weights,
)
from weightpipe.result import WeightResult, collect_weights, weight_factors

__all__ = [
    "BootstrapResult",
    "Recipe",
    "WeightFrame",
    "WeightResult",
    "__version__",
    "boot_mean",
    "boot_total",
    "bootstrap_estimate",
    "bootstrap_weights",
    "collect_weights",
    "design_effect",
    "ess",
    "weight_factors",
]

__version__ = "0.1.0"
