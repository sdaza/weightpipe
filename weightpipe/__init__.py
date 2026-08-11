"""weightpipe: declarative survey weighting with recipe-aware replicates."""

from weightpipe.design import Design
from weightpipe.diagnostics import design_effect, ess
from weightpipe.estimate import estimate, point_estimate
from weightpipe.frame import WeightFrame
from weightpipe.methods import design_matrix, population_totals
from weightpipe.recipe import Recipe
from weightpipe.replicates import (
    BootstrapResult,
    JackknifeResult,
    boot_mean,
    boot_median,
    boot_proportion,
    boot_ratio,
    boot_total,
    bootstrap_estimate,
    bootstrap_weights,
    jack_mean,
    jack_median,
    jack_proportion,
    jack_ratio,
    jack_total,
    jackknife_estimate,
    jackknife_weights,
)
from weightpipe.result import WeightResult, collect_weights, weight_factors

__all__ = [
    "BootstrapResult",
    "Design",
    "JackknifeResult",
    "Recipe",
    "WeightFrame",
    "WeightResult",
    "__version__",
    "boot_mean",
    "boot_median",
    "boot_proportion",
    "boot_ratio",
    "boot_total",
    "bootstrap_estimate",
    "bootstrap_weights",
    "collect_weights",
    "design_effect",
    "design_matrix",
    "ess",
    "estimate",
    "jack_mean",
    "jack_median",
    "jack_proportion",
    "jack_ratio",
    "jack_total",
    "jackknife_estimate",
    "jackknife_weights",
    "point_estimate",
    "population_totals",
    "weight_factors",
]

__version__ = "0.1.0"
