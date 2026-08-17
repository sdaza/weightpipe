"""weightpipe: declarative survey weighting with recipe-aware replicates."""

from weightpipe.design import Design
from weightpipe.diagnostics import design_effect, ess
from weightpipe.estimate import estimate, point_estimate
from weightpipe.frame import WeightFrame
from weightpipe.methods import design_matrix, population_totals
from weightpipe.pipeline import WeightPipe
from weightpipe.planning import (
    allocate_strata,
    allocation_table,
    margin_of_error,
    sample_size,
    stratified_margin_of_error,
)
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
    "WeightPipe",
    "WeightResult",
    "__version__",
    "allocate_strata",
    "allocation_table",
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
    "margin_of_error",
    "point_estimate",
    "population_totals",
    "sample_size",
    "stratified_margin_of_error",
    "weight_factors",
]

__version__ = "0.1.0"
