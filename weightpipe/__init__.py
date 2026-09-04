"""weightpipe: declarative survey weighting with recipe-aware replicates."""

from weightpipe._logging import set_log_level, setup_logging
from weightpipe.design import Design
from weightpipe.diagnostics import BalanceReport, balance, design_effect, ess, margins
from weightpipe.estimate import Estimation, estimate, estimate_glm, point_estimate
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
    "BalanceReport",
    "BootstrapResult",
    "Design",
    "Estimation",
    "JackknifeResult",
    "Recipe",
    "WeightFrame",
    "WeightPipe",
    "WeightResult",
    "__version__",
    "allocate_strata",
    "allocation_table",
    "balance",
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
    "estimate_glm",
    "jack_mean",
    "jack_median",
    "jack_proportion",
    "jack_ratio",
    "jack_total",
    "jackknife_estimate",
    "jackknife_weights",
    "margin_of_error",
    "margins",
    "point_estimate",
    "population_totals",
    "sample_size",
    "set_log_level",
    "setup_logging",
    "stratified_margin_of_error",
    "weight_factors",
]

__version__ = "0.1.2"
