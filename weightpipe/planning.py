"""Pre-field sample-size and precision planning."""

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

AllocationMethod = Literal["mixed", "error", "neyman", "root", "stdev"]


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    if value.ndim == 0 or value.size == 1:
        return float(value.reshape(-1)[0])
    return value


def _z_critical(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return float(round(abs(norm.ppf((1 - confidence) / 2)), 2))


def sample_size(
    margin: float | list[float] | np.ndarray,
    *,
    deff: float | list[float] | np.ndarray = 1,
    response_rate: float | list[float] | np.ndarray = 1,
    population: float | list[float] | np.ndarray | None = None,
    confidence: float = 0.95,
    proportion: float | list[float] | np.ndarray = 0.5,
) -> int | np.ndarray:
    """Required sample size for estimating a proportion.

    ``population=None`` uses the infinite-population formula. ``deff`` and
    ``response_rate`` inflate the number of sampled units required.
    """
    margin_a = _array(margin)
    deff_a = _array(deff)
    response_a = _array(response_rate)
    proportion_a = _array(proportion)

    if np.any((margin_a <= 0) | (margin_a >= 1)):
        raise ValueError("margin must be between 0 and 1")
    if np.any(deff_a <= 0):
        raise ValueError("deff must be positive")
    if np.any((response_a <= 0) | (response_a > 1)):
        raise ValueError("response_rate must be in (0, 1]")
    if np.any((proportion_a <= 0) | (proportion_a >= 1)):
        raise ValueError("proportion must be between 0 and 1")

    z = _z_critical(confidence)
    n0 = z**2 * proportion_a * (1 - proportion_a) / margin_a**2
    if population is None:
        planned = np.round(n0 * deff_a / response_a)
    else:
        population_a = _array(population)
        if np.any(population_a <= 0):
            raise ValueError("population must be positive")
        planned = np.round(((n0 * population_a) / (n0 + population_a - 1)) * deff_a / response_a)
        planned = np.minimum(planned, population_a)

    planned = np.asarray(planned, dtype=int)
    if planned.ndim == 0 or planned.size == 1:
        return int(planned.reshape(-1)[0])
    return planned


def margin_of_error(
    sample: float | list[float] | np.ndarray,
    *,
    deff: float | list[float] | np.ndarray = 1,
    response_rate: float | list[float] | np.ndarray = 1,
    population: float | list[float] | np.ndarray | None = None,
    confidence: float = 0.95,
    proportion: float | list[float] | np.ndarray = 0.5,
    relative: bool = False,
) -> float | np.ndarray:
    """Margin of error for a planned proportion estimate."""
    sample_a = _array(sample)
    deff_a = _array(deff)
    response_a = _array(response_rate)
    proportion_a = _array(proportion)

    if np.any(sample_a <= 0):
        raise ValueError("sample must be positive")
    if np.any(deff_a <= 0):
        raise ValueError("deff must be positive")
    if np.any((response_a <= 0) | (response_a > 1)):
        raise ValueError("response_rate must be in (0, 1]")
    if np.any((proportion_a <= 0) | (proportion_a >= 1)):
        raise ValueError("proportion must be between 0 and 1")

    effective_n = sample_a / deff_a * response_a
    error = _z_critical(confidence) * np.sqrt(proportion_a * (1 - proportion_a) / effective_n)
    if population is not None:
        population_a = _array(population)
        if np.any(population_a <= 0):
            raise ValueError("population must be positive")
        if np.any(sample_a > population_a):
            raise ValueError("sample cannot exceed population")
        error *= np.sqrt((population_a - effective_n) / (population_a - 1))

    error = np.round(error, 4)
    if relative:
        error = np.round(error / proportion_a, 4)
    return _scalar_or_array(np.asarray(error, dtype=float))


def allocate_strata(
    sample: int | None,
    population: list[float] | np.ndarray | pd.Series,
    *,
    method: AllocationMethod = "mixed",
    minimum: int = 1,
    proportional_weight: float = 1,
    margin: float | list[float] | np.ndarray | None = None,
    deff: float | list[float] | np.ndarray = 1,
    response_rate: float | list[float] | np.ndarray = 1,
    proportion: float | list[float] | np.ndarray = 0.5,
) -> np.ndarray | pd.Series:
    """Allocate a planned sample across strata.

    Methods are ``mixed`` (equal/proportional blend), ``root``, ``neyman``,
    ``stdev``, and ``error`` (a margin target for every stratum).
    """
    index = population.index if isinstance(population, pd.Series) else None
    population_a = _array(population).reshape(-1)
    if np.any(population_a <= 0):
        raise ValueError("population must be positive")
    strata = len(population_a)
    proportion_a = _array(proportion)
    response_a = _array(response_rate)
    deff_a = _array(deff)
    for name, value in (("proportion", proportion_a), ("response_rate", response_a)):
        if value.size not in (1, strata):
            raise ValueError(f"{name} must be scalar or match population length")

    if method == "error":
        if margin is None:
            raise ValueError("margin is required when method='error'")
        planned = _array(
            sample_size(
                margin,
                deff=deff_a,
                response_rate=response_a,
                population=population_a,
                proportion=proportion_a,
            )
        )
    else:
        if sample is None or sample <= 0:
            raise ValueError("sample must be positive")
        total = float(sample)
        if method == "mixed":
            if not 0 <= proportional_weight <= 1:
                raise ValueError("proportional_weight must be between 0 and 1")
            equal = np.round(total / strata)
            proportional = np.round(total * population_a / population_a.sum())
            planned = np.round(equal * (1 - proportional_weight) + proportional * proportional_weight)
        elif method == "root":
            planned = np.round(total * np.sqrt(population_a) / np.sqrt(population_a).sum())
        elif method in ("neyman", "stdev"):
            sd = np.sqrt(proportion_a * (1 - proportion_a))
            basis = population_a * sd if method == "neyman" else np.broadcast_to(sd, population_a.shape)
            planned = np.round(total * basis / basis.sum())
        else:
            raise ValueError(f"unknown allocation method: {method!r}")
        planned = np.round(planned * deff_a / response_a)

    planned = np.maximum(planned, minimum)
    planned = np.minimum(planned, population_a).astype(int)
    if index is not None:
        return pd.Series(planned, index=index, name="sample")
    return planned


def stratified_margin_of_error(
    sample: list[float] | np.ndarray | pd.Series,
    *,
    population: list[float] | np.ndarray | pd.Series | None = None,
    weights: list[float] | np.ndarray | None = None,
    deff: float | list[float] | np.ndarray = 1,
    response_rate: float | list[float] | np.ndarray = 1,
    proportion: float | list[float] | np.ndarray = 0.5,
    confidence: float = 0.95,
    relative: bool = False,
) -> float:
    """Overall margin of error for a stratified proportion."""
    sample_a = _array(sample).reshape(-1)
    if sample_a.size < 2:
        raise ValueError("sample must contain at least two strata")
    if weights is None:
        if population is None:
            raise ValueError("provide population or weights")
        population_a = _array(population).reshape(-1)
        if population_a.size != sample_a.size:
            raise ValueError("population must match sample length")
        weights_a = population_a / population_a.sum()
        fpc: float | np.ndarray = (population_a - sample_a) / (population_a - 1)
    else:
        weights_a = _array(weights).reshape(-1)
        if weights_a.size != sample_a.size:
            raise ValueError("weights must match sample length")
        fpc = 1.0

    proportion_a = _array(proportion)
    effective_n = sample_a / _array(deff) * _array(response_rate)
    variance = np.sum(weights_a**2 * proportion_a * (1 - proportion_a) / effective_n * fpc)
    error = float(round(_z_critical(confidence) * np.sqrt(variance), 4))
    if relative:
        aggregate = float(np.average(np.broadcast_to(proportion_a, sample_a.shape), weights=weights_a))
        return float(round(error / aggregate, 4))
    return error


def allocation_table(
    population: dict[Any, float] | pd.Series,
    *,
    sample: int | None,
    method: AllocationMethod = "mixed",
    **kwargs: Any,
) -> pd.DataFrame:
    """Return ``stratum``, ``population``, and planned ``sample`` columns."""
    population_s = pd.Series(population, dtype=float)
    planned = allocate_strata(sample, population_s, method=method, **kwargs)
    result = pd.DataFrame(
        {
            "stratum": population_s.index,
            "population": population_s.to_numpy(),
            "sample": np.asarray(planned),
        }
    )
    return result.reset_index(drop=True)
