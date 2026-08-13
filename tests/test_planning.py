"""Analytical tests for pre-field sample planning."""

import numpy as np
import pandas as pd
import pytest

from weightpipe import (
    Design,
    allocate_strata,
    allocation_table,
    margin_of_error,
    sample_size,
    stratified_margin_of_error,
)


def test_sample_size_and_margin_are_inverse_at_standard_inputs() -> None:
    assert sample_size(0.05) == 384
    assert margin_of_error(384) == pytest.approx(0.05)
    assert sample_size(0.05, deff=1.2, response_rate=0.9) == 512
    assert sample_size(0.05, deff=1.2, response_rate=0.9, population=1000) == 370
    assert margin_of_error(370, deff=1.2, response_rate=0.9, population=1000) == pytest.approx(0.05)


def test_sample_size_is_vectorized_and_capped_by_population() -> None:
    result = sample_size([0.05, 0.03], population=[100, 10_000])
    assert isinstance(result, np.ndarray)
    assert result.tolist() == [80, 964]
    assert sample_size(0.01, population=20) == 20


def test_relative_margin_of_error() -> None:
    absolute = margin_of_error(384, proportion=0.5)
    relative = margin_of_error(384, proportion=0.5, relative=True)
    assert relative == pytest.approx(round(absolute / 0.5, 4))


def test_mixed_and_neyman_allocation() -> None:
    population = np.array([1000, 3000])
    assert allocate_strata(400, population).tolist() == [100, 300]
    assert allocate_strata(400, population, method="mixed", proportional_weight=0).tolist() == [200, 200]
    neyman = allocate_strata(400, population, method="neyman", proportion=[0.1, 0.5])
    assert int(neyman.sum()) == 400
    assert neyman[1] > neyman[0]


def test_error_allocation_and_stratified_margin() -> None:
    population = np.array([1000, 2000, 3000])
    planned = allocate_strata(
        None,
        population,
        method="error",
        margin=0.1,
        proportion=[0.3, 0.5, 0.4],
    )
    error = stratified_margin_of_error(planned, population=population, proportion=[0.3, 0.5, 0.4])
    assert error > 0
    assert error < 0.1


def test_allocation_table_hands_off_to_stratified_design() -> None:
    populations = {"North": 1000, "South": 3000}
    table = allocation_table(populations, sample=200)
    assert list(table.columns) == ["stratum", "population", "sample"]

    sample = pd.DataFrame(
        {
            "region": np.repeat(table["stratum"], table["sample"]),
            "y": 1.0,
        }
    ).reset_index(drop=True)
    design = Design.stratified(sample, stratum="region", N_h=populations)
    assert design.kind == "stratified"
    assert design.data.groupby("region")[design.weight].first().to_dict() == {
        "North": pytest.approx(20.0),
        "South": pytest.approx(20.0),
    }


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: sample_size(0), "margin"),
        (lambda: sample_size(0.05, response_rate=0), "response_rate"),
        (lambda: margin_of_error(0), "sample"),
        (
            lambda: allocate_strata(None, [100, 200], method="mixed"),
            "sample",
        ),
        (
            lambda: stratified_margin_of_error([10, 20]),
            "population or weights",
        ),
    ],
)
def test_planning_validation(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]
