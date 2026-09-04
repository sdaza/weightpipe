"""Smoke tests for the public package surface."""

import tomllib
from pathlib import Path

import pandas as pd
import pytest

from weightpipe import Recipe, WeightFrame, WeightResult, __version__


def test_version() -> None:
    # Releases bump pyproject.toml and __init__.py together; keep them in step.
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml is not available outside a source checkout")
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert __version__ == declared


def test_weight_frame_from_base(toy_sample: pd.DataFrame) -> None:
    frame = WeightFrame.from_base(toy_sample, base_weight="design_weight", unit_id="unit_id")
    assert list(frame.step_names) == []
    assert frame.weights.equals(frame.data["base_weight"])
    assert "unit_id" in frame.data.columns


def test_weight_frame_allows_zero_rejects_negative() -> None:
    ok = WeightFrame(data=pd.DataFrame({"base_weight": [1.0, 0.0]}))
    assert float(ok.weights.sum()) == 1.0
    bad = pd.DataFrame({"base_weight": [1.0, -0.1]})
    with pytest.raises(ValueError, match="non-negative"):
        WeightFrame(data=bad)


def test_recipe_fit_base_only(toy_sample: pd.DataFrame) -> None:
    result = Recipe(data=toy_sample, base_weight="design_weight", unit_id="unit_id").fit()
    assert isinstance(result, WeightResult)
    assert len(result.weights) == 4
    assert result.diagnostics["steps_applied"] == []
