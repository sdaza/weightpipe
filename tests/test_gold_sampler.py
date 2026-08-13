"""Frozen CSV + optional live gold checks against R ``sampler``.

CSV tests always run when ``tests/gold/planning_*_r_sampler.csv`` exist
(committed for CI). Regenerate:

  SAMPLER_R_DIR=... Rscript tests/gold/generate_sampler_gold.R
  # or with installed package:
  Rscript tests/gold/generate_sampler_gold.R

Live rpy2 tests skip without extra ``r-gold`` / R package ``sampler``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from weightpipe import allocate_strata, margin_of_error, sample_size, stratified_margin_of_error

GOLD = Path(__file__).resolve().parent / "gold"
pytestmark = pytest.mark.gold

ALLOC_COLS = {
    "aprop": {"sample": 1000, "proportional_weight": 1.0},
    "afixed": {"sample": 1000, "proportional_weight": 0.0},
    "a40": {"sample": 1000, "proportional_weight": 0.4},
    "a60": {"sample": 1000, "proportional_weight": 0.6},
    "aroot": {"sample": 1000, "method": "root"},
    "aneyman": {"sample": 1000, "method": "neyman"},
    "astdev": {"sample": 1000, "method": "stdev"},
    "aerr": {"sample": None, "method": "error", "margin": 0.11},
}


def _read_optional(name: str) -> pd.DataFrame | None:
    path = GOLD / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def test_planning_ssize_serr_matches_r_sampler_csv() -> None:
    g = _read_optional("planning_ssize_serr_r_sampler.csv")
    if g is None:
        pytest.skip("missing planning_ssize_serr_r_sampler.csv — run generate_sampler_gold.R")

    for _, row in g.iterrows():
        deff = float(row["deff"])
        rr = float(row["rr"])
        p = float(row["p"])
        cl = float(row["cl"])
        n_pop = None if pd.isna(row["N"]) else float(row["N"])
        expected = float(row["result_r_sampler"])
        if row["kind"] == "ssize":
            ours = sample_size(
                float(row["e"]),
                deff=deff,
                response_rate=rr,
                population=n_pop,
                confidence=cl,
                proportion=p,
            )
            assert ours == pytest.approx(expected)
        else:
            ours = margin_of_error(
                float(row["n"]),
                deff=deff,
                response_rate=rr,
                population=n_pop,
                confidence=cl,
                proportion=p,
            )
            assert ours == pytest.approx(expected, abs=1e-12)


def test_planning_chile_alloc_matches_r_sampler_csv() -> None:
    g = _read_optional("planning_chile_alloc_r_sampler.csv")
    if g is None:
        pytest.skip("missing planning_chile_alloc_r_sampler.csv — run generate_sampler_gold.R")

    population = g["pob"].to_numpy(dtype=float)
    proportion = g["pr"].to_numpy(dtype=float)
    for col, kwargs in ALLOC_COLS.items():
        call_kwargs = dict(kwargs)
        if call_kwargs.get("method") in ("neyman", "stdev", "error"):
            call_kwargs["proportion"] = proportion
        ours = allocate_strata(population=population, **call_kwargs)
        np.testing.assert_array_equal(np.asarray(ours), g[col].to_numpy())


def test_planning_chile_moe_matches_r_sampler_csv() -> None:
    alloc = _read_optional("planning_chile_alloc_r_sampler.csv")
    moe = _read_optional("planning_chile_moe_r_sampler.csv")
    if alloc is None or moe is None:
        pytest.skip("missing planning chile MOE gold — run generate_sampler_gold.R")

    population = alloc["pob"].to_numpy(dtype=float)
    proportion = alloc["pr"].to_numpy(dtype=float)
    for _, row in moe.iterrows():
        name = str(row["allocation"])
        ours = stratified_margin_of_error(
            alloc[name].to_numpy(dtype=float),
            population=population,
            proportion=proportion,
        )
        assert ours == pytest.approx(float(row["moe_r_sampler"]), abs=1e-12)


def test_planning_matches_r_sampler_live() -> None:
    """Live rpy2 check against R sampler (skips without rpy2 / package)."""
    pytest.importorskip("rpy2")
    from rpy2.robjects.packages import importr

    try:
        importr("sampler")
    except Exception:
        pytest.skip("R package 'sampler' not installed (sdaza/sampler)")

    from rpy2 import robjects as ro

    ssize = ro.r["ssize"]
    serr = ro.r["serr"]
    astrata = ro.r["astrata"]
    serrst = ro.r["serrst"]

    assert sample_size(0.05) == int(ssize(0.05)[0])
    assert margin_of_error(384) == pytest.approx(float(serr(384)[0]))
    assert sample_size(0.05, deff=1.2, response_rate=0.9, population=1000) == int(
        ssize(0.05, deff=1.2, rr=0.9, N=1000)[0]
    )

    pob = ro.FloatVector([1000.0, 3000.0])
    r_n = np.array(list(astrata(400, pob, wp=1.0)), dtype=int)
    ours = allocate_strata(400, [1000.0, 3000.0], proportional_weight=1.0)
    np.testing.assert_array_equal(np.asarray(ours), r_n)
    assert stratified_margin_of_error(ours, population=[1000.0, 3000.0]) == pytest.approx(
        float(serrst(n=ro.IntVector(r_n.tolist()), N=pob)[0])
    )
