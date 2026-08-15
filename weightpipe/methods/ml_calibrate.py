"""Model-assisted linear calibration via tree embeddings (forest / GBM)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomTreesEmbedding

from weightpipe.frame import WeightFrame
from weightpipe.methods.design_matrix import design_matrix, parse_formula, population_totals
from weightpipe.methods.linear import linear_calibrate
from weightpipe.steps.base import StepResult

CalEngine = Literal["linear", "forest", "gbm"]


def _feature_matrix(data: pd.DataFrame, formula: str | list[str] | tuple[str, ...]) -> pd.DataFrame:
    x = design_matrix(data, formula)
    cols = [c for c in x.columns if c != "(Intercept)"]
    if not cols:
        raise ValueError("ML calibration engine needs at least one non-intercept formula term")
    return x.loc[:, cols]


def _leaf_indicator_matrix(
    leaves_fit: np.ndarray,
    leaves_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One-hot leaf indicators with columns defined on the fitting sample."""
    n_fit, n_trees = leaves_fit.shape
    n_other = leaves_other.shape[0]
    blocks_fit: list[np.ndarray] = []
    blocks_other: list[np.ndarray] = []
    names: list[str] = []
    for t in range(n_trees):
        levels = np.unique(leaves_fit[:, t])
        # drop first level for rank (like drop_first dummies)
        keep = levels[1:] if len(levels) > 1 else levels
        for lev in keep:
            names.append(f"_leaf_t{t}_l{int(lev)}")
            blocks_fit.append((leaves_fit[:, t] == lev).astype(float))
            blocks_other.append((leaves_other[:, t] == lev).astype(float))
    if not names:
        # constant leaves — single column of ones for structure
        names = ["_leaf_const"]
        return np.ones((n_fit, 1)), np.ones((n_other, 1)), names
    return np.column_stack(blocks_fit), np.column_stack(blocks_other), names


def build_ml_auxiliaries(
    sample: pd.DataFrame,
    population: pd.DataFrame,
    formula: str | list[str] | tuple[str, ...],
    *,
    engine: Literal["forest", "gbm"],
    n_estimators: int = 40,
    max_depth: int = 3,
    seed: int | None = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Fit a tree embedding on the sample and map sample + population to auxiliaries."""
    missing_s = [t for t in parse_formula(formula) if t not in sample.columns]
    missing_p = [t for t in parse_formula(formula) if t not in population.columns]
    if missing_s:
        raise KeyError(f"sample missing formula columns: {missing_s}")
    if missing_p:
        raise KeyError(f"population missing formula columns: {missing_p}")

    xs = _feature_matrix(sample, formula).to_numpy(dtype=float)
    xp = _feature_matrix(population, formula).to_numpy(dtype=float)
    rng = 0 if seed is None else int(seed)

    if engine == "forest":
        embed = RandomTreesEmbedding(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            random_state=rng,
            sparse_output=False,
        )
        zs = np.asarray(embed.fit_transform(xs), dtype=float)
        zp = np.asarray(embed.transform(xp), dtype=float)
        # Drop all-zero and duplicate-ish first column patterns; keep finite cols with variation on pop or sample
        keep = [j for j in range(zs.shape[1]) if np.nanstd(zs[:, j]) > 0 or np.nanstd(zp[:, j]) > 0]
        if not keep:
            keep = list(range(min(1, zs.shape[1])))
        zs = zs[:, keep]
        zp = zp[:, keep]
        names = [f"_emb{j}" for j in range(zs.shape[1])]
        detail = {"embedder": "RandomTreesEmbedding", "n_features": len(names)}
    else:
        # GBM leaf codes from a synthetic working model on X
        y_syn = xs.mean(axis=1)
        model = GradientBoostingRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=0.08,
            random_state=rng,
        )
        model.fit(xs, y_syn)
        leaves_s = np.asarray(model.apply(xs), dtype=int)
        leaves_p = np.asarray(model.apply(xp), dtype=int)
        if leaves_s.ndim == 1:
            leaves_s = leaves_s.reshape(-1, 1)
            leaves_p = leaves_p.reshape(-1, 1)
        zs, zp, names = _leaf_indicator_matrix(leaves_s, leaves_p)
        detail = {"embedder": "GradientBoostingRegressor.leaves", "n_features": len(names)}

    sample_extra = pd.DataFrame(zs, index=sample.index, columns=names)
    pop_extra = pd.DataFrame(zp, index=population.index, columns=names)
    return sample_extra, pop_extra, names, detail


def ml_linear_calibrate(
    weights: pd.Series,
    data: pd.DataFrame,
    *,
    formula: str | list[str] | tuple[str, ...],
    population: pd.DataFrame,
    engine: Literal["forest", "gbm"] = "forest",
    totals: dict[str, Any] | pd.Series | None = None,
    include_linear: bool = True,
    bounds: tuple[float, float] | list[float] | None = None,
    penalty: float | dict[str, float] | None = None,
    calfun: str = "linear",
    n_estimators: int = 40,
    max_depth: int = 3,
    seed: int | None = 0,
    max_iter: int = 100,
    warn: bool = True,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Linear/GREG calibration on tree-embedding auxiliaries (+ optional linear terms).

    Requires a ``population`` microdata frame with the same formula columns so
    embedding totals can be computed. This is model-assisted calibration, not
    classical GREG with only census margins.
    """
    if engine not in ("forest", "gbm"):
        raise ValueError(f"unknown ML calibrate engine: {engine!r}")

    sample_extra, pop_extra, emb_names, emb_detail = build_ml_auxiliaries(
        data,
        population,
        formula,
        engine=engine,
        n_estimators=n_estimators,
        max_depth=max_depth,
        seed=seed,
    )

    work = data.copy()
    for c in emb_names:
        work[c] = sample_extra[c].to_numpy(dtype=float)

    if include_linear:
        base_terms = parse_formula(formula)
        full_formula = "~ " + " + ".join([*base_terms, *emb_names])
        if totals is None:
            merged_totals = population_totals(population, formula)
        else:
            merged_totals = dict(totals)
    else:
        full_formula = "~ " + " + ".join(emb_names)
        merged_totals = {} if totals is None else dict(totals)

    pop_with = population.copy()
    for c in emb_names:
        pop_with[c] = pop_extra[c].to_numpy(dtype=float)
    for c in emb_names:
        if c not in merged_totals:
            merged_totals[c] = float(pop_with[c].sum())
    if "(Intercept)" not in merged_totals:
        merged_totals["(Intercept)"] = float(len(population))

    # High-dimensional embeddings are often singular — default mild ridge if unbounded linear
    use_penalty = penalty
    if use_penalty is None and calfun == "linear" and bounds is None and len(emb_names) >= 2:
        use_penalty = 10.0

    weights_out, factors, diag = linear_calibrate(
        weights,
        work,
        formula=full_formula,
        totals=merged_totals,
        bounds=bounds,
        penalty=use_penalty,
        calfun=calfun,  # type: ignore[arg-type]
        max_iter=max_iter,
        warn=warn,
    )
    diag = dict(diag)
    diag["engine"] = engine
    diag["include_linear"] = include_linear
    diag["embedding"] = emb_detail
    diag["n_estimators"] = int(n_estimators)
    diag["max_depth"] = int(max_depth)
    diag["seed"] = seed
    return weights_out, factors, diag


def apply_ml_linear_calibrate(
    frame: WeightFrame,
    *,
    formula: str | list[str] | tuple[str, ...],
    population: pd.DataFrame,
    engine: Literal["forest", "gbm"] = "forest",
    totals: dict[str, Any] | pd.Series | None = None,
    include_linear: bool = True,
    bounds: tuple[float, float] | list[float] | None = None,
    penalty: float | dict[str, float] | None = None,
    calfun: str = "linear",
    n_estimators: int = 40,
    max_depth: int = 3,
    seed: int | None = 0,
    max_iter: int = 100,
    warn: bool = True,
) -> StepResult:
    weights, factors, diag = ml_linear_calibrate(
        frame.weights,
        frame.data,
        formula=formula,
        population=population,
        engine=engine,
        totals=totals,
        include_linear=include_linear,
        bounds=bounds,
        penalty=penalty,
        calfun=calfun,
        n_estimators=n_estimators,
        max_depth=max_depth,
        seed=seed,
        max_iter=max_iter,
        warn=warn,
    )
    return StepResult(weights=weights, factors=factors, diagnostics=diag)
