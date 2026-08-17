"""Sampling design: base weights and variance structure (strata / PSU)."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from weightpipe._logging import get_logger

logger = get_logger(__name__)


def _as_col_list(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    cols = [str(c) for c in value]
    if not cols:
        raise ValueError("stage column list must be non-empty")
    return cols


def _infer_kind(
    *,
    N: Any,
    N_h: Any,
    weight: str | None,
    probabilities: list[str] | None,
    stage_weights: list[str] | None,
    strata: str | None,
    psu: str | None,
) -> str:
    if N is not None:
        return "srs"
    if N_h is not None:
        return "stratified"
    if probabilities is not None or stage_weights is not None:
        return "stratified_multistage" if strata is not None else "multistage"
    if psu is not None:
        return "stratified_cluster" if strata is not None else "cluster"
    return "custom"


def _product_weights(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    as_probabilities: bool,
) -> np.ndarray:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"stage columns not found: {missing}")
    mat = frame.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(mat).all():
        raise ValueError("stage columns must be finite")
    if as_probabilities:
        if np.any((mat <= 0) | (mat > 1)):
            raise ValueError("inclusion probabilities must be in (0, 1]")
        with np.errstate(divide="raise", invalid="raise"):
            return 1.0 / np.prod(mat, axis=1)
    if np.any(mat < 0):
        raise ValueError("stage weights must be non-negative")
    return np.prod(mat, axis=1)


@dataclass(frozen=True, init=False)
class Design:
    """Sampling design metadata and base weights.

    Pass the inputs that define the design; ``kind`` is inferred:

    - *(none)* → unit weights ``1`` (with a log message)
    - ``N=...`` → SRS, weights ``N/n``
    - ``strata=...``, ``N_h=...`` → stratified SRS, weights ``N_h/n_h``
    - ``weight=...``, ``psu=...`` → cluster (add ``strata=`` if stratified)
    - ``probabilities=[...]``, ``psu=...`` → multi-stage, ``w = 1 / ∏ π_k``
    - ``stage_weights=[...]``, ``psu=...`` → multi-stage, ``w = ∏ w_k``
    - ``weight=...`` only → use existing weights

    For multi-stage / cluster designs, ``psu`` is the ultimate cluster used for
    bootstrap/jackknife variance. Stage details are folded into the weight.

    Base weights live in ``data[weight]``.
    """

    data: pd.DataFrame
    weight: str
    strata: str | None = None
    psu: str | None = None
    kind: str = "custom"
    meta: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        weight: str | None = None,
        strata: str | None = None,
        psu: str | None = None,
        N: float | int | None = None,
        N_h: dict[Any, float] | pd.Series | None = None,
        probabilities: str | Sequence[str] | None = None,
        stage_weights: str | Sequence[str] | None = None,
        copy: bool = True,
    ) -> None:
        prob_cols = _as_col_list(probabilities)
        stage_cols = _as_col_list(stage_weights)
        sources = (N, N_h, weight, prob_cols, stage_cols)
        n_sources = sum(x is not None for x in sources)
        if n_sources > 1:
            raise ValueError("provide at most one of N=, N_h=, weight=, probabilities=, or stage_weights=")
        if N is not None and (strata is not None or psu is not None):
            raise ValueError("N= (SRS) cannot be combined with strata= or psu=")
        if N_h is not None and strata is None:
            raise ValueError("N_h= requires strata=")
        if N_h is not None and psu is not None:
            raise ValueError("N_h= stratified SRS cannot be combined with psu=")
        if (prob_cols is not None or stage_cols is not None) and psu is None:
            raise ValueError("multi-stage designs require psu= (ultimate cluster for variance)")

        frame = data.copy() if copy else data
        meta: dict[str, Any] = {}
        weight_col: str
        kind = _infer_kind(
            N=N,
            N_h=N_h,
            weight=weight,
            probabilities=prob_cols,
            stage_weights=stage_cols,
            strata=strata,
            psu=psu,
        )

        if n_sources == 0:
            weight_col = "base_weight"
            if len(frame) < 1:
                raise ValueError("data must have at least one row")
            if weight_col in frame.columns:
                raise ValueError(f"column {weight_col!r} already exists")
            frame = frame.assign(**{weight_col: 1.0})
            meta = {"unit_weights": True, "note": "no design weight provided; using 1.0"}
            logger.info("No design weight provided; using base_weight=1.0 for all rows")
        elif N is not None:
            weight_col = "base_weight"
            n = len(frame)
            if n < 1:
                raise ValueError("data must have at least one row")
            if float(N) <= 0:
                raise ValueError("N must be positive")
            if weight_col in frame.columns:
                raise ValueError(f"column {weight_col!r} already exists")
            frame = frame.assign(**{weight_col: float(N) / n})
            meta = {"N": float(N), "n": n}
            strata = None
            psu = None
        elif N_h is not None:
            assert strata is not None
            weight_col = "base_weight"
            if strata not in frame.columns:
                raise KeyError(f"strata column not found: {strata}")
            if weight_col in frame.columns:
                raise ValueError(f"column {weight_col!r} already exists")
            n_h = frame.groupby(strata, observed=True).size()
            pop = {str(k): float(v) for k, v in dict(N_h).items()}
            weights = np.empty(len(frame), dtype=float)
            meta_nh: dict[str, dict[str, float]] = {}
            labels = frame[strata].astype(str)
            for label, n in n_h.items():
                key = str(label)
                if key not in pop:
                    raise KeyError(f"N_h missing stratum {label!r}")
                Nh = pop[key]
                if n < 1 or Nh <= 0:
                    raise ValueError(f"invalid N_h/n_h for stratum {label!r}: N={Nh}, n={n}")
                w = Nh / float(n)
                weights[labels.to_numpy() == key] = w
                meta_nh[key] = {"N": Nh, "n": float(n), "weight": w}
            frame = frame.assign(**{weight_col: weights})
            meta = {"N_h": meta_nh}
            psu = None
        elif prob_cols is not None or stage_cols is not None:
            weight_col = "base_weight"
            if weight_col in frame.columns:
                raise ValueError(f"column {weight_col!r} already exists")
            if prob_cols is not None:
                weights = _product_weights(frame, prob_cols, as_probabilities=True)
                meta = {
                    "stages": len(prob_cols),
                    "probabilities": list(prob_cols),
                    "formula": "1 / prod(probabilities)",
                }
            else:
                assert stage_cols is not None
                weights = _product_weights(frame, stage_cols, as_probabilities=False)
                meta = {
                    "stages": len(stage_cols),
                    "stage_weights": list(stage_cols),
                    "formula": "prod(stage_weights)",
                }
            frame = frame.assign(**{weight_col: weights})
        else:
            assert weight is not None
            weight_col = weight
            if weight_col not in frame.columns:
                raise KeyError(f"weight column not found: {weight_col}")

        if strata is not None and strata not in frame.columns:
            raise KeyError(f"strata column not found: {strata}")
        if psu is not None and psu not in frame.columns:
            raise KeyError(f"psu column not found: {psu}")
        w = frame[weight_col]
        if (w < 0).any():
            raise ValueError("design weights must be non-negative")

        object.__setattr__(self, "data", frame)
        object.__setattr__(self, "weight", weight_col)
        object.__setattr__(self, "strata", strata)
        object.__setattr__(self, "psu", psu)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "meta", meta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight,
            "strata": self.strata,
            "psu": self.psu,
            "kind": self.kind,
            "meta": dict(self.meta),
            "n": len(self.data),
        }
