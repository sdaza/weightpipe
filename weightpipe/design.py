"""Sampling design: base weights and variance structure (strata / PSU)."""

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Design:
    """Sampling design metadata and base weights.

    Base weights live in ``data[weight]``. Variance resampling uses ``strata``
    and ``psu`` (Rao–Wu / jackknife). Omit ``psu`` for unit-level designs
    (SRS or stratified SRS); omit ``strata`` for a single overall stratum.
    """

    data: pd.DataFrame
    weight: str
    strata: str | None = None
    psu: str | None = None
    kind: str = "custom"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.weight not in self.data.columns:
            raise KeyError(f"weight column not found: {self.weight}")
        w = self.data[self.weight]
        if (w < 0).any():
            raise ValueError("design weights must be non-negative")
        if self.strata is not None and self.strata not in self.data.columns:
            raise KeyError(f"strata column not found: {self.strata}")
        if self.psu is not None and self.psu not in self.data.columns:
            raise KeyError(f"psu column not found: {self.psu}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight,
            "strata": self.strata,
            "psu": self.psu,
            "kind": self.kind,
            "meta": dict(self.meta),
            "n": len(self.data),
        }

    @classmethod
    def from_weights(
        cls,
        data: pd.DataFrame,
        *,
        weight: str,
        strata: str | None = None,
        psu: str | None = None,
        copy: bool = True,
    ) -> Self:
        """Use an existing base-weight column (any design)."""
        frame = data.copy() if copy else data
        return cls(data=frame, weight=weight, strata=strata, psu=psu, kind="custom")

    @classmethod
    def srs(
        cls,
        data: pd.DataFrame,
        *,
        N: float | int,
        weight: str = "base_weight",
        copy: bool = True,
    ) -> Self:
        """Simple random sampling without replacement: ``w = N / n``."""
        frame = data.copy() if copy else data
        n = len(frame)
        if n < 1:
            raise ValueError("data must have at least one row")
        if float(N) <= 0:
            raise ValueError("N must be positive")
        if weight in frame.columns:
            raise ValueError(f"column {weight!r} already exists; pass a new weight name")
        frame = frame.assign(**{weight: float(N) / n})
        return cls(
            data=frame,
            weight=weight,
            strata=None,
            psu=None,
            kind="srs",
            meta={"N": float(N), "n": n},
        )

    @classmethod
    def stratified(
        cls,
        data: pd.DataFrame,
        *,
        stratum: str,
        N_h: dict[Any, float] | pd.Series,
        weight: str = "base_weight",
        copy: bool = True,
    ) -> Self:
        """Stratified SRS: ``w_i = N_h / n_h`` within each stratum.

        ``N_h`` maps stratum labels to population sizes. Sample sizes ``n_h``
        are counted from ``data``.
        """
        if stratum not in data.columns:
            raise KeyError(f"stratum column not found: {stratum}")
        frame = data.copy() if copy else data
        if weight in frame.columns:
            raise ValueError(f"column {weight!r} already exists; pass a new weight name")

        n_h = frame.groupby(stratum, observed=True).size()
        pop = {str(k): float(v) for k, v in dict(N_h).items()}
        weights = np.empty(len(frame), dtype=float)
        meta_nh: dict[str, dict[str, float]] = {}
        labels = frame[stratum].astype(str)
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

        frame = frame.assign(**{weight: weights})
        return cls(
            data=frame,
            weight=weight,
            strata=stratum,
            psu=None,
            kind="stratified",
            meta={"N_h": meta_nh},
        )

    @classmethod
    def cluster(
        cls,
        data: pd.DataFrame,
        *,
        weight: str,
        psu: str,
        strata: str | None = None,
        copy: bool = True,
    ) -> Self:
        """Cluster / multi-stage design with precomputed base weights.

        Pass ultimate-cluster ids in ``psu``. Optional ``strata`` for stratified
        cluster samples. Multi-stage inclusion probabilities should already be
        folded into ``weight``.
        """
        frame = data.copy() if copy else data
        return cls(
            data=frame,
            weight=weight,
            strata=strata,
            psu=psu,
            kind="cluster" if strata is None else "stratified_cluster",
        )
