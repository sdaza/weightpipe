"""Eligibility helpers: drop ineligible and unknown-eligibility redistribution."""

from typing import Any

import numpy as np
import pandas as pd

from weightpipe.frame import WeightFrame
from weightpipe.methods.cluster_utils import assert_cluster_column, cluster_table
from weightpipe.steps.base import StepResult, as_logical_mask, make_cells


def drop_ineligible_weights(
    weights: pd.Series,
    data: pd.DataFrame,
    *,
    ineligible: str,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Zero out active units flagged as ineligible (not redistributed)."""
    w = weights.astype(float).to_numpy(copy=True)
    active = w > 0
    inelig = as_logical_mask(data, ineligible).to_numpy()
    drop = active & inelig
    w_before = w.copy()
    w[drop] = 0.0
    factors = np.ones_like(w)
    factors[drop] = 0.0
    factors[~active] = 1.0
    diag = {
        "n_dropped": int(drop.sum()),
        "weight_dropped": float(w_before[drop].sum()),
        "n_remaining": int((w > 0).sum()),
    }
    return (
        pd.Series(w, index=weights.index, name="weights"),
        pd.Series(factors, index=weights.index, name="factors"),
        diag,
    )


def unknown_eligibility_weights(
    weights: pd.Series,
    data: pd.DataFrame,
    *,
    unknown: str,
    by: list[str] | None = None,
    cluster: str | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Redistribute unknown-eligibility weight among known units within cells.

    With ``cluster``, each cluster counts once (mean member weight). A cluster is
    unknown if any member is unknown; the factor is applied to every member.
    """
    w0 = weights.astype(float).to_numpy(copy=True)
    n = len(w0)
    unknown_mask = as_logical_mask(data, unknown).to_numpy()
    eligible = w0 > 0
    cells = make_cells(data, by, n).to_numpy()
    new_w = w0.copy()
    factors = np.ones(n, dtype=float)
    cell_rows: list[dict[str, Any]] = []

    if cluster is None:
        for g in pd.unique(cells):
            idx = np.where((cells == g) & eligible)[0]
            if idx.size == 0:
                continue
            unk = idx[unknown_mask[idx]]
            known = idx[~unknown_mask[idx]]
            w_known = float(w0[known].sum()) if known.size else 0.0
            w_tot = float(w0[idx].sum())
            factor = (w_tot / w_known) if w_known > 0 else np.nan
            if np.isfinite(factor):
                factors[known] = factor
                factors[unk] = 0.0
                new_w[known] = w0[known] * factor
                new_w[unk] = 0.0
            cell_rows.append(
                {
                    "cell": str(g),
                    "level": "person",
                    "n_known": int(known.size),
                    "n_unknown": int(unk.size),
                    "factor": float(factor) if np.isfinite(factor) else None,
                }
            )
    else:
        assert_cluster_column(data, cluster)
        cl = data[cluster].astype(str).to_numpy()
        for g in pd.unique(cells):
            idx = np.where((cells == g) & eligible)[0]
            if idx.size == 0:
                continue
            tbl, _ = cluster_table(
                w0,
                data,
                cluster=cluster,
                eligible=(cells == g) & eligible,
                flag=unknown_mask,
                flag_reduce="any",
                cells=cells,
            )
            if tbl.empty:
                continue
            w_tot = float(tbl["weight"].sum())
            known_mask = ~tbl["flag"].to_numpy(dtype=bool)
            w_known = float(tbl.loc[known_mask, "weight"].sum())
            factor = (w_tot / w_known) if w_known > 0 else np.nan
            unk_clusters = set(tbl.loc[~known_mask, "cluster"].astype(str))
            if np.isfinite(factor):
                member_unk = np.isin(cl[idx], list(unk_clusters))
                known_idx = idx[~member_unk]
                unk_idx = idx[member_unk]
                factors[known_idx] = factor
                factors[unk_idx] = 0.0
                new_w[known_idx] = w0[known_idx] * factor
                new_w[unk_idx] = 0.0
            cell_rows.append(
                {
                    "cell": str(g),
                    "level": "household",
                    "n_known": int(known_mask.sum()),
                    "n_unknown": int((~known_mask).sum()),
                    "factor": float(factor) if np.isfinite(factor) else None,
                }
            )

    diag = {"cells": cell_rows, "method": "unknown_eligibility", "cluster": cluster}
    return (
        pd.Series(new_w, index=weights.index, name="weights"),
        pd.Series(factors, index=weights.index, name="factors"),
        diag,
    )


def apply_drop_ineligible(frame: WeightFrame, *, ineligible: str) -> StepResult:
    weights, factors, diag = drop_ineligible_weights(frame.weights, frame.data, ineligible=ineligible)
    return StepResult(weights=weights, factors=factors, diagnostics=diag)


def apply_unknown_eligibility(
    frame: WeightFrame,
    *,
    unknown: str,
    by: list[str] | None = None,
    cluster: str | None = None,
) -> StepResult:
    weights, factors, diag = unknown_eligibility_weights(
        frame.weights,
        frame.data,
        unknown=unknown,
        by=by,
        cluster=cluster,
    )
    return StepResult(weights=weights, factors=factors, diagnostics=diag)
