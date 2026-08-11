"""Shared helpers for cluster-level adjustments."""

from typing import Any

import numpy as np
import pandas as pd


def assert_cluster_column(data: pd.DataFrame, cluster: str) -> None:
    if cluster not in data.columns:
        raise KeyError(f"cluster column not found: {cluster}")


def cluster_table(
    weights: np.ndarray,
    data: pd.DataFrame,
    *,
    cluster: str,
    eligible: np.ndarray,
    flag: np.ndarray,
    flag_reduce: str,
    cells: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Collapse active units to one row per cluster.

    ``flag_reduce`` is ``\"any\"`` (unknown eligibility) or ``\"all\"`` (response).
    Cluster weight is the mean of member weights (weightflow convention).
    """
    assert_cluster_column(data, cluster)
    if flag_reduce not in ("any", "all"):
        raise ValueError(f"flag_reduce must be 'any' or 'all', got {flag_reduce!r}")

    idx = np.where(eligible)[0]
    if idx.size == 0:
        empty = pd.DataFrame(columns=["cluster", "weight", "flag", "cell"])
        return empty, idx

    cl = data[cluster].astype(str).to_numpy()[idx]
    rows: list[dict[str, Any]] = []
    for h in pd.unique(cl):
        mem = idx[cl == h]
        fl = flag[mem]
        flag_h = bool(fl.any()) if flag_reduce == "any" else bool(fl.all())
        rows.append(
            {
                "cluster": str(h),
                "weight": float(weights[mem].mean()),
                "flag": flag_h,
                "cell": str(cells[mem[0]]),
            }
        )
    return pd.DataFrame(rows), idx
