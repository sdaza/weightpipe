"""Regenerate frozen svy gold CSVs under tests/gold/.

Usage:
    uv run --extra gold python tests/gold/generate_svy_gold.py
"""

from pathlib import Path

import pandas as pd
import polars as pl
from svy import Cat, Design, Sample

OUT = Path(__file__).resolve().parent


def _sample(df: pd.DataFrame, *, wgt: str = "pw") -> Sample:
    return Sample(data=pl.from_pandas(df), design=Design(wgt=wgt))


def _weights(sample: Sample, column: str) -> pd.Series:
    return (
        sample.data.sort("svy_row_index")
        .select(pl.col(column))
        .to_series()
        .to_pandas()
        .astype(float)
        .reset_index(drop=True)
    )


def main() -> None:
    df = pd.DataFrame(
        {
            "unit_id": [1, 2, 3, 4],
            "sex": ["M", "M", "F", "F"],
            "region": ["N", "S", "N", "S"],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    control = {"sex": {"M": 60.0, "F": 40.0}, "region": {"N": 30.0, "S": 70.0}}
    raked = _sample(df).weighting.rake(controls=control, wgt_name="rk_wgt", tol=1e-12, max_iter=200)
    g = df.copy()
    g["weight_svy"] = _weights(raked, "rk_wgt")
    g.to_csv(OUT / "raking_2x2_svy.csv", index=False)

    df2 = pd.DataFrame({"unit_id": [1, 2, 3, 4], "region": ["N", "N", "S", "S"], "pw": [1.0, 1.0, 1.0, 1.0]})
    ps = _sample(df2).weighting.poststratify(controls={"N": 10.0, "S": 30.0}, by="region", wgt_name="ps_wgt")
    g2 = df2.copy()
    g2["weight_svy"] = _weights(ps, "ps_wgt")
    g2.to_csv(OUT / "poststrat_region_svy.csv", index=False)

    df3 = pd.DataFrame(
        {
            "unit_id": [1, 2, 3, 4, 5],
            "region": ["N", "N", "N", "S", "S"],
            "responded": [1, 1, 0, 1, 0],
            "pw": [1.0, 1.0, 1.0, 2.0, 2.0],
            "resp_status": ["rr", "rr", "nr", "rr", "nr"],
        }
    )
    adj = _sample(df3).weighting.adjust(
        resp_status="resp_status",
        by="region",
        wgt_name="nr_wgt",
        unknown_to_inelig=False,
        respondents_only=False,
    )
    g3 = df3.drop(columns=["resp_status"])
    g3["weight_svy"] = _weights(adj, "nr_wgt")
    g3.to_csv(OUT / "nr_weighting_class_svy.csv", index=False)

    df4 = pd.DataFrame(
        {
            "unit_id": [1, 2, 3, 4],
            "region": ["N", "N", "S", "S"],
            "age": [20.0, 40.0, 30.0, 50.0],
            "pw": [1.0, 1.0, 1.0, 1.0],
        }
    )
    cal = _sample(df4).weighting.calibrate(
        controls={Cat("region"): {"N": 50.0, "S": 50.0}, "age": 3500.0},
        wgt_name="cal_wgt",
    )
    g4 = df4.copy()
    g4["weight_svy"] = _weights(cal, "cal_wgt")
    g4.to_csv(OUT / "linear_calibrate_svy.csv", index=False)

    print(f"Wrote gold CSVs under {OUT}")


if __name__ == "__main__":
    main()
