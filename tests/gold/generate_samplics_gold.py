"""Regenerate frozen samplics gold CSVs under tests/gold/.

Usage:
    uv run --extra gold python tests/gold/generate_samplics_gold.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from samplics.weighting import SampleWeight

OUT = Path(__file__).resolve().parent


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
    w = SampleWeight().rake(
        samp_weight=df["pw"],
        margins={"sex": df["sex"], "region": df["region"]},
        control=control,
        tol=1e-12,
        ctrl_tol=1e-12,
        max_iter=200,
    )
    g = df.copy()
    g["weight_samplics"] = w
    g.to_csv(OUT / "raking_2x2_samplics.csv", index=False)

    df2 = pd.DataFrame({"unit_id": [1, 2, 3, 4], "region": ["N", "N", "S", "S"], "pw": [1.0, 1.0, 1.0, 1.0]})
    w2 = SampleWeight().poststratify(
        samp_weight=df2["pw"],
        control={"N": 10.0, "S": 30.0},
        domain=df2["region"],
    )
    g2 = df2.copy()
    g2["weight_samplics"] = w2
    g2.to_csv(OUT / "poststrat_region_samplics.csv", index=False)

    df3 = pd.DataFrame(
        {
            "unit_id": [1, 2, 3, 4, 5],
            "region": ["N", "N", "N", "S", "S"],
            "responded": [1, 1, 0, 1, 0],
            "pw": [1.0, 1.0, 1.0, 2.0, 2.0],
        }
    )
    status = np.where(df3["responded"] == 1, "rr", "nr")
    w3 = SampleWeight().adjust(
        samp_weight=df3["pw"].to_numpy(),
        adj_class=df3["region"].to_numpy(),
        resp_status=status,
        resp_dict={"rr": "respondent", "nr": "non-respondent", "in": "ineligible", "uk": "unknown"},
        unknown_to_inelig=False,
    )
    g3 = df3.copy()
    g3["weight_samplics"] = w3
    g3.to_csv(OUT / "nr_weighting_class_samplics.csv", index=False)

    print(f"Wrote gold CSVs under {OUT}")


if __name__ == "__main__":
    main()
