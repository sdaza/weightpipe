#!/usr/bin/env python3
"""Size and memory microbenchmarks for weightpipe (optional raking peers).

Times the public APIs users would call. Peak RSS is the whole subprocess
(interpreter + frame + method). Python-heap peak is tracemalloc during the job.

Usage:
    uv run python scripts/bench_size_memory.py
    uv run python scripts/bench_size_memory.py --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

PROPS = {
    "sex": {"M": 0.48, "F": 0.52},
    "region": {"N": 0.25, "S": 0.25, "E": 0.25, "W": 0.25},
    "educ": {"hs": 0.40, "ba": 0.40, "grad": 0.20},
}
WEIGHTIPY_PCT = {
    var: {lev: 100.0 * p for lev, p in dist.items()} for var, dist in PROPS.items()
}
REGIONS = ("N", "S", "E", "W")
SIZES_DEFAULT = (5_000, 25_000, 100_000, 250_000)
SIZES_QUICK = (5_000, 25_000)
VAR_SIZES_DEFAULT = (10_000, 50_000)
VAR_SIZES_QUICK = (10_000,)
N_PSU_VARIANCE = 40
BOOT_REPLICATES = 50


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024.0


def make_frame(n: int, *, n_psu: int | None = None, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_psu = n_psu if n_psu is not None else max(20, n // 80)
    n_psu = min(n_psu, n)
    return pd.DataFrame(
        {
            "sex": rng.choice(["M", "F"], n),
            "region": rng.choice(REGIONS, n),
            "educ": rng.choice(["hs", "ba", "grad"], n, p=[0.45, 0.40, 0.15]),
            "age": rng.normal(40.0, 12.0, n).clip(18.0, 80.0),
            "unknown": (rng.random(n) < 0.05).astype(np.int8),
            "ineligible": (rng.random(n) < 0.03).astype(np.int8),
            "responded": (rng.random(n) < 0.80).astype(np.int8),
            "psu": rng.integers(0, n_psu, n),
            "stratum": rng.choice(["A", "B"], n),
            "pw": rng.uniform(0.8, 2.5, n),
            "y": rng.normal(50.0, 10.0, n),
        }
    )


def _rake_controls(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    total = float(df["pw"].sum())
    return {var: {k: v * total for k, v in dist.items()} for var, dist in PROPS.items()}


def job_baseline(df: pd.DataFrame) -> None:
    float(df["pw"].sum())


def job_rake_weightpipe(df: pd.DataFrame) -> None:
    from weightpipe import WeightPipe

    pipe = WeightPipe(df, weight="pw").calibrate(
        method="raking",
        proportions=PROPS,
        max_iter=50,
        tol=1e-6,
    )
    _ = pipe.weights


def job_rake_samplics(df: pd.DataFrame) -> None:
    from samplics.weighting import SampleWeight

    control = _rake_controls(df)
    SampleWeight().rake(
        samp_weight=df["pw"],
        margins={"sex": df["sex"], "region": df["region"], "educ": df["educ"]},
        control=control,
        tol=1e-6,
        ctrl_tol=1e-6,
        max_iter=50,
    )


def job_rake_weightipy(df: pd.DataFrame) -> None:
    import weightipy as wp

    scheme = wp.scheme_from_dict(WEIGHTIPY_PCT)
    wp.weight_dataframe(df, scheme, weight_column="w")


def job_linear_weightpipe(df: pd.DataFrame) -> None:
    from weightpipe import WeightPipe, population_totals

    rng = np.random.default_rng(1)
    pop = pd.DataFrame(
        {
            "region": rng.choice(REGIONS, 8_000),
            "sex": rng.choice(["M", "F"], 8_000),
            "age": rng.normal(42.0, 12.0, 8_000).clip(18.0, 80.0),
        }
    )
    totals = population_totals(pop, "~ region + sex + age")
    pipe = WeightPipe(df, weight="pw").calibrate(
        method="linear",
        formula="~ region + sex + age",
        totals=totals,
    )
    _ = pipe.weights


def job_cascade_weightpipe(df: pd.DataFrame) -> None:
    from weightpipe import WeightPipe

    pipe = (
        WeightPipe(df, weight="pw", psu="psu", strata="stratum")
        .unknown_eligibility(unknown="unknown", by=["region"])
        .drop_ineligible(ineligible="ineligible")
        .nonresponse(respondent="responded", method="weighting_class", by=["region"])
        .calibrate(method="raking", proportions=PROPS, max_iter=50, tol=1e-6)
        .trim(max_ratio=5.0, reference="median", redistribute=True)
    )
    _ = pipe.weights


def job_boot_cascade_weightpipe(df: pd.DataFrame) -> None:
    from weightpipe import WeightPipe

    pipe = (
        WeightPipe(df, weight="pw", psu="psu", strata="stratum")
        .unknown_eligibility(unknown="unknown", by=["region"])
        .drop_ineligible(ineligible="ineligible")
        .nonresponse(respondent="responded", method="weighting_class", by=["region"])
        .calibrate(method="raking", proportions=PROPS, max_iter=50, tol=1e-6)
        .trim(max_ratio=5.0, reference="median", redistribute=True)
    )
    pipe.estimate("y", estimand="mean", variance="bootstrap", replicates=BOOT_REPLICATES, seed=1)


def job_lin_cascade_weightpipe(df: pd.DataFrame) -> None:
    from weightpipe import WeightPipe

    pipe = (
        WeightPipe(df, weight="pw", psu="psu", strata="stratum")
        .unknown_eligibility(unknown="unknown", by=["region"])
        .drop_ineligible(ineligible="ineligible")
        .nonresponse(respondent="responded", method="weighting_class", by=["region"])
        .calibrate(method="raking", proportions=PROPS, max_iter=50, tol=1e-6)
        .trim(max_ratio=5.0, reference="median", redistribute=True)
    )
    pipe.estimate("y", estimand="mean", variance="linearization")


JOBS = {
    "baseline": job_baseline,
    "rake_weightpipe": job_rake_weightpipe,
    "rake_samplics": job_rake_samplics,
    "rake_weightipy": job_rake_weightipy,
    "linear_weightpipe": job_linear_weightpipe,
    "cascade_weightpipe": job_cascade_weightpipe,
    "boot_cascade_weightpipe": job_boot_cascade_weightpipe,
    "lin_cascade_weightpipe": job_lin_cascade_weightpipe,
}


def preload(job: str) -> None:
    """Import libraries before the timer so wall time is the weighting call."""
    if job == "baseline":
        return
    if job == "rake_samplics":
        from samplics.weighting import SampleWeight  # noqa: F401
        return
    if job == "rake_weightipy":
        import weightipy as wp  # noqa: F401
        return
    from weightpipe import WeightPipe, population_totals  # noqa: F401


def run_worker(job: str, n: int, n_psu: int | None) -> dict[str, object]:
    fn = JOBS[job]
    preload(job)
    df = make_frame(n, n_psu=n_psu)
    df_mb = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        fn(df)
        err = None
    except Exception as exc:  # noqa: BLE001 — surface missing optional deps
        err = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "job": job,
        "n": n,
        "n_psu": n_psu,
        "seconds": None if err else round(seconds, 4),
        "rss_mb": round(peak_rss_mb(), 2),
        "heap_mb": round(peak / (1024 * 1024), 2),
        "df_mb": round(df_mb, 2),
        "error": err,
    }


def probe_optional() -> dict[str, bool]:
    out = {"samplics": False, "weightipy": False}
    try:
        import samplics  # noqa: F401

        out["samplics"] = True
    except ImportError:
        pass
    try:
        import weightipy  # noqa: F401

        out["weightipy"] = True
    except ImportError:
        pass
    return out


def run_child(job: str, n: int, n_psu: int | None) -> dict[str, object]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", job, "--n", str(n)]
    if n_psu is not None:
        cmd.extend(["--n-psu", str(n_psu)])
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[-500:]
        return {
            "job": job,
            "n": n,
            "n_psu": n_psu,
            "seconds": None,
            "rss_mb": None,
            "heap_mb": None,
            "df_mb": None,
            "error": err,
        }
    return json.loads(proc.stdout)


def plan(quick: bool, available: dict[str, bool]) -> list[tuple[str, int, int | None]]:
    sizes = SIZES_QUICK if quick else SIZES_DEFAULT
    var_sizes = VAR_SIZES_QUICK if quick else VAR_SIZES_DEFAULT
    jobs = ["baseline", "rake_weightpipe"]
    if available["samplics"]:
        jobs.append("rake_samplics")
    if available["weightipy"]:
        jobs.append("rake_weightipy")
    jobs.extend(["linear_weightpipe", "cascade_weightpipe"])
    rows: list[tuple[str, int, int | None]] = []
    for n in sizes:
        for job in jobs:
            rows.append((job, n, None))
    for n in var_sizes:
        rows.append(("lin_cascade_weightpipe", n, N_PSU_VARIANCE))
        rows.append(("boot_cascade_weightpipe", n, N_PSU_VARIANCE))
    return rows


def fmt_s(v: object) -> str:
    if v is None:
        return "—"
    x = float(v)
    if x < 0.01:
        return f"{x * 1000:.1f} ms"
    return f"{x:.3f} s"


def fmt_mb(v: object) -> str:
    if v is None:
        return "—"
    return f"{float(v):.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=sorted(JOBS), default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--n-psu", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(__file__).resolve().parent / "bench_size_memory_results.json",
    )
    args = parser.parse_args()

    if args.worker is not None:
        if args.n is None:
            raise SystemExit("--n is required with --worker")
        print(json.dumps(run_worker(args.worker, args.n, args.n_psu)), flush=True)
        return 0

    available = probe_optional()
    rows = plan(args.quick, available)
    print(
        f"machine={platform.machine()} python={platform.python_version()} "
        f"samplics={available['samplics']} weightipy={available['weightipy']} "
        f"jobs={len(rows)}",
        flush=True,
    )
    results: list[dict[str, object]] = []
    for job, n, n_psu in rows:
        rec = run_child(job, n, n_psu)
        results.append(rec)
        extra = f" psu={n_psu}" if n_psu else ""
        status = rec["error"] or f"{fmt_s(rec['seconds'])}  rss={fmt_mb(rec['rss_mb'])} MiB"
        print(f"  {job:24s} n={n:<8d}{extra:10s} {status}", flush=True)

    payload = {
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "available": available,
        "results": results,
    }
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
