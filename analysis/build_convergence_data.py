"""
Build docs/data/convergence.json for the interactive site.

Reads the per-seed convergence traces written by the experiment runs and
resamples them onto a common cost grid, storing the mean and standard
deviation per method.

    python analysis/build_convergence_data.py [trace_dir]

Defaults to results/. Each trace file is expected to contain a "seeds" list
whose entries hold a "pilot_trace" of [cost, titer] pairs.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUDGET = 40_000
N_GRID = 60

PATTERNS = {
    "gp": "*conv_traces_gp_*.json",
    "bark": "*conv_traces_bark_*.json",
    "ng": "*conv_traces_ngboost_*.json",
}


def newest(trace_dir: str, pattern: str) -> str | None:
    hits = glob.glob(os.path.join(trace_dir, pattern))
    return sorted(hits)[-1] if hits else None


def main() -> None:
    trace_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "results")
    grid = np.linspace(0, BUDGET, N_GRID)
    out: dict = {"grid": [round(float(g)) for g in grid], "methods": {}}

    for key, pattern in PATTERNS.items():
        path = newest(trace_dir, pattern)
        if path is None:
            print(f"  {key:5s} no trace file matching {pattern}")
            continue

        with open(path) as fh:
            data = json.load(fh)

        seeds = data["seeds"]
        mat = np.full((len(seeds), N_GRID), np.nan)
        for i, seed in enumerate(seeds):
            trace = seed.get("pilot_trace", [])
            if not trace:
                continue
            cost = np.array([p[0] for p in trace], float)
            titer = np.array([p[1] for p in trace], float)
            mat[i] = np.interp(grid, cost, titer, left=np.nan, right=float(titer[-1]))

        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        clean = lambda a: [None if not np.isfinite(v) else round(float(v), 2) for v in a]
        out["methods"][key] = {
            "n_seeds": len(seeds),
            "mean": clean(mean),
            "std": clean(std),
        }
        print(f"  {key:5s} {os.path.basename(path):44s} seeds={len(seeds)}  "
              f"final={mean[-1]:.1f} g/L")

    if not out["methods"]:
        print(f"\n  nothing written, no trace files found in {trace_dir}")
        return

    dest = os.path.join(_ROOT, "docs", "data", "convergence.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"\n  wrote {os.path.relpath(dest, _ROOT)} "
          f"({os.path.getsize(dest)} bytes, {len(out['methods'])} methods)")


if __name__ == "__main__":
    main()
