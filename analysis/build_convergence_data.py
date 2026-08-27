"""
Build docs/data/convergence.json for the interactive site.

Reads the per-seed convergence traces produced by the experiment runs and
resamples them onto a common cost grid, storing the mean and standard
deviation per method. Run once after new experiments:

    python analysis/build_convergence_data.py
"""
from __future__ import annotations
import json
import os
import glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

BUDGET = 40_000
N_GRID = 60

# Trace files live wherever the runs wrote them. Point this at the folder
# holding ch3_conv_traces_*.json if they are not in results/.
SEARCH_DIRS = [
    os.path.join(_ROOT, "results"),
    os.path.expanduser("~/Desktop/Research Project/Step 1 (apply decision tree as "
                       "surrogate and compare)/chapter3_constrained_multifidelity_discrete/results"),
]

PATTERNS = {
    "gp":   "*conv_traces_gp_*.json",
    "bark": "*conv_traces_bark_*.json",
    "ng":   "*conv_traces_ngboost_*.json",
}


def newest(pattern: str) -> str | None:
    hits: list[str] = []
    for d in SEARCH_DIRS:
        hits += glob.glob(os.path.join(d, pattern))
    return sorted(hits)[-1] if hits else None


def main() -> None:
    grid = np.linspace(0, BUDGET, N_GRID)
    out = {"grid": [round(float(g)) for g in grid], "methods": {}}

    for key, pattern in PATTERNS.items():
        path = newest(pattern)
        if path is None:
            print(f"  {key:5s} no trace file found for {pattern}")
            continue

        with open(path) as fh:
            data = json.load(fh)

        mat = np.full((len(data["seeds"]), N_GRID), np.nan)
        for i, seed in enumerate(data["seeds"]):
            trace = seed.get("pilot_trace", [])
            if not trace:
                continue
            cost = np.array([p[0] for p in trace], float)
            titer = np.array([p[1] for p in trace], float)
            mat[i] = np.interp(grid, cost, titer, left=np.nan, right=float(titer[-1]))

        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        clean = lambda a: [None if not np.isfinite(v) else round(float(v), 2) for v in a]
        out["methods"][key] = {"n_seeds": len(data["seeds"]),
                               "mean": clean(mean), "std": clean(std)}
        print(f"  {key:5s} {os.path.basename(path):48s} seeds={len(data['seeds'])}  "
              f"final={mean[-1]:.1f} g/L")

    dest = os.path.join(_ROOT, "docs", "data", "convergence.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"\n  wrote {os.path.relpath(dest, _ROOT)}  "
          f"({os.path.getsize(dest)} bytes, {len(out['methods'])} methods)")


if __name__ == "__main__":
    main()
