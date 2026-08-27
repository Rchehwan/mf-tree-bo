"""
EXP B2 — GP paper-engine baseline with NO pilot warm-start, on the CURRENT frozen objective, to fill
the missing 2x2 cell (counterpart to Exp B GP+warmstart 59.97 and MF-BARK Exp A no-warmstart 50.7).

Benchmark init only (11 MTP + 1 MBR, no seeded Pilot), run to the 40,000 cost budget.
The init sampling is not patched, so this is the benchmark's own
default init, matching MF-BARK Exp A. Objective FROZEN.

Run with the mf-bo4bio environment:  python experiments/run_gp_baseline.py [n_seeds]
"""
from __future__ import annotations
import os, sys, json
from datetime import datetime
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
# Point MFBO_ROOT at a local clone of github.com/adrian-martens/mf-bo4bio
_MFBO = os.environ.get("MFBO_ROOT", os.path.expanduser("~/mf-bo4bio"))
for _p in (os.path.join(_MFBO, "src"), os.path.join(_REPO, "src"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from objective import make_objective
print("Loading objective (frozen) …", flush=True)
_f_obj, _meta = make_objective(distribution="alpha", noise=True)
FIXED_INIT_GLC, FIXED_INIT_GLN = 6.0, 3.0
_eval_log: list[dict] = []
BATCH = {0: 12, 1: 4, 2: 1}   # paper batch sizes -> rounds = eval_count / batch


def _shim(X, initial_conditions=None, mbr_level: int = 7, clone_distribution: str = "alpha", rng=None):
    fid_map = {0: 0, mbr_level: 1, 10: 2}
    out = []
    for row in np.asarray(X):
        T, pH, F1, F2, F3, fidelity, clone = row
        pf = int(round(float(fidelity))); mf = fid_map[pf]; ci = int(round(float(clone)))
        f1, f3 = (0.0, 0.0) if pf == 0 else (float(F1), float(F3))
        y = _f_obj([float(T), float(pH), FIXED_INIT_GLC, FIXED_INIT_GLN, f1, float(F2), f3, mf, ci])
        out.append(float(y)); _eval_log.append({"fid_internal": mf, "y": y})
    return out


import mfbo4bio.virtual_lab as vl
vl.conduct_experiment = _shim
print("  patched conduct_experiment ✓  (NO warm-start: paper default init 11 MTP + 1 MBR)", flush=True)

import mfbo4bio.mfbo_qLogEI as paper_engine


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = os.path.join(_CH3, "results")
    print(f"\n{'='*70}\n  EXP B2 — GP paper engine, NO warm-start  ({n_seeds} seeds)\n{'='*70}", flush=True)
    finals, mixes, per_seed = [], [], []
    for seed in range(n_seeds):
        _eval_log.clear()
        try:
            paper_engine.run(n_iterations=50, output_name=f"gp_baseline_{seed+1}", seed=seed,
                             clone_distribution="alpha", mbr_level=7, task_representation="HYBRID", embed_dim=3)
        except Exception as exc:
            print(f"  seed={seed} ERROR: {exc}", flush=True); import traceback; traceback.print_exc(); continue
        pilot_ys = [e["y"] for e in _eval_log if e["fid_internal"] == 2]
        bp = float(max(pilot_ys)) if pilot_ys else float("nan")
        cnt = {f: sum(1 for e in _eval_log if e["fid_internal"] == f) for f in (0, 1, 2)}
        # BO rounds = (evals - init evals) / batch; init = 11 MTP + 1 MBR (12 evals). Approx rounds via batch.
        rounds = {f: cnt[f] // BATCH[f] for f in (0, 1, 2)}
        finals.append(bp); mixes.append(cnt)
        per_seed.append(dict(seed=seed, best_pilot=bp, eval_counts=cnt, rounds=rounds,
                             uses_mbr=cnt[1] > 0, reaches_pilot=cnt[2] > 0))
        print(f"  seed={seed}  best_pilot={bp:.3f}  evals MTP/MBR/Pilot={cnt[0]}/{cnt[1]}/{cnt[2]}  "
              f"rounds~{rounds[0]}/{rounds[1]}/{rounds[2]}", flush=True)
    finals = np.array(finals, float); ok = finals[np.isfinite(finals)]
    mbr_seeds = int(sum(p["uses_mbr"] for p in per_seed))
    no_pilot = int(sum(not p["reaches_pilot"] for p in per_seed))
    mean_rounds = {f: float(np.mean([p["rounds"][f] for p in per_seed])) for f in (0, 1, 2)}
    print(f"\n  EXP B2 SUMMARY: GP NO-warmstart titer = {np.mean(ok):.2f} ± {np.std(ok):.2f}  (n={ok.size})", flush=True)
    print(f"  per-seed: {[f'{v:.2f}' for v in finals]}", flush=True)
    print(f"  mean rounds MTP/MBR/Pilot = {mean_rounds[0]:.1f}/{mean_rounds[1]:.1f}/{mean_rounds[2]:.1f}  "
          f"| seeds using MBR = {mbr_seeds}/{n_seeds} | seeds never reaching Pilot = {no_pilot}/{n_seeds}", flush=True)
    print(f"  (this run is on the CURRENT frozen objective; supersedes the old 52.3 which was an earlier objective)", flush=True)
    with open(os.path.join(rd, f"gp_baseline_{ts}_summary.json"), "w") as fh:
        json.dump(dict(method="GP_paper_engine_NO_warmstart", n_seeds=n_seeds,
                       mean=float(np.mean(ok)) if ok.size else float("nan"),
                       std=float(np.std(ok)) if ok.size else float("nan"),
                       finals=finals.tolist(), mean_rounds=mean_rounds,
                       seeds_using_mbr=mbr_seeds, seeds_never_pilot=no_pilot, per_seed=per_seed), fh, indent=1)
    print(f"  saved gp_baseline_{ts}_summary.json", flush=True)


if __name__ == "__main__":
    main()
