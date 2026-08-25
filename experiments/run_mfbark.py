"""
MF-BARK final 10-seed no-warmstart run.

Exactly matches Exp A config but extends from 5 → 10 seeds:
  - Surrogate: MF-BARK (n_trees=30, clone_rank=3, n_burn=220, n_keep=25, thin=6, n_max_train=160)
  - Acquisition: MC-MF-MES (unchanged, all 3 fidelities)
  - Init: 11 MTP + 1 MBR, NO pilot warm-start (n_pilot_init=0)
  - Runs to the full 40,000 cost budget
  - Tag: mfbark_<ts>

Comparison targets: GP no-warmstart 56.3±5.6, NGBoost-ensemble 49.8±9.1, BARK 5-seed 50.7±15.8.

Usage: python experiments/run_mfbark.py [n_seeds]   (default 10)
"""
from __future__ import annotations
import os, sys, json, pickle, time
from datetime import datetime
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CH3  = os.path.dirname(_HERE)
_SRC  = os.path.join(_CH3, "src")
for p in (_HERE, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from objective import make_objective
from mf_bark import MFBARK
import bo_loop as loop

print("Loading objective (frozen: MBR sf0.7, noise ON, paper-style MTP) …", flush=True)
_f_obj, _meta = make_objective(distribution="alpha", noise=True)

BARK_KW = dict(n_clones=30, n_fidelities=3, pilot_fidelity=2, n_trees=30, clone_rank=3,
               n_burn=220, n_keep=25, thin=6, n_max_train=160)


def make_surrogate(rng):
    return MFBARK(seed=int(rng.integers(0, 2**31)), warm=True, **BARK_KW)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"mfbark_{ts}"
    rd  = os.path.join(_CH3, "results")
    os.makedirs(rd, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  MF-BARK  ({n_seeds} seeds, no warm start, 40k budget)")
    print(f"  Config: n_trees=30 clone_rank=3 n_burn=220 n_keep=25 thin=6 n_max_train=160")
    print(f"  Init: 11 MTP + 1 MBR, n_pilot_init=0   tag={tag}")
    print(f"{'='*72}", flush=True)

    t_total = time.time()
    seed_res = []

    for seed in range(n_seeds):
        print(f"\n  seed={seed}", flush=True)
        res = loop.run_mfmes_bo(make_surrogate, _f_obj, seed, verbose=True, n_pilot_init=0)
        seed_res.append(res)

        ckpt = os.path.join(rd, f"{tag}_ckpt_s{seed}.pkl")
        with open(ckpt, "wb") as fh:
            pickle.dump(res, fh)

        cs = res["clone_stats"][-1] if res["clone_stats"] else {}
        print(f"    → best_pilot={res['best_pilot_final']:.3f}  "
              f"mix MTP/MBR/Plt={res['fid_counts'][0]}/{res['fid_counts'][1]}/{res['fid_counts'][2]}  "
              f"pilot_rds={res['pilot_rounds']}  never_pilot={res['guard_needed']}  "
              f"budget={res['budget_frac']*100:.0f}%  ei_rounds={res.get('ei_rounds', 0)}  "
              f"n_rounds={res['n_rounds']}  ({res['elapsed']:.0f}s)  "
              f"Bclone off={cs.get('off_mean', float('nan')):.2f}±{cs.get('off_std', float('nan')):.2f}",
              flush=True)

    finals   = np.array([r["best_pilot_final"] for r in seed_res], float)
    ok       = finals[np.isfinite(finals)]
    fc       = {f: float(np.mean([r["fid_counts"][f] for r in seed_res])) for f in [0, 1, 2]}
    bud      = float(np.mean([r["budget_frac"] for r in seed_res]) * 100)
    n_mbr    = int(np.sum([r["fid_counts"][1] > 0 for r in seed_res]))
    n_never  = int(np.sum([r["guard_needed"]       for r in seed_res]))
    ei_total = int(np.sum([r.get("ei_rounds", 0)   for r in seed_res]))
    wall     = time.time() - t_total

    print(f"\n{'='*72}")
    print(f"  MF-BARK SUMMARY  ({n_seeds} seeds, no warm start)")
    print(f"{'='*72}")
    print(f"  best pilot titer   = {np.mean(ok):.2f} ± {np.std(ok):.2f}  g/L")
    print(f"  budget spent       = {bud:.0f}%")
    print(f"  fidelity mix (avg) MTP/MBR/Pilot = {fc[0]:.1f}/{fc[1]:.1f}/{fc[2]:.1f}")
    print(f"  seeds using MBR    = {n_mbr}/{n_seeds}")
    print(f"  seeds never Pilot  = {n_never}/{n_seeds}")
    print(f"  ei_rounds total    = {ei_total}  (target 0)")
    print(f"  wall time          = {wall/60:.1f} min")
    print(f"  per-seed finals:   {[f'{v:.2f}' for v in finals]}")
    print(f"  per-seed mix:      {[(r['fid_counts'][0],r['fid_counts'][1],r['fid_counts'][2]) for r in seed_res]}")
    print(f"\n  Comparison:")
    print(f"    GP no-warmstart:        56.3 ± 5.6  g/L")
    print(f"    NGBoost-ensemble:       49.8 ± 9.1  g/L")
    print(f"    MF-BARK 5-seed (Exp A): 50.7 ± 15.8 g/L")
    print(f"    MF-BARK 10-seed (this): {np.mean(ok):.1f} ± {np.std(ok):.1f} g/L", flush=True)

    summary = dict(
        method="MF-BARK", tag=tag, n_seeds=n_seeds,
        mean=float(np.mean(ok)) if ok.size else float("nan"),
        std=float(np.std(ok))   if ok.size else float("nan"),
        finals=finals.tolist(),
        fid_mix=fc, budget_pct=bud,
        seeds_using_mbr=n_mbr, seeds_never_pilot=n_never,
        ei_rounds_total=ei_total, wall_min=wall / 60,
        bark_kw=BARK_KW, run_kw=dict(n_pilot_init=0),
        per_seed=[{k: r[k] for k in ("fid_counts", "best_pilot_final", "pilot_rounds",
                  "guard_needed", "budget_frac", "n_rounds", "elapsed", "ei_rounds")}
                 for r in seed_res],
    )
    sfile = os.path.join(rd, f"{tag}_summary.json")
    with open(sfile, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\n  Saved: {os.path.basename(sfile)}")


if __name__ == "__main__":
    main()
