"""
NGBoost ensemble: full 10-seed run.

K=10 bootstrap-diverse NGBoost members with varied tree depths [2,3,4,5,6,7].
Functional posterior via ensemble mixture; max-value samples draw a member then
sample from that member's predictive distribution → genuine spread over y*.
The ensemble posterior keeps the MES score finite, so the EI fallback is not used.

Init: 11 MTP + 1 MBR (no pilot warm-start — MF-BARK Exp A parity).
Acquisition: MC-MF-MES.  Budget: 40 000.

Usage:  python experiments/run_ngboost_ensemble.py [n_seeds]   (default 10)
"""
from __future__ import annotations
import os, sys, json, pickle
from datetime import datetime
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CH3  = os.path.dirname(_HERE)
_SRC  = os.path.join(_CH3, "src")
for p in (_HERE, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from objective      import make_objective
from ngboost_ensemble import NGBoostEnsembleCokrig
import bo_loop as loop

print("Loading objective (MBR sf0.7, noise ON, paper-style MTP) …")
_f_obj, _meta = make_objective(distribution="alpha", noise=True)


def _jsonable(o):
    if isinstance(o, dict):         return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):return [_jsonable(v) for v in o]
    if isinstance(o, np.bool_):     return bool(o)
    if isinstance(o, np.integer):   return int(o)
    if isinstance(o, np.floating):  return float(o)
    if isinstance(o, np.ndarray):   return o.tolist()
    return o


def make_surrogate(rng):
    return NGBoostEnsembleCokrig(
        pilot_fidelity=2, n_estimators=50, K=10,
        corr_rng=np.random.default_rng(int(rng.integers(0, 2**31))))


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(_CH3, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  NGBoost Ensemble (K=10) + MF-MES  ({n_seeds} seeds)")
    print(f"  Init: 11 MTP + 1 MBR (no pilot warm-start — MF-BARK Exp A parity)")
    print(f"  Tag: ngboost_ensemble_{ts}")
    print(f"{'='*72}")

    seed_res = []
    for seed in range(n_seeds):
        print(f"\n  seed={seed}")
        res = loop.run_mfmes_bo(
            make_surrogate, _f_obj, seed, verbose=True,
            n_pilot_init=0)
        seed_res.append(res)

        pkl_path = os.path.join(results_dir,
                                f"ngboost_ensemble_{ts}_ckpt_s{seed}.pkl")
        with open(pkl_path, "wb") as fh:
            pickle.dump(res, fh)

        es = res["clone_stats"][-1] if res["clone_stats"] else {}
        print(f"    → best_pilot={res['best_pilot_final']:.3f}  "
              f"mix MTP/MBR/Plt={res['fid_counts'][0]}/{res['fid_counts'][1]}/{res['fid_counts'][2]}  "
              f"pilot_rounds={res['pilot_rounds']}  guard_needed={res['guard_needed']}  "
              f"ei_rounds={res['ei_rounds']}  ({res['elapsed']:.0f}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    finals = np.array([r["best_pilot_final"] for r in seed_res], float)
    ok     = finals[np.isfinite(finals)]
    fc     = {f: float(np.mean([r["fid_counts"][f] for r in seed_res])) for f in [0, 1, 2]}
    n_mbr         = int(np.sum([r["fid_counts"][1] > 0     for r in seed_res]))
    n_guard       = int(np.sum([r["guard_needed"]           for r in seed_res]))
    n_never_pilot = int(np.sum([r["pilot_rounds"] == 0      for r in seed_res]))
    ei_per_seed   = [r["ei_rounds"] for r in seed_res]
    ei_total      = int(sum(ei_per_seed))
    budget_pct    = float(np.mean([r["budget_frac"] for r in seed_res])) * 100

    # Comparison targets
    REF = {
        "Single NGBoost fixed, no-warmstart (10s)": "45.60 ±  9.30",
        "MF-BARK no-warmstart (10s)":               "50.67 ± 15.79",
        "MF-BARK warmstart (10s)":                  "51.61 ± 10.93",
        "GP no-warmstart (10s)":                    "56.34 ±  5.64",
        "GP warmstart (10s)":                       "59.97 ±  8.46",
    }

    print(f"\n{'='*72}")
    print(f"  FINAL RESULTS — NGBoost Ensemble (K=10) + MF-MES")
    print(f"{'='*72}")
    print(f"  Best pilot titer: {ok.mean():.2f} ± {ok.std():.2f}  (n_ok={ok.size})")
    print(f"  Fidelity mix MTP/MBR/Pilot = {fc[0]:.1f}/{fc[1]:.1f}/{fc[2]:.1f}")
    print(f"  Seeds using MBR      = {n_mbr}/{n_seeds}")
    print(f"  Seeds never Pilot    = {n_never_pilot}/{n_seeds}   guard_needed = {n_guard}/{n_seeds}")
    print(f"  ei_rounds per seed   = {ei_per_seed}  total = {ei_total}")
    print(f"  Budget spent (mean)  = {budget_pct:.0f}%")
    print(f"  Per-seed finals:  {[f'{v:.2f}' for v in finals]}")
    print(f"  Per-seed pilot:   {[r['fid_counts'][2] for r in seed_res]}")
    print(f"  Per-seed MBR:     {[r['fid_counts'][1] for r in seed_res]}")

    print(f"\n  {'Method':<46} {'Titer':>22}")
    print(f"  {'-'*68}")
    print(f"  {'NGBoost Ensemble K=10 (this, no-warmstart)':<46} "
          f"{ok.mean():.2f} ± {ok.std():.2f}")
    for name, val in REF.items():
        print(f"  {name:<46} {val}")

    gap_bark = ok.mean() - 50.67
    gap_sngb = ok.mean() - 45.60
    gap_gp   = ok.mean() - 56.34
    print(f"\n  Δ vs MF-BARK no-warmstart: {gap_bark:+.2f}  "
          f"Δ vs single-NGBoost: {gap_sngb:+.2f}  "
          f"Δ vs GP no-warmstart: {gap_gp:+.2f}")

    print(f"\n  INTERPRETATION:")
    if n_never_pilot == 0 and ei_total == 0:
        print(f"  All seeds reached the Pilot scale; the EI fallback was never used.")
    elif n_never_pilot == 0:
        print(f"  ✓ All seeds reached Pilot.  ei_total={ei_total} (minor).")
    else:
        print(f"  ✗ {n_never_pilot}/{n_seeds} seeds never reached Pilot.")

    print(f"  Interpretability: ensemble of {10} readable NGBoost members.")
    print(f"  Per-member IF-THEN rules extractable; diversity visible across members.")

    # Save summary JSON
    summary = dict(
        method="NGBoost_Ensemble_K10_MFMES",
        tag=f"ngboost_ensemble_{ts}",
        n_seeds=n_seeds,
        mean=float(ok.mean()) if ok.size else float("nan"),
        std=float(ok.std())  if ok.size else float("nan"),
        finals=finals.tolist(),
        fid_mix=fc, seeds_using_mbr=n_mbr, guard_needed=n_guard,
        seeds_never_pilot=n_never_pilot,
        ei_rounds_per_seed=ei_per_seed, ei_rounds_total=ei_total,
        budget_pct=budget_pct,
        per_seed=[{k: r[k] for k in
                   ("fid_counts", "best_pilot_final", "pilot_rounds",
                    "guard_needed", "n_rounds", "ei_rounds")}
                  for r in seed_res],
    )
    summary_path = os.path.join(results_dir,
                                f"ngboost_ensemble_{ts}_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(_jsonable(summary), fh, indent=1)
    print(f"\n  Saved: {os.path.basename(summary_path)}")


if __name__ == "__main__":
    main()
