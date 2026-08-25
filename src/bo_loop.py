"""
Shared multi-fidelity BO loop with the MF-MES acquisition (numpy only, no torch).

Surrogate-agnostic: any surrogate exposing
    fit(X7, y) ; predict_pilot(X7) -> (mu, sigma) ; get_rho(fid) -> rho
can be plugged in.

The fidelity is chosen by the acquisition alone; no part of the budget is reserved for the
pilot scale. `guard_needed` records whether a seed ever selected the pilot unaided. If one did
not, a single final pilot evaluation of the best-predicted point is taken so the run still
reports a pilot titer.

X7 convention = [T, pH, F1, F2, F3, fidelity, clone]; init_glc/init_gln fixed (6/3) in the
objective.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE,):
    if p not in sys.path:
        sys.path.insert(0, p)

from acquisition import sample_pilot_maxvalues, select_fidelity_mfmes, select_fidelity_ei   # noqa: E402

# ── problem constants ─────────────────────────────────────────────────────────
FIXED_INIT_GLC, FIXED_INIT_GLN = 6.0, 3.0
N_CLONES = 30
CONT_BOUNDS_5 = np.array([[30.0, 40.0], [6.0, 8.0], [0.0, 50.0], [0.0, 50.0], [0.0, 50.0]])
BUDGET = 40_000
# Initial design. N_PILOT_INIT is 0 for the reported runs, so the method starts with no
# pilot data and must decide to buy it.
N_MTP_INIT, N_MBR_INIT, N_PILOT_INIT = 11, 1, 3
COST = {0: 10, 1: 575, 2: 2100}
BATCH_SIZE = {0: 12, 1: 4, 2: 1}
PILOT_FIDELITY = 2
FIDELITY_LEVELS = [0, 1, 2]
INIT_COST = N_MTP_INIT * COST[0] + N_MBR_INIT * COST[1] + N_PILOT_INIT * COST[2]

N_MTP_PLATES = 25           # paper-style T,pH grid (5x5) of MTP plates per round
N_POOL_CONT = 48            # free continuous samples for MBR/Pilot pool (x30 clones)
N_MAXVAL = 30               # pilot max-value samples for MES
N_MAXVAL_CONT = 5           # small ref pool (x30 clones=150) for max-value sampling (avoids inflation)


def _f7(f_obj, x7):
    T, pH, F1, F2, F3 = (float(x7[i]) for i in range(5))
    fid, clone = int(round(x7[5])), int(round(x7[6]))
    return float(f_obj([T, pH, FIXED_INIT_GLC, FIXED_INIT_GLN, F1, F2, F3, fid, clone]))


def _mtp_plates(rng, n_plates, wells=12, grid_res=5):
    gT = np.linspace(*CONT_BOUNDS_5[0], grid_res)
    gP = np.linspace(*CONT_BOUNDS_5[1], grid_res)
    grid = np.array([(t, p) for t in gT for p in gP])
    idx = rng.choice(len(grid), size=n_plates, replace=(n_plates > len(grid)))
    plates = []
    for (T, pH) in grid[idx]:
        F2 = CONT_BOUNDS_5[3, 0] + (CONT_BOUNDS_5[3, 1] - CONT_BOUNDS_5[3, 0]) * rng.random(wells)
        clone = rng.integers(0, N_CLONES, wells)
        plate = np.column_stack([np.full(wells, T), np.full(wells, pH), np.zeros(wells),
                                 F2, np.zeros(wells), np.zeros(wells), clone])
        plates.append(plate)
    return plates


def _free_pool(rng, fid, n_cont=N_POOL_CONT):
    cont = CONT_BOUNDS_5[:, 0] + rng.random((n_cont, 5)) * (CONT_BOUNDS_5[:, 1] - CONT_BOUNDS_5[:, 0])
    cont = np.repeat(cont, N_CLONES, axis=0)
    clone = np.tile(np.arange(N_CLONES), n_cont)
    return np.column_stack([cont, np.full(len(clone), fid), clone])


def _mixed_init(rng, n_pilot_init=None):
    n_plt = N_PILOT_INIT if n_pilot_init is None else int(n_pilot_init)
    # 11 MTP: paper-style — random (T,pH) per init well is fine for init; F1=F3=0, F2 free, clone free
    T = rng.uniform(*CONT_BOUNDS_5[0], N_MTP_INIT); pH = rng.uniform(*CONT_BOUNDS_5[1], N_MTP_INIT)
    F2 = rng.uniform(*CONT_BOUNDS_5[3], N_MTP_INIT)
    X_mtp = np.column_stack([T, pH, np.zeros(N_MTP_INIT), F2, np.zeros(N_MTP_INIT),
                             np.zeros(N_MTP_INIT, int), rng.integers(0, N_CLONES, N_MTP_INIT)])
    cont = CONT_BOUNDS_5[:, 0] + rng.random((N_MBR_INIT, 5)) * (CONT_BOUNDS_5[:, 1] - CONT_BOUNDS_5[:, 0])
    X_mbr = np.column_stack([cont, np.ones(N_MBR_INIT, int), rng.integers(0, N_CLONES, N_MBR_INIT)])
    parts = [X_mtp, X_mbr]
    if n_plt > 0:                                 # optional pilot warm-start (Exp A tests n_plt=0)
        contp = CONT_BOUNDS_5[:, 0] + rng.random((n_plt, 5)) * (CONT_BOUNDS_5[:, 1] - CONT_BOUNDS_5[:, 0])
        parts.append(np.column_stack([contp, np.full(n_plt, PILOT_FIDELITY, int),
                                      rng.integers(0, N_CLONES, n_plt)]))
    return np.vstack(parts)


def run_mfmes_bo(make_surrogate, f_obj, seed: int, verbose: bool = True,
                 n_pilot_init=None, fidelities=None, harvest=False) -> dict:
    """make_surrogate: callable(rng) -> surrogate with fit/predict_pilot/get_rho.
    n_pilot_init: override the 3-point pilot warm-start (Exp A uses 0).
    fidelities:  override the active fidelity set (Exp D uses [0, 2] to drop MBR).
    harvest:     add the cost-aware EI harvest term on Pilot (Exp C lock-in fix). Opt-in so the
                 committed baselines are unchanged."""
    fids = FIDELITY_LEVELS if fidelities is None else list(fidelities)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    surro = make_surrogate(rng)

    X = _mixed_init(rng, n_pilot_init=n_pilot_init)
    y = np.array([_f7(f_obj, x) for x in X])
    cumcost = int(sum(COST[int(r)] for r in X[:, 5]))     # per-point init cost (adapts to init mix)
    pm = X[:, 5] == PILOT_FIDELITY
    best_pilot = float(y[pm].max()) if pm.any() else float("-inf")

    fids_chosen, clone_stats = [], []
    pilot_trace = []                       # [(cumcost, best_pilot)] after each round once pilot is seen
    round_num = 0
    ei_rounds = 0                         # rounds that fell back to the EI tie-breaker (MI ~0)
    while cumcost < BUDGET:               # run until the COST BUDGET is spent — NO early termination
        round_num += 1
        surro.fit(X, y)
        if hasattr(surro, "clone_cov_stats"):
            clone_stats.append(surro.clone_cov_stats())
        elif hasattr(surro, "encoding_stats"):
            clone_stats.append(surro.encoding_stats())

        # pilot max-value samples. Prefer JOINT sampling (respects surrogate covariance -> realistic,
        # un-inflated max); fall back to the capped marginal CDF-product for surrogates without a joint.
        mv_pool = _free_pool(rng, PILOT_FIDELITY, n_cont=N_MAXVAL_CONT)
        if hasattr(surro, "sample_pilot_max"):
            maxsamples = surro.sample_pilot_max(mv_pool, N_MAXVAL, rng)
            if np.isfinite(best_pilot):
                maxsamples = np.maximum(maxsamples, best_pilot)
        else:
            mu_mv, sig_mv = surro.predict_pilot(mv_pool)
            yb = best_pilot if np.isfinite(best_pilot) else float(np.max(mu_mv))
            maxsamples = sample_pilot_maxvalues(mu_mv, sig_mv, N_MAXVAL, rng, y_best=yb)

        pools = {f: _free_pool(rng, f, n_cont=(N_POOL_CONT if f == 2 else 48))
                 for f in fids if f != 0}                 # MTP(0) uses plates; others use free pools
        plates = _mtp_plates(rng, N_MTP_PLATES) if 0 in fids else None
        best_fid, batch, scores, rhos = select_fidelity_mfmes(
            pools, surro.predict_pilot, surro.get_rho, COST, BATCH_SIZE,
            fids, maxsamples, mtp_plates=plates,
            incumbent=(best_pilot if harvest else None), pilot_fidelity=PILOT_FIDELITY)

        # EI tie-breaker (NO early-stop): if the MF-MES MI is ~0 for EVERY fidelity (surrogate briefly
        # sees no candidate beating the incumbent), fall back to cost-aware EI so the loop keeps
        # EXPLORING toward improvement points instead of stalling / quitting. We run until the cost
        # budget is spent — never terminate early on an "uninformative" round.
        used_ei = False
        if max(scores.values()) < 1e-10:
            inc = best_pilot if np.isfinite(best_pilot) else float("-inf")
            best_fid, batch, scores, rhos = select_fidelity_ei(
                pools, surro.predict_pilot, surro.get_rho, COST, BATCH_SIZE,
                fids, inc, mtp_plates=plates)
            used_ei = True

        y_batch = np.array([_f7(f_obj, x) for x in batch])
        X = np.vstack([X, batch]); y = np.append(y, y_batch)
        cumcost += BATCH_SIZE[best_fid] * COST[best_fid]
        fids_chosen.append(best_fid)
        pm = X[:, 5] == PILOT_FIDELITY
        best_pilot = float(y[pm].max()) if pm.any() else float("-inf")
        if verbose:
            sc = lambda f: scores.get(f, float("nan")); rh = lambda f: rhos.get(f, float("nan"))
            print(f"    r{round_num:2d} MTP={sc(0):+.3e}(ρ{rh(0):.2f}) "
                  f"MBR={sc(1):+.3e}(ρ{rh(1):.2f}) Plt={sc(2):+.3e}(ρ{rh(2):.2f}) "
                  f"→fid={best_fid}{'(EI)' if used_ei else ''}  best_pilot={best_pilot:.3f}  "
                  f"cost={cumcost} ({time.time()-t0:.0f}s)")
        if used_ei:
            ei_rounds += 1
        if np.isfinite(best_pilot):
            pilot_trace.append([cumcost, best_pilot])

    # NO GUARD. best_pilot is ONLY over actually-evaluated pilot points (warm-start init + any Pilot
    # rounds MES chose on its own). We do NOT force a final pilot harvest — that would be a guard and
    # defeat the point of a genuine multi-fidelity acquisition. `never_picked_pilot` is a pure
    # diagnostic: did MES select Pilot at all beyond the warm-start?
    pilot_rounds = fids_chosen.count(PILOT_FIDELITY)
    never_picked_pilot = pilot_rounds == 0

    counts = {f: fids_chosen.count(f) for f in FIDELITY_LEVELS}
    return dict(fids_chosen=fids_chosen, fid_counts=counts, best_pilot_final=best_pilot,
                pilot_rounds=pilot_rounds, guard_needed=never_picked_pilot, ei_rounds=ei_rounds,
                cumcost=cumcost, budget_frac=cumcost / BUDGET,
                clone_stats=clone_stats, n_rounds=round_num, elapsed=time.time() - t0,
                pilot_trace=pilot_trace)


