"""
Multi-fidelity bioprocess objective with a categorical clone choice.

Built on the `digital_biolab` CHO simulator, with 30 clones mapped into its CellType fields and
the three reactor scales of the benchmark.

Objective vector x = [temperature, pH, init_glucose, init_glutamine, feed_amount, fidelity, clone]
  - fidelity in {0:MTP, 1:MBR, 2:Pilot}
  - clone in 0..29
Returns PEAK ACTIVE product over the run ("harvest at the best moment"; higher = better).

Also exposes CLONE DESCRIPTORS (clone represented BY PARAMETERS for the tree surrogate).
The MTP controllability mask is applied by the BO loop (candidate generation), not here — this
function evaluates any x at any fidelity.
"""
from __future__ import annotations

import os
import sys
import numpy as np

DBIOLAB_ROOT = os.environ.get("DBIOLAB_ROOT", os.path.expanduser("~/Desktop/Research Project/digital_biolab"))
MFBO_ROOT = os.environ.get("MFBO_ROOT", os.path.expanduser("~/Desktop/Research Project/mf-bo4bio"))
for p in (os.path.join(DBIOLAB_ROOT, "src"), os.path.join(MFBO_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dbiolab.mammalian_cellculture_simulation import (  # noqa: E402
    CellType, FeedConcentrations, MammalianCellcultureReactor,
    MammalianRecipe, ReactorType, FeedingEvent, NoiseParameters,
)

# Load the paper's clone data file DIRECTLY (its package __init__ imports torch, which we don't need)
import importlib.util as _ilu  # noqa: E402
_cd_path = os.path.join(MFBO_ROOT, "src", "mfbo4bio", "conditions_data.py")
_spec = _ilu.spec_from_file_location("paper_conditions_data", _cd_path)
paper_data = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(paper_data)


# ── Map one paper clone dict -> Sam CellType ──────────────────────────────────
def _paper_to_sam_celltype(name: str, p: dict) -> CellType:
    return CellType(
        name=name,
        mu_max=p["mu_max"], k_lysis=p["K_lysis"],
        k_deg_gln=p["k"][0], k_death_max=p["k"][1], k_death_mu=p["k"][2],
        k_sat_lac=p["K"][0], k_sat_ammonia=p["K"][1], k_sat_glc=p["K"][2], k_sat_gln=p["K"][3],
        y_cells_glc=p["Y"][0], y_cells_gln=p["Y"][1], y_lactate=p["Y"][2], y_ammonia=p["Y"][3],
        y_product_from_total=p["Y"][4], y_product_from_viable=p["Y"][5],
        m_glc=p["m"][0], m_gln=p["m"][1], ph_opt=p["pH_opt"], e_activation=p["E_a"],
        # Sam's extras (no paper equivalent) -> defaults; keep aggregation ON (=> peak objective)
        ph_robustness_factor=1.0, temperature_robustness_factor=1.0,
        hcp_per_cell=1e-6, k_agg_product=0.1,
    )


def _build_clones(distribution: str = "alpha"):
    src = (paper_data.process_parameters_alpha if distribution == "alpha"
           else paper_data.process_parameters_beta)
    clones = [_paper_to_sam_celltype(f"clone_{i}", p) for i, p in enumerate(src.values())]
    return clones


# ── Fidelity reactors — Sam's semantics:
#   scale_factor    multiplies the DEATH rate   (higher → more death → LESS product)
#   growth_inhibition multiplies the GROWTH rate (higher → more growth → MORE product)
#
# Scale ordering, matching the benchmark: MBR >= Pilot > MTP.
# MBR is a LOW-BIAS, well-controlled scale-DOWN of Pilot (slightly higher/equal yield, tightly
# correlated). Calibrated on 500 random configurations of the simulator:
#   mean titer  MTP 0.88  <  Pilot 9.02  <  MBR 10.01   (MBR>=Pilot>MTP ✓)
#   corr(MBR,Pilot)=0.998 ✓ | corr(MTP,Pilot)=0.13 (Sam's real 5× MTP + MTP mask → genuinely crude)
# WHY THE CHANGE: the old MBR (sf=2.00/gi=0.97 → mean 6.02 < Pilot 9.02) made MBR strictly
# DOMINATED — after any good Pilot obs, MBR EI collapsed to ~0 and MBR was NEVER selected (Stages
# 2C–2H, all methods). With MBR mean ABOVE Pilot, MBR EI survives → the middle fidelity revives.
# MTP kept at Sam's REAL MTP_1.json scales (death-multiplier 5.0) as the anchor — NOT changed.
REACTORS = {
    0: ReactorType(name="MTP",   volume=0.003, scale_factor=5.00, growth_inhibition=0.95),
    1: ReactorType(name="MBR",   volume=0.25,  scale_factor=0.70, growth_inhibition=1.00),
    2: ReactorType(name="PILOT", volume=15.0,  scale_factor=1.00, growth_inhibition=1.00),
}
BATCH_SIZE = {0: 12, 1: 4, 2: 1}
COST = {0: 10, 1: 575, 2: 2100}
PILOT_FIDELITY = 2

# ── Per-fidelity measurement noise — benchmark ratio (MTP 0.15, MBR 1e-5, Pilot 1e-8)
# scaled to Sam's titer magnitude. Gaussian measurement noise on c_product only; Brownian PROCESS
# noise stays OFF. MTP σ is a meaningful fraction of the MTP titer spread (~0.30 std) → repeated
# MTP evals vary clearly (~10-20% CV); MBR/Pilot are effectively deterministic.
FIDELITY_NOISE = {0: 0.20, 1: 1e-4, 2: 1e-6}

# Bounds — 7 continuous vars (paper: 3 feeds at fixed times 70/140/210 h):
#   0:temperature 1:pH 2:init_glucose 3:init_glutamine 4:feed1 5:feed2 6:feed3
CONT_BOUNDS = np.array([[30.0, 40.0], [6.0, 8.0], [2.0, 10.0], [1.0, 6.0],
                        [0.0, 50.0], [0.0, 50.0], [0.0, 50.0]])
FEED_TIMES = [70.0, 140.0, 210.0]   # paper feed schedule
RUNTIME = 320.0                     # paper experiment time

# MTP controllability mask, following the benchmark:
#   T and pH at MTP are BATCH-WIDE but CHOOSABLE — ONE (T,pH) value shared across all wells of a
#   plate, selected by the optimizer (grid search over T,pH), NOT a hard 37/7 constant. This mirrors
#   the paper engine's `custom_optimization` (mf-bo4bio/optimization.py: 5×5 T,pH grid → plate of 12
#   wells sharing that (T,pH)). A fixed 37/7 mask instead crushes
#   corr(MTP,Pilot) to ~0.04 (MTP a near-constant floor); choosable/batch-wide T,pH restores it to
#   ~0.73. feed1/feed3 stay off; feed2 + clone free per well.
MTP_BATCHWIDE_CONT = [0, 1]                       # T, pH — shared per plate, CHOSEN (not fixed)
MTP_FREE_CONT      = [2, 3, 5]                     # init_glc, init_gln, feed2 — free per well
MTP_FIXED_FEEDS    = {4: 0.0, 6: 0.0}             # feed1 & feed3 off at MTP (paper-style)
# LEGACY masked mask (T,pH hard-fixed to platform) — kept for backward-compat with the pre-Stage-0
# must NOT use this — use gen_mtp_plate_candidates (choosable batch-wide T,pH) instead.
MTP_FIXED          = {0: 37.0, 1: 7.0, 4: 0.0, 6: 0.0}


def gen_mtp_plate_candidates(rng, n_plates, wells_per_plate, tp_grid_res=None,
                             fixed_init=None):
    """Paper-style MTP candidate PLATES for tree-BO loops.

    Each plate shares ONE (T,pH) — chosen from either a `tp_grid_res`×`tp_grid_res` grid over the
    T/pH bounds (grid search, like the paper's resolution=5) or, if `tp_grid_res is None`, a random
    uniform draw per plate. Within a plate, `wells_per_plate` wells vary feed2 (free) and clone
    (free); feed1=feed3=0; init_glc/init_gln set to `fixed_init` (default = CONT_BOUNDS midpoints,
    matching the tree loops' FIXED_INIT_GLC/GLN=6/3) or free if fixed_init="free".

    Returns (n_plates, wells_per_plate, 9) array of raw x rows
    [T,pH,iG,iQ,F1,F2,F3, fidelity=0, clone]. The caller scores each plate and picks the best.
    """
    n_clones = 30
    if fixed_init is None:
        iG0, iQ0 = 6.0, 3.0
    # T,pH plate values
    if tp_grid_res is not None:
        gT = np.linspace(CONT_BOUNDS[0, 0], CONT_BOUNDS[0, 1], tp_grid_res)
        gP = np.linspace(CONT_BOUNDS[1, 0], CONT_BOUNDS[1, 1], tp_grid_res)
        grid = np.array([(t, p) for t in gT for p in gP])
        idx  = rng.choice(len(grid), size=n_plates, replace=(n_plates > len(grid)))
        tp   = grid[idx]
    else:
        tp = np.column_stack([
            CONT_BOUNDS[0, 0] + (CONT_BOUNDS[0, 1] - CONT_BOUNDS[0, 0]) * rng.random(n_plates),
            CONT_BOUNDS[1, 0] + (CONT_BOUNDS[1, 1] - CONT_BOUNDS[1, 0]) * rng.random(n_plates)])
    plates = np.empty((n_plates, wells_per_plate, 9))
    for p in range(n_plates):
        T, pH = tp[p]
        feed2 = CONT_BOUNDS[5, 0] + (CONT_BOUNDS[5, 1] - CONT_BOUNDS[5, 0]) * rng.random(wells_per_plate)
        clone = rng.integers(0, n_clones, wells_per_plate)
        if fixed_init == "free":
            iG = CONT_BOUNDS[2, 0] + (CONT_BOUNDS[2, 1] - CONT_BOUNDS[2, 0]) * rng.random(wells_per_plate)
            iQ = CONT_BOUNDS[3, 0] + (CONT_BOUNDS[3, 1] - CONT_BOUNDS[3, 0]) * rng.random(wells_per_plate)
        else:
            iG = np.full(wells_per_plate, iG0); iQ = np.full(wells_per_plate, iQ0)
        plates[p, :, 0] = T; plates[p, :, 1] = pH
        plates[p, :, 2] = iG; plates[p, :, 3] = iQ
        plates[p, :, 4] = 0.0; plates[p, :, 5] = feed2; plates[p, :, 6] = 0.0
        plates[p, :, 7] = 0; plates[p, :, 8] = clone
    return plates


# ── Clone descriptors (represent clone BY PARAMETERS for the tree) ────────────
DESCRIPTOR_NAMES = ["mu_max", "ph_opt", "e_activation", "k_death_max",
                    "y_product_from_viable", "y_cells_glc"]

def clone_descriptors(clones):
    """(n_clones, n_descriptors) array — the tree's view of each clone."""
    return np.array([[getattr(c, n) for n in DESCRIPTOR_NAMES] for c in clones], float)


# ── The objective ─────────────────────────────────────────────────────────────
def make_objective(distribution: str = "alpha", noise: bool = True):
    clones = _build_clones(distribution)
    n_clones = len(clones)
    # Per-fidelity measurement noise. noise=False forces all sigma to 0 (deterministic).
    if noise:
        _noise_by_fid = {fid: NoiseParameters(measurement_noise_c_product=sigma)
                         for fid, sigma in FIDELITY_NOISE.items()}
    else:
        _noise_by_fid = {fid: NoiseParameters.from_dict({}) for fid in REACTORS}

    def f(x):
        temperature, ph, init_glc, init_gln, feed1, feed2, feed3 = (float(x[i]) for i in range(7))
        fidelity = int(round(float(x[7]))); clone_i = int(round(float(x[8])))
        reactor = REACTORS[fidelity]
        celltype = clones[clone_i]
        init = FeedConcentrations(c_product=0, x_total=8e5, x_viable=8e5, x_dead=0,
                                  c_glc=init_glc, c_gln=init_gln, c_lac=0, c_ammonia=0, c_aggregates=0)
        amounts = [feed1, feed2, feed3]
        feeding = [FeedingEvent(time_point=t, feed_concentrations=FeedConcentrations(
                       c_product=0, x_total=0, x_viable=0, x_dead=0, c_glc=a, c_gln=0,
                       c_lac=0, c_ammonia=0, c_aggregates=0))
                   for t, a in zip(FEED_TIMES, amounts)]
        rec = MammalianRecipe(name="ch3", celltype=celltype, temperature=temperature, ph_value=ph,
                              initial_conditions=init, feeding_strategy=feeding, runtime=RUNTIME,
                              measurement_intervals=2.0, noise_parameters=_noise_by_fid[fidelity])
        sol = MammalianCellcultureReactor(reactor_type=reactor).run_experiment(rec)
        return float(np.max(sol.c_product))   # PEAK active product (higher = better)

    meta = dict(n_clones=n_clones, cont_bounds=CONT_BOUNDS, reactors=REACTORS,
                batch_size=BATCH_SIZE, cost=COST, pilot_fidelity=PILOT_FIDELITY,
                mtp_free_cont=MTP_FREE_CONT, mtp_fixed=MTP_FIXED,
                clone_descriptors=clone_descriptors(clones), descriptor_names=DESCRIPTOR_NAMES)
    return f, meta


if __name__ == "__main__":
    f, meta = make_objective()
    print(f"clones: {meta['n_clones']}  | descriptors: {meta['descriptor_names']}")
    print(f"batch sizes {meta['batch_size']}  costs {meta['cost']}")
    nc = CONT_BOUNDS.shape[0]  # 7 continuous vars now
    rng = np.random.default_rng(0)
    print(f"\ncontinuous dims={nc} (T,pH,iG,iQ,feed1,feed2,feed3). random evals -> peak product:")
    for _ in range(6):
        cont = CONT_BOUNDS[:, 0] + (CONT_BOUNDS[:, 1] - CONT_BOUNDS[:, 0]) * rng.random(nc)
        fid = rng.integers(0, 3); clone = rng.integers(0, meta["n_clones"])
        x = list(cont) + [fid, clone]
        print(f"  fid={fid} clone={clone:2d}  T={cont[0]:.1f} pH={cont[1]:.2f} feeds=({cont[4]:.0f},{cont[5]:.0f},{cont[6]:.0f}) -> {f(x):7.2f}")
    print("\nsame conditions across the 3 fidelities (scale bias check):")
    base = [37.5, 7.2, 6.0, 3.0, 10.0, 10.0, 10.0]
    for fid in (0, 1, 2):
        print(f"  {meta['reactors'][fid].name:5s} -> peak product {f(base + [fid, 0]):7.2f}")
