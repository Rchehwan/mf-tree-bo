"""
Leakage-free impact (target) encoding of the 30 clones for the NGBoost tree surrogate.

Replaces the 30 one-hot clone columns with a few NUMERIC features per clone derived from OBSERVED
PILOT titer, recomputed each round from ONLY already-evaluated data (no leakage). Keeps the tree
interpretable ("clones with high observed pilot titer") while sharing strength across clones via a
Bayesian-smoothed estimate for clones with few/zero observations.

Features per clone (all from pilot-fidelity observations up to the current round):
  1. impact_mean  : Bayesian-smoothed mean pilot titer,
                    (n_c * ybar_c + kappa * global_mean) / (n_c + kappa)
                    -> shrinks low-count clones toward the global mean (no wild estimates).
  2. impact_count : log1p(n_c) — how many pilot observations back the estimate (confidence).
  3. impact_best  : best observed pilot titer for the clone (shrunk to global best for n_c=0).

NO LEAKAGE: encodings are a pure function of PAST (X_train, y_train); candidates are encoded with the
SAME fitted map, never using their own (unobserved) outcome. `fit(X, y)` uses all rows but only the
pilot-fidelity ones inform the target statistics — and a point's own outcome is never used to encode
that same point at prediction time because encoding is keyed by CLONE ID, and the candidate's clone
statistic is built only from other observed points of that clone.
"""
from __future__ import annotations
import numpy as np

_FID_COL, _CLONE_COL = 5, 6


class CloneImpactEncoder:
    def __init__(self, n_clones: int = 30, pilot_fidelity: int = 2, kappa: float = 3.0):
        self.n_clones = n_clones
        self.pilot_fidelity = pilot_fidelity
        self.kappa = kappa           # smoothing strength (pseudo-count toward global mean)
        self._mean = None            # (n_clones,)
        self._count = None
        self._best = None
        self._global_mean = 0.0
        self._global_best = 0.0

    def fit(self, X_raw: np.ndarray, y: np.ndarray) -> "CloneImpactEncoder":
        X_raw = np.asarray(X_raw, float); y = np.asarray(y, float)
        fid = X_raw[:, _FID_COL].astype(int)
        clone = X_raw[:, _CLONE_COL].astype(int).clip(0, self.n_clones - 1)
        pm = fid == self.pilot_fidelity
        if pm.any():
            g_mean = float(y[pm].mean()); g_best = float(y[pm].max())
        else:
            # fall back to all-fidelity stats before any pilot data exists
            g_mean = float(y.mean()) if len(y) else 0.0
            g_best = float(y.max()) if len(y) else 0.0
        self._global_mean, self._global_best = g_mean, g_best

        mean = np.full(self.n_clones, g_mean)
        count = np.zeros(self.n_clones)
        best = np.full(self.n_clones, g_best)
        for c in range(self.n_clones):
            m = pm & (clone == c)
            n_c = int(m.sum())
            count[c] = n_c
            if n_c > 0:
                ybar = float(y[m].mean())
                mean[c] = (n_c * ybar + self.kappa * g_mean) / (n_c + self.kappa)
                best[c] = float(y[m].max())
        self._mean, self._count, self._best = mean, count, best
        return self

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Return (n, 5+3): [T,pH,F1,F2,F3] + [impact_mean, log1p(count), impact_best] for the clone.

        Fidelity column is dropped from features (the surrogate is fitted per fidelity).
        """
        X_raw = np.asarray(X_raw, float)
        cont = X_raw[:, :_FID_COL]
        clone = X_raw[:, _CLONE_COL].astype(int).clip(0, self.n_clones - 1)
        enc = np.column_stack([self._mean[clone], np.log1p(self._count[clone]), self._best[clone]])
        return np.hstack([cont, enc])

    def encoding_stats(self) -> dict:
        return dict(global_mean=self._global_mean, global_best=self._global_best,
                    n_clones_observed=int((self._count > 0).sum()),
                    impact_mean_spread=float(np.std(self._mean)))


# ── Fix P2: all-fidelity clone encoding ──────────────────────────────────────────────────────────


class CloneImpactEncoderAllFid:
    """
    Leakage-free clone impact encoding built from all fidelities.

    Identical interface to CloneImpactEncoder.  The key difference: observations from MTP,
    MBR, AND Pilot all feed the per-clone statistics, after being corrected to pilot scale via

        scale_f  =  mean(y_pilot) / mean(y_f)

    so "cheap" MTP titer values are translated to their pilot-equivalent before computing the
    Bayesian-shrunk clone mean.  With 200+ MTP evaluations across 30 clones in a typical BO
    run, every clone gets data from round 1 onward — fixing the "3-6 of 30 clones observed"
    bug that plagued the pilot-only encoder.

    Features: [T, pH, F1, F2, F3]  +  [impact_mean, log1p(count), impact_best]
    The clone features remain human-readable ("clone's fidelity-corrected historical yield").
    IF-THEN rules like "high-yield clones + T<35 → high titer" are still extractable.

    NO LEAKAGE: fit() uses only past (X_train, y_train); candidates are encoded with the
    fitted scale and per-clone statistics, never using their own (unobserved) outcome.
    """

    def __init__(self, n_clones: int = 30, pilot_fidelity: int = 2,
                 kappa: float = 3.0, n_fidelities: int = 3):
        self.n_clones = n_clones
        self.pilot_fidelity = pilot_fidelity
        self.kappa = kappa
        self.n_fidelities = n_fidelities
        self._mean = None
        self._count = None
        self._best = None
        self._global_mean = 0.0
        self._global_best = 0.0
        # per-fidelity scale factors: y_f * scale_f ≈ y_pilot
        self._fid_scale: dict[int, float] = {}

    def fit(self, X_raw: np.ndarray, y: np.ndarray) -> "CloneImpactEncoderAllFid":
        X_raw = np.asarray(X_raw, float); y = np.asarray(y, float)
        fid   = X_raw[:, _FID_COL].astype(int)
        clone = X_raw[:, _CLONE_COL].astype(int).clip(0, self.n_clones - 1)

        # ── (a) Global pilot statistics ───────────────────────────────────────
        pm = fid == self.pilot_fidelity
        if pm.any():
            g_mean = float(y[pm].mean())
            g_best = float(y[pm].max())
        else:
            # before any pilot eval: fall back to all-fidelity stats
            g_mean = float(y.mean()) if len(y) else 0.0
            g_best = float(y.max())  if len(y) else 0.0
        self._global_mean, self._global_best = g_mean, g_best

        # ── (b) Cross-fidelity scale: scale_f * mean(y_f) ≈ mean(y_pilot) ───
        self._fid_scale = {}
        for f in range(self.n_fidelities):
            if f == self.pilot_fidelity:
                self._fid_scale[f] = 1.0
            else:
                fm = fid == f
                if fm.any():
                    f_mean = float(y[fm].mean())
                    self._fid_scale[f] = (g_mean / f_mean) if abs(f_mean) > 1e-8 else 1.0
                else:
                    self._fid_scale[f] = 1.0   # no data at fidelity f yet

        # ── (c) Fidelity-corrected yields (all observations) ─────────────────
        scale_vec = np.array([self._fid_scale.get(int(fid[i]), 1.0) for i in range(len(y))])
        y_corr = y * scale_vec

        # ── (d) Per-clone Bayesian-smoothed statistics from ALL fidelities ────
        mean  = np.full(self.n_clones, g_mean)
        count = np.zeros(self.n_clones)
        best  = np.full(self.n_clones, g_best)

        for c in range(self.n_clones):
            m = clone == c
            n_c = int(m.sum())
            count[c] = n_c
            if n_c > 0:
                ybar   = float(y_corr[m].mean())
                mean[c] = (n_c * ybar + self.kappa * g_mean) / (n_c + self.kappa)
                best[c] = float(y_corr[m].max())

        self._mean, self._count, self._best = mean, count, best
        return self

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Return (n, 8): [T,pH,F1,F2,F3] + [impact_mean, log1p(count), impact_best]."""
        X_raw = np.asarray(X_raw, float)
        cont  = X_raw[:, :_FID_COL]
        clone = X_raw[:, _CLONE_COL].astype(int).clip(0, self.n_clones - 1)
        enc   = np.column_stack([
            self._mean[clone],
            np.log1p(self._count[clone]),
            self._best[clone],
        ])
        return np.hstack([cont, enc])

    def encoding_stats(self) -> dict:
        return dict(
            global_mean=self._global_mean,
            global_best=self._global_best,
            n_clones_observed=int((self._count > 0).sum()),
            impact_mean_spread=float(np.std(self._mean)),
            fidelity_scales={str(k): round(v, 4) for k, v in self._fid_scale.items()},
        )
