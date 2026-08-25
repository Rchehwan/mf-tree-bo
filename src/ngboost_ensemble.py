"""
NGBoost co-kriging tree surrogate with leakage-free impact-encoded clones.
Also contains NGBoostEnsembleCokrig — an ensemble of K bootstrap-diverse members whose
disagreement provides the functional posterior MF-MES requires.

Per-fidelity NGBoost (calibrated Normal mean+std, no kernel). Clones are represented by leakage-free
IMPACT ENCODING (CloneImpactEncoder) recomputed each round from observed pilot titer — NOT 30 one-hot
columns — so similar clones share strength and the model stays interpretable. Exposes the same
interface the shared MF-MES loop needs: fit / predict_pilot / get_rho.

rho(f, pilot) = Pearson correlation of per-fidelity NGBoost predictions on a shared candidate set
(genuine < 1 because the per-fidelity models are trained on disjoint data).

Two features matter for the acquisition:
  - Epistemic uncertainty: a distance-to-nearest-training-point term so sigma grows away from data.
  - All-fidelity clone encoding: CloneImpactEncoderAllFid uses MTP, MBR and Pilot observations.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import KDTree

from impact_encoding import CloneImpactEncoderAllFid

try:
    from acquisition import sample_pilot_maxvalues as _sample_maxvalues
    _MF_MES_OK = True
except ImportError:
    _MF_MES_OK = False

try:
    from sklearn.tree import DecisionTreeRegressor as _DTR
except ImportError:
    _DTR = None

try:
    from ngboost import NGBRegressor as _NGB
    from ngboost.distns import Normal as _Normal
    _OK = True
except ImportError:
    _OK = False

_FID_COL, _CLONE_COL = 5, 6
N_CLONES = 30
_CONT_BOUNDS_5 = np.array([[30.0, 40.0], [6.0, 8.0], [0.0, 50.0], [0.0, 50.0], [0.0, 50.0]])


# Fixed per-feature normalization ranges for 1NN distance in the 8-dim feature space:
# [T, pH, F1, F2, F3, impact_mean, log1p(count), impact_best]
_FEAT_NORMS_8 = np.array([10.0, 2.0, 50.0, 50.0, 50.0, 60.0, 4.0, 60.0])


# ── Ensemble surrogate (Option A: fix EI-fallback / functional posterior) ─────────────────────────

# K=10 member depths: structural diversity through varied tree depth (2–7).
# Bootstrap alone isn't sufficient diversity for small n (members learn similar functions);
# depth variation forces genuinely different function approximations so members disagree
# where data is sparse — exactly the property BARK's sampled forests have.
_MEMBER_DEPTHS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 7]


class NGBoostEnsembleCokrig:
    """
    Ensemble of K NGBoost surrogates with structural diversity for a genuine functional posterior.

    WHY: a single NGBoost gives only a per-point Normal posterior; its max-value samples from
    the CDF-product collapse to one value when the surrogate is confident -> MF-MES MI
    underflows to 0 -> EI fallback fires every round.  An ensemble whose members genuinely
    DISAGREE about the function shape (different tree depths + bootstrap) gives a real
    distribution over functions (the tree analogue of BARK sampling many forests): draw
    member k -> sample that member's CDF-product max -> members with different depths
    predict different maxima locations -> genuine spread -> MF-MES stays informative ->
    EI fallback no longer needed.

    DIVERSITY MECHANISM — two sources:
      1. Bootstrap resampling (different training subsets per member)
      2. Varied tree depths [2,3,4,5,6,7] — depth-2 members learn a linear/piecewise-linear
         function, depth-7 members capture fine-grained interactions; at unseen pool points
         they predict genuinely different values even on the same data.

    EPISTEMIC UNCERTAINTY — holdout calibration:
      Hold out 20% of training data; fit K cal-members on 80% (bootstrapped + varied depth);
      calibrate alpha on the ENSEMBLE MEAN predictions vs holdout y so that std(z_far)=1.
      This gives a non-trivially positive alpha (unlike OOB calibration which drives alpha→0
      because the ensemble mean is already well-calibrated).

    SIGMA FOR SAMPLING — in sample_pilot_max, per-member sigma is inflated by the ensemble
      DISAGREEMENT (std across members): large where members genuinely differ (data-sparse)
      and small where they agree (well-observed). This is belt-and-suspenders: even if
      alpha is moderate, the disagreement sigma ensures genuine spread in maxima samples.

    Provides:
      epistemic proximity term (alpha * dist_1NN, holdout-calibrated)
      all-fidelity clone encoding (CloneImpactEncoderAllFid)

    Interpretability: each member is gradient-boosted trees; feature importances and IF-THEN
    rules are readable from any individual member.

    Interface: fit / predict_pilot / sample_pilot_max / get_rho
    """

    MIN_FIT_N = 5
    N_CORR    = 15
    CAL_FRAC  = 0.20

    def __init__(self, pilot_fidelity: int = 2, n_estimators: int = 50,
                 n_fidelities: int = 3, corr_rng: np.random.Generator | None = None,
                 rho_floor: float = 0.05, sigma_floor_frac: float = 0.05,
                 K: int = 10):
        if not _OK:
            raise ImportError("pip install ngboost")
        self.pilot_fidelity   = pilot_fidelity
        self.n_estimators     = n_estimators
        self.n_fidelities     = n_fidelities
        self.rho_floor        = rho_floor
        self.sigma_floor_frac = sigma_floor_frac
        self.K                = K
        self._corr_rng        = corr_rng or np.random.default_rng(0)
        self._enc             = CloneImpactEncoderAllFid(n_clones=N_CLONES,
                                                         pilot_fidelity=pilot_fidelity)
        self._member_models: list[dict] = [{} for _ in range(K)]
        self._y_std:    float           = 5.0
        self._kdtree:   KDTree | None   = None
        self._alpha_epist: float        = 1.0
        self._base_seed: int            = 1000

    # ── helpers ───────────────────────────────────────────────────────────────
    def _norm(self, feat: np.ndarray) -> np.ndarray:
        return feat / _FEAT_NORMS_8[None, :]

    def _make_ngb(self, k: int) -> "_NGB":
        """NGBoost with depth-varied base learner for member k.

        min_samples_leaf=2 prevents the base tree from perfectly memorising a small
        bootstrap sample (which drives NGBoost's scale→0 → log(0)=-inf → NaN gradients).
        """
        depth = _MEMBER_DEPTHS[k % len(_MEMBER_DEPTHS)]
        if _DTR is not None:
            base = _DTR(criterion="squared_error", max_depth=depth, min_samples_leaf=2)
            return _NGB(Dist=_Normal, n_estimators=self.n_estimators,
                        learning_rate=0.05, verbose=False, Base=base)
        return _NGB(Dist=_Normal, n_estimators=self.n_estimators,
                    learning_rate=0.05, verbose=False)

    def _make_ngb_safe(self) -> "_NGB":
        """Shallow fallback (depth=2) used when primary member raises NaN errors."""
        if _DTR is not None:
            base = _DTR(criterion="squared_error", max_depth=2, min_samples_leaf=2)
            return _NGB(Dist=_Normal, n_estimators=self.n_estimators,
                        learning_rate=0.05, verbose=False, Base=base)
        return _NGB(Dist=_Normal, n_estimators=self.n_estimators,
                    learning_rate=0.05, verbose=False)

    def _fit_k_members(self, X_raw: np.ndarray, y: np.ndarray,
                       enc: "CloneImpactEncoderAllFid",
                       target: list) -> None:
        """Fit K bootstrap+depth-varied members into `target` (list of K dicts)."""
        n   = len(y)
        fid = X_raw[:, _FID_COL].astype(int)
        for k in range(self.K):
            rng_k    = np.random.default_rng(self._base_seed + k)
            boot_idx = rng_k.integers(0, n, n)
            X_b  = X_raw[boot_idx]; y_b  = y[boot_idx]; fid_b = fid[boot_idx]
            target[k] = {}
            for f in range(self.n_fidelities):
                m = fid_b == f
                if m.sum() < self.MIN_FIT_N:
                    continue
                feat_b = enc.transform(X_b[m])
                # Primary attempt: depth-varied member; fallback: safe shallow model
                for mdl in [self._make_ngb(k), self._make_ngb_safe()]:
                    try:
                        mdl.fit(feat_b, y_b[m])
                        target[k][f] = mdl
                        break
                    except Exception:
                        continue   # try the safe fallback

    # ── main fit ──────────────────────────────────────────────────────────────
    def fit(self, X_raw: np.ndarray, y: np.ndarray) -> "NGBoostEnsembleCokrig":
        X_raw = np.asarray(X_raw, float); y = np.asarray(y, float)
        n     = len(y)
        self._y_std = float(np.std(y)) or 5.0
        fid   = X_raw[:, _FID_COL].astype(int)

        # ── Step 1: fit shared all-fidelity encoder on full data ──────────────
        self._enc.fit(X_raw, y)
        feat_full      = self._enc.transform(X_raw)
        feat_full_norm = self._norm(feat_full)
        self._kdtree   = KDTree(feat_full_norm)

        # ── Step 2: holdout calibration of alpha (20% holdout / 80% train) ───
        # Using HOLDOUT (not OOB) because the ensemble mean on OOB is already so
        # accurate that OOB calibration drives alpha→0 (no epistemic needed) which
        # collapses max-value samples. The 80%-trained ensemble has higher residuals
        # on the 20% holdout → alpha calibrates to a non-trivially positive value.
        if n >= 10:
            cal_sz  = max(3, int(n * self.CAL_FRAC))
            rng_cal = np.random.default_rng(42)       # fixed — not the BO rng
            perm    = rng_cal.permutation(n)
            cal_idx = perm[:cal_sz];  tr_idx = perm[cal_sz:]
            X_tr, y_tr   = X_raw[tr_idx],  y[tr_idx]
            X_cal, y_cal = X_raw[cal_idx], y[cal_idx]

            enc_cal = CloneImpactEncoderAllFid(n_clones=N_CLONES,
                                               pilot_fidelity=self.pilot_fidelity)
            enc_cal.fit(X_tr, y_tr)
            feat_tr      = enc_cal.transform(X_tr)
            feat_tr_norm = self._norm(feat_tr)

            # K cal-members (bootstrapped from X_tr, varied depth)
            cal_members: list[dict] = [{} for _ in range(self.K)]
            self._fit_k_members(X_tr, y_tr, enc_cal, cal_members)

            # Ensemble mean + mean aleatoric on holdout
            feat_cal      = enc_cal.transform(X_cal)
            feat_cal_norm = self._norm(feat_cal)
            fid_cal       = X_cal[:, _FID_COL].astype(int)
            mus_cal        = np.zeros((self.K, len(X_cal)))
            sig_as_cal     = np.full((self.K, len(X_cal)), max(self._y_std, 1e-6))

            for k in range(self.K):
                for f in range(self.n_fidelities):
                    fm = fid_cal == f
                    if not fm.any():
                        continue
                    mdl = (cal_members[k].get(f) or
                           cal_members[k].get(self.pilot_fidelity))
                    if mdl is None:
                        continue
                    d = mdl.pred_dist(feat_cal[fm])
                    mus_cal[k, fm]    = np.asarray(d.loc,   float)
                    sig_as_cal[k, fm] = np.maximum(np.asarray(d.scale, float), 1e-8)

            mu_ens_cal  = np.mean(mus_cal,   axis=0)
            sig_a_ens_cal = np.mean(sig_as_cal, axis=0)

            tree_cal     = KDTree(feat_tr_norm)
            dist_cal, _  = tree_cal.query(feat_cal_norm, k=1)
            dist_cal     = np.asarray(dist_cal, float)

            med = float(np.median(dist_cal))
            far = dist_cal >= med
            if far.sum() < 3:
                far = np.ones(len(y_cal), bool)

            res = y_cal[far] - mu_ens_cal[far]    # SIGNED residuals; target std(z)=1
            s_a = sig_a_ens_cal[far]
            d_f = dist_cal[far]

            def z_std_err(log_a: float) -> float:
                a       = np.exp(log_a)
                sig_tot = np.sqrt(s_a**2 + (a * d_f)**2)
                z       = res / np.maximum(sig_tot, 1e-8)
                return abs(float(np.std(z)) - 1.0)

            log_alphas = np.linspace(np.log(0.01), np.log(1000.0), 80)
            errs       = [z_std_err(la) for la in log_alphas]
            self._alpha_epist = float(np.exp(log_alphas[int(np.argmin(errs))]))

        # ── Step 3: fit K final members on full data ──────────────────────────
        self._member_models = [{} for _ in range(self.K)]
        self._fit_k_members(X_raw, y, self._enc, self._member_models)

        return self

    # ── mixture pilot prediction (used by MF-MES scoring) ────────────────────
    def predict_pilot(self, X_raw_cands: np.ndarray):
        """Mixture predictive: mu = mean(mu_k),
        sigma^2 = mean(sig_a_k^2) + var(mu_k) + sig_e^2  (law of total variance)."""
        X_raw_cands = np.asarray(X_raw_cands, float)
        feat      = self._enc.transform(X_raw_cands)
        feat_norm = self._norm(feat)
        floor     = max(self.sigma_floor_frac * self._y_std, 1e-6)
        pid       = self.pilot_fidelity

        mus    = np.empty((self.K, len(feat)))
        sig_as = np.empty((self.K, len(feat)))
        for k in range(self.K):
            mm = self._member_models[k]
            if pid in mm:
                d          = mm[pid].pred_dist(feat)
                mus[k]    = np.asarray(d.loc,   float)
                sig_as[k] = np.maximum(np.asarray(d.scale, float), 1e-8)
            else:
                mus[k]    = feat[:, 5]
                sig_as[k] = np.full(len(feat), max(self._y_std, 1e-6))

        mu_mix = np.mean(mus, axis=0)

        if self._kdtree is not None and self._alpha_epist > 0:
            dists, _ = self._kdtree.query(feat_norm, k=1)
            sig_e    = self._alpha_epist * np.asarray(dists, float)
        else:
            sig_e = np.zeros(len(feat))

        sig_mix = np.sqrt(np.mean(sig_as**2, axis=0) + np.var(mus, axis=0) + sig_e**2)
        return mu_mix, np.maximum(sig_mix, floor)

    # ── THE FIX: member-sampling with ensemble-disagreement-inflated sigma ────
    def sample_pilot_max(self, X_raw_cands: np.ndarray, n_samples: int,
                         rng: np.random.Generator) -> np.ndarray:
        """Sample n_samples Pilot maxima for MF-MES.

        TWO SOURCES OF SPREAD:
          (a) Different member k's predict different mu at pool points (depth diversity)
              -> different members identify different "best" locations -> spread in lo.
          (b) Per-member sigma is inflated by ensemble_std (disagreement across members)
              -> CDF product is wide where members differ -> spread within each member's draw.

        Together these ensure std(maxsamples) is large even if individual member's
        aleatoric sigma is small, so MF-MES sees a genuine distribution over y*.
        """
        if not _MF_MES_OK:
            raise ImportError("sample_pilot_max requires acquisition on sys.path")
        X_raw_cands = np.asarray(X_raw_cands, float)
        feat      = self._enc.transform(X_raw_cands)
        feat_norm = self._norm(feat)
        floor     = max(self.sigma_floor_frac * self._y_std, 1e-6)
        pid       = self.pilot_fidelity

        # Pre-compute ALL member means and aleatoric sigmas at the pool (vectorised)
        all_mus  = np.empty((self.K, len(feat)))
        all_sigs = np.empty((self.K, len(feat)))
        for k in range(self.K):
            mm = self._member_models[k]
            if pid in mm:
                d          = mm[pid].pred_dist(feat)
                all_mus[k]  = np.asarray(d.loc,   float)
                all_sigs[k] = np.maximum(np.asarray(d.scale, float), 1e-8)
            else:
                all_mus[k]  = feat[:, 5]
                all_sigs[k] = np.full(len(feat), max(self._y_std, 1e-6))

        # Ensemble disagreement: std across members at each pool point
        ens_std = np.std(all_mus, axis=0)                  # (n_pool,)

        # Shared proximity epistemic
        if self._kdtree is not None and self._alpha_epist > 0:
            dists, _ = self._kdtree.query(feat_norm, k=1)
            sig_e    = self._alpha_epist * np.asarray(dists, float)
        else:
            sig_e = np.zeros(len(feat))

        maxsamples = np.empty(n_samples, float)
        for i in range(n_samples):
            member_k = int(rng.integers(0, self.K))
            mu_k     = all_mus[member_k]
            sig_a_k  = all_sigs[member_k]
            # Sigma = member's aleatoric + ensemble disagreement + proximity epistemic
            sig_k = np.maximum(np.sqrt(sig_a_k**2 + ens_std**2 + sig_e**2), floor)
            ms    = _sample_maxvalues(mu_k, sig_k, 1, rng)
            maxsamples[i] = ms[0]
        return maxsamples

    # ── rho: ensemble-averaged fidelity-pilot correlation ────────────────────
    def get_rho(self, fid: int) -> float:
        pid = self.pilot_fidelity
        if fid == pid:
            return 1.0

        rows = []
        for c in range(N_CLONES):
            cont = _CONT_BOUNDS_5[:, 0] + self._corr_rng.random((self.N_CORR, 5)) * (
                _CONT_BOUNDS_5[:, 1] - _CONT_BOUNDS_5[:, 0])
            rows.append(np.column_stack(
                [cont, np.full(self.N_CORR, fid), np.full(self.N_CORR, c)]))
        X    = np.vstack(rows)
        feat = self._enc.transform(X)

        rho_list = []
        for k in range(self.K):
            mm = self._member_models[k]
            if fid not in mm or pid not in mm:
                rho_list.append(self.rho_floor); continue
            mu_f = np.asarray(mm[fid].pred_dist(feat).loc, float)
            mu_p = np.asarray(mm[pid].pred_dist(feat).loc, float)
            if np.std(mu_f) < 1e-8 or np.std(mu_p) < 1e-8:
                rho_list.append(self.rho_floor); continue
            rho_k = float(np.clip(np.corrcoef(mu_f, mu_p)[0, 1], self.rho_floor, 1.0))
            rho_list.append(rho_k)

        return float(np.clip(np.mean(rho_list) if rho_list else self.rho_floor,
                             self.rho_floor, 1.0))

    def encoding_stats(self) -> dict:
        s = self._enc.encoding_stats()
        s["alpha_epist"] = round(self._alpha_epist, 4)
        s["K"]           = self.K
        return s
