"""
MF-BARK — Multi-Fidelity Bayesian Additive Regression Kernel (fully Bayesian tree kernel GP).

Novel extension of BARK (Boyne et al., "BARK: A Fully Bayesian Tree Kernel for Black-box
Optimization") from single-fidelity to MULTI-FIDELITY with cross-fidelity AND cross-clone
coregionalization. Correctness over speed (pure numpy/scipy, no torch).

Kernel:  K((x,c,f),(x',c',f')) = B_fid[f,f'] * B_clone[c,c'] * K_tree(x,x') + noise_f * delta
  - K_tree(x,x') = fraction of trees in the SAMPLED forest placing x, x' in the same leaf.
                   Trees split on the 5 CONTINUOUS design vars ONLY; fidelity and clone are
                   handled by the coregionalization terms instead.
  - B_fid    = L_fid @ L_fid.T   (3x3 PSD over {MTP,MBR,Pilot}).
  - B_clone  = correlation of (W @ W.T + d I)  (30x30 PSD, LOW-RANK rank ~2-3 so it is estimable
               from limited data; diag normalised to 1 so B_fid carries the scale).
  - noise_f  : per-fidelity observation noise (MTP noisy, MBR/Pilot near-zero), consistent with the
               committed objective FIDELITY_NOISE.

FULLY BAYESIAN: the forest STRUCTURE (grow/prune/change moves with MH accept-reject), L_fid AND W
(clone) are sampled JOINTLY by MCMC. Prediction integrates over the retained posterior samples
(mixture of GPs) — the spread of per-sample means across forests is the epistemic uncertainty that a
single fixed forest (CTKG) or aleatoric NGBoost cannot represent.

*** CLONE STRUCTURE LEARNS FROM ALL FIDELITIES ***  Because the kernel factorises as
B_fid * B_clone * K_tree, an observation of clone c at ANY fidelity (incl. the 200+ MTP evals) enters
clone c's row of B_clone, with B_fid supplying the cross-fidelity scale. Clone learning is NOT
restricted to pilot data.

NOT an interpretable tree: MF-BARK is a GP with a tree kernel.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve

_FID_COL, _CLONE_COL, _N_CONT = 5, 6, 5
# design-space bounds for the 5 continuous vars [T, pH, F1, F2, F3] — split thresholds are drawn
# uniformly over these KNOWN bounds so splits can fall in UNEXPLORED regions (-> epistemic variance).
CONT_BOUNDS = np.array([[30.0, 40.0], [6.0, 8.0], [0.0, 50.0], [0.0, 50.0], [0.0, 50.0]])
# per-fidelity raw measurement noise (matches committed objective FIDELITY_NOISE)
RAW_NOISE = {0: 0.20, 1: 1e-4, 2: 1e-6}


# ── tree: dict of nodes keyed by int id; node = [feat, thr, left, right], feat=-1 => leaf ──────────
class Tree:
    __slots__ = ("nodes", "_ctr")

    def __init__(self):
        self.nodes = {0: [-1, 0.0, -1, -1]}     # root is a leaf
        self._ctr = 1

    def copy(self) -> "Tree":
        t = Tree.__new__(Tree)
        t.nodes = {k: v[:] for k, v in self.nodes.items()}
        t._ctr = self._ctr
        return t

    def leaves(self):
        return [i for i, n in self.nodes.items() if n[0] == -1]

    def nogs(self):
        """internal nodes whose BOTH children are leaves (prunable)."""
        out = []
        for i, n in self.nodes.items():
            if n[0] != -1 and self.nodes[n[2]][0] == -1 and self.nodes[n[3]][0] == -1:
                out.append(i)
        return out

    def internals(self):
        return [i for i, n in self.nodes.items() if n[0] != -1]

    def depth_of(self, target):
        # BFS from root
        stack = [(0, 0)]
        while stack:
            i, d = stack.pop()
            if i == target:
                return d
            n = self.nodes[i]
            if n[0] != -1:
                stack.append((n[2], d + 1)); stack.append((n[3], d + 1))
        return 0

    def apply(self, Xc):
        """Vectorised leaf label for each row of Xc (n,5). Labels are compact node indices, consistent
        within THIS tree object (dict insertion order is stable and preserved by copy()), so they are
        valid for same-leaf equality between any two apply() calls on the same tree."""
        ids = list(self.nodes.keys())
        idx = {nid: i for i, nid in enumerate(ids)}
        feat = np.array([self.nodes[nid][0] for nid in ids])
        thr = np.array([self.nodes[nid][1] for nid in ids])
        left = np.array([idx[self.nodes[nid][2]] if self.nodes[nid][0] != -1 else -1 for nid in ids])
        right = np.array([idx[self.nodes[nid][3]] if self.nodes[nid][0] != -1 else -1 for nid in ids])
        n = len(Xc)
        cur = np.full(n, idx[0], dtype=np.int64)
        for _ in range(64):                     # max depth guard
            f = feat[cur]
            internal = f != -1
            if not internal.any():
                break
            ii = np.where(internal)[0]
            go_left = Xc[ii, f[ii]] <= thr[cur[ii]]
            cur[ii] = np.where(go_left, left[cur[ii]], right[cur[ii]])
        return cur

    def grow(self, leaf_id, feat, thr):
        a, b = self._ctr, self._ctr + 1
        self._ctr += 2
        self.nodes[leaf_id] = [feat, thr, a, b]
        self.nodes[a] = [-1, 0.0, -1, -1]
        self.nodes[b] = [-1, 0.0, -1, -1]

    def prune(self, nog_id):
        n = self.nodes[nog_id]
        del self.nodes[n[2]]; del self.nodes[n[3]]
        self.nodes[nog_id] = [-1, 0.0, -1, -1]


def _p_split(depth, alpha=0.95, beta=2.0):
    return alpha * (1.0 + depth) ** (-beta)


class MFBARK:
    def __init__(self, n_clones=30, n_fidelities=3, pilot_fidelity=2, n_trees=30, clone_rank=3,
                 n_burn=300, n_keep=40, thin=8, p_grow=0.4, p_prune=0.4,
                 step_Lfid=0.15, step_W=0.10, sigma_W_prior=0.5, noise_floor=1e-3,
                 jitter=1e-6, n_max_train=180, seed=0, warm=True):
        self.n_clones, self.n_fidelities, self.pilot = n_clones, n_fidelities, pilot_fidelity
        self.n_trees, self.rank = n_trees, clone_rank
        self.n_burn, self.n_keep, self.thin = n_burn, n_keep, thin
        self.p_grow, self.p_prune = p_grow, p_prune
        self.step_Lfid, self.step_W = step_Lfid, step_W
        self.sigma_W_prior, self.noise_floor, self.jitter = sigma_W_prior, noise_floor, jitter
        self.n_max_train = n_max_train
        self.rng = np.random.default_rng(seed)
        self.warm = warm
        # persistent warm-start state across fit() calls (BO rounds)
        self._warm_forest = None
        self._warm_Lfid = None
        self._warm_W = None
        # diagnostics from last fit
        self.diag = {}
        self._samples = []          # retained posterior samples
        self._y_mean = 0.0
        self._y_std = 1.0

    # ── coregionalization matrices ────────────────────────────────────────────
    def _Bfid(self, Lvec):
        n = self.n_fidelities
        L = np.zeros((n, n)); k = 0
        for i in range(n):
            for j in range(i + 1):
                L[i, j] = Lvec[k]; k += 1
        return L @ L.T

    def _Bclone(self, W):
        raw = W @ W.T + np.eye(self.n_clones)      # W (n_clones, rank); + I => PSD, corr->I when W=0
        d = np.sqrt(np.clip(np.diag(raw), 1e-12, None))
        return raw / np.outer(d, d)                # correlation (diag 1)

    # ── kernel assembly + marginal likelihood ─────────────────────────────────
    def _Ktree(self, leaf_ids):
        n = leaf_ids[0].shape[0]
        K = np.zeros((n, n))
        for lid in leaf_ids:
            K += (lid[:, None] == lid[None, :])
        return K / len(leaf_ids)

    def _noise_diag(self):
        nv = {f: max((RAW_NOISE[f] / self._y_std) ** 2, self.noise_floor) for f in range(self.n_fidelities)}
        return np.array([nv[f] for f in self._f]) + self.jitter

    def _logml(self, Kt, Bf_pairs, Bc_pairs, noise):
        K = Bf_pairs * Bc_pairs * Kt
        K[np.diag_indices_from(K)] += noise
        try:
            c, low = cho_factor(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf, None
        alpha = cho_solve((c, low), self._y, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(c)))
        ml = -0.5 * self._y @ alpha - 0.5 * logdet - 0.5 * len(self._y) * np.log(2 * np.pi)
        return ml, (c, low, alpha)

    def _propose_thr(self, feat, mask):
        """Data-driven split threshold for `feat`: a midpoint between two adjacent OBSERVED values
        (among the training points in `mask`, or all if mask is None). BART-style — makes trees adapt
        to the data so nearby points stay in the same leaf (variance shrinks near data) and far/sparse
        regions isolate (variance grows). Falls back to the design bounds if <2 distinct values."""
        vals = self._Xc[:, feat] if mask is None else self._Xc[mask, feat]
        v = np.unique(vals)
        if len(v) >= 2:
            i = int(self.rng.integers(len(v) - 1))
            return float(0.5 * (v[i] + v[i + 1]))
        return float(self.rng.uniform(*CONT_BOUNDS[feat]))

    # ── one structural MH move on a random tree ────────────────────────────────
    def _tree_move(self, m, Kt, Bf_pairs, Bc_pairs, noise, cur_ml):
        tree = self._forest[m]
        lids = self._leaf_ids[m]
        u = self.rng.random()
        prop = tree.copy()
        move = None
        if u < self.p_grow or len(tree.nogs()) == 0:
            move = "grow"
            leaves = tree.leaves(); b = len(leaves)
            leaf = leaves[self.rng.integers(b)]
            feat = int(self.rng.integers(_N_CONT))
            thr = self._propose_thr(feat, lids == leaf)   # DATA-DRIVEN split (between observed values)
            prop.grow(leaf, feat, thr)
            d = tree.depth_of(leaf)
            nog_after = len(prop.nogs())
            log_prior = (np.log(_p_split(d)) + 2 * np.log(1 - _p_split(d + 1)) - np.log(1 - _p_split(d)))
            log_trans = np.log(self.p_prune / self.p_grow) + np.log(b) - np.log(max(nog_after, 1))
        elif u < self.p_grow + self.p_prune:
            move = "prune"
            nogs = tree.nogs(); w = len(nogs)
            nog = nogs[self.rng.integers(w)]
            d = tree.depth_of(nog)
            prop.prune(nog)
            b_after = len(prop.leaves())
            log_prior = -(np.log(_p_split(d)) + 2 * np.log(1 - _p_split(d + 1)) - np.log(1 - _p_split(d)))
            log_trans = np.log(self.p_grow / self.p_prune) + np.log(w) - np.log(max(b_after, 1))
        else:
            move = "change"
            ints = tree.internals()
            if not ints:
                return Kt, cur_ml, None, False
            node = ints[self.rng.integers(len(ints))]
            feat = int(self.rng.integers(_N_CONT))
            thr = self._propose_thr(feat, None)          # DATA-DRIVEN (global obs of this feature)
            prop.nodes[node] = [feat, thr, prop.nodes[node][2], prop.nodes[node][3]]
            log_prior = 0.0; log_trans = 0.0

        lids_new = prop.apply(self._Xc)
        eq_old = (lids[:, None] == lids[None, :]).astype(float)
        eq_new = (lids_new[:, None] == lids_new[None, :]).astype(float)
        Kt_prop = Kt + (eq_new - eq_old) / self.n_trees
        ml_prop, cache = self._logml(Kt_prop, Bf_pairs, Bc_pairs, noise)
        log_acc = (ml_prop - cur_ml) + log_prior + log_trans
        if np.log(self.rng.random() + 1e-300) < log_acc:
            self._forest[m] = prop
            self._leaf_ids[m] = lids_new
            return Kt_prop, ml_prop, move, True
        return Kt, cur_ml, move, False

    # ── MH on L_fid / W (global) ───────────────────────────────────────────────
    def _param_move(self, which, Kt, cur_ml):
        if which == "Lfid":
            prop = self._Lfid + self.step_Lfid * self.rng.standard_normal(self._Lfid.shape)
            Bf = self._Bfid(prop); Bf_pairs = Bf[self._f[:, None], self._f[None, :]]
            Bc_pairs = self._Bc_pairs
            log_prior = -0.5 * (np.sum(prop ** 2) - np.sum(self._Lfid ** 2))     # N(0,1) prior
        else:
            # update ONE random column of W (n_clones params) — far better acceptance than all
            # n_clones*rank at once for a random-walk proposal.
            prop = self._W.copy()
            col = int(self.rng.integers(self.rank))
            prop[:, col] = prop[:, col] + self.step_W * self.rng.standard_normal(self.n_clones)
            Bc = self._Bclone(prop); Bc_pairs = Bc[self._c[:, None], self._c[None, :]]
            Bf_pairs = self._Bf_pairs
            log_prior = -0.5 / self.sigma_W_prior ** 2 * (np.sum(prop ** 2) - np.sum(self._W ** 2))
        ml_prop, cache = self._logml(Kt, Bf_pairs, Bc_pairs, self._noise)
        if np.log(self.rng.random() + 1e-300) < (ml_prop - cur_ml) + log_prior:
            if which == "Lfid":
                self._Lfid = prop; self._Bf_pairs = Bf_pairs
            else:
                self._W = prop; self._Bc_pairs = Bc_pairs
            return ml_prop, True
        return cur_ml, False

    # ── fit: run the MCMC ──────────────────────────────────────────────────────
    def fit(self, X_raw, y):
        X_raw = np.asarray(X_raw, float)
        y = np.asarray(y, float)
        # cap training set: keep all non-MTP + most-recent MTP (like CTKG)
        if len(y) > self.n_max_train:
            f = X_raw[:, _FID_COL].astype(int)
            nonc = np.where(f != 0)[0]; cheap = np.where(f == 0)[0]
            slots = max(0, self.n_max_train - len(nonc))
            keep = np.sort(np.concatenate([nonc, cheap[-slots:] if slots else np.array([], int)]))
            X_raw, y = X_raw[keep], y[keep]
        self._Xc = X_raw[:, :_N_CONT].copy()
        self._f = X_raw[:, _FID_COL].astype(int)
        self._c = X_raw[:, _CLONE_COL].astype(int).clip(0, self.n_clones - 1)
        self._y_mean, self._y_std = float(y.mean()), float(y.std() or 1.0)
        self._y = (y - self._y_mean) / self._y_std
        n = len(y)

        # init forest + params (warm-start from previous round if available)
        if self.warm and self._warm_forest is not None:
            self._forest = [t.copy() for t in self._warm_forest]
            self._Lfid = self._warm_Lfid.copy()
            self._W = self._warm_W.copy()
            n_burn = max(60, self.n_burn // 4)
        else:
            self._forest = [Tree() for _ in range(self.n_trees)]
            L0 = np.linalg.cholesky(0.6 * np.ones((self.n_fidelities, self.n_fidelities))
                                    + 0.4 * np.eye(self.n_fidelities))
            self._Lfid = L0[np.tril_indices(self.n_fidelities)].copy()
            self._W = 0.1 * self.rng.standard_normal((self.n_clones, self.rank))
            n_burn = self.n_burn

        self._leaf_ids = [t.apply(self._Xc) for t in self._forest]
        self._noise = self._noise_diag()
        self._Bf_pairs = self._Bfid(self._Lfid)[self._f[:, None], self._f[None, :]]
        self._Bc_pairs = self._Bclone(self._W)[self._c[:, None], self._c[None, :]]
        Kt = self._Ktree(self._leaf_ids)
        cur_ml, _ = self._logml(Kt, self._Bf_pairs, self._Bc_pairs, self._noise)

        acc = {"grow": [0, 0], "prune": [0, 0], "change": [0, 0], "Lfid": [0, 0], "W": [0, 0]}
        ll_trace, self._samples = [], []
        total = n_burn + self.n_keep * self.thin
        for it in range(total):
            for m in range(self.n_trees):
                Kt, cur_ml, move, ok = self._tree_move(m, Kt, self._Bf_pairs, self._Bc_pairs,
                                                        self._noise, cur_ml)
                if move:
                    acc[move][1] += 1; acc[move][0] += int(ok)
            cur_ml, ok = self._param_move("Lfid", Kt, cur_ml); acc["Lfid"][1] += 1; acc["Lfid"][0] += int(ok)
            for _ in range(self.rank):     # one W-column update per column per sweep
                cur_ml, ok = self._param_move("W", Kt, cur_ml); acc["W"][1] += 1; acc["W"][0] += int(ok)
            ll_trace.append(cur_ml)
            if it >= n_burn and (it - n_burn) % self.thin == 0:
                _, cache = self._logml(Kt, self._Bf_pairs, self._Bc_pairs, self._noise)
                c, low, alpha = cache
                self._samples.append(dict(
                    forest=[t.copy() for t in self._forest],
                    leaf_ids=[lid.copy() for lid in self._leaf_ids],
                    Bfid=self._Bfid(self._Lfid), Bclone=self._Bclone(self._W),
                    chol=(c.copy(), low), alpha=alpha.copy()))
        # warm state for next round
        self._warm_forest = [t.copy() for t in self._forest]
        self._warm_Lfid = self._Lfid.copy(); self._warm_W = self._W.copy()
        self.diag = dict(acc_rates={k: (v[0] / v[1] if v[1] else 0.0) for k, v in acc.items()},
                         ll_trace=ll_trace, n_samples=len(self._samples), n_train=n,
                         distinct_clones=int(len(np.unique(self._c))),
                         distinct_clones_by_fid={f: int(len(np.unique(self._c[self._f == f])))
                                                 for f in range(self.n_fidelities)})
        return self

    # ── prediction: mixture over retained posterior samples ────────────────────
    def _pred_one(self, s, Xc_cand, c_cand, fid):
        Kt_cs = np.zeros((len(Xc_cand), len(self._Xc)))
        for t, ltr in zip(s["forest"], s["leaf_ids"]):
            lc = t.apply(Xc_cand)
            Kt_cs += (lc[:, None] == ltr[None, :])
        Kt_cs /= self.n_trees
        Bf = s["Bfid"]; Bc = s["Bclone"]
        Bf_col = Bf[fid, self._f]                                   # (n,)
        Bc_cs = Bc[c_cand[:, None], self._c[None, :]]              # (m,n)
        K_xs = Bf_col[None, :] * Bc_cs * Kt_cs
        mu = K_xs @ s["alpha"]
        v = cho_solve(s["chol"], K_xs.T, check_finite=False)       # (n,m)
        K_ss = Bf[fid, fid]                                        # Bclone diag=1, Ktree diag=1
        var = np.maximum(K_ss - np.einsum("ij,ji->i", K_xs, v), 1e-9)
        return mu, var

    def predict(self, X_raw_cands, fidelity=None):
        fid = self.pilot if fidelity is None else fidelity
        X = np.asarray(X_raw_cands, float)
        Xc = X[:, :_N_CONT]; c_cand = X[:, _CLONE_COL].astype(int).clip(0, self.n_clones - 1)
        mus, vars = [], []
        for s in self._samples:
            mu, var = self._pred_one(s, Xc, c_cand, fid)
            mus.append(mu); vars.append(var)
        mus = np.array(mus); vars = np.array(vars)                 # (S, m)
        mean = mus.mean(0)
        # law of total variance: E[var] + Var[mean]  (the 2nd term = epistemic across forests)
        tot_var = vars.mean(0) + mus.var(0)
        mean_raw = mean * self._y_std + self._y_mean
        sig_raw = np.sqrt(tot_var) * self._y_std
        return mean_raw, sig_raw

    # ── acquisition interface (for bo_loop) ─────────────────────────────────
    def predict_pilot(self, X_raw_cands):
        return self.predict(X_raw_cands, self.pilot)

    def sample_pilot_max(self, X_pool, n_samples, rng):
        """Pool max-value samples across the retained forest samples (integrates over forests)."""
        X = np.asarray(X_pool, float)
        Xc = X[:, :_N_CONT]; c_cand = X[:, _CLONE_COL].astype(int).clip(0, self.n_clones - 1)
        per = max(1, n_samples // max(len(self._samples), 1))
        maxs = []
        for s in self._samples:
            mu, var = self._pred_one(s, Xc, c_cand, self.pilot)
            sig = np.sqrt(var)
            draws = mu[None, :] + rng.standard_normal((per, len(mu))) * sig[None, :]
            maxs.extend(draws.max(1).tolist())
        maxs = np.array(maxs) * self._y_std + self._y_mean
        return maxs[:n_samples] if len(maxs) >= n_samples else np.pad(maxs, (0, n_samples - len(maxs)), mode="edge")

    def get_rho(self, fid):
        if not self._samples:
            return 1.0 if fid == self.pilot else 0.3
        rs = []
        for s in self._samples:
            B = s["Bfid"]; p = self.pilot
            rs.append(B[fid, p] / np.sqrt(max(B[fid, fid] * B[p, p], 1e-12)))
        return float(np.clip(np.mean(rs), 0.05, 1.0))

    # ── diagnostics for the gates ──────────────────────────────────────────────
    def Bfid_corr(self):
        Bs = np.array([s["Bfid"] for s in self._samples])
        B = Bs.mean(0)
        d = np.sqrt(np.clip(np.diag(B), 1e-12, None))
        return B / np.outer(d, d)

    def Bclone_stats(self):
        Bs = np.array([s["Bclone"] for s in self._samples])       # (S, C, C)
        Bmean = Bs.mean(0)
        off = Bmean[~np.eye(self.n_clones, dtype=bool)]
        # stability: std across MCMC samples of each off-diagonal, averaged
        off_sample_std = Bs.std(0)[~np.eye(self.n_clones, dtype=bool)].mean()
        return dict(off_mean=float(off.mean()), off_std=float(off.std()),
                    off_min=float(off.min()), off_max=float(off.max()),
                    mcmc_stability_std=float(off_sample_std), Bmean=Bmean)

    def clone_cov_stats(self):
        st = self.Bclone_stats()
        return {k: st[k] for k in ("off_mean", "off_std", "off_min", "off_max", "mcmc_stability_std")}
