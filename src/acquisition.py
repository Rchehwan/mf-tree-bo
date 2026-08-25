"""
Torch-free multi-fidelity Max-value Entropy Search (MF-MES) acquisition.

This is the SAME information-theoretic criterion BoTorch's GIBBON
(qMultiFidelityLowerBoundMaxValueEntropy) approximates, implemented directly on a surrogate
posterior by sampling — no BoTorch, no gradients, so it works with a non-differentiable tree
kernel. Both tree surrogates share this acquisition and differ only in the surrogate that
supplies the pilot posterior (mu_p, sigma_p) and the fidelity-to-pilot correlation rho(f).

Criterion (moment-matched MF-MES, Takeno et al. 2020 / GIBBON lower bound):
  For a candidate evaluated at fidelity f, observing y_{x,f} informs the PILOT latent g(x) with
  correlation rho(f,pilot). The pilot max y* constrains g(x) <= y*. Under a joint-Gaussian model
  the entropy reduction of the observation given the truncation g<=y* is

      alpha_MES(x,f) = -1/(2M) * sum_m log( 1 - rho^2 * W(beta_m) ),
      beta_m = (y*_m - mu_p(x)) / sigma_p(x),   W(beta) = beta*z + z^2,   z = phi(beta)/Phi(beta),

  where {y*_m} are M sampled pilot max-values and W in [0,1) is the one-sided (G<=y*) truncation
  variance-reduction factor. rho=1 (f==pilot) recovers single-fidelity MES.

Batch = JOINT mutual information, not the sum of per-point EIs. Built greedily: pick the top
candidate, then FANTASISE its pilot observation by raising every max-value sample to at least its
pilot mean (the max is now known to be >= what we just observed). A near-duplicate of an already
picked high-value point then sees beta_m >> 0 -> W -> 0 -> ~0 marginal information, so redundant
points add ~nothing (no local-penalisation / Lipschitz trick — the redundancy falls out of the
max-value conditioning). The greedy sum of marginal gains is the joint-MI lower bound.

Fidelity score = joint_MI(batch) / (batch_size * cost).  The optimiser picks the fidelity with the
best information-per-cost and returns its batch.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def sample_pilot_maxvalues(mu: np.ndarray, sigma: np.ndarray, n_samples: int,
                           rng: np.random.Generator, y_best: float | None = None) -> np.ndarray:
    """Sample M pilot max-values via the approximate max-CDF F(y)=prod_i Phi((y-mu_i)/sigma_i).

    mu, sigma: pilot posterior over a reference candidate pool (m,). Returns (M,) max-values, each
    >= y_best (the incumbent pilot best) when provided. Robust bisection on log F(y)=log u.
    """
    mu = np.asarray(mu, float); sigma = np.maximum(np.asarray(sigma, float), 1e-9)
    lo = float(mu.max())
    # Cap the upper search so the CDF-product (which treats pool points as INDEPENDENT and thus
    # over-estimates the max) cannot inflate max-values far above the design's plausible range.
    hi = float(mu.max() + 3.0 * np.median(sigma))
    if y_best is not None and np.isfinite(y_best):
        lo = max(lo, float(y_best))
    hi = max(hi, lo + 1e-2)
    us = rng.random(n_samples)
    ys = np.empty(n_samples)
    for k in range(n_samples):
        target = np.log(us[k] + 1e-12)
        a, b = lo, hi
        # expand hi if F(hi) still below target (rare)
        for _ in range(40):
            mid = 0.5 * (a + b)
            logF = np.sum(norm.logcdf((mid - mu) / sigma))
            if logF < target:
                a = mid
            else:
                b = mid
        ys[k] = 0.5 * (a + b)
    # max must be at least the best posterior mean location
    return np.maximum(ys, lo)


def _W(beta: np.ndarray) -> np.ndarray:
    """One-sided (G <= y*) truncation variance-reduction factor W(beta) = beta*z + z^2 in [0,1)."""
    Phi = np.clip(norm.cdf(beta), 1e-12, 1.0)
    z = norm.pdf(beta) / Phi                     # inverse Mills ratio (upper truncation)
    W = beta * z + z ** 2
    return np.clip(W, 0.0, 1.0 - 1e-6)


def mfmes_scores(mu_p: np.ndarray, sigma_p: np.ndarray, rho: float,
                 maxsamples: np.ndarray) -> np.ndarray:
    """Per-candidate MF-MES alpha (m,). mu_p,sigma_p: pilot posterior at candidates; rho: corr(f,pilot)."""
    mu_p = np.asarray(mu_p, float)
    sigma_p = np.maximum(np.asarray(sigma_p, float), 1e-9)
    beta = (maxsamples[None, :] - mu_p[:, None]) / sigma_p[:, None]   # (m, M)
    W = _W(beta)
    arg = np.clip(1.0 - (rho ** 2) * W, 1e-8, 1.0)
    alpha = -0.5 * np.mean(np.log(arg), axis=1)                       # (m,)
    return np.maximum(alpha, 0.0)


def greedy_joint_batch(mu_p: np.ndarray, sigma_p: np.ndarray, rho: float,
                       maxsamples: np.ndarray, q: int,
                       allowed: np.ndarray | None = None) -> tuple[list[int], float]:
    """Greedily select q candidate indices maximising JOINT MI about the pilot max.

    After each pick, fantasise its pilot observation by raising all max-samples to at least the
    picked point's pilot mean -> near-duplicates of it lose their marginal information. Returns
    (chosen_indices, joint_MI) where joint_MI is the greedy sum of marginal gains.
    """
    m = len(mu_p)
    avail = np.ones(m, bool) if allowed is None else allowed.copy()
    ms = np.asarray(maxsamples, float).copy()
    chosen, total = [], 0.0
    q = int(min(q, int(avail.sum())))
    for _ in range(q):
        a = mfmes_scores(mu_p, sigma_p, rho, ms)
        a[~avail] = -np.inf
        i = int(np.argmax(a))
        if not np.isfinite(a[i]):
            break
        chosen.append(i); avail[i] = False
        total += float(max(a[i], 0.0))
        # fantasy: the pilot max is now known to be >= the observed pilot value at i
        ms = np.maximum(ms, float(mu_p[i]))
    return chosen, total


def _ei(mu, sigma, best):
    """Expected improvement over `best` for a Normal(mu, sigma) predictive."""
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - best) / sigma
    return sigma * (z * norm.cdf(z) + norm.pdf(z))


def select_fidelity_ei(candidate_pools, predict_pilot, rho_of, cost, batch_size,
                       fidelity_levels, incumbent, mtp_plates=None):
    """Cost-aware EI tie-breaker: score = rho * sum(top-batch EI) / (batch*cost). Used only when the
    MF-MES MI is ~0 for every fidelity, so the loop keeps EXPLORING (toward best-improvement points)
    instead of stalling. rho weights the cheap fidelities' EI by their pilot correlation."""
    scores, batches, rhos = {}, {}, {}
    for fid in fidelity_levels:
        rho = float(rho_of(fid)); rhos[fid] = rho
        bs = batch_size[fid]
        if fid == 0 and mtp_plates is not None:
            best_pl, best_s = None, -np.inf
            for plate in mtp_plates:
                mu, sg = predict_pilot(plate)
                s = rho * float(np.sort(_ei(mu, sg, incumbent))[::-1][:len(plate)].sum())
                if s > best_s:
                    best_s, best_pl = s, plate
            batches[fid] = best_pl
            scores[fid] = best_s / (bs * cost[fid])
        else:
            pool = candidate_pools[fid]
            mu, sg = predict_pilot(pool)
            ei = _ei(mu, sg, incumbent)
            idx = np.argsort(ei)[::-1][:bs]
            batches[fid] = pool[idx]
            scores[fid] = rho * float(ei[idx].sum()) / (bs * cost[fid])
    best_fid = max(scores, key=lambda k: scores[k])
    return best_fid, batches[best_fid], scores, rhos


def select_fidelity_mfmes(candidate_pools: dict, predict_pilot, rho_of, cost, batch_size,
                          fidelity_levels, maxsamples, mtp_plates=None,
                          incumbent=None, pilot_fidelity=2, harvest_w=1.0):
    """Score every fidelity by joint-MI-per-cost and return the winner's batch.

    Parameters
    ----------
    candidate_pools : {fid: X_pool (n,7)} free pools for MBR/Pilot (and MTP if mtp_plates is None).
    predict_pilot   : callable X -> (mu_p, sigma_p) pilot posterior.
    rho_of          : callable fid -> rho(fid, pilot) in [~0,1].
    cost, batch_size: dicts fid -> cost per point / batch size.
    mtp_plates      : optional list of MTP plates (each (12,7) sharing one T,pH). If given, the MTP
                      batch is the single best plate (paper-style batch-wide T/pH), scored by its
                      joint MI; otherwise MTP is treated as a free pool like MBR/Pilot.

    Returns (best_fid, best_batch (q,7), scores dict, rhos dict).
    """
    scores, batches, rhos = {}, {}, {}
    for fid in fidelity_levels:
        rho = float(rho_of(fid)); rhos[fid] = rho
        bs = batch_size[fid]
        if fid == 0 and mtp_plates is not None:
            best_plate, best_mi = None, -np.inf
            for plate in mtp_plates:
                mu_p, sig_p = predict_pilot(plate)
                _, mi = greedy_joint_batch(mu_p, sig_p, rho, maxsamples, len(plate))
                if mi > best_mi:
                    best_mi, best_plate = mi, plate
            batches[fid] = best_plate
            scores[fid] = best_mi / (bs * cost[fid])
        elif fid == pilot_fidelity and incumbent is not None and np.isfinite(incumbent):
            # HARVEST fix (Exp C): Pilot is the ONLY fidelity that can improve best_pilot. Pure MES
            # keeps buying cheap MBR *information about* the max while the incumbent stays frozen
            # (lock-in). Add a cost-aware EI term (standardised by sigma -> O(1), comparable to the MES
            # nats) so Pilot is HARVESTED when the surrogate confidently predicts a candidate above the
            # incumbent; when nothing beats it, EI~0 and MES governs (so early exploration is unchanged).
            pool = candidate_pools[fid]
            mu_p, sig_p = predict_pilot(pool)
            alpha = mfmes_scores(mu_p, sig_p, rho, maxsamples)
            ei_std = _ei(mu_p, sig_p, incumbent) / np.maximum(sig_p, 1e-9)
            comb = alpha + harvest_w * ei_std
            idx = np.argsort(comb)[::-1][:bs]
            batches[fid] = pool[idx]
            scores[fid] = float(comb[idx].sum()) / (bs * cost[fid])
        else:
            pool = candidate_pools[fid]
            mu_p, sig_p = predict_pilot(pool)
            idx, mi = greedy_joint_batch(mu_p, sig_p, rho, maxsamples, bs)
            batches[fid] = pool[idx]
            scores[fid] = mi / (bs * cost[fid])
    best_fid = max(scores, key=lambda k: scores[k])
    return best_fid, batches[best_fid], scores, rhos
