"""Calibrate skipping policies from one streaming pass.

Inputs (from ``scripts/policy_calib.py``):
  ``scale[l]``   (E,) mean ||E_e(x)|| over tokens routed to e     -> ContributionPolicy
  ``traces[l]``  (T, k) sorted top-k router probs per token         -> MassRatio medians,
                                                                       headroom, fair tau
Outputs are plain dicts so a policy can be built without the model.
"""

from __future__ import annotations

import torch


def mass_ratio_medians(traces: dict[int, torch.Tensor], k: int) -> dict[int, list[float]]:
    """beta[l][m-1] = median over tokens of (share of the m lowest experts) -- arXiv:2512.21911."""
    out = {}
    for l, w in traces.items():
        total = w.sum(1, keepdim=True).clamp_min(1e-12)
        tail = w.flip(1).cumsum(1).flip(1) / total          # tail[:, j] = share of j..k-1
        out[l] = [float(tail[:, k - m].median()) for m in range(1, k)]
    return out


def _mean_k_for_tau(cum_share: torch.Tensor, tau: float, min_keep: int) -> float:
    thr = 1.0 - tau
    keep = torch.cat([torch.ones_like(cum_share[:, :1], dtype=torch.bool), cum_share[:, :-1] < thr], 1)
    keep[:, :min_keep] = True
    return float(keep.sum(1).float().mean())


def tau_for_target_k(cum_share: torch.Tensor, target_k: float, min_keep: int = 1,
                     iters: int = 40) -> float:
    """Bisect tau in [0, 1] so the mean kept count equals target_k (monotone in tau)."""
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _mean_k_for_tau(cum_share, mid, min_keep) > target_k:
            lo = mid                                         # too many kept -> raise tau
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cumulative_share(traces: dict[int, torch.Tensor], scale: dict[int, torch.Tensor] | None,
                     indices: dict[int, torch.Tensor] | None) -> dict[int, torch.Tensor]:
    """Per layer, (T, k) cumulative share of the sorted ranking quantity.

    scale=None -> gate-mass ranking (the fair score-only baseline).
    scale given -> contribution ranking w * s[indices].
    """
    out = {}
    for l, w in traces.items():
        c = w if scale is None else w * scale[l][indices[l]]
        c_sorted = c.sort(1, descending=True).values
        out[l] = (c_sorted / c_sorted.sum(1, keepdim=True).clamp_min(1e-12)).cumsum(1)
    return out


def calibrate_taus(traces, scale, indices, target_k: float, min_keep: int = 1) -> dict[int, float]:
    """One tau per layer, each hitting target_k on its own tokens (per-layer budget)."""
    cs = cumulative_share(traces, scale, indices)
    return {l: tau_for_target_k(cs[l], target_k, min_keep) for l in cs}


def scale_dispersion(scale: dict[int, torch.Tensor]) -> dict[int, float]:
    """Coefficient of variation of the calibrated output scale within each layer -- gate G1."""
    return {l: float(s.std() / s.mean().clamp_min(1e-12)) for l, s in scale.items()}


def error_curve(traces, scale, indices, renorm: bool) -> dict[int, torch.Tensor]:
    """Per layer, (T, k) normalised err(P_j) for prefixes j=1..k in w*s order --
    the quantity ContributionRenormPolicy thresholds.  Mirrors its select()."""
    out = {}
    for l, w in traces.items():
        s = scale[l][indices[l]]; c2 = (w * s).pow(2)
        order = c2.argsort(1, descending=True); w_s, c2_s = w.gather(1, order), c2.gather(1, order)
        total = c2_s.sum(1, keepdim=True).clamp_min(1e-30); kept = c2_s.cumsum(1); dropped = total - kept
        amp = kept * (w_s.sum(1, keepdim=True).clamp_min(1e-12) / w_s.cumsum(1).clamp_min(1e-12) - 1).pow(2) if renorm else torch.zeros_like(kept)
        out[l] = (dropped + amp) / total
    return out


def calibrate_taus_err(traces, scale, indices, target_k: float, renorm: bool, min_keep: int = 1) -> dict[int, float]:
    """Per-layer tau for the error-model policies hitting target_k (bisection; mean k is monotone in tau)."""
    curves = error_curve(traces, scale, indices, renorm); out = {}
    for l, err in curves.items():
        k = err.shape[1]
        def mean_k(tau):
            ok = err <= tau; ok[:, k - 1] = True
            j = torch.clamp(ok.float().argmax(1), min=min_keep - 1)
            return float((j + 1).float().mean())
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if mean_k(mid) > target_k: lo = mid
            else: hi = mid
        out[l] = 0.5 * (lo + hi)
    return out
