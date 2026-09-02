"""Singular spectra and rank statistics.

Used for two distinct purposes, which must not be confused:

  * *diagnosis* -- is there low-dimensional structure to exploit at all?
  * *budgeting* -- given a byte budget, how much rank does this slot deserve?

Both reduce to the singular values, but the second must be computed on the
**whitened** residual (see ``factorize.whiten``), because an unweighted spectrum
answers a Frobenius question while deployment asks an operator-norm one.  That
mismatch is the diagnosis in arXiv:2606.03465 for why plain tensor
decompositions underperform, so we keep the distinction explicit in the API.
"""

from __future__ import annotations

import torch


def spectrum(w: torch.Tensor, driver: str | None = None) -> torch.Tensor:
    """Singular values of a 2-D matrix, descending, as float64 on CPU."""
    if w.ndim != 2:
        raise ValueError(f"expected 2-D, got shape {tuple(w.shape)}")
    return torch.linalg.svdvals(w.to(torch.float64))


def energy_at_rank(sv: torch.Tensor, r: int) -> float:
    """Fraction of squared Frobenius energy captured by the top-``r`` components."""
    if r <= 0:
        return 0.0
    e = sv.pow(2)
    total = e.sum()
    if total <= 0:
        return 1.0
    return float(e[:r].sum() / total)


def rank_for_energy(sv: torch.Tensor, target: float) -> int:
    """Smallest rank capturing at least ``target`` of the energy."""
    e = sv.pow(2)
    total = e.sum()
    if total <= 0:
        return 0
    c = torch.cumsum(e, 0) / total
    return int(torch.searchsorted(c, torch.tensor(target, dtype=c.dtype)).item()) + 1


def effective_rank(sv: torch.Tensor) -> float:
    """Roy & Vetterli entropy-based effective rank: exp(H(p)), p = sv / sum(sv).

    Continuous, and far more informative than a hard energy threshold when
    comparing slots whose spectra decay at different speeds.
    """
    s = sv[sv > 0]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def stable_rank(sv: torch.Tensor) -> float:
    """||W||_F^2 / ||W||_2^2 -- a lower bound on rank, robust to tail noise."""
    if sv.numel() == 0 or sv[0] <= 0:
        return 0.0
    return float(sv.pow(2).sum() / sv[0] ** 2)


def spectrum_summary(w: torch.Tensor) -> dict[str, float]:
    sv = spectrum(w)
    full = min(w.shape)
    return {
        "n": float(full),
        "effective_rank": effective_rank(sv),
        "stable_rank": stable_rank(sv),
        "r90": float(rank_for_energy(sv, 0.90)),
        "r99": float(rank_for_energy(sv, 0.99)),
        "r90_frac": rank_for_energy(sv, 0.90) / full,
        "r99_frac": rank_for_energy(sv, 0.99) / full,
        "sv_max": float(sv[0]) if sv.numel() else 0.0,
    }
