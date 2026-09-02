"""Principal angles between subspaces.

This is the measurement instrument for **gate G0**: does the dominant subspace of
a community's expert dictionary rotate *smoothly* with layer depth?

The answer decides the project's headline claim.

  * smooth drift            -> a global orthogonal-polynomial chart over depth works
  * piecewise-flat, with jumps -> a *piecewise* chart works; the jumps are chart boundaries
  * no structure            -> the depth axis is dead and the idea is dropped

ConMoE (arXiv:2605.29350) measured 50.4% cross-layer nearest-neighbour rates on
Qwen3-30B-A3B within 4-layer scopes, which predicts the second outcome.  We
measure it directly rather than assuming either way.
"""

from __future__ import annotations

import torch


def top_subspace(w: torch.Tensor, r: int) -> torch.Tensor:
    """Orthonormal basis (d_out, r) for the dominant left singular subspace."""
    if w.ndim != 2:
        raise ValueError(f"expected 2-D, got {tuple(w.shape)}")
    r = min(r, *w.shape)
    u, _, _ = torch.linalg.svd(w.to(torch.float64), full_matrices=False)
    return u[:, :r].contiguous()


def principal_angles(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians, ascending) between the column spans of a and b.

    Both must have orthonormal columns.  Uses the singular values of ``a^T b``,
    clamped for numerical safety; this is the standard stable formulation for
    angles away from zero.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"ambient dims differ: {a.shape[0]} vs {b.shape[0]}")
    s = torch.linalg.svdvals(a.T.to(torch.float64) @ b.to(torch.float64))
    return torch.arccos(s.clamp(-1.0, 1.0)).flip(0)


def grassmann_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Geodesic distance on the Grassmannian: sqrt(sum theta_i^2)."""
    return float(principal_angles(a, b).pow(2).sum().sqrt())


def subspace_affinity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cos^2 of the principal angles, in [0, 1].

    1.0 means the subspaces coincide; 0.0 means they are mutually orthogonal.
    Normalised by dimension so subspaces of different rank stay comparable.
    """
    th = principal_angles(a, b)
    if th.numel() == 0:
        return 0.0
    return float(th.cos().pow(2).mean())


def chordal_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """sqrt(sum sin^2 theta_i) -- the projection F-norm distance."""
    return float(principal_angles(a, b).sin().pow(2).sum().sqrt())
