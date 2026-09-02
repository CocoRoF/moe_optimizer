"""Orthogonal-polynomial charts.

Two things live here, and the difference between them is the central finding of
``my_paper/REVIEW_and_REDIRECTION_2026-09-02.md``:

``expert_chart``
    A chart over a *learned* expert coordinate.  Compresses an (E, r) coefficient
    table.  Its saving is bounded by E/(E+D) of the code -- under 0.1% for every
    production MoE -- so it is implemented as a **regulariser and progressive
    code**, never as a compression mechanism.  Claims to the contrary are false
    and the ``SlotCode`` accounting will show it.

``depth_chart``
    A chart over *layer depth*, which has a genuine total order and no learned
    coordinate.  It compresses the dictionary itself, which is replicated L times
    across the stack.  This is the position where the polynomial can be a primary
    compressor.

Both share the same basis machinery, and in both cases we prefer an *empirically
orthogonalised* basis over textbook Legendre: standard Legendre is orthogonal
under the uniform measure on [-1, 1], but the measure that matters is the
importance weight (routing frequency, or layer sensitivity), which is never
uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb

import torch


def legendre_vandermonde(z: torch.Tensor, degree: int) -> torch.Tensor:
    """Univariate Legendre features P_0..P_degree at points z in [-1, 1].

    Built by the three-term recurrence (2n+1) z P_n = (n+1) P_{n+1} + n P_{n-1},
    which is numerically stable far beyond the degrees we use -- unlike forming
    monomials and orthogonalising, which loses conditioning by degree ~15.
    """
    z = z.to(torch.float64).flatten()
    cols = [torch.ones_like(z)]
    if degree >= 1:
        cols.append(z)
    for n in range(1, degree):
        cols.append(((2 * n + 1) * z * cols[n] - n * cols[n - 1]) / (n + 1))
    return torch.stack(cols, dim=1)


def multi_indices(q: int, p: int) -> list[tuple[int, ...]]:
    """Total-degree-<= p multi-indices in q variables, ordered by degree.

    len(result) == comb(q + p, p), the M of the report's section 11.8.
    """
    out: list[tuple[int, ...]] = []
    for total in range(p + 1):
        for combo in combinations_with_replacement(range(q), total):
            idx = [0] * q
            for c in combo:
                idx[c] += 1
            out.append(tuple(idx))
    assert len(out) == comb(q + p, p)
    return out


def multivariate_vandermonde(z: torch.Tensor, degree: int) -> torch.Tensor:
    """Tensor-product Legendre features for z of shape (n, q), total degree <= p."""
    z = z.to(torch.float64)
    if z.ndim == 1:
        z = z.unsqueeze(1)
    n, q = z.shape
    uni = [legendre_vandermonde(z[:, j], degree) for j in range(q)]
    idxs = multi_indices(q, degree)
    cols = []
    for alpha in idxs:
        col = torch.ones(n, dtype=torch.float64)
        for j, a in enumerate(alpha):
            col = col * uni[j][:, a]
        cols.append(col)
    return torch.stack(cols, dim=1)


@dataclass
class OrthoBasis:
    """A basis orthonormalised against an explicit importance measure.

    ``Psi`` satisfies ``Psi^T diag(w) Psi = I`` up to numerical error, where w is
    the importance weight.  ``R_inv`` maps raw Vandermonde columns to Psi, so a
    fitted coefficient set can be converted back to the raw basis for storage.
    """

    Psi: torch.Tensor        # (n_points, M)
    R_inv: torch.Tensor      # (M, M)
    weights: torch.Tensor    # (n_points,)
    degree: int
    q: int

    @property
    def n_features(self) -> int:
        return self.Psi.shape[1]

    def orthogonality_error(self) -> float:
        g = self.Psi.T @ torch.diag(self.weights) @ self.Psi
        return float((g - torch.eye(g.shape[0], dtype=g.dtype)).norm())


def weighted_orthogonal_basis(
    z: torch.Tensor, degree: int, weights: torch.Tensor | None = None
) -> OrthoBasis:
    """Orthonormalise the Legendre Vandermonde against an importance measure.

    This is section 11.9 of the report, implemented as a weighted QR:

        diag(w)^{1/2} V = Q R      =>      Psi = V R^-1

    With uniform weights this reduces to a well-conditioned reparameterisation of
    standard Legendre, so the ablation "standard vs routing-weighted" is a change
    of one argument rather than of the code path.
    """
    V = multivariate_vandermonde(z, degree)
    n, M = V.shape
    if weights is None:
        w = torch.full((n,), 1.0 / n, dtype=torch.float64)
    else:
        w = weights.to(torch.float64).flatten().clamp_min(0)
        w = w / w.sum().clamp_min(1e-30)

    _, R = torch.linalg.qr(w.sqrt().unsqueeze(1) * V, mode="reduced")
    # Fix sign convention so the basis is deterministic across runs.
    sign = torch.sign(torch.diagonal(R))
    sign[sign == 0] = 1.0
    R = sign.unsqueeze(1) * R
    R_inv = torch.linalg.solve_triangular(R, torch.eye(M, dtype=torch.float64), upper=True)
    return OrthoBasis(Psi=V @ R_inv, R_inv=R_inv, weights=w, degree=degree,
                      q=1 if z.ndim == 1 else z.shape[1])


def fit_chart(
    targets: torch.Tensor, basis: OrthoBasis, weights: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted least squares of ``targets`` (n, k) onto the basis.

    Returns ``(Theta, fitted)`` with Theta of shape (M, k).  Because Psi is
    orthonormal under the weight measure, the normal equations reduce to a single
    weighted projection -- no iteration, no learning rate, no gradient.  This is
    what keeps the method inside tier T0 of the training-free taxonomy.
    """
    Y = targets.to(torch.float64)
    w = basis.weights if weights is None else (weights.to(torch.float64) /
                                               weights.sum().clamp_min(1e-30))
    Theta = basis.Psi.T @ (w.unsqueeze(1) * Y)
    return Theta, basis.Psi @ Theta


def truncate(Theta: torch.Tensor, basis: OrthoBasis, degree: int) -> torch.Tensor:
    """Keep only features of total degree <= ``degree``.

    Nested truncation is the property that makes a single artifact expose a
    continuum of operating points: because the basis is orthonormal, dropping
    high-degree rows of Theta *is* the optimal lower-degree fit -- no refitting
    required.  This is the mechanism behind the progressive-code claim.
    """
    keep = [i for i, a in enumerate(multi_indices(basis.q, basis.degree)) if sum(a) <= degree]
    return Theta[keep]
