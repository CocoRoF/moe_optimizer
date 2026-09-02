"""Activation-aware whitening.

Plain low-rank factorisation minimises ||W - W_hat||_F.  What deployment cares
about is ||(W - W_hat) x|| for x drawn from the actual token distribution.  These
differ whenever the input covariance is far from isotropic -- which it always is
in a trained transformer.

arXiv:2606.03465 identifies precisely this gap as the reason Frobenius-optimal
tensor decompositions (HOSVD, TT-SVD, Tucker) underperform on LLMs: they
"optimize a norm misaligned with operator-norm preservation".  On that evidence
whitening is not an optional refinement here; it is the default path, and the
un-whitened variant is kept only as an ablation.

Given input second-moment C_x = E[x x^T] with Cholesky factor L (C_x = L L^T),

    W x  =  (W L) (L^-1 x)

and ``L^-1 x`` is white.  So factorising ``W L`` under the Frobenius norm is
equivalent to factorising W under the data-weighted norm.  We factorise the
whitened matrix and undo the transform on reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Whitener:
    """A per-slot input whitening transform and its inverse.

    ``forward(W) = W @ L`` and ``inverse(Ww) = Ww @ L_inv``.
    """

    L: torch.Tensor          # (d_in, d_in) lower-triangular Cholesky factor
    L_inv: torch.Tensor
    ridge: float
    trace_ratio: float       # conditioning diagnostic; see ``from_covariance``

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w @ self.L

    def inverse(self, w: torch.Tensor) -> torch.Tensor:
        return w @ self.L_inv

    @classmethod
    def identity(cls, d: int, dtype: torch.dtype = torch.float64) -> "Whitener":
        eye = torch.eye(d, dtype=dtype)
        return cls(L=eye, L_inv=eye.clone(), ridge=0.0, trace_ratio=1.0)

    @classmethod
    def from_covariance(cls, cov: torch.Tensor, ridge: float = 1e-4) -> "Whitener":
        """Build from an input second-moment matrix.

        ``ridge`` is relative to the mean diagonal, so the same setting behaves
        consistently across layers with very different activation scales.  A
        calibration covariance is rank-deficient whenever fewer tokens than
        dimensions were seen, so the ridge is load-bearing, not cosmetic.
        """
        cov = cov.to(torch.float64)
        d = cov.shape[0]
        if cov.shape != (d, d):
            raise ValueError(f"covariance must be square, got {tuple(cov.shape)}")
        cov = 0.5 * (cov + cov.T)
        scale = float(cov.diagonal().mean().clamp_min(1e-30))
        reg = cov + ridge * scale * torch.eye(d, dtype=torch.float64)
        try:
            L = torch.linalg.cholesky(reg)
        except RuntimeError:
            # Fall back to an eigen-based square root when Cholesky fails.
            evals, evecs = torch.linalg.eigh(reg)
            L = evecs @ torch.diag(evals.clamp_min(ridge * scale).sqrt())
        L_inv = torch.linalg.solve_triangular(
            L, torch.eye(d, dtype=torch.float64), upper=False
        ) if L.is_contiguous() and torch.allclose(L.triu(1), torch.zeros_like(L)) else torch.linalg.pinv(L)
        return cls(L=L, L_inv=L_inv, ridge=ridge,
                   trace_ratio=float(cov.diagonal().max() / cov.diagonal().mean().clamp_min(1e-30)))


def whiten_stack(stack: torch.Tensor, whitener: Whitener) -> torch.Tensor:
    """Apply the transform to every expert of an (E, d_out, d_in) table."""
    return torch.einsum("eoi,ij->eoj", stack.to(torch.float64), whitener.L)


def unwhiten_stack(stack: torch.Tensor, whitener: Whitener) -> torch.Tensor:
    return torch.einsum("eoi,ij->eoj", stack.to(torch.float64), whitener.L_inv)


def covariance_from_activations(x: torch.Tensor) -> torch.Tensor:
    """Second moment E[x x^T] from a (n_tokens, d) activation sample."""
    x = x.to(torch.float64)
    return (x.T @ x) / max(x.shape[0], 1)
