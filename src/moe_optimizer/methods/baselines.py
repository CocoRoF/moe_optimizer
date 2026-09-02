"""Reference implementations of the prior-art families.

These exist so the Pareto comparison is run under identical conditions --
identical byte accounting, identical error metrics, identical calibration -- and
so a difference in the table is a difference in method rather than in
experimental setup.  They are faithful to the *structural idea* of each paper,
not to every engineering detail, and are labelled "-like" throughout for that
reason.  Published numbers are not comparable to these and must not be mixed
into the same table.
"""

from __future__ import annotations

from typing import Any

import torch

from ..factorize.base import Compressor, SlotCode
from ..factorize.whiten import Whitener
from ..registry import COMPRESSORS


def _svd_lowrank(w: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """W ~= A @ B with A (d_out, r), B (r, d_in); singular values folded in evenly."""
    u, s, vh = torch.linalg.svd(w.to(torch.float64), full_matrices=False)
    r = min(r, s.numel())
    root = s[:r].clamp_min(0).sqrt()
    return u[:, :r] * root, root.unsqueeze(1) * vh[:r]


def _apply_whitener(stack: torch.Tensor, stats: dict[str, Any] | None,
                    enabled: bool) -> tuple[torch.Tensor, Whitener | None]:
    """Whiten the input mode when a calibration covariance is available."""
    if not enabled or not stats or "input_cov" not in stats:
        return stack.to(torch.float64), None
    wh = Whitener.from_covariance(torch.as_tensor(stats["input_cov"]))
    return torch.einsum("eoi,ij->eoj", stack.to(torch.float64), wh.L), wh


def _unwhiten(x: torch.Tensor, wh: Whitener | None) -> torch.Tensor:
    return x if wh is None else torch.einsum("eoi,ij->eoj", x, wh.L_inv)


@COMPRESSORS.register("per_expert_svd")
class PerExpertSVD(Compressor):
    """Each expert factorised independently.  No cross-expert sharing at all.

    The floor that any sharing-based method must beat; if it does not, the shared
    structure it assumes does not exist.
    """

    name = "per_expert_svd"

    def __init__(self, rank: int, whiten: bool = True, dtype: str = "float16") -> None:
        self.rank, self.whiten, self.dtype = rank, whiten, dtype

    def fit(self, stack: torch.Tensor, stats: dict[str, Any] | None = None) -> SlotCode:
        E, d_out, d_in = stack.shape
        sw, wh = _apply_whitener(stack, stats, self.whiten)
        A = torch.empty((E, d_out, self.rank), dtype=torch.float64)
        B = torch.empty((E, self.rank, d_in), dtype=torch.float64)
        for e in range(E):
            A[e], B[e] = _svd_lowrank(sw[e], self.rank)

        code = SlotCode(
            method=self.name, shape=(E, d_out, d_in),
            per_expert={"A": A, "B": B},
            dtypes={"A": self.dtype, "B": self.dtype},
            meta={"rank": self.rank, "whitened": wh is not None},
        )
        code.meta["_reconstruct"] = lambda c, wh=wh: _unwhiten(
            torch.einsum("eor,eri->eoi", c.per_expert["A"], c.per_expert["B"]), wh
        )
        return code


@COMPRESSORS.register("shared_base_delta")
class SharedBaseDelta(Compressor):
    """D^2-MoE-like: one shared base plus a per-expert low-rank delta.

    The base is an importance-weighted mean rather than a plain mean, so
    frequently routed experts pull it toward themselves -- reconstructing a rare
    expert badly costs less than reconstructing a hot one badly.
    """

    name = "shared_base_delta"

    def __init__(self, rank: int, whiten: bool = True, dtype: str = "float16") -> None:
        self.rank, self.whiten, self.dtype = rank, whiten, dtype

    def fit(self, stack: torch.Tensor, stats: dict[str, Any] | None = None) -> SlotCode:
        E, d_out, d_in = stack.shape
        sw, wh = _apply_whitener(stack, stats, self.whiten)

        w = torch.as_tensor(stats["importance"], dtype=torch.float64) if stats and \
            "importance" in stats else torch.ones(E, dtype=torch.float64)
        w = (w.clamp_min(0) / w.clamp_min(0).sum().clamp_min(1e-30))
        base = torch.einsum("e,eoi->oi", w, sw)

        resid = sw - base
        A = torch.empty((E, d_out, self.rank), dtype=torch.float64)
        B = torch.empty((E, self.rank, d_in), dtype=torch.float64)
        for e in range(E):
            A[e], B[e] = _svd_lowrank(resid[e], self.rank)

        code = SlotCode(
            method=self.name, shape=(E, d_out, d_in),
            shared={"base": base}, per_expert={"A": A, "B": B},
            dtypes={"base": self.dtype, "A": self.dtype, "B": self.dtype},
            meta={"rank": self.rank, "whitened": wh is not None},
        )
        code.meta["_reconstruct"] = lambda c, wh=wh: _unwhiten(
            c.shared["base"].unsqueeze(0)
            + torch.einsum("eor,eri->eoi", c.per_expert["A"], c.per_expert["B"]),
            wh,
        )
        return code


@COMPRESSORS.register("shared_basis")
class SharedBasis(Compressor):
    """MoBE-like: W_e = A_e B_e with B_e a mixture of m shared basis matrices.

    Note the accounting this produces: A_e is both per-expert *and*
    dictionary-sized, so it dominates the code at 80-86% of stored bytes.  That
    asymmetry -- visible in ``SlotCode.per_expert_share`` -- is what identifies
    A_e, not the mixture coefficients, as the remaining compression lever.

    Fitted by alternating least squares: with the basis fixed the coefficients
    are a linear solve and vice versa.  No gradients, so this stays in tier T1.
    """

    name = "shared_basis"

    def __init__(self, rank: int, n_basis: int = 8, iters: int = 8,
                 whiten: bool = True, dtype: str = "float16", seed: int = 0) -> None:
        self.rank, self.n_basis, self.iters = rank, n_basis, iters
        self.whiten, self.dtype, self.seed = whiten, dtype, seed

    def fit(self, stack: torch.Tensor, stats: dict[str, Any] | None = None) -> SlotCode:
        E, d_out, d_in = stack.shape
        r, m = self.rank, self.n_basis
        sw, wh = _apply_whitener(stack, stats, self.whiten)

        A = torch.empty((E, d_out, r), dtype=torch.float64)
        B = torch.empty((E, r, d_in), dtype=torch.float64)
        for e in range(E):
            A[e], B[e] = _svd_lowrank(sw[e], r)

        # Initialise the shared basis from the principal components of {B_e}.
        Bf = B.reshape(E, -1)
        _, _, vh = torch.linalg.svd(Bf, full_matrices=False)
        basis = vh[:m].reshape(m, r, d_in).clone()
        alpha = torch.zeros((E, m), dtype=torch.float64)

        for _ in range(self.iters):
            # coefficients: least squares of B_e onto the basis
            G = (basis.reshape(m, -1) @ basis.reshape(m, -1).T)
            G += 1e-9 * torch.eye(m, dtype=torch.float64) * G.diagonal().mean().clamp_min(1e-30)
            alpha = torch.linalg.solve(G, basis.reshape(m, -1) @ Bf.T).T
            # basis: least squares of {B_e} onto the coefficients
            H = alpha.T @ alpha
            H += 1e-9 * torch.eye(m, dtype=torch.float64) * H.diagonal().mean().clamp_min(1e-30)
            basis = torch.linalg.solve(H, alpha.T @ Bf).reshape(m, r, d_in)
            # refresh A_e against the reconstructed B_e
            Bhat = torch.einsum("em,mri->eri", alpha, basis)
            for e in range(E):
                gram = Bhat[e] @ Bhat[e].T
                gram += 1e-9 * torch.eye(r, dtype=torch.float64) * gram.diagonal().mean().clamp_min(1e-30)
                A[e] = torch.linalg.solve(gram, Bhat[e] @ sw[e].T).T
            Bf = torch.einsum("em,mri->eri", alpha, basis).reshape(E, -1)

        code = SlotCode(
            method=self.name, shape=(E, d_out, d_in),
            shared={"basis": basis}, per_expert={"A": A, "alpha": alpha},
            dtypes={"basis": self.dtype, "A": self.dtype, "alpha": "float32"},
            meta={"rank": r, "n_basis": m, "whitened": wh is not None},
        )
        code.meta["_reconstruct"] = lambda c, wh=wh: _unwhiten(
            torch.einsum(
                "eor,eri->eoi", c.per_expert["A"],
                torch.einsum("em,mri->eri", c.per_expert["alpha"], c.shared["basis"]),
            ),
            wh,
        )
        return code
