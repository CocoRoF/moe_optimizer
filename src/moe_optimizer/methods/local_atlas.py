"""POEM-Atlas: local functional charts over expert communities.

Structure, per community c (report section 12, corrected):

    W_e  ~=  Anchor_c  +  U_c M_e V_c^T  +  S_e

``U_c``, ``V_c``  local dictionary shared by the community  -- O(K * (d_out + d_in) * r)
``M_e``           per-expert coefficients, diagonal or full -- O(E * r) or O(E * r^2)
``S_e``           optional sparse outlier residual

Two design decisions here are consequences of the review in
``my_paper/REVIEW_and_REDIRECTION_2026-09-02.md``:

1.  The factorisation is **matrix-based**, not Tucker.  arXiv:2606.03465 reports
    that joint tensor decomposition (TD-MoE) substantially underperforms the
    matrix baseline (MoBE) on exactly the models we target, and traces it to a
    Frobenius/operator-norm mismatch.  A Tucker arm is kept for ablation but is
    not the default.

2.  ``expert_chart`` re-codes the per-expert coefficients through an orthogonal
    polynomial in a learned coordinate.  It is **off by default and it does not
    reduce size**: per-expert bytes are bounded by E/(E+D) of the code, which is
    under 0.1% for every production MoE.  It is retained because the nested
    truncation of an orthogonal basis is what makes the artifact *progressive* --
    one checkpoint, many operating points -- and because the accounting it emits
    is the direct experimental demonstration of that bound.
"""

from __future__ import annotations

from typing import Any

import torch

from ..community.cluster import cluster_experts
from ..factorize.base import Compressor, SlotCode
from ..factorize.chart import (fit_chart, multivariate_vandermonde,
                               weighted_orthogonal_basis)
from ..registry import COMPRESSORS
from .baselines import _apply_whitener, _unwhiten


def _hosvd_bases(resid: torch.Tensor, r_out: int, r_in: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Dominant output/input subspaces of a (n, d_out, d_in) residual stack.

    Built from Gram accumulations rather than unfoldings, which avoids
    materialising the (d_out, n*d_in) matrix.
    """
    n, d_out, d_in = resid.shape
    g_out = torch.einsum("eoi,epi->op", resid, resid)
    g_in = torch.einsum("eoi,eoj->ij", resid, resid)
    eo, vo = torch.linalg.eigh(g_out)
    ei, vi = torch.linalg.eigh(g_in)
    return (vo.flip(1)[:, : min(r_out, d_out)].contiguous(),
            vi.flip(1)[:, : min(r_in, d_in)].contiguous())


def _learn_coordinates(coeffs: torch.Tensor, q: int) -> torch.Tensor:
    """A q-dimensional coordinate for each expert, from its coefficient vector.

    PCA of the coefficients, rescaled to [-1, 1].  The report proposes a graph
    embedding of the functional affinity instead; that is strictly better
    motivated but needs calibration traces, so PCA is the weights-only default
    and the graph variant is a drop-in replacement.
    """
    x = coeffs.to(torch.float64)
    x = x - x.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    z = x @ vh[: min(q, vh.shape[0])].T
    lo, hi = z.min(0).values, z.max(0).values
    return (2 * (z - lo) / (hi - lo).clamp_min(1e-12) - 1)


@COMPRESSORS.register("local_atlas")
class LocalAtlas(Compressor):
    name = "local_atlas"

    def __init__(
        self,
        n_communities: int = 8,
        rank: int = 64,
        coupling: str = "full",          # "full" (r^2 per expert) or "diag" (r)
        affinity: str = "weight",
        clusterer: str = "spectral",
        anchor: str = "weighted_mean",   # or "dominant" or "none"
        whiten: bool = True,
        expert_chart: bool = False,
        chart_q: int = 2,
        chart_degree: int = 3,
        dtype: str = "float16",
        seed: int = 0,
    ) -> None:
        # "full" is the default because HOSVD determines each mode's subspace only
        # up to an unrelated rotation, which the diagonal form cannot absorb --
        # see finding F3.  "diag" is retained for the ablation, and costs ~1.6
        # points of relative error at matched bytes on OLMoE layer 0.
        if coupling not in ("diag", "full"):
            raise ValueError(f"coupling must be 'diag' or 'full', got {coupling!r}")
        self.n_communities, self.rank, self.coupling = n_communities, rank, coupling
        self.affinity, self.clusterer, self.anchor = affinity, clusterer, anchor
        self.whiten, self.dtype, self.seed = whiten, dtype, seed
        self.expert_chart, self.chart_q, self.chart_degree = expert_chart, chart_q, chart_degree

    # -- anchors ----------------------------------------------------------

    def _anchor(self, members: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.anchor == "none":
            return torch.zeros_like(members[0])
        if self.anchor == "dominant":
            return members[int(w.argmax())].clone()
        return torch.einsum("e,eoi->oi", w / w.sum().clamp_min(1e-30), members)

    # -- fit --------------------------------------------------------------

    def fit(self, stack: torch.Tensor, stats: dict[str, Any] | None = None) -> SlotCode:
        E, d_out, d_in = stack.shape
        r = self.rank
        sw, wh = _apply_whitener(stack, stats, self.whiten)

        imp = torch.as_tensor(stats["importance"], dtype=torch.float64) if stats and \
            "importance" in stats else torch.ones(E, dtype=torch.float64)
        imp = imp.clamp_min(0)

        labels = cluster_experts(sw, self.n_communities, signal=self.affinity,
                                 algorithm=self.clusterer, stats=stats, seed=self.seed)
        k = int(labels.max()) + 1

        anchors = torch.zeros((k, d_out, d_in), dtype=torch.float64)
        U = torch.zeros((k, d_out, r), dtype=torch.float64)
        V = torch.zeros((k, d_in, r), dtype=torch.float64)
        M = (torch.zeros((E, r), dtype=torch.float64) if self.coupling == "diag"
             else torch.zeros((E, r, r), dtype=torch.float64))

        for c in range(k):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            members = sw[idx]
            anchors[c] = self._anchor(members, imp[idx])
            resid = members - anchors[c]
            uo, vi = _hosvd_bases(resid, r, r)
            U[c, :, : uo.shape[1]] = uo
            V[c, :, : vi.shape[1]] = vi
            # With orthonormal U, V the optimal coefficient block is the projection.
            proj = torch.einsum("or,eoi,is->ers", U[c], resid, V[c])
            M[idx] = torch.diagonal(proj, dim1=1, dim2=2) if self.coupling == "diag" else proj

        per_expert: dict[str, torch.Tensor] = {"M": M}
        dtypes = {"anchors": self.dtype, "U": self.dtype, "V": self.dtype,
                  "M": "float32", "labels": "int8"}
        meta: dict[str, Any] = {
            "n_communities": k, "rank": r, "coupling": self.coupling,
            "affinity": self.affinity, "anchor": self.anchor,
            "whitened": wh is not None, "expert_chart": self.expert_chart,
        }

        chart: dict[str, torch.Tensor] = {}
        if self.expert_chart:
            if self.coupling != "diag":
                raise NotImplementedError("expert_chart currently supports coupling='diag'")
            Z = _learn_coordinates(M, self.chart_q)
            basis = weighted_orthogonal_basis(Z, self.chart_degree, weights=imp)
            Theta, fitted = fit_chart(M, basis, weights=imp)
            delta = M - fitted
            chart = {"Z": Z, "Theta": Theta, "R_inv": basis.R_inv}
            per_expert = {"Z": Z, "delta": delta}
            dtypes |= {"Z": "float32", "Theta": "float32", "delta": "float32",
                       "R_inv": "float32"}
            meta |= {
                "chart_q": self.chart_q, "chart_degree": self.chart_degree,
                "chart_features": basis.n_features,
                "chart_rel_resid": float(delta.norm() / M.norm().clamp_min(1e-30)),
                "chart_orth_err": basis.orthogonality_error(),
            }

        code = SlotCode(
            method=self.name, shape=(E, d_out, d_in),
            shared={"anchors": anchors, "U": U, "V": V,
                    **({"Theta": chart["Theta"], "R_inv": chart["R_inv"]}
                       if chart else {})},
            per_expert={**per_expert, "labels": labels.to(torch.int16)},
            dtypes=dtypes, meta=meta,
        )
        code.meta["_reconstruct"] = self._make_reconstructor(wh, k)
        return code

    def _make_reconstructor(self, wh, k: int):
        coupling, q, degree = self.coupling, self.chart_q, self.chart_degree
        use_chart = self.expert_chart

        def rec(c: SlotCode) -> torch.Tensor:
            E, d_out, d_in = c.shape
            labels = c.per_expert["labels"].long()
            if use_chart:
                Z = c.per_expert["Z"]
                # Psi = V @ R_inv, so this reproduces the fitted basis exactly
                # regardless of the importance weights used at fit time.
                Psi = multivariate_vandermonde(Z, degree) @ c.shared["R_inv"]
                M = Psi @ c.shared["Theta"] + c.per_expert["delta"]
            else:
                M = c.per_expert["M"]

            out = torch.empty((E, d_out, d_in), dtype=torch.float64)
            for cc in range(k):
                idx = (labels == cc).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                U, V = c.shared["U"][cc], c.shared["V"][cc]
                block = (torch.einsum("or,er,ir->eoi", U, M[idx], V) if coupling == "diag"
                         else torch.einsum("or,ers,is->eoi", U, M[idx], V))
                out[idx] = c.shared["anchors"][cc].unsqueeze(0) + block
            return _unwhiten(out, wh)

        return rec
