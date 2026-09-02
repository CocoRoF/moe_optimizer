"""The depth chart: generate the per-layer dictionary from a polynomial in depth.

This is the method proposed in ``my_paper/REVIEW_and_REDIRECTION_2026-09-02.md``
to replace the report's original design, and the reasoning is worth stating
plainly because it is the whole point of the project:

An orthogonal-polynomial chart over the *expert* index cannot compress, because
the per-expert coefficient table is E/(E+D) of the code and D >> E always.  The
same chart over *layer depth* attacks the dictionary itself, which is stored L
times -- L = 16 for OLMoE, 48 for Qwen3-30B-A3B.  Depth also has what the expert
index lacks: a genuine total order, so no coordinate has to be learned, and the
usual objection that "experts are an unordered set" simply does not apply.

    U_l  ~=  sum_{p=0..P} Theta_p * Ptilde_p(tau_l)          tau_l in [-1, 1]

Storage falls from L*(d_out + d_in)*r to (P+1)*(d_out + d_in)*r, plus per-expert
coefficients that were already negligible.

The gauge problem
-----------------
The subspace of layer l is well defined; the *basis* returned by an SVD is not --
it is fixed only up to an r x r orthogonal rotation.  Fitting a polynomial to raw
SVD bases would therefore be fitting to arbitrary gauge noise and would fail even
if the underlying subspaces drifted perfectly smoothly.

We remove the gauge by sequential orthogonal Procrustes: each layer's basis is
rotated to best match the previous layer's, and the per-expert coefficients are
counter-rotated so that every expert's reconstruction is exactly unchanged.  Only
then is the polynomial fitted.  Skipping this step is the most likely way to get
a false negative on gate G0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..factorize.base import DTYPE_BYTES, SlotCode
from ..factorize.chart import fit_chart, multivariate_vandermonde, truncate, weighted_orthogonal_basis


def _orthonormalise(x: torch.Tensor) -> torch.Tensor:
    """QR-orthonormalise a batch of bases, with a deterministic sign convention.

    The chart output is not orthonormal (a polynomial in depth does not preserve
    the Stiefel manifold), and the diagonal coupling is only meaningful against an
    orthonormal pair.  The sign fix makes encoder and decoder agree exactly.
    """
    q, r = torch.linalg.qr(x.to(torch.float64))
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign.unsqueeze(-2)


def procrustes_rotation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Orthogonal R minimising ||a R - b||_F, for a, b with orthonormal columns.

    R = U V^T from the SVD of a^T b.  This is the exact solution, not an
    approximation, and it preserves orthonormality exactly.
    """
    u, _, vh = torch.linalg.svd(a.T.to(torch.float64) @ b.to(torch.float64))
    return u @ vh


def gauge_align(bases: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Rotate a sequence of orthonormal bases into a common, smoothly varying gauge.

    Returns ``(aligned, rotations)`` with ``aligned[l] = bases[l] @ rotations[l]``.
    Alignment is sequential (each to its predecessor) rather than all-to-first, so
    the chain tracks slow drift even when the endpoints are far apart.
    """
    if not bases:
        return [], []
    aligned = [bases[0]]
    rots = [torch.eye(bases[0].shape[1], dtype=torch.float64)]
    for b in bases[1:]:
        r = procrustes_rotation(b, aligned[-1])
        rots.append(r)
        aligned.append(b @ r)
    return aligned, rots


def depth_coordinates(layers: list[int]) -> torch.Tensor:
    """Map layer indices to [-1, 1].  A single layer maps to 0."""
    t = torch.tensor(layers, dtype=torch.float64)
    if t.numel() < 2:
        return torch.zeros_like(t)
    return 2 * (t - t.min()) / (t.max() - t.min()) - 1


@dataclass
class DepthAtlasCode:
    """A compressed *stack* of expert tables -- all layers of one matrix type."""

    matrix: str
    layers: list[int]
    shape: tuple[int, int, int]                # per-layer (E, d_out, d_in)
    Theta_U: torch.Tensor                      # (P+1, d_out, r)
    Theta_V: torch.Tensor                      # (P+1, d_in,  r)
    R_inv: torch.Tensor                        # (P+1, P+1) raw-basis conversion
    coeffs: torch.Tensor                       # (L, E, r) per-expert, per-layer
    anchors: torch.Tensor | None = None        # (L, d_out, d_in), or None if charted
    Theta_A: torch.Tensor | None = None        # (P+1, d_out, d_in) charted anchor
    coupling: str = "full"
    degree: int = 4
    dtypes: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- accounting -------------------------------------------------------

    def _b(self, t: torch.Tensor | None, key: str) -> float:
        return 0.0 if t is None else t.numel() * DTYPE_BYTES[self.dtypes.get(key, "float16")]

    @property
    def component_bytes(self) -> dict[str, float]:
        return {
            "chart": self._b(self.Theta_U, "Theta") + self._b(self.Theta_V, "Theta")
            + self._b(self.R_inv, "R_inv"),
            "per_expert": self._b(self.coeffs, "coeffs"),
            "anchors": self._b(self.anchors, "anchors") + self._b(self.Theta_A, "Theta"),
        }

    @property
    def nbytes(self) -> float:
        return sum(self.component_bytes.values())

    def dense_bytes(self, dtype: str = "bfloat16") -> float:
        e, o, i = self.shape
        return len(self.layers) * e * o * i * DTYPE_BYTES[dtype]

    def ratio(self, dtype: str = "bfloat16") -> float:
        return self.nbytes / self.dense_bytes(dtype)

    # -- reconstruction ---------------------------------------------------

    def dictionaries(self, degree: int | None = None,
                     rank: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the chart at every layer, optionally truncating the degree.

        Two progressive knobs, and they are not equally well behaved:

        ``rank``   exactly nested.  The dictionaries are orthonormal and the
                   coefficients are independent per column, so dropping trailing
                   columns is precisely the optimal lower-rank code.
        ``degree`` approximately nested.  A lower degree perturbs the dictionary
                   itself, so the stored coefficients -- fitted against the full
                   -degree dictionary -- are no longer optimal for it.  The
                   degradation is graceful but this is not a free knob, and the
                   distinction should be reported rather than glossed.
        """
        tau = depth_coordinates(self.layers)
        V = multivariate_vandermonde(tau, self.degree)
        Psi = V @ self.R_inv
        tu, tv = self.Theta_U, self.Theta_V
        if degree is not None and degree < self.degree:
            keep = degree + 1                      # univariate: one feature per degree
            Psi, tu, tv = Psi[:, :keep], tu[:keep], tv[:keep]
        U = torch.einsum("lp,por->lor", Psi, tu)
        V = torch.einsum("lp,pir->lir", Psi, tv)
        if rank is not None and rank < U.shape[-1]:
            U, V = U[..., :rank], V[..., :rank]
        return _orthonormalise(U), _orthonormalise(V)

    def reconstruct(self, degree: int | None = None,
                    rank: int | None = None) -> torch.Tensor:
        """(L, E, d_out, d_in)."""
        U, V = self.dictionaries(degree, rank)
        c = self.coeffs
        if rank is not None:
            c = c[..., :rank] if self.coupling == "diag" else c[..., :rank, :rank]
        out = (torch.einsum("lor,ler,lir->leoi", U, c, V) if self.coupling == "diag"
               else torch.einsum("lor,lers,lis->leoi", U, c, V))
        anc = self.anchors
        if anc is None and self.Theta_A is not None:
            tau = depth_coordinates(self.layers)
            Psi = multivariate_vandermonde(tau, self.degree) @ self.R_inv
            anc = torch.einsum("lp,poi->loi", Psi, self.Theta_A)
        return out if anc is None else out + anc.unsqueeze(1)

    def summary(self) -> dict[str, Any]:
        cb = self.component_bytes
        return {"matrix": self.matrix, "n_layers": len(self.layers), "degree": self.degree,
                "bytes": self.nbytes, "ratio_vs_bf16": self.ratio(),
                **{f"bytes_{k}": v for k, v in cb.items()},
                **{k: v for k, v in self.meta.items() if not k.startswith("_")}}


class DepthAtlas:
    """Fit a depth chart across all layers of one matrix type.

    Unlike the per-slot compressors this consumes the whole stack, so it is not a
    ``Compressor``: the object it compresses is different, and pretending
    otherwise would let it be scored against per-slot methods on incomparable
    footings.  Compare via bytes-per-original-parameter instead.
    """

    name = "depth_atlas"

    def __init__(self, rank: int = 64, degree: int = 4, anchor: bool = True,
                 coupling: str = "full", chart_anchor: bool = True,
                 dtype: str = "float16") -> None:
        if coupling not in ("full", "diag"):
            raise ValueError(f"coupling must be 'full' or 'diag', got {coupling!r}")
        self.rank, self.degree, self.anchor, self.dtype = rank, degree, anchor, dtype
        self.coupling, self.chart_anchor = coupling, chart_anchor

    def fit(self, stacks: dict[int, torch.Tensor], matrix: str = "gate") -> DepthAtlasCode:
        """``stacks`` maps layer index -> (E, d_out, d_in) table."""
        layers = sorted(stacks)
        if not layers:
            raise ValueError("no layers supplied")
        r = self.rank
        E, d_out, d_in = stacks[layers[0]].shape

        raw_U: list[torch.Tensor] = []
        raw_V: list[torch.Tensor] = []
        anchors: list[torch.Tensor] = []
        resids: list[torch.Tensor] = []

        for l in layers:
            s = stacks[l].to(torch.float64)
            if s.shape != (E, d_out, d_in):
                raise ValueError(f"layer {l} shape {tuple(s.shape)} != {(E, d_out, d_in)}")
            a = s.mean(0) if self.anchor else torch.zeros(d_out, d_in, dtype=torch.float64)
            res = s - a
            g_out = torch.einsum("eoi,epi->op", res, res)
            g_in = torch.einsum("eoi,eoj->ij", res, res)
            raw_U.append(torch.linalg.eigh(g_out)[1].flip(1)[:, :r].contiguous())
            raw_V.append(torch.linalg.eigh(g_in)[1].flip(1)[:, :r].contiguous())
            anchors.append(a)
            resids.append(res)

        # Remove the SVD gauge before fitting anything.
        U_al, rot_U = gauge_align(raw_U)
        V_al, rot_V = gauge_align(raw_V)

        tau = depth_coordinates(layers)
        basis = weighted_orthogonal_basis(tau, self.degree)
        stackU = torch.stack(U_al).reshape(len(layers), -1)
        stackV = torch.stack(V_al).reshape(len(layers), -1)
        Theta_U, fit_U = fit_chart(stackU, basis)
        Theta_V, fit_V = fit_chart(stackV, basis)

        # Coefficients must be projected onto the dictionaries the *decoder* will
        # produce, not onto the exact ones.  Fitting against the exact bases and
        # decoding against the charted ones leaves a rotation mismatch that the
        # diagonal coupling amplifies badly -- the reconstruction error ends up an
        # order of magnitude worse than the chart residual would suggest.
        U_hat = _orthonormalise(fit_U.reshape(len(layers), d_out, r))
        V_hat = _orthonormalise(fit_V.reshape(len(layers), d_in, r))
        cores = torch.stack([
            torch.einsum("or,eoi,is->ers", U_hat[i], resids[i], V_hat[i])
            for i in range(len(layers))
        ])
        coeffs = torch.diagonal(cores, dim1=2, dim2=3) if self.coupling == "diag" else cores

        anchor_stack = torch.stack(anchors) if self.anchor else None
        Theta_A = None
        anchor_resid = 0.0
        if anchor_stack is not None and self.chart_anchor and len(layers) > self.degree + 1:
            # The anchor is L * d_out * d_in -- by far the largest term in the code.
            # Charting it over depth is where most of the depth-axis saving comes from.
            Theta_A, fit_A = fit_chart(anchor_stack.reshape(len(layers), -1), basis)
            anchor_resid = float((anchor_stack.reshape(len(layers), -1) - fit_A).norm()
                                 / anchor_stack.norm().clamp_min(1e-30))
            Theta_A = Theta_A.reshape(-1, d_out, d_in)
            anchor_stack = None

        code = DepthAtlasCode(
            matrix=matrix, layers=layers, shape=(E, d_out, d_in),
            Theta_U=Theta_U.reshape(-1, d_out, r), Theta_V=Theta_V.reshape(-1, d_in, r),
            R_inv=basis.R_inv, coeffs=coeffs,
            anchors=anchor_stack, Theta_A=Theta_A, coupling=self.coupling,
            degree=self.degree,
            dtypes={"Theta": self.dtype, "R_inv": "float32",
                    "coeffs": "float32", "anchors": self.dtype},
            meta={
                "rank": r,
                "chart_rel_resid_U": float(
                    (torch.stack(U_al).reshape(len(layers), -1) - fit_U).norm()
                    / torch.stack(U_al).norm().clamp_min(1e-30)),
                "chart_rel_resid_V": float(
                    (torch.stack(V_al).reshape(len(layers), -1) - fit_V).norm()
                    / torch.stack(V_al).norm().clamp_min(1e-30)),
                "gauge_aligned": True, "coupling": self.coupling,
                "chart_rel_resid_anchor": anchor_resid,
            },
        )
        return code
