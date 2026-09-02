"""Gate G0 -- does the expert dictionary rotate smoothly with layer depth?

This is the first experiment the project should run, and the one that decides
whether the orthogonal-polynomial idea survives.  It needs **weights only**: no
forward pass, no calibration data, no GPU.

Method
------
For one matrix type, the "dictionary" of layer l is the dominant subspace shared
by all E experts of that layer.  We never form the (d_out, E*d_in) stack: the
same subspace is the top eigenspace of the Gram accumulation

    M_out(l) = sum_e W_{l,e} W_{l,e}^T          (d_out x d_out)
    M_in(l)  = sum_e W_{l,e}^T W_{l,e}          (d_in  x d_in)

which is cheap to build one expert at a time and is invariant to the neuron
permutation symmetry on the *opposite* mode.  We then measure principal angles
between the dictionaries of every pair of layers.

Reading the result
------------------
``affinity[l, l']`` is mean cos^2 of the principal angles, in [0, 1].

  * decays smoothly with |l - l'|         -> global polynomial chart over depth
  * block-diagonal with sharp jumps       -> piecewise chart; blocks are the charts
  * flat/near-zero off the diagonal       -> no depth structure; drop the depth axis

The third outcome kills the redirection proposed in
``my_paper/REVIEW_and_REDIRECTION_2026-09-02.md``; the first two support it, and
distinguishing them sets the polynomial degree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..io.checkpoint import ExpertStore
from ..types import MatrixType, Slot
from .subspace import principal_angles, subspace_affinity


@dataclass
class DepthProfile:
    """Result of the G0 measurement for one matrix type and one side."""

    matrix: MatrixType
    side: str                      # "out" or "in"
    rank: int
    layers: list[int]
    affinity: torch.Tensor         # (L, L) mean cos^2 of principal angles
    energy: torch.Tensor           # (L, rank) eigenvalue energy fractions
    meta: dict = field(default_factory=dict)

    def neighbour_affinity(self, gap: int = 1) -> torch.Tensor:
        """Affinity between layers ``gap`` apart -- the smoothness signal."""
        n = len(self.layers)
        if gap >= n:
            return torch.empty(0)
        return torch.tensor([self.affinity[i, i + gap] for i in range(n - gap)])

    def decay_curve(self) -> torch.Tensor:
        """Mean affinity as a function of layer gap, gap = 0..L-1."""
        n = len(self.layers)
        return torch.tensor([float(self.neighbour_affinity(g).mean()) if g < n else 0.0
                             for g in range(n)])

    def verdict(self, smooth_thresh: float = 0.7, dead_thresh: float = 0.25) -> str:
        """A blunt three-way call on the gate.  Judgement still belongs to a human."""
        curve = self.decay_curve()
        if curve.numel() < 3:
            return "inconclusive (too few layers)"
        near = float(curve[1])
        far = float(curve[max(1, len(curve) // 2)])
        if near < dead_thresh:
            return f"DEAD: adjacent-layer affinity {near:.3f} < {dead_thresh} -- drop the depth axis"
        if near >= smooth_thresh and far >= dead_thresh:
            return f"SMOOTH: adjacent {near:.3f}, mid-range {far:.3f} -- global chart plausible"
        return (f"PIECEWISE: adjacent {near:.3f} but mid-range {far:.3f} "
                f"-- use local charts over depth")


def layer_dictionary(
    store: ExpertStore, slot: Slot, rank: int, side: str = "out",
    gram_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dominant subspace shared by all experts of one slot.

    Returns ``(basis, energy_fractions)`` where basis is (d, rank) orthonormal.

    Accumulates the Gram matrix one expert at a time, so peak memory is one
    expert plus a (d, d) accumulator -- never the full table.

    ``gram_dtype=torch.float32`` is ~4x faster on CPU and is accurate to ~1e-6
    in the top-rank subspace, which is far below the affinity differences the
    G0 verdict turns on; the eigendecomposition itself always runs in float64.
    """
    if side not in ("out", "in"):
        raise ValueError(f"side must be 'out' or 'in', got {side!r}")
    d_out, d_in = store.arch.shape(slot.matrix)
    d = d_out if side == "out" else d_in
    gram = torch.zeros((d, d), dtype=gram_dtype)

    for e in range(store.arch.n_experts):
        w = store.expert(slot, e).to(gram_dtype)
        gram += (w @ w.T) if side == "out" else (w.T @ w)
        del w

    # Gram is symmetric PSD; eigh is stable and cheaper than svd here.
    evals, evecs = torch.linalg.eigh(gram.to(torch.float64))
    evals = evals.flip(0).clamp_min(0)
    evecs = evecs.flip(1)
    r = min(rank, d)
    total = evals.sum().clamp_min(1e-30)
    return evecs[:, :r].contiguous(), (evals[:r] / total)


def depth_profile(
    store: ExpertStore,
    matrix: MatrixType,
    rank: int = 64,
    side: str = "out",
    layers: list[int] | None = None,
    progress: bool = True,
    gram_dtype: torch.dtype = torch.float64,
) -> DepthProfile:
    """Run the G0 measurement across all MoE layers."""
    layers = list(layers if layers is not None else store.arch.moe_layers)
    bases: list[torch.Tensor] = []
    energies: list[torch.Tensor] = []

    for n, layer in enumerate(layers):
        if progress:
            print(f"  [{n + 1:>3}/{len(layers)}] layer {layer:>3} dictionary ({matrix}, {side})",
                  flush=True)
        b, en = layer_dictionary(store, Slot(layer, matrix), rank=rank, side=side,
                                 gram_dtype=gram_dtype)
        bases.append(b)
        energies.append(en)

    n = len(bases)
    aff = torch.eye(n, dtype=torch.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a = subspace_affinity(bases[i], bases[j])
            aff[i, j] = aff[j, i] = a

    return DepthProfile(
        matrix=matrix,
        side=side,
        rank=rank,
        layers=layers,
        affinity=aff,
        energy=torch.stack(energies),
        meta={"model": store.arch.model_id, "n_experts": store.arch.n_experts},
    )


def angle_spectrum(
    store: ExpertStore, matrix: MatrixType, l1: int, l2: int, rank: int = 64, side: str = "out"
) -> torch.Tensor:
    """Full principal-angle spectrum between two layers, in degrees.

    Useful when the scalar affinity is ambiguous: a subspace that agrees on 40 of
    64 directions and is orthogonal on the rest tells a very different story from
    one that is uniformly half-aligned, and only the spectrum distinguishes them.
    """
    a, _ = layer_dictionary(store, Slot(l1, matrix), rank, side)
    b, _ = layer_dictionary(store, Slot(l2, matrix), rank, side)
    return torch.rad2deg(principal_angles(a, b))
