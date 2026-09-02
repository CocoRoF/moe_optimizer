"""Neuron permutation alignment.

A transformer FFN expert is invariant to a joint permutation of its intermediate
units: permuting the rows of gate/up and the columns of down leaves the function
unchanged.  Two experts that implement nearly the same function can therefore
have near-zero raw weight cosine similarity.

Any similarity measured on raw weights is consequently a measurement of an
arbitrary coordinate choice, not of function.  LorExperts' report that experts are
"near-orthogonal in raw weight space" is exactly what this symmetry predicts, and
is not by itself evidence against shared structure.  We remove the symmetry
before drawing any geometric conclusion.

The optimal permutation is a linear assignment problem on the neuron-similarity
matrix, solved exactly by Jonker-Volgenant in O(n^3).
"""

from __future__ import annotations

import torch


def _row_normalise(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def permutation_alignment(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Permutation ``perm`` of b's rows maximising similarity to a.

    Both are (n_units, d).  Returns an index tensor such that ``b[perm]`` is the
    best row-wise match to ``a``.  For gate/up matrices the "units" are the
    intermediate neurons (rows); for down, pass the transpose.
    """
    from scipy.optimize import linear_sum_assignment

    sim = _row_normalise(a.to(torch.float32)) @ _row_normalise(b.to(torch.float32)).T
    rows, cols = linear_sum_assignment(-sim.numpy())
    perm = torch.empty(a.shape[0], dtype=torch.long)
    perm[torch.from_numpy(rows)] = torch.from_numpy(cols)
    return perm


def align_expert_to(
    reference: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Permute ``target``'s intermediate units to match ``reference``.

    Both dicts hold the three matrices of one expert with keys gate/up/down in
    (d_out, d_in) orientation.  The permutation is derived from gate and up
    jointly -- they share the neuron axis -- and then applied consistently to all
    three, so the expert's function is provably unchanged.
    """
    need = {"gate", "up", "down"}
    if not need <= reference.keys() or not need <= target.keys():
        raise KeyError(f"both experts need keys {sorted(need)}")

    ref_units = torch.cat([reference["gate"], reference["up"]], dim=1)
    tgt_units = torch.cat([target["gate"], target["up"]], dim=1)
    perm = permutation_alignment(ref_units, tgt_units)

    return (
        {
            "gate": target["gate"][perm],
            "up": target["up"][perm],
            "down": target["down"][:, perm],
        },
        perm,
    )
