"""Byte-exact parameter accounting for MoE expert-table re-parameterizations.

This module exists to answer one question before any code is written:
*where do the parameters of a compressed MoE expert table actually live?*

Every method below stores an expert table {W_e}_{e<E}, W_e of shape (d_out, d_in),
for one (layer, matrix-type) slot.  We count storage exactly, in parameters, and
report the share contributed by each structural component.  The decisive quantity
is the **per-expert share**: the fraction of the compressed size that scales with
E rather than with d_out/d_in.  Any technique that compresses only the per-expert
table (e.g. coding the expert-mode factor with an orthogonal polynomial chart)
is upper-bounded by that share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb


@dataclass(frozen=True)
class Slot:
    """One (layer, matrix-type) expert table."""

    name: str
    E: int          # number of routed experts
    d_out: int
    d_in: int

    @property
    def dense(self) -> int:
        return self.E * self.d_out * self.d_in


@dataclass(frozen=True)
class Budget:
    """Storage decomposed into the parts that matter for the argument.

    ``per_expert``  scales with E (coefficient tables, coordinates, expert-mode factors)
    ``shared``      scales with d_out/d_in (dictionaries, bases, anchors, cores)
    ``residual``    per-expert corrective terms kept outside the model
    """

    method: str
    per_expert: int
    shared: int
    residual: int = 0
    notes: str = ""

    @property
    def total(self) -> int:
        return self.per_expert + self.shared + self.residual

    def ratio(self, slot: Slot) -> float:
        return self.total / slot.dense

    @property
    def per_expert_share(self) -> float:
        return self.per_expert / self.total if self.total else 0.0


def n_poly_features(q: int, p: int) -> int:
    """Number of total-degree-<=p multivariate polynomial features in q variables."""
    return comb(q + p, p)


# --- storage models -------------------------------------------------------


def per_expert_svd(slot: Slot, r: int) -> Budget:
    """Each expert factorized independently: W_e ~= A_e B_e."""
    return Budget(
        method=f"per-expert SVD (r={r})",
        per_expert=slot.E * r * (slot.d_out + slot.d_in),
        shared=0,
        notes="no cross-expert sharing at all",
    )


def shared_base_delta(slot: Slot, r: int) -> Budget:
    """D^2-MoE style: one global base + per-expert low-rank delta."""
    return Budget(
        method=f"shared base + low-rank delta (r={r})",
        per_expert=slot.E * r * (slot.d_out + slot.d_in),
        shared=slot.d_out * slot.d_in,
        notes="base amortized over E experts; delta still per-expert",
    )


def shared_basis(slot: Slot, r: int, m: int) -> Budget:
    """MoBE style: W_e = A_e B_e, with B_e a mixture of m shared basis matrices.

    A_e (d_out x r) stays expert-specific; B_e is generated from the shared basis
    via m mixture coefficients per expert.
    """
    return Budget(
        method=f"shared basis / MoBE-like (r={r}, m={m})",
        per_expert=slot.E * slot.d_out * r + slot.E * m,
        shared=m * r * slot.d_in,
        notes="A_e dominates: it is per-expert AND dictionary-sized",
    )


def tucker_joint(slot: Slot, rE: int, rO: int, rI: int) -> Budget:
    """TD-MoE style: Tucker over the (expert, out, in) tensor."""
    return Budget(
        method=f"joint Tucker (rE={rE}, rO={rO}, rI={rI})",
        per_expert=slot.E * rE,
        shared=rE * rO * rI + slot.d_out * rO + slot.d_in * rI,
        notes="U_E is the ONLY per-expert term",
    )


def tucker_poly_chart(slot: Slot, rE: int, rO: int, rI: int, q: int, p: int) -> Budget:
    """POEM-Atlas as written in the report: replace U_E by a polynomial chart.

    U_E[e,:] = Phi(z_e) Theta, with z_e in R^q and total degree <= p.
    """
    M = n_poly_features(q, p)
    return Budget(
        method=f"Tucker + Legendre chart on U_E (rE={rE}, q={q}, p={p}, M={M})",
        per_expert=slot.E * q,
        shared=rE * rO * rI + slot.d_out * rO + slot.d_in * rI + M * rE,
        notes="chart replaces E*rE with E*q + M*rE",
    )


def local_atlas_matrix(slot: Slot, K: int, r: int, m: int) -> Budget:
    """LorExperts/POEM-flavoured local charts: K communities, each with a dictionary."""
    dict_per_community = m * r * slot.d_in + slot.d_out * r
    return Budget(
        method=f"local atlas, matrix factors (K={K}, r={r}, m={m})",
        per_expert=slot.E * (r + m),
        shared=K * dict_per_community,
        notes="K local dictionaries; per-expert part is coefficients only",
    )


def stack(slot: Slot, L: int) -> Slot:
    """Treat L layers of the same matrix type as one slot (for depth-axis methods)."""
    return Slot(name=f"{slot.name} x{L}L", E=slot.E * L, d_out=slot.d_out, d_in=slot.d_in)


def depth_chart(slot: Slot, L: int, r: int, m: int, p_depth: int) -> Budget:
    """Cross-LAYER chart: the dictionary is a degree-p polynomial in depth.

    Counts the whole *stack* of L layers for this matrix type.  The shared
    dictionary, normally stored L times, is instead generated by evaluating
    (p_depth + 1) Legendre coefficient tensors at each layer's depth coordinate.
    Compare against ``stack(slot, L)``.
    """
    dict_per_layer = m * r * slot.d_in + slot.d_out * r
    return Budget(
        method=f"depth chart, {L} layers (r={r}, m={m}, p_depth={p_depth})",
        per_expert=L * slot.E * m,
        shared=(p_depth + 1) * dict_per_layer,
        notes=f"dictionary stored {L}x -> generated from {p_depth + 1} coefficient sets",
    )
