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


# --- report ---------------------------------------------------------------

MODELS = {
    # name: (E, d_model, d_ff_expert, n_moe_layers)
    "OLMoE-1B-7B":     (64, 2048, 1024, 16),
    "Qwen3-30B-A3B":   (128, 2048, 768, 48),
    "DeepSeekMoE-16B": (64, 2048, 1408, 27),
    "Mixtral-8x7B":    (8, 4096, 14336, 32),
}


def hdr(t): print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")


def row(b, slot):
    print(f"  {b.method:<58} {b.total/1e6:>9.2f}M  {b.ratio(slot)*100:>7.2f}%"
          f" {b.per_expert_share*100:>10.2f}%")


def report() -> None:
    hdr("1. PER-EXPERT SHARE OF COMPRESSED STORAGE  (one gate/up slot, E x d_ff x d_model)")
    print("  The last column is the ceiling on anything that compresses only the")
    print("  per-expert table -- which is exactly what a chart on U_E does.\n")
    print(f"  {'method':<58} {'stored':>10}  {'of dense':>7} {'per-expert':>11}")

    for name, (E, d_model, d_ff, _L) in MODELS.items():
        slot = Slot(name=name, E=E, d_out=d_ff, d_in=d_model)
        print(f"\n  --- {name}: E={E}, W_e = ({d_ff} x {d_model}), dense = {slot.dense/1e6:.1f}M params")
        r = max(64, d_ff // 4)
        for b in (
            per_expert_svd(slot, r=r),
            shared_base_delta(slot, r=r),
            shared_basis(slot, r=r, m=8),
            tucker_joint(slot, rE=E // 2, rO=d_ff // 2, rI=d_model // 2),
            local_atlas_matrix(slot, K=8, r=r, m=8),
        ):
            row(b, slot)

    hdr("2. WHAT THE LEGENDRE CHART ON U_E ACTUALLY BUYS  (report sections 7.3 / 11.7)")
    print("  Replacing the free lookup U_E (E x rE) with Phi(z_e)Theta.\n")
    print(f"  {'model':<18} {'Tucker total':>13} {'U_E':>10} {'U_E share':>10}"
          f" {'chart total':>13} {'net saving':>11}")
    for name, (E, d_model, d_ff, _L) in MODELS.items():
        slot = Slot(name=name, E=E, d_out=d_ff, d_in=d_model)
        rE, rO, rI = E // 2, d_ff // 2, d_model // 2
        base = tucker_joint(slot, rE, rO, rI)
        chart = tucker_poly_chart(slot, rE, rO, rI, q=2, p=3)
        saving = (base.total - chart.total) / base.total
        print(f"  {name:<18} {base.total/1e6:>12.2f}M {base.per_expert:>10,}"
              f" {base.per_expert_share*100:>9.3f}% {chart.total/1e6:>12.2f}M"
              f" {saving*100:>10.3f}%")

    hdr("3. THE STRUCTURAL REASON  (independent of method and of rank)")
    print("  Any 'shared dictionary + per-expert coefficients' scheme stores")
    print("      per-expert  = E * K        shared = K * D,   D ~ d_out or d_out*d_in/r")
    print("  so the per-expert share is E / (E + D) -- and D >> E for every real MoE.\n")
    print(f"  {'model':<18} {'E':>5} {'D = d_out*r':>13} {'E/(E+D)':>10}")
    for name, (E, d_model, d_ff, _L) in MODELS.items():
        D = d_ff * max(64, d_ff // 4)
        print(f"  {name:<18} {E:>5} {D:>13,} {E/(E+D)*100:>9.3f}%")

    hdr("4. THE DEPTH AXIS: AN UPPER BOUND, NOT A TARGET")
    print("  Same chart idea, applied along layer depth instead of expert index.")
    print("  The object being compressed is now the dictionary itself (stored L times).")
    print()
    print("  !! These figures assume (a) one global polynomial covers every layer and")
    print("  !! (b) BOTH mode dictionaries are compressible.  Finding F2 in")
    print("  !! docs/FINDINGS.md falsifies both on OLMoE-1B-7B: only the")
    print("  !! residual-stream-facing mode carries cross-layer structure, and affinity")
    print("  !! reaches chance by gap 6-8, so charts must be piecewise over ~4 layers.")
    print("  !! Read this table as an upper bound on the lever, never as a target.\n")
    print(f"  {'model':<18} {'L':>4} {'flat atlas':>12} {'p=2':>10} {'p=4':>10} {'p=8':>10}")
    for name, (E, d_model, d_ff, L) in MODELS.items():
        slot = Slot(name=name, E=E, d_out=d_ff, d_in=d_model)
        st = stack(slot, L)
        r, m = max(64, d_ff // 4), 8
        flat = local_atlas_matrix(st, K=L * 8, r=r, m=m)
        cells = []
        for p in (2, 4, 8):
            d = depth_chart(slot, L=L, r=r, m=m, p_depth=p)
            cells.append(f"{d.ratio(st)*100:>9.2f}%")
        print(f"  {name:<18} {L:>4} {flat.ratio(st)*100:>11.2f}% " + " ".join(cells))

    hdr("5. MULTIVARIATE FEATURE COUNT  M = C(q+p, p)")
    print(f"  {'q\\\\p':>5}" + "".join(f"{p:>7}" for p in range(1, 9)))
    for q in (1, 2, 3, 4, 6, 8):
        print(f"  {q:>5}" + "".join(f"{n_poly_features(q, p):>7}" for p in range(1, 9)))


