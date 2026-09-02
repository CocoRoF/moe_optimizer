#!/usr/bin/env python3
"""Where do the parameters of a compressed MoE expert table actually live?

Run:  python3 scripts/param_economics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_optimizer.param_economics import (  # noqa: E402
    Slot, depth_chart, local_atlas_matrix, n_poly_features, per_expert_svd,
    shared_base_delta, shared_basis, stack, tucker_joint, tucker_poly_chart,
)

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


def main() -> None:
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

    hdr("4. THE DEPTH AXIS: WHERE AN ORTHOGONAL CHART IS *NOT* NEGLIGIBLE")
    print("  Same chart idea, applied along layer depth instead of expert index.")
    print("  The object being compressed is now the dictionary itself (stored L times).\n")
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


if __name__ == "__main__":
    main()
