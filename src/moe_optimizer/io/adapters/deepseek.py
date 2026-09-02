"""DeepSeekMoE / DeepSeek-V2/V3 family.

Two structural features matter here.  ``first_k_dense_replace`` leading layers
are dense MLPs with no routed experts, and ``n_shared_experts`` always-on shared
experts sit alongside the routed pool.  Both are recorded and both are excluded
from compression.
"""

from __future__ import annotations

from typing import Any

from ...types import MatrixType, MoEArch
from .base import Adapter, moe_layer_indices, register


def _arch(cfg: dict[str, Any], model_id: str) -> MoEArch:
    n_layers = cfg["num_hidden_layers"]
    first_dense = cfg.get("first_k_dense_replace", 0) or 0
    step = cfg.get("moe_layer_freq", 1) or 1
    return MoEArch(
        model_id=model_id,
        n_layers=n_layers,
        d_model=cfg["hidden_size"],
        d_expert=cfg.get("moe_intermediate_size") or cfg["intermediate_size"],
        n_experts=cfg["n_routed_experts"],
        top_k=cfg["num_experts_per_tok"],
        moe_layers=moe_layer_indices(n_layers, first_dense=first_dense, step=step),
        n_shared_experts=cfg.get("n_shared_experts", 0) or 0,
        dtype=cfg.get("torch_dtype", "bfloat16"),
    )


def _key(layer: int, expert: int, matrix: MatrixType) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{matrix}_proj.weight"


register(Adapter(name="deepseek", model_types=("deepseek", "deepseek_v2", "deepseek_v3"),
                 arch_fn=_arch, key_fn=_key))
