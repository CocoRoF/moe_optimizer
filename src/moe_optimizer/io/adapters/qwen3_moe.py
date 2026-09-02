"""Qwen3-MoE / Qwen2-MoE / Qwen1.5-MoE.

Qwen stores each expert as three separate 2-D tensors under
``model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight``, already in
(d_out, d_in) orientation.  Qwen1.5/2-MoE additionally carry a *shared* expert,
which we record in the arch but never compress: it is dense, always active, and
compressing it would confound the expert-table result we are after.
"""

from __future__ import annotations

from typing import Any

from ...types import MatrixType, MoEArch
from .base import Adapter, moe_layer_indices, register


def _arch(cfg: dict[str, Any], model_id: str) -> MoEArch:
    n_layers = cfg["num_hidden_layers"]
    step = cfg.get("decoder_sparse_step", 1) or 1
    shared_dim = cfg.get("shared_expert_intermediate_size", 0) or 0
    d_expert = cfg.get("moe_intermediate_size") or cfg["intermediate_size"]
    return MoEArch(
        model_id=model_id,
        n_layers=n_layers,
        d_model=cfg["hidden_size"],
        d_expert=d_expert,
        n_experts=cfg["num_experts"],
        top_k=cfg["num_experts_per_tok"],
        moe_layers=moe_layer_indices(n_layers, step=step),
        n_shared_experts=1 if shared_dim else 0,
        dtype=cfg.get("torch_dtype", "bfloat16"),
    )


def _key(layer: int, expert: int, matrix: MatrixType) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{matrix}_proj.weight"


register(Adapter(name="qwen_moe", model_types=("qwen3_moe", "qwen2_moe"),
                 arch_fn=_arch, key_fn=_key))
