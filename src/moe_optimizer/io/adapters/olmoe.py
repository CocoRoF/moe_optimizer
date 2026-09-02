"""OLMoE-1B-7B (allenai).

The primary local testbed: 64 experts, 16 layers, 1B active parameters, fully
open including intermediate checkpoints.  Small enough that the entire pipeline
-- including calibration forward passes -- runs on CPU.
"""

from __future__ import annotations

from typing import Any

from ...types import MatrixType, MoEArch
from .base import Adapter, moe_layer_indices, register

_OLMOE_NAMES = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


def _arch(cfg: dict[str, Any], model_id: str) -> MoEArch:
    n_layers = cfg["num_hidden_layers"]
    return MoEArch(
        model_id=model_id,
        n_layers=n_layers,
        d_model=cfg["hidden_size"],
        d_expert=cfg["intermediate_size"],
        n_experts=cfg["num_experts"],
        top_k=cfg["num_experts_per_tok"],
        moe_layers=moe_layer_indices(n_layers),
        dtype=cfg.get("torch_dtype", "bfloat16"),
    )


def _key(layer: int, expert: int, matrix: MatrixType) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{_OLMOE_NAMES[matrix]}.weight"


register(Adapter(name="olmoe", model_types=("olmoe",), arch_fn=_arch, key_fn=_key))
