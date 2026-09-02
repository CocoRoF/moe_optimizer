"""Mixtral (8 experts).

Kept deliberately, but as a *negative* control rather than a headline testbed:
with E=8 and d_expert=14336 the shared-dictionary family is structurally
inapplicable (see docs/param_economics.md).  Mixtral names the three matrices
w1/w3/w2 for gate/up/down.
"""

from __future__ import annotations

from typing import Any

from ...types import MatrixType, MoEArch
from .base import Adapter, moe_layer_indices, register

_MIXTRAL_NAMES = {"gate": "w1", "up": "w3", "down": "w2"}


def _arch(cfg: dict[str, Any], model_id: str) -> MoEArch:
    n_layers = cfg["num_hidden_layers"]
    return MoEArch(
        model_id=model_id,
        n_layers=n_layers,
        d_model=cfg["hidden_size"],
        d_expert=cfg["intermediate_size"],
        n_experts=cfg["num_local_experts"],
        top_k=cfg["num_experts_per_tok"],
        moe_layers=moe_layer_indices(n_layers),
        dtype=cfg.get("torch_dtype", "bfloat16"),
    )


def _key(layer: int, expert: int, matrix: MatrixType) -> str:
    return f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{_MIXTRAL_NAMES[matrix]}.weight"


register(Adapter(name="mixtral", model_types=("mixtral",), arch_fn=_arch, key_fn=_key))
