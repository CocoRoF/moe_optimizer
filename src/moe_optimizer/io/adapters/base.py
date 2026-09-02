"""Model-family adapters.

An adapter's whole job is to answer two questions for a given checkpoint:

    1. what is the MoE structure?                       -> ``arch(config)``
    2. what tensor name holds expert ``e``'s ``gate``   -> ``weight_key(...)``
       matrix in layer ``l``, and does it need a transpose?

Nothing else in the codebase is allowed to know about naming conventions.  This
is what makes it possible to run the identical algorithm over Qwen3, OLMoE,
DeepSeek and Mixtral and to trust that a difference in the results is a
difference in the models rather than in the plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ...types import MatrixType, MoEArch

ADAPTERS: dict[str, "Adapter"] = {}


@dataclass(frozen=True)
class Adapter:
    """A model family."""

    name: str
    # config["model_type"] values this adapter claims
    model_types: tuple[str, ...]
    arch_fn: Callable[[dict[str, Any], str], MoEArch]
    key_fn: Callable[[int, int, MatrixType], str]
    # True when the checkpoint stores W as (d_in, d_out) and we must transpose
    transposed: bool = False

    def arch(self, config: dict[str, Any], model_id: str) -> MoEArch:
        return self.arch_fn(config, model_id)

    def weight_key(self, layer: int, expert: int, matrix: MatrixType) -> str:
        return self.key_fn(layer, expert, matrix)


def register(adapter: Adapter) -> Adapter:
    ADAPTERS[adapter.name] = adapter
    return adapter


def get_adapter(config: dict[str, Any]) -> Adapter:
    mt = config.get("model_type", "")
    for a in ADAPTERS.values():
        if mt in a.model_types:
            return a
    raise KeyError(
        f"no adapter for model_type={mt!r}; registered: "
        + ", ".join(f"{a.name}{a.model_types}" for a in ADAPTERS.values())
    )


def moe_layer_indices(n_layers: int, first_dense: int = 0, step: int = 1) -> tuple[int, ...]:
    """Layers carrying a routed MoE.

    ``first_dense`` leading layers are dense (DeepSeek does this); thereafter
    every ``step``-th layer is sparse (Qwen exposes ``decoder_sparse_step``).
    """
    return tuple(i for i in range(first_dense, n_layers) if (i - first_dense) % step == 0)
