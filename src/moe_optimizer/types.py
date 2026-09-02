"""Shared dataclasses.  Deliberately dependency-light so they can be imported
from analysis scripts without pulling in torch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MatrixType = Literal["gate", "up", "down"]
MATRIX_TYPES: tuple[MatrixType, ...] = ("gate", "up", "down")


@dataclass(frozen=True)
class MoEArch:
    """Everything about a model's MoE structure that the algorithms need.

    Shapes are given in the *mathematical* orientation used throughout this
    codebase: every expert weight is materialised as ``(d_out, d_in)`` and applied
    as ``y = W @ x``.  Adapters are responsible for transposing whatever the
    checkpoint happens to store.
    """

    model_id: str
    n_layers: int
    d_model: int
    d_expert: int          # expert FFN intermediate size
    n_experts: int         # routed experts per MoE layer
    top_k: int
    moe_layers: tuple[int, ...]      # which layer indices actually have a routed MoE
    n_shared_experts: int = 0
    dtype: str = "bfloat16"

    def shape(self, t: MatrixType) -> tuple[int, int]:
        """(d_out, d_in) of one expert weight of the given type."""
        return (self.d_model, self.d_expert) if t == "down" else (self.d_expert, self.d_model)

    @property
    def expert_params_per_layer(self) -> int:
        return self.n_experts * sum(o * i for o, i in (self.shape(t) for t in MATRIX_TYPES))

    @property
    def total_expert_params(self) -> int:
        return len(self.moe_layers) * self.expert_params_per_layer


@dataclass(frozen=True)
class Slot:
    """Addresses one expert table: all experts of one matrix type in one layer."""

    layer: int
    matrix: MatrixType

    def __str__(self) -> str:
        return f"L{self.layer:03d}.{self.matrix}"


@dataclass
class RoutingStats:
    """Calibration-derived routing statistics for one MoE layer.

    ``counts[e]``      : how often expert e was selected
    ``gate_mass[e]``   : summed gate weight assigned to e
    ``coactivation``   : (E, E) counts of experts selected for the same token
    ``n_tokens``       : tokens observed
    """

    layer: int
    n_experts: int
    counts: list[int] = field(default_factory=list)
    gate_mass: list[float] = field(default_factory=list)
    coactivation: list[list[int]] = field(default_factory=list)
    n_tokens: int = 0
