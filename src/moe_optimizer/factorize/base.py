"""The contract every compression method must satisfy.

The research claim is a Pareto comparison, so the interface is designed to make
dishonest comparisons hard:

  * ``SlotCode.nbytes`` counts **serialised bytes**, including every scale,
    zero-point, index and shape needed to reconstruct -- not parameter counts,
    and not just the "interesting" tensors.  Parameter counts hide quantisation;
    omitting metadata hides sparse-residual index overhead, which is where
    sparse methods usually lose.
  * ``SlotCode.component_bytes`` breaks that total down, so the per-expert versus
    shared split (the subject of the C1 finding) is always visible.
  * ``reconstruct`` must return the full (E, d_out, d_in) table, so every method
    is scored by the same error metrics on the same object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

Tensors = dict[str, torch.Tensor]

# Bytes per element for the storage dtypes we support.
DTYPE_BYTES: dict[str, float] = {
    "float32": 4.0, "float16": 2.0, "bfloat16": 2.0,
    "int8": 1.0, "uint8": 1.0, "int4": 0.5, "fp8": 1.0,
    "int32": 4.0, "int64": 8.0, "int16": 2.0,
}


def tensor_bytes(t: torch.Tensor, dtype: str | None = None) -> float:
    """Serialised size of one tensor at a declared storage dtype."""
    name = dtype or str(t.dtype).removeprefix("torch.")
    if name not in DTYPE_BYTES:
        raise KeyError(f"no byte size declared for dtype {name!r}")
    return t.numel() * DTYPE_BYTES[name]


@dataclass
class SlotCode:
    """A compressed expert table, plus everything needed to score it honestly.

    ``shared``    : tensors amortised over all experts (dictionaries, anchors, cores)
    ``per_expert``: tensors whose size scales with E (coefficients, coordinates)
    ``residual``  : corrective terms
    ``dtypes``    : storage dtype per tensor name; absent means the tensor's own dtype
    """

    method: str
    shape: tuple[int, int, int]                      # (E, d_out, d_in)
    shared: Tensors = field(default_factory=dict)
    per_expert: Tensors = field(default_factory=dict)
    residual: Tensors = field(default_factory=dict)
    dtypes: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def _group_bytes(self, group: Tensors) -> float:
        return sum(tensor_bytes(t, self.dtypes.get(k)) for k, t in group.items())

    @property
    def component_bytes(self) -> dict[str, float]:
        return {
            "shared": self._group_bytes(self.shared),
            "per_expert": self._group_bytes(self.per_expert),
            "residual": self._group_bytes(self.residual),
        }

    @property
    def nbytes(self) -> float:
        return sum(self.component_bytes.values())

    def dense_bytes(self, dtype: str = "bfloat16") -> float:
        e, o, i = self.shape
        return e * o * i * DTYPE_BYTES[dtype]

    def ratio(self, dtype: str = "bfloat16") -> float:
        """Compressed / dense.  Lower is smaller."""
        return self.nbytes / self.dense_bytes(dtype)

    @property
    def per_expert_share(self) -> float:
        """Fraction of the code that scales with E.

        This is the quantity bounded by E/(E+D) in the C1 analysis, and hence the
        ceiling on any scheme that re-codes only the per-expert table.
        """
        total = self.nbytes
        return self.component_bytes["per_expert"] / total if total else 0.0

    def reconstruct(self) -> torch.Tensor:
        fn = self.meta.get("_reconstruct")
        if fn is None:
            raise NotImplementedError(f"{self.method} did not attach a reconstructor")
        return fn(self)

    def summary(self) -> dict[str, Any]:
        cb = self.component_bytes
        return {
            "method": self.method,
            "shape": list(self.shape),
            "bytes": self.nbytes,
            "ratio_vs_bf16": self.ratio(),
            "shared_bytes": cb["shared"],
            "per_expert_bytes": cb["per_expert"],
            "residual_bytes": cb["residual"],
            "per_expert_share": self.per_expert_share,
            **{k: v for k, v in self.meta.items() if not k.startswith("_")},
        }


class Compressor(ABC):
    """Compress one expert table.

    Implementations are constructed from config, so every hyperparameter must be
    a constructor argument.  ``fit`` sees the table and optional calibration
    statistics and must not touch anything else -- in particular it must not read
    the router or other layers, which keeps every method comparable and keeps the
    pipeline streamable one slot at a time.
    """

    name: str = "abstract"

    @abstractmethod
    def fit(self, stack: torch.Tensor, stats: dict[str, Any] | None = None) -> SlotCode:
        """``stack`` is (E, d_out, d_in) float32."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, **{k: v for k, v in vars(self).items()
                                      if not k.startswith("_")}}


def reconstruction_report(
    original: torch.Tensor, code: SlotCode, weights: torch.Tensor | None = None
) -> dict[str, float]:
    """Error metrics for one compressed slot.

    ``weights`` optionally supplies a per-expert importance (routing frequency),
    which turns the unweighted Frobenius numbers into workload-weighted ones.
    Reporting both is deliberate: the gap between them is the practical size of
    the Frobenius/operator mismatch for this slot.
    """
    approx = code.reconstruct().to(torch.float64)
    orig = original.to(torch.float64)
    err = orig - approx

    per_expert_sq = err.pow(2).flatten(1).sum(1)
    orig_sq = orig.pow(2).flatten(1).sum(1).clamp_min(1e-30)
    rel_per_expert = (per_expert_sq / orig_sq).sqrt()

    out: dict[str, float] = {
        "rel_fro": float(err.norm() / orig.norm().clamp_min(1e-30)),
        "rel_fro_worst_expert": float(rel_per_expert.max()),
        "rel_fro_median_expert": float(rel_per_expert.median()),
        "cos_mean": float(
            torch.nn.functional.cosine_similarity(
                orig.flatten(1), approx.flatten(1), dim=1
            ).mean()
        ),
    }
    if weights is not None:
        w = weights.to(torch.float64).clamp_min(0)
        w = w / w.sum().clamp_min(1e-30)
        out["rel_fro_routing_weighted"] = float(
            (w * per_expert_sq).sum().sqrt() / (w * orig_sq).sum().sqrt()
        )
    return out
