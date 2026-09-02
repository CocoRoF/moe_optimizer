"""Lazy, shard-aware access to a checkpoint's expert tables.

The whole codebase is built around one memory rule: **never hold more than one
expert table in RAM at a time.**  A single table -- all E experts of one matrix
type in one layer -- is 0.5-0.8 GB in float32 for the models we target, while the
full checkpoint is 14-60 GB.  Every algorithm here is therefore written to
consume ``ExpertStore.stack(slot)`` and release it before requesting the next.

Tensors are read through ``safetensors``' memory-mapped reader, so a shard is
paged in on demand rather than deserialised whole.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open

from ..types import MATRIX_TYPES, MatrixType, MoEArch, Slot
from .adapters import Adapter, get_adapter


@dataclass
class ResolvedModel:
    path: Path
    config: dict
    arch: MoEArch
    adapter: Adapter


def resolve_model(model_id_or_path: str, cache_dir: str | None = None,
                  allow_download: bool = True) -> ResolvedModel:
    """Locate a checkpoint locally, downloading only if permitted.

    ``model_id_or_path`` may be a local directory or a HuggingFace repo id.
    """
    p = Path(model_id_or_path)
    if p.is_dir():
        path = p
    else:
        if not allow_download:
            raise FileNotFoundError(
                f"{model_id_or_path} is not a local directory and downloads are disabled"
            )
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                model_id_or_path,
                cache_dir=cache_dir,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
            )
        )

    config = json.loads((path / "config.json").read_text())
    adapter = get_adapter(config)
    arch = adapter.arch(config, model_id_or_path)
    return ResolvedModel(path=path, config=config, arch=arch, adapter=adapter)


class ExpertStore:
    """Read-only, memory-mapped view of a checkpoint's expert weights."""

    def __init__(self, model: ResolvedModel, dtype: torch.dtype = torch.float32) -> None:
        self.model = model
        self.arch = model.arch
        self.adapter = model.adapter
        self.dtype = dtype
        self._index = self._build_index(model.path)
        self._open: dict[str, object] = {}

    # -- shard index ------------------------------------------------------

    @staticmethod
    def _build_index(path: Path) -> dict[str, str]:
        idx = path / "model.safetensors.index.json"
        if idx.exists():
            weight_map = json.loads(idx.read_text())["weight_map"]
            return {k: str(path / v) for k, v in weight_map.items()}
        single = path / "model.safetensors"
        if not single.exists():
            shards = sorted(path.glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError(f"no safetensors found under {path}")
            out: dict[str, str] = {}
            for s in shards:
                with safe_open(str(s), framework="pt") as f:
                    for k in f.keys():
                        out[k] = str(s)
            return out
        with safe_open(str(single), framework="pt") as f:
            return {k: str(single) for k in f.keys()}

    def _handle(self, shard: str):
        h = self._open.get(shard)
        if h is None:
            h = safe_open(shard, framework="pt")
            self._open[shard] = h
        return h

    def has(self, key: str) -> bool:
        return key in self._index

    def get(self, key: str) -> torch.Tensor:
        shard = self._index.get(key)
        if shard is None:
            raise KeyError(f"{key!r} not in checkpoint ({len(self._index)} tensors indexed)")
        return self._handle(shard).get_tensor(key).to(self.dtype)

    # -- expert tables ----------------------------------------------------

    def slots(self, matrices: tuple[MatrixType, ...] = MATRIX_TYPES) -> Iterator[Slot]:
        for layer in self.arch.moe_layers:
            for m in matrices:
                yield Slot(layer=layer, matrix=m)

    def expert(self, slot: Slot, e: int) -> torch.Tensor:
        """One expert weight as (d_out, d_in)."""
        w = self.get(self.adapter.weight_key(slot.layer, e, slot.matrix))
        if self.adapter.transposed:
            w = w.T
        return w.contiguous()

    def stack(self, slot: Slot) -> torch.Tensor:
        """The full expert table as (E, d_out, d_in).

        This is the single largest allocation the pipeline makes.  Callers must
        drop the reference before requesting another slot.
        """
        d_out, d_in = self.arch.shape(slot.matrix)
        out = torch.empty((self.arch.n_experts, d_out, d_in), dtype=self.dtype)
        for e in range(self.arch.n_experts):
            w = self.expert(slot, e)
            if w.shape != (d_out, d_in):
                raise ValueError(
                    f"{slot} expert {e}: checkpoint gives {tuple(w.shape)}, "
                    f"arch says {(d_out, d_in)} -- adapter orientation is wrong"
                )
            out[e] = w
        return out

    def slot_bytes(self, slot: Slot, dtype_bytes: int = 2) -> int:
        d_out, d_in = self.arch.shape(slot.matrix)
        return self.arch.n_experts * d_out * d_in * dtype_bytes

    def close(self) -> None:
        self._open.clear()

    def __enter__(self) -> "ExpertStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
