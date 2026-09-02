"""Matched-budget Pareto sweeps.

The comparison that matters is *quality at equal serialised bytes*, so a sweep
records, for every (method, hyperparameter) point, both the error and the exact
byte count -- never a nominal "compression ratio", which each paper defines
differently (report section 1.3).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from ..factorize.base import reconstruction_report
from ..registry import COMPRESSORS


@dataclass
class SweepResult:
    slot: str
    method: str
    params: dict[str, Any]
    bytes: float
    ratio: float
    per_expert_share: float
    error: dict[str, float]
    seconds: float
    meta: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        return {
            "slot": self.slot, "method": self.method, **self.params,
            "bytes": self.bytes, "ratio": self.ratio,
            "per_expert_share": self.per_expert_share,
            **self.error, "seconds": self.seconds,
        }


def sweep_slot(
    stack: torch.Tensor,
    slot_name: str,
    points: list[dict[str, Any]],
    stats: dict[str, Any] | None = None,
    importance: torch.Tensor | None = None,
    verbose: bool = True,
) -> list[SweepResult]:
    """Run every configuration in ``points`` against one expert table.

    Each point is ``{"name": <compressor>, **kwargs}``.  A point that raises is
    recorded with the exception rather than aborting the sweep, so one bad
    hyperparameter cannot cost an entire run.
    """
    out: list[SweepResult] = []
    for i, spec in enumerate(points):
        spec = dict(spec)
        name = spec.pop("name")
        t0 = time.perf_counter()
        try:
            code = COMPRESSORS.get(name)(**spec).fit(stack, stats)
            err = reconstruction_report(stack, code, weights=importance)
            res = SweepResult(
                slot=slot_name, method=name, params=spec, bytes=code.nbytes,
                ratio=code.ratio(), per_expert_share=code.per_expert_share,
                error=err, seconds=time.perf_counter() - t0,
                meta={k: v for k, v in code.meta.items() if not k.startswith("_")},
            )
            del code
        except Exception as exc:  # noqa: BLE001 - a failed point must not kill the sweep
            res = SweepResult(
                slot=slot_name, method=name, params=spec, bytes=float("nan"),
                ratio=float("nan"), per_expert_share=float("nan"),
                error={"rel_fro": float("nan")}, seconds=time.perf_counter() - t0,
                meta={"error": f"{type(exc).__name__}: {exc}"},
            )
        out.append(res)
        if verbose:
            e = res.error.get("rel_fro", float("nan"))
            note = res.meta.get("error", "")
            print(f"  [{i + 1:>3}/{len(points)}] {name:<20} "
                  f"ratio={res.ratio:>6.3f}  rel_fro={e:>8.5f}  "
                  f"peshare={res.per_expert_share:>6.3f}  {res.seconds:>5.1f}s {note}",
                  flush=True)
    return out


def write_results(results: list[SweepResult], path: str) -> None:
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


def pareto_front(results: list[SweepResult], metric: str = "rel_fro") -> list[SweepResult]:
    """Points not dominated on (bytes, error) -- both smaller is better."""
    valid = [r for r in results if r.bytes == r.bytes and r.error.get(metric, float("nan"))
             == r.error.get(metric, float("nan"))]
    front: list[SweepResult] = []
    for r in sorted(valid, key=lambda x: x.bytes):
        if not front or r.error[metric] < front[-1].error[metric]:
            front.append(r)
    return front
