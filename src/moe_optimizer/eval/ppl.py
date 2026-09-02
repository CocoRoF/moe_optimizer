"""F12: does a compressed expert table still model language?

Every number before this one is a reconstruction error.  This module writes a
compressed table *back into the live model* and measures perplexity, which is
the first quantity here a reader can weigh against the literature.

The HF OLMoE/Qwen3-MoE experts module is fused: ``gate_up_proj`` is
(E, 2*d_ff, d_model) with gate in the first half of dim 1 and up in the second,
``down_proj`` is (E, d_model, d_ff).  Our slots are (E, d_out, d_in) in the same
orientation, so write-back is a slice assignment.
"""

from __future__ import annotations

import math
import time

import torch

from ..calib.stats import slot_stats
from ..factorize.base import reconstruction_report
from ..io.checkpoint import ExpertStore
from ..registry import COMPRESSORS
from ..types import Slot


def write_slot(experts_module, matrix: str, table: torch.Tensor) -> None:
    """Overwrite one matrix type for all experts inside a fused experts module."""
    d_ff = experts_module.intermediate_dim
    t = table.to(experts_module.down_proj.dtype)
    with torch.no_grad():
        if matrix == "gate":
            experts_module.gate_up_proj[:, :d_ff].copy_(t)
        elif matrix == "up":
            experts_module.gate_up_proj[:, d_ff:].copy_(t)
        elif matrix == "down":
            experts_module.down_proj.copy_(t)
        else:
            raise ValueError(matrix)


def materialize(model, store: ExpertStore, spec: dict, calib: dict | None,
                matrices=("gate", "up", "down"), layers=None, verbose=True) -> dict:
    """Compress every slot with ``spec`` and write it into ``model``.  Returns
    per-slot error rows so PPL can be read against reconstruction error."""
    import moe_optimizer.methods  # noqa: F401
    spec = dict(spec); name = spec.pop("name")
    layers = list(layers if layers is not None else store.arch.moe_layers)
    rows, t0 = [], time.time()
    total_bytes = total_dense = 0.0
    for i, l in enumerate(layers):
        for m in matrices:
            slot = Slot(l, m)
            stack = store.stack(slot)
            stats = slot_stats(calib, l, m) if calib is not None else None
            code = COMPRESSORS.get(name)(**spec).fit(stack, stats)
            err = reconstruction_report(stack, code, weights=stats["importance"] if stats else None,
                                        input_cov=stats.get("input_cov") if stats else None)
            write_slot(model.model.layers[l].mlp.experts, m, code.reconstruct())
            total_bytes += code.nbytes; total_dense += code.dense_bytes()
            rows.append({"slot": str(slot), "ratio": code.ratio(), **err})
            if verbose:
                print(f"  [{i+1:>2}/{len(layers)}] {slot}  ratio={code.ratio():.3f}  "
                      f"rel_act={err.get('rel_act', float('nan')):.4f}  {time.time()-t0:5.0f}s", flush=True)
            del stack, code
    return {"rows": rows, "expert_ratio": total_bytes / total_dense}


@torch.no_grad()
def perplexity(model, tok, text: str, seq_len: int = 512, max_tokens: int = 16384,
               verbose=True) -> float:
    ids = tok(text, return_tensors="pt").input_ids[0][:max_tokens]
    n = ids.numel() // seq_len
    ids = ids[: n * seq_len].view(n, seq_len)
    nll, count, t0 = 0.0, 0, time.time()
    for i in range(n):
        out = model(ids[i:i+1], labels=ids[i:i+1])
        nll += float(out.loss) * (seq_len - 1); count += seq_len - 1
        if verbose and (i + 1) % 4 == 0:
            print(f"    ppl so far {math.exp(nll/count):8.3f}  [{i+1}/{n}]  {time.time()-t0:4.0f}s", flush=True)
    return math.exp(nll / count)


def run_f12(model_id: str, spec: dict, calib_path: str, cache_dir=".cache",
            max_tokens: int = 16384, seq_len: int = 512) -> dict:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..io.checkpoint import resolve_model

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                               split="test")["text"] if t.strip())
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 cache_dir=cache_dir, low_cpu_mem_usage=True).eval()
    print("== baseline perplexity ==", flush=True)
    base = perplexity(model, tok, text, seq_len, max_tokens)
    print(f"baseline ppl = {base:.3f}", flush=True)

    store = ExpertStore(resolve_model(model_id, cache_dir=cache_dir, allow_download=False))
    calib = torch.load(calib_path)["layers"]
    print(f"== materialize {spec} ==", flush=True)
    info = materialize(model, store, spec, calib)
    print(f"expert-table ratio = {info['expert_ratio']:.3f}", flush=True)
    print("== compressed perplexity ==", flush=True)
    comp = perplexity(model, tok, text, seq_len, max_tokens)
    print(f"compressed ppl = {comp:.3f}   (baseline {base:.3f}, x{comp/base:.3f})", flush=True)
    return {"baseline_ppl": base, "compressed_ppl": comp, **info}
