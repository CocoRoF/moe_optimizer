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


def compress_to_dir(store: ExpertStore, spec: dict, calib: dict | None, out_dir: str,
                    matrices=("gate", "up", "down"), layers=None, verbose=True) -> dict:
    """Phase 1 of F12: compress every slot with only the weight store resident.

    Writes each reconstructed slot as fp16 safetensors plus a rows.json of
    per-slot errors.  Peak memory is one slot's factorisation (~3 GB on OLMoE),
    never the model.  Holding both -- as the first F12 attempt did -- reaches
    ~21 GB and is SIGKILLed on this machine.
    """
    import json
    from pathlib import Path
    from safetensors.torch import save_file
    import moe_optimizer.methods  # noqa: F401

    spec = dict(spec); name = spec.pop("name")
    layers = list(layers if layers is not None else store.arch.moe_layers)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
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
            save_file({"w": code.reconstruct().to(torch.float16).contiguous()},
                      str(out / f"L{l:03d}.{m}.safetensors"))
            total_bytes += code.nbytes; total_dense += code.dense_bytes()
            rows.append({"slot": str(slot), "layer": l, "matrix": m, "ratio": code.ratio(), **err})
            if verbose:
                print(f"  [{i+1:>2}/{len(layers)}] {slot}  ratio={code.ratio():.3f}  "
                      f"rel_act={err.get('rel_act', float('nan')):.4f}  {time.time()-t0:5.0f}s", flush=True)
            del stack, code
    info = {"spec": {"name": name, **spec}, "rows": rows,
            "expert_ratio": total_bytes / total_dense if total_dense else float("nan")}
    (out / "rows.json").write_text(json.dumps(info))
    return info


def load_compressed_into(model, in_dir: str, verbose=True) -> dict:
    """Phase 2 of F12: stream the compressed slots from disk into the live model."""
    import json
    from pathlib import Path
    from safetensors.torch import load_file

    info = json.loads((Path(in_dir) / "rows.json").read_text())
    for r in info["rows"]:
        w = load_file(str(Path(in_dir) / f"L{r['layer']:03d}.{r['matrix']}.safetensors"))["w"]
        write_slot(model.model.layers[r["layer"]].mlp.experts, r["matrix"], w)
        del w
    if verbose:
        print(f"  loaded {len(info['rows'])} slots from {in_dir}", flush=True)
    return info


def materialize(model, store: ExpertStore, spec: dict, calib: dict | None,
                matrices=("gate", "up", "down"), layers=None, verbose=True) -> dict:
    """Kept for callers that already hold a model; prefer the two-phase path."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="moeopt_mat_") as d:
        compress_to_dir(store, spec, calib, d, matrices, layers, verbose)
        return load_compressed_into(model, d, verbose)


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
            max_tokens: int = 16384, seq_len: int = 512, work_dir: str | None = None) -> dict:
    """Two-phase: compress with only the weight store resident, then load the
    model once and evaluate baseline and compressed perplexity."""
    import gc
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..io.checkpoint import resolve_model

    tag = f"{spec['name']}_" + "_".join(f"{k}{v}" for k, v in spec.items() if k != "name")
    work_dir = work_dir or f"runs/compressed/{model_id.split('/')[-1]}/{tag}"

    print(f"== phase 1: compress -> {work_dir} ==", flush=True)
    store = ExpertStore(resolve_model(model_id, cache_dir=cache_dir, allow_download=False))
    calib = torch.load(calib_path)["layers"]
    info = compress_to_dir(store, spec, calib, work_dir)
    print(f"expert-table ratio = {info['expert_ratio']:.3f}", flush=True)
    del store, calib; gc.collect()

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                               split="test")["text"] if t.strip())
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 cache_dir=cache_dir, low_cpu_mem_usage=True).eval()
    print("== baseline perplexity ==", flush=True)
    base = perplexity(model, tok, text, seq_len, max_tokens)
    print(f"baseline ppl = {base:.3f}", flush=True)
    print("== phase 2: load compressed slots ==", flush=True)
    load_compressed_into(model, work_dir)
    print("== compressed perplexity ==", flush=True)
    comp = perplexity(model, tok, text, seq_len, max_tokens)
    print(f"compressed ppl = {comp:.3f}   (baseline {base:.3f}, x{comp/base:.3f})", flush=True)
    return {"baseline_ppl": base, "compressed_ppl": comp, **info}
