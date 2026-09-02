"""Run a calibration corpus through a model and save per-layer statistics."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .hooks import CalibrationCollector


def _corpus(tokenizer, n_tokens: int, seq_len: int, name: str = "wikitext"):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0][: n_tokens]
    n_seq = ids.numel() // seq_len
    return ids[: n_seq * seq_len].view(n_seq, seq_len)


def run_calibration(model_id: str, out: str, n_tokens: int = 32768, seq_len: int = 512,
                    batch: int = 4, dtype=torch.bfloat16, cache_dir: str | None = None,
                    layers: list[int] | None = None, max_batches: int | None = None) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir,
                                                 low_cpu_mem_usage=True)
    model.eval()
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

    layers = layers if layers is not None else list(range(model.config.num_hidden_layers))
    col = CalibrationCollector(model, layers)
    seqs = _corpus(tok, n_tokens, seq_len)
    print(f"corpus: {seqs.shape[0]} x {seq_len} = {seqs.numel():,} tokens", flush=True)

    torch.set_grad_enabled(False)
    done = 0
    t0 = time.time()
    for i in range(0, seqs.shape[0], batch):
        if max_batches is not None and i // batch >= max_batches:
            break
        model(seqs[i:i + batch])
        done += seqs[i:i + batch].numel()
        el = time.time() - t0
        print(f"  {done:>7,} tokens  {el:6.0f}s  {done/el:6.1f} tok/s", flush=True)

    stats = col.stats()
    col.remove()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model_id, "n_tokens": done, "layers": stats}, out)
    print(f"saved {out}", flush=True)
    return stats
