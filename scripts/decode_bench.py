"""E3 (correct form): batch-1 decode bandwidth and speed per policy."""
import sys, json, gc, torch
sys.path.insert(0, "src")
from datasets import load_dataset
from transformers import AutoTokenizer
from moe_optimizer.io.checkpoint import ExpertStore, resolve_model
from moe_optimizer.runtime.stream import StreamingOLMoE, TopKPolicy, MassRatioPolicy, ContributionPolicy, decode_benchmark
from moe_optimizer.runtime.calibrate import calibrate_taus
MODEL = next((a for a in sys.argv[1:] if "/" in a), "allenai/OLMoE-1B-7B-0924")
ARGS = [a for a in sys.argv[1:] if "/" not in a]
SHORT = MODEL.split("/")[-1].split("-")[0].lower()          # olmoe | qwen3
def _engine_cls(cfg):
    from moe_optimizer.runtime.stream import StreamingOLMoE, StreamingQwen3MoE
    return StreamingQwen3MoE if cfg.get("model_type", "").startswith("qwen") else StreamingOLMoE
STEPS = int(ARGS[0]) if ARGS else 64
TARGETS = [float(x) for x in ARGS[1].split(",")] if len(ARGS) > 1 else [6.0, 5.0, 4.0]
cal = torch.load(f"runs/policy_calib_{SHORT}.pt"); K, tr, ix, sc = cal["k"], cal["traces"], cal["indices"], cal["scale"]
tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=".cache")
text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:32 + STEPS + 1]
rm = resolve_model(MODEL, cache_dir=".cache", allow_download=False); store = ExpertStore(rm)
pols = [TopKPolicy(K)]
b = MassRatioPolicy(K, cal["beta_median"]); b.name = "mass_ratio(median)"; pols.append(b)
for t in TARGETS:
    p = TopKPolicy(int(round(t))); p.name = f"top{int(round(t))}(static)"; pols.append(p)
    q = ContributionPolicy(K, {l: torch.ones_like(sc[l]) for l in sc}, calibrate_taus(tr, None, ix, t)); q.name = f"score_only@{t}"; pols.append(q)
    c = ContributionPolicy(K, sc, calibrate_taus(tr, sc, ix, t)); c.name = f"contribution@{t}"; pols.append(c)
rows = []
for pol in pols:
    eng = _engine_cls(rm.config)(store, rm.config, policy=pol)
    r = decode_benchmark(eng, ids, prefill=32, steps=STEPS); rows.append(r); del eng; gc.collect()
    print(f"  {r['policy']:<22} k'={r['mean_k']:.2f}  {r['MB_per_tok']:6.1f} MB/tok  {r['expert_loads_per_tok']:5.1f} loads/tok  "
          f"{r['tok_per_s']:.2f} tok/s  decode-ppl={r['decode_ppl']:.2f}  cache-check={r['cache_consistency_max_dlogit']:.3f}", flush=True)
json.dump(rows, open(f"runs/decode_bench_{SHORT}.json", "w"), indent=1); print("DONE")
