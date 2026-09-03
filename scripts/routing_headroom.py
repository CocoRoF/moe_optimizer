"""How concentrated is gate mass across the top-k experts?  (headroom for skipping)

For each MoE layer, record the *sorted, normalised* top-k router weights per
token and report: the k-th weight's distribution, and the fraction of tokens
whose cumulative mass reaches 90 / 95 / 99 % by k' < k.  If most tokens reach
95 % by k'=4 of 8, half the expert loads are buying < 5 % of the gate mass --
that is the headroom any dynamic-skipping mechanism is spending.
"""
import sys, time, torch
torch.set_num_threads(11)                       # 30 % of 16 cores stays free
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL = sys.argv[1] if len(sys.argv) > 1 else "allenai/OLMoE-1B-7B-0924"
N_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=".cache")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, cache_dir=".cache", low_cpu_mem_usage=True).eval()
L = model.config.num_hidden_layers; K = model.config.num_experts_per_tok
rec = {l: [] for l in range(L)}
def hook(l):
    def f(mod, args, out):
        _, w, _ = out                            # (T, k) top-k weights, normalised if norm_topk_prob
        rec[l].append(w.float().sort(dim=1, descending=True).values)
    return f
hs = [model.model.layers[l].mlp.gate.register_forward_hook(hook(l)) for l in range(L)]
text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_TOK]; ids = ids[: ids.numel() // 512 * 512].view(-1, 512)
t0 = time.time()
with torch.no_grad():
    for i in range(0, ids.shape[0], 2): model(ids[i:i+2])
print(f"{MODEL}: {ids.numel():,} tokens, top-{K}, {L} layers, {time.time()-t0:.0f}s", flush=True)
allw = torch.cat([torch.cat(rec[l]) for l in range(L)])          # (L*T, k)
cum = allw.cumsum(1)
print("\nsorted top-k weight, median over all tokens & layers:")
print("  " + "  ".join(f"w{j+1}={allw[:, j].median():.3f}" for j in range(K)))
print("\nfraction of tokens whose cumulative gate mass reaches the target by k' experts:")
for tgt in (0.90, 0.95, 0.99):
    row = [(cum[:, j] >= tgt).float().mean().item() for j in range(K)]
    print(f"  {int(tgt*100)}%: " + "  ".join(f"k'={j+1}:{row[j]*100:5.1f}%" for j in range(K)))
print("\nper-layer: median w_k (last kept expert) and mean experts needed for 95% mass:")
for l in range(L):
    w = torch.cat(rec[l]); c = w.cumsum(1)
    need = (c < 0.95).sum(1).float() + 1
    print(f"  L{l:02d}: median w{K}={w[:, -1].median():.3f}   mean k'(95%)={need.mean():.2f}   p90 k'={need.quantile(0.9):.0f}")
print("DONE")
