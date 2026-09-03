"""E1: one streaming pass over calibration text -> everything the policies need."""
import sys, time, torch
sys.path.insert(0, "src")
from datasets import load_dataset
from transformers import AutoTokenizer
from moe_optimizer.io.checkpoint import ExpertStore, resolve_model
from moe_optimizer.runtime.stream import StreamingOLMoE, ExpertPolicy
from moe_optimizer.runtime.calibrate import mass_ratio_medians, scale_dispersion
MODEL, N = "allenai/OLMoE-1B-7B-0924", int(sys.argv[1]) if len(sys.argv) > 1 else 2048

class Trace(ExpertPolicy):
    def __init__(self, k): self.k, self.w, self.i, self.name = k, {}, {}, "trace"
    def select(self, probs, layer):
        w, i = probs.topk(self.k, dim=-1)
        self.w.setdefault(layer, []).append(w.cpu()); self.i.setdefault(layer, []).append(i.cpu())
        return i, w

tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=".cache")
text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N]; ids = ids[: ids.numel() // 512 * 512].view(-1, 512)
rm = resolve_model(MODEL, cache_dir=".cache", allow_download=False)
pol = Trace(rm.config["num_experts_per_tok"])
eng = StreamingOLMoE(ExpertStore(rm), rm.config, policy=pol); eng.record_output_norms = {}
t0 = time.time()
for j in range(ids.shape[0]):
    eng.forward(ids[j]); print(f"  seq {j+1}/{ids.shape[0]}  {time.time()-t0:.0f}s", flush=True)
traces = {l: torch.cat(v) for l, v in pol.w.items()}; indices = {l: torch.cat(v) for l, v in pol.i.items()}
scale = {l: (r[:, 0] / r[:, 1].clamp_min(1)).float() for l, r in eng.record_output_norms.items()}
for l in scale:                                       # experts never routed: fill with layer mean
    m = scale[l] > 0; scale[l][~m] = scale[l][m].mean()
K = pol.k
torch.save({"model": MODEL, "n_tokens": ids.numel(), "k": K, "traces": traces, "indices": indices,
            "scale": scale, "beta_median": mass_ratio_medians(traces, K)}, "runs/policy_calib_olmoe.pt")
cv = scale_dispersion(scale)
allw = torch.cat(list(traces.values())); cum = allw.cumsum(1)
print(f"\n=== E1 done: {ids.numel():,} tokens, top-{K} ===")
print("G1 -- CV of calibrated output scale s[l,e] within layer (kill line 0.15):")
print("  " + " ".join(f"L{l:02d}:{cv[l]:.2f}" for l in sorted(cv)))
print(f"  mean CV {sum(cv.values())/len(cv):.3f}   min {min(cv.values()):.3f}   max {max(cv.values()):.3f}")
print("\nheadroom -- sorted top-k weight medians:  " + " ".join(f"w{j+1}={allw[:, j].median():.3f}" for j in range(K)))
for tgt in (0.90, 0.95, 0.99):
    print(f"  tokens reaching {int(tgt*100)}% gate mass by k': " + " ".join(f"{j+1}:{(cum[:, j] >= tgt).float().mean()*100:4.0f}%" for j in range(K)))
print("\nscale vs gate weight -- correlation of s[e] with mean w_e per layer (is scale redundant with score?):")
cs = []
for l in sorted(traces):
    mw = torch.zeros(scale[l].numel()).index_add_(0, indices[l].flatten(), traces[l].flatten()) / torch.bincount(indices[l].flatten(), minlength=scale[l].numel()).clamp_min(1)
    cs.append(torch.corrcoef(torch.stack([scale[l], mw]))[0, 1].item())
print("  " + " ".join(f"L{l:02d}:{c:+.2f}" for l, c in enumerate(cs)) + f"   mean {sum(cs)/len(cs):+.3f}")
print("DONE")
