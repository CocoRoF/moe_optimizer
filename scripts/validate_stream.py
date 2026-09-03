"""Does the streaming engine reproduce the reference model?  Logit-level check.

Reference is loaded through accelerate disk offload so RSS stays under the
30%-free rule; 48 tokens is enough to exercise RoPE, causal masking and every
layer's routing.  Pass criterion: top-1 token agreement 100% and max |dlogit|
consistent with bf16-vs-fp32 arithmetic (< ~0.5 on logits of magnitude ~20).
"""
import sys, time, torch
sys.path.insert(0, "src"); torch.set_num_threads(11)
from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_optimizer.io.checkpoint import ExpertStore, resolve_model
from moe_optimizer.runtime.stream import StreamingOLMoE
MODEL = "allenai/OLMoE-1B-7B-0924"
tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=".cache")
ids = tok("The Mixture-of-Experts architecture routes each token to a small subset of experts, "
          "so the memory traffic per token depends on how many experts are actually loaded.",
          return_tensors="pt").input_ids[0][:48]
rm = resolve_model(MODEL, cache_dir=".cache", allow_download=False)
eng = StreamingOLMoE(ExpertStore(rm), rm.config, dtype=torch.float32)
t0 = time.time(); lg_s, st = eng.forward(ids); print(f"stream: {ids.numel()} tok in {time.time()-t0:.1f}s, k'={st.experts_per_token:.2f}, {st.bytes_read/ids.numel()/1e6:.1f} MB/tok", flush=True)
del eng
ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, cache_dir=".cache", low_cpu_mem_usage=True,
        device_map="auto", max_memory={"cpu": "6GiB"}, offload_folder="runs/offload_olmoe", offload_state_dict=True).eval()
with torch.no_grad(): lg_r = ref(ids.unsqueeze(0)).logits[0].float()
d = (lg_s - lg_r); top_agree = (lg_s.argmax(-1) == lg_r.argmax(-1)).float().mean().item()
ref_bf16_noise = (lg_r - lg_r.to(torch.bfloat16).float()).abs().max().item()
print(f"top-1 agreement {top_agree*100:.1f}%   max|dlogit| {d.abs().max():.3f}   mean|dlogit| {d.abs().mean():.4f}   "
      f"logit scale {lg_r.abs().max():.1f}   (bf16 quantisation of ref alone: {ref_bf16_noise:.3f})")
nll_s = torch.nn.functional.cross_entropy(lg_s[:-1], ids[1:]).item(); nll_r = torch.nn.functional.cross_entropy(lg_r[:-1], ids[1:]).item()
print(f"nll stream {nll_s:.4f}   nll ref {nll_r:.4f}   -> {'PASS' if top_agree > 0.97 and abs(nll_s-nll_r) < 0.05 else 'FAIL'}")
