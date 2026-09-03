"""E2/E3: perplexity, mean k', bytes/token, tok/s for every policy at matched target k'."""
import sys, json, time, torch
sys.path.insert(0, "src")
from datasets import load_dataset
from transformers import AutoTokenizer
from moe_optimizer.io.checkpoint import ExpertStore, resolve_model
from moe_optimizer.runtime.stream import StreamingOLMoE, TopKPolicy, MassRatioPolicy, ContributionPolicy
from moe_optimizer.runtime.calibrate import calibrate_taus, cumulative_share, mass_ratio_medians
MODEL = "allenai/OLMoE-1B-7B-0924"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
TARGETS = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [6.0, 5.0, 4.0]
cal = torch.load("runs/policy_calib_olmoe.pt"); K = cal["k"]; tr, ix, sc = cal["traces"], cal["indices"], cal["scale"]
tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=".cache")
text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N]
rm = resolve_model(MODEL, cache_dir=".cache", allow_download=False); store = ExpertStore(rm)

def run(policy):
    eng = StreamingOLMoE(store, rm.config, policy=policy)
    ppl, st = eng.perplexity(ids, verbose=False)
    del eng; gc.collect()
    r = {"policy": policy.name, "ppl": ppl, "mean_k": st.experts_per_token,
         "MB_per_tok": st.bytes_read / st.tokens / 1e6, "tok_per_s": st.tokens / st.seconds}
    print(f"  {r['policy']:<22} ppl={ppl:8.3f}  k'={r['mean_k']:.2f}  {r['MB_per_tok']:6.1f} MB/tok  {r['tok_per_s']:.2f} tok/s", flush=True)
    return r

rows = [run(TopKPolicy(K))]
b = MassRatioPolicy(K, cal["beta_median"]); b.name = "mass_ratio(median)"; rows.append(run(b))
for t in TARGETS:
    p = TopKPolicy(int(round(t))); p.name = f"top{int(round(t))}(static)"; rows.append(run(p))
    tau_s = calibrate_taus(tr, None, ix, t); q = ContributionPolicy(K, {l: torch.ones_like(sc[l]) for l in sc}, tau_s); q.name = f"score_only@k'={t}"; rows.append(run(q))
    tau_c = calibrate_taus(tr, sc, ix, t); c = ContributionPolicy(K, sc, tau_c); c.name = f"contribution@k'={t}"; rows.append(run(c))
json.dump(rows, open("runs/policy_sweep_olmoe.json", "w"), indent=1)
print("DONE")
