"""E4: worst-domain check.  Same policies, three corpora, report the ratio of
per-domain perplexity degradation to the WikiText degradation (MoEXBench's
warning: the mean hides the tail).  Runs only for policies E2 has justified."""
import sys, json, torch
sys.path.insert(0, "src")
from datasets import load_dataset
from transformers import AutoTokenizer
from moe_optimizer.io.checkpoint import ExpertStore, resolve_model
from moe_optimizer.runtime.stream import StreamingOLMoE, TopKPolicy, ContributionPolicy
from moe_optimizer.runtime.calibrate import calibrate_taus
MODEL = "allenai/OLMoE-1B-7B-0924"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
TARGET = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
cal = torch.load("runs/policy_calib_olmoe.pt"); K, tr, ix, sc = cal["k"], cal["traces"], cal["indices"], cal["scale"]
tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=".cache")
corpora = {
 "wikitext": "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip()),
 "gsm8k":    "\n\n".join(r["question"] + " " + r["answer"] for r in load_dataset("openai/gsm8k", "main", split="test").select(range(400))),
 "code":     "\n\n".join(r["content"] for r in load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train").select(range(60))),
}
rm = resolve_model(MODEL, cache_dir=".cache", allow_download=False); store = ExpertStore(rm)
pols = [TopKPolicy(K)]
q = ContributionPolicy(K, {l: torch.ones_like(sc[l]) for l in sc}, calibrate_taus(tr, None, ix, TARGET)); q.name = f"score_only@{TARGET}"; pols.append(q)
c = ContributionPolicy(K, sc, calibrate_taus(tr, sc, ix, TARGET)); c.name = f"contribution@{TARGET}"; pols.append(c)
res = {}
for name, text in corpora.items():
    ids = tok(text, return_tensors="pt").input_ids[0][:N]
    for pol in pols:
        ppl, st = StreamingOLMoE(store, rm.config, policy=pol).perplexity(ids, verbose=False)
        res.setdefault(pol.name, {})[name] = {"ppl": ppl, "k": st.experts_per_token}
        print(f"  {name:<9} {pol.name:<20} ppl={ppl:8.3f}  k'={st.experts_per_token:.2f}", flush=True)
base = res[pols[0].name]
print("\nrelative degradation vs top-8 (worst domain / wikitext):")
for pn in list(res)[1:]:
    d = {dom: res[pn][dom]["ppl"] / base[dom]["ppl"] - 1 for dom in corpora}
    print(f"  {pn:<20} " + "  ".join(f"{dom}:{d[dom]*100:+5.1f}%" for dom in corpora) + f"   tail/mean ratio {max(d.values())/max(d['wikitext'],1e-9):.2f}")
json.dump(res, open("runs/policy_tail_olmoe.json", "w"), indent=1); print("DONE")
