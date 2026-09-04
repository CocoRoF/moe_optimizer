"""F24: oracle diagnostic.  Rank by the TRUE per-token contribution w*||E_e(x)||
(all experts computed; not deployable) at a target mean k', against score-only
and mean-scale contribution.  If the oracle wins where the proxy ties, the
proxy is the problem; if the oracle also ties, the signal is."""
import sys, json, gc, torch
sys.path.insert(0, "src")
from datasets import load_dataset
from transformers import AutoTokenizer
from moe_optimizer.io.checkpoint import ExpertStore, resolve_model
from moe_optimizer.runtime.stream import TopKPolicy, ContributionPolicy
from moe_optimizer.runtime.calibrate import calibrate_taus
MODEL = next((a for a in sys.argv[1:] if "/" in a), "allenai/OLMoE-1B-7B-0924")
ARGS = [a for a in sys.argv[1:] if "/" not in a]; SHORT = MODEL.split("/")[-1].split("-")[0].lower()
N = int(ARGS[0]) if ARGS else 4096; TARGET = float(ARGS[1]) if len(ARGS) > 1 else 5.0
def _engine_cls(cfg):
    from moe_optimizer.runtime.stream import StreamingOLMoE, StreamingQwen3MoE, StreamingQwen15MoE
    mt = cfg.get("model_type", "")
    if mt == "qwen2_moe": return StreamingQwen15MoE
    return StreamingQwen3MoE if mt.startswith("qwen") else StreamingOLMoE
cal = torch.load(f"runs/policy_calib_{SHORT}.pt"); K, tr, ix, sc = cal["k"], cal["traces"], cal["indices"], cal["scale"]
tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=".cache")
text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N]
rm = resolve_model(MODEL, cache_dir=".cache", allow_download=False); store = ExpertStore(rm); Eng = _engine_cls(rm.config)

def run(policy, oracle_tau=None, name=None):
    eng = Eng(store, rm.config, policy=policy)
    if oracle_tau is not None: eng.oracle_tau = oracle_tau
    ppl, st = eng.perplexity(ids, verbose=False)
    k = sum(eng._oracle_k) / len(eng._oracle_k) if oracle_tau is not None else st.experts_per_token
    r = {"policy": name or policy.name, "ppl": ppl, "mean_k": k, "per_seq_nll": st.per_seq_nll}
    print(f"  {r['policy']:<26} ppl={ppl:8.3f}  k'={k:.2f}", flush=True); del eng; gc.collect(); return r

rows = [run(TopKPolicy(K))]
tau_s = calibrate_taus(tr, None, ix, TARGET); q = ContributionPolicy(K, {l: torch.ones_like(sc[l]) for l in sc}, tau_s); q.name = f"score_only@{TARGET}"; rows.append(run(q))
tau_c = calibrate_taus(tr, sc, ix, TARGET); c = ContributionPolicy(K, sc, tau_c); c.name = f"contribution@{TARGET}"; rows.append(run(c))
# oracle: same threshold search, but on the calibration set's *true* contributions we do not have;
# use the mean-scale taus as the starting point and bisect on the test run's realised k' (3 probes).
lo, hi, best = 0.0, 1.0, None
for _ in range(3):
    mid = 0.5 * (lo + hi); r = run(TopKPolicy(K), oracle_tau={l: mid for l in sc}, name=f"oracle(tau={mid:.3f})")
    best = r if best is None or abs(r["mean_k"] - TARGET) < abs(best["mean_k"] - TARGET) else best
    if r["mean_k"] > TARGET: lo = mid
    else: hi = mid
best["policy"] = f"oracle@{best['mean_k']:.2f}"; rows.append(best)
json.dump(rows, open(f"runs/policy_oracle_{SHORT}.json", "w"), indent=1)
import math, random
random.seed(0); B = 5000
def ci(a, b):
    n = len(a); rs = []
    for _ in range(B):
        i_ = [random.randrange(n) for _ in range(n)]; rs.append(math.exp(sum(a[i] for i in i_)/n)/math.exp(sum(b[i] for i in i_)/n) - 1)
    rs.sort(); return rs[int(.025*B)], rs[int(.5*B)], rs[int(.975*B)]
R = {r["policy"]: r["per_seq_nll"] for r in rows}
for a, b in ((best["policy"], q.name), (best["policy"], c.name), (c.name, q.name)):
    lo_, md, hi_ = ci(R[a], R[b]); print(f"  {a} vs {b}: {md*100:+5.1f}% [{lo_*100:+5.1f}, {hi_*100:+5.1f}] {'SIGNIFICANT' if hi_ < 0 or lo_ > 0 else 'n.s.'}")
print("DONE")
