"""F30: per-layer signal selection from a calibration pass, then a matched-k'
comparison of the resulting MixedPolicy against score-only and contribution."""
import sys, json, gc, torch
sys.path.insert(0, "src")
from datasets import load_dataset
from transformers import AutoTokenizer
from moe_optimizer.io.checkpoint import ExpertStore, resolve_model
from moe_optimizer.runtime.stream import TopKPolicy, ContributionPolicy, MixedPolicy, signal_selection_pass, mode_selection_pass
from moe_optimizer.runtime.calibrate import calibrate_taus, cumulative_share, tau_for_target_k
MODEL = next((a for a in sys.argv[1:] if "/" in a), "allenai/OLMoE-1B-7B-0924")
ARGS = [a for a in sys.argv[1:] if "/" not in a and not a.startswith("--")]; SHORT = MODEL.split("/")[-1].split("-")[0].lower()
N_CAL = int(ARGS[0]) if ARGS else 1024; N_TEST = int(ARGS[1]) if len(ARGS) > 1 else 4096
TARGETS = [float(x) for x in ARGS[2].split(",")] if len(ARGS) > 2 else [5.0]
def _engine_cls(cfg):
    from moe_optimizer.runtime.stream import StreamingOLMoE, StreamingQwen3MoE, StreamingQwen15MoE
    mt = cfg.get("model_type", "")
    if mt == "qwen2_moe": return StreamingQwen15MoE
    return StreamingQwen3MoE if mt.startswith("qwen") else StreamingOLMoE
cal = torch.load(f"runs/policy_calib_{SHORT}.pt"); K, tr, ix, sc = cal["k"], cal["traces"], cal["indices"], cal["scale"]
tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=".cache")
rm = resolve_model(MODEL, cache_dir=".cache", allow_download=False); store = ExpertStore(rm); Eng = _engine_cls(rm.config)
train = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip())
cal_ids = tok(train, return_tensors="pt").input_ids[0][:N_CAL]
eng = Eng(store, rm.config); sel = signal_selection_pass(eng, cal_ids, sc); del eng; gc.collect()

use = {l: v["use_scale"] for l, v in sel.items()}

MODE = "--mode" in sys.argv; modes = {}
if MODE:
    print("=== F31: per-layer dynamic-vs-static at the first target ===", flush=True)
    eng = Eng(store, rm.config); modes = mode_selection_pass(eng, cal_ids, sc, use, TARGETS[0]); del eng; gc.collect()
    print(f"  layers choosing dynamic: {sum(1 for l in modes if modes[l]['mode']=='dynamic')} / {len(modes)}", flush=True)
    for l in sorted(modes)[:: max(1, len(modes) // 8)]: print(f"  L{l:02d} err static {modes[l]['err_static']:.4f}  dynamic {modes[l]['err_dynamic']:.4f}  -> {modes[l]['mode']}")
    json.dump({str(l): v for l, v in modes.items()}, open(f"runs/mode_select_{SHORT}.json", "w"), indent=1)
print(f"=== F30 signal selection, {MODEL}, {cal_ids.numel()} calibration tokens ===")
print("  layers choosing contribution:", sum(use.values()), "/", len(use), " ->", [l for l, u in use.items() if u])
for l in sorted(sel)[:: max(1, len(sel) // 8)]:
    print(f"  L{l:02d} rel.err drop-2: score {sel[l]['score_err'][1]:.4f}  contrib {sel[l]['contrib_err'][1]:.4f}  -> {'contrib' if use[l] else 'score'}")
json.dump({"model": MODEL, "selection": {int(l): v for l, v in sel.items()}}, open(f"runs/signal_select_{SHORT}.json", "w"), indent=1)
test = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(test, return_tensors="pt").input_ids[0][:N_TEST]
def run(pol):
    e = Eng(store, rm.config, policy=pol); ppl, st = e.perplexity(ids, verbose=False); del e; gc.collect()
    r = {"policy": pol.name, "ppl": ppl, "mean_k": st.experts_per_token, "per_seq_nll": st.per_seq_nll}
    print(f"  {pol.name:<26} ppl={ppl:8.3f}  k'={st.experts_per_token:.2f}", flush=True); return r
rows = [run(TopKPolicy(K))]
for t in TARGETS:
    p = TopKPolicy(int(round(t))); p.name = f"top{int(round(t))}(static)"; rows.append(run(p))
    q = ContributionPolicy(K, {l: torch.ones_like(sc[l]) for l in sc}, calibrate_taus(tr, None, ix, t)); q.name = f"score_only@k'={t}"; rows.append(run(q))
    c = ContributionPolicy(K, sc, calibrate_taus(tr, sc, ix, t)); c.name = f"contribution@k'={t}"; rows.append(run(c))
    # mixed: tau per layer under its chosen signal
    tau = {}
    for l in sc:
        cs = cumulative_share({l: tr[l]}, sc if use[l] else None, ix)[l]; tau[l] = tau_for_target_k(cs, t)
    m = MixedPolicy(K, sc, tau, use); m.name = f"mixed@k'={t}"; rows.append(run(m))
    if MODE:
        mm = MixedPolicy(K, sc, tau, use, mode={l: v["mode"] for l, v in modes.items()}, static_k={l: t for l in modes}); mm.name = f"mixed+mode@k'={t}"; rows.append(run(mm))
json.dump(rows, open(f"runs/policy_sweep_{SHORT}_mixed.json", "w"), indent=1)
import math, random
random.seed(0); B = 5000; R = {r["policy"]: r["per_seq_nll"] for r in rows}
def ci(a, b):
    n = len(a); rs = []
    for _ in range(B):
        i_ = [random.randrange(n) for _ in range(n)]; rs.append(math.exp(sum(a[i] for i in i_)/n)/math.exp(sum(b[i] for i in i_)/n) - 1)
    rs.sort(); return rs[int(.025*B)], rs[int(.5*B)], rs[int(.975*B)]
def lab(lo, hi): return "SIGNIFICANT (better)" if hi < 0 else "SIGNIFICANT (worse)" if lo > 0 else "n.s."
for t in TARGETS:
    for a, b in ((f"mixed@k'={t}", f"score_only@k'={t}"), (f"mixed@k'={t}", f"contribution@k'={t}"), (f"mixed@k'={t}", f"top{int(round(t))}(static)")):
        lo, md, hi = ci(R[a], R[b]); print(f"  {a} vs {b}: {md*100:+5.1f}% [{lo*100:+5.1f}, {hi*100:+5.1f}] {lab(lo, hi)}")
print("F30 DONE")
