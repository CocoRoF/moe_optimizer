"""E2/E3: perplexity, mean k', bytes/token, tok/s for every policy at matched target k'."""
import sys, json, gc, torch
sys.path.insert(0, "src")
from datasets import load_dataset
from transformers import AutoTokenizer
from moe_optimizer.io.checkpoint import ExpertStore, resolve_model
from moe_optimizer.runtime.stream import TopKPolicy, MassRatioPolicy, ContributionPolicy, ContributionRenormPolicy, LayerTopKPolicy
from moe_optimizer.runtime.calibrate import calibrate_taus, calibrate_taus_err, allocate_layer_budgets, calibrate_taus_per_layer_target
MODEL = next((a for a in sys.argv[1:] if "/" in a), "allenai/OLMoE-1B-7B-0924")
ARGS = [a for a in sys.argv[1:] if "/" not in a]
SHORT = MODEL.split("/")[-1].split("-")[0].lower()          # olmoe | qwen3
NO_RENORM = "--no-renorm" in sys.argv                 # F25 counterfactual (Qwen3 only)
FULL_RENORM = "--renorm-full" in sys.argv             # F25b clean counterfactual (Qwen3 only)
ARGS = [a for a in ARGS if a not in ("--no-renorm", "--renorm-full")]
def _engine_cls(cfg):
    from moe_optimizer.runtime.stream import StreamingOLMoE, StreamingQwen3MoE
    if cfg.get("model_type", "").startswith("qwen"):
        if NO_RENORM: return lambda *a, **k: StreamingQwen3MoE(*a, renorm=False, **k)
        if FULL_RENORM: return lambda *a, **k: StreamingQwen3MoE(*a, renorm="full", **k)
        return StreamingQwen3MoE
    return StreamingOLMoE
N = int(ARGS[0]) if ARGS else 2048
TARGETS = [float(x) for x in ARGS[1].split(",")] if len(ARGS) > 1 else [6.0, 5.0, 4.0]
cal = torch.load(f"runs/policy_calib_{SHORT}.pt"); K = cal["k"]; tr, ix, sc = cal["traces"], cal["indices"], cal["scale"]
tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=".cache")
text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N]
rm = resolve_model(MODEL, cache_dir=".cache", allow_download=False); store = ExpertStore(rm)

def _per_layer_mean(per_layer_k, n_seq):
    """per_layer_k is appended layer by layer for each sequence -> (n_seq, L) -> mean over sequences."""
    L = len(per_layer_k) // max(n_seq, 1)
    return [float(x) for x in torch.tensor(per_layer_k).view(n_seq, L).mean(0)] if L else []


def run(policy):
    eng = _engine_cls(rm.config)(store, rm.config, policy=policy)
    ppl, st = eng.perplexity(ids, verbose=False)
    del eng; gc.collect()
    r = {"policy": policy.name, "ppl": ppl, "mean_k": st.experts_per_token,
         "MB_per_tok": st.bytes_read / st.tokens / 1e6, "tok_per_s": st.tokens / st.seconds,
         "per_seq_nll": st.per_seq_nll,
         "per_layer_k": _per_layer_mean(st.per_layer_k, len(st.per_seq_nll))}
    print(f"  {r['policy']:<22} ppl={ppl:8.3f}  k'={r['mean_k']:.2f}  {r['MB_per_tok']:6.1f} MB/tok  {r['tok_per_s']:.2f} tok/s", flush=True)
    return r

rows = [run(TopKPolicy(K))]
b = MassRatioPolicy(K, cal["beta_median"]); b.name = "mass_ratio(median)"; rows.append(run(b))
for t in TARGETS:
    p = TopKPolicy(int(round(t))); p.name = f"top{int(round(t))}(static)"; rows.append(run(p))
    tau_s = calibrate_taus(tr, None, ix, t); q = ContributionPolicy(K, {l: torch.ones_like(sc[l]) for l in sc}, tau_s); q.name = f"score_only@k'={t}"; rows.append(run(q))
    tau_c = calibrate_taus(tr, sc, ix, t); c = ContributionPolicy(K, sc, tau_c); c.name = f"contribution@k'={t}"; rows.append(run(c))
    renorm = bool(rm.config.get("norm_topk_prob", False)) and not NO_RENORM and not FULL_RENORM
    tau_r = calibrate_taus_err(tr, sc, ix, t, renorm); r_ = ContributionRenormPolicy(K, sc, tau_r, renorm=renorm); r_.name = f"{r_.name}@k'={t}"; rows.append(run(r_))
    # ---- layer-adaptive budgets (training-free, from the same calibration traces)
    b_gate = allocate_layer_budgets(tr, None, ix, t)                     # budgets from gate-weight curves
    b_contrib = allocate_layer_budgets(tr, sc, ix, t)                    # budgets from contribution curves
    lt = LayerTopKPolicy(K, b_gate); lt.name = f"layer_topk(static)@k'={t}"; rows.append(run(lt))
    ql = ContributionPolicy(K, {l: torch.ones_like(sc[l]) for l in sc}, calibrate_taus_per_layer_target(tr, None, ix, b_gate)); ql.name = f"score_only+layerbudget@k'={t}"; rows.append(run(ql))
    cl = ContributionPolicy(K, sc, calibrate_taus_per_layer_target(tr, sc, ix, b_contrib)); cl.name = f"contribution+layerbudget@k'={t}"; rows.append(run(cl))
    rows[-1]["budgets"] = {int(l): v for l, v in b_contrib.items()}
json.dump(rows, open(f"runs/policy_sweep_{SHORT}{'_norenorm' if NO_RENORM else '_renormfull' if FULL_RENORM else ''}.json", "w"), indent=1)
print("DONE")
