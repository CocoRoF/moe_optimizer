"""Downstream accuracy by log-likelihood scoring on the streaming engine.
Multiple-choice tasks need no generation: score = sum of continuation-token
log-probs given the context; predict argmax.  CPU-feasible at a few hundred
examples per task.  Reports accuracy per policy at one target k'."""
import sys, json, gc, math, torch, torch.nn.functional as F
sys.path.insert(0, "src")
from datasets import load_dataset
from transformers import AutoTokenizer
from moe_optimizer.io.checkpoint import ExpertStore, resolve_model
from moe_optimizer.runtime.stream import TopKPolicy, ContributionPolicy, LayerTopKPolicy
from moe_optimizer.runtime.calibrate import calibrate_taus, allocate_layer_budgets, calibrate_taus_per_layer_target
MODEL = next((a for a in sys.argv[1:] if "/" in a), "allenai/OLMoE-1B-7B-0924")
ARGS = [a for a in sys.argv[1:] if "/" not in a]; SHORT = MODEL.split("/")[-1].split("-")[0].lower()
N_EX = int(ARGS[0]) if ARGS else 300; TARGET = float(ARGS[1]) if len(ARGS) > 1 else 5.0
TASKS = ARGS[2].split(",") if len(ARGS) > 2 else ["hellaswag", "arc_easy", "arc_challenge", "openbookqa"]
def _engine_cls(cfg):
    from moe_optimizer.runtime.stream import StreamingOLMoE, StreamingQwen3MoE, StreamingQwen15MoE
    mt = cfg.get("model_type", "")
    if mt == "qwen2_moe": return StreamingQwen15MoE
    return StreamingQwen3MoE if mt.startswith("qwen") else StreamingOLMoE

def load_task(name, n):
    """-> list of (context, [choices], label)."""
    if name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation").select(range(n))
        return [(r["ctx"], [" " + e for e in r["endings"]], int(r["label"])) for r in ds]
    if name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(n))
        return [("Question: " + r["question"] + "\nAnswer:", [" " + t for t in r["choices"]["text"]], r["choices"]["label"].index(r["answerKey"])) for r in ds]
    if name == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(n))
        return [("Question: " + r["question"] + "\nAnswer:", [" " + t for t in r["choices"]["text"]], r["choices"]["label"].index(r["answerKey"])) for r in ds]
    if name == "openbookqa":
        ds = load_dataset("allenai/openbookqa", "main", split="test").select(range(n))
        return [("Question: " + r["question_stem"] + "\nAnswer:", [" " + t for t in r["choices"]["text"]], r["choices"]["label"].index(r["answerKey"])) for r in ds]
    if name == "piqa":   # script-based on the Hub; unsupported by datasets >= 4 -> skipped by the caller
        ds = load_dataset("ybisk/piqa", split="validation").select(range(n))
        return [("Question: " + r["goal"] + "\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], int(r["label"])) for r in ds]
    raise ValueError(name)

def loglik(eng, tok, ctx, cont, cache=None, ctx_logit=None):
    """Sum of log-probs of the continuation tokens given the context.  When a
    KV cache of the context (and the context's last-position logits) is passed,
    only the continuation is forwarded -- the context is shared across choices."""
    c = tok(ctx, return_tensors="pt").input_ids[0]; full = tok(ctx + cont, return_tensors="pt").input_ids[0]
    n_ctx = c.numel()
    if cache is None or not torch.equal(full[:n_ctx], c):
        lg, _ = eng.forward(full)
        lp = F.log_softmax(lg[n_ctx - 1:-1], -1)
    else:
        cc = {l: (k.clone(), v.clone()) for l, (k, v) in cache.items()}
        lg_c, _ = eng.forward(full[n_ctx:], cc)
        lg = torch.cat([ctx_logit.unsqueeze(0), lg_c[:-1]], 0) if full.numel() - n_ctx > 1 else ctx_logit.unsqueeze(0)
        lp = F.log_softmax(lg, -1)
    return float(lp.gather(1, full[n_ctx:].unsqueeze(1)).sum())

def accuracy(eng, tok, examples):
    """-> (accuracy, per-example correctness list) so policies can be compared paired."""
    hits = []
    for ctx, choices, label in examples:
        c = tok(ctx, return_tensors="pt").input_ids[0]
        cache = {}; lg_ctx, _ = eng.forward(c, cache)
        scores = [loglik(eng, tok, ctx, ch, cache, lg_ctx[-1]) for ch in choices]
        hits.append(int(max(range(len(scores)), key=lambda j: scores[j]) == label))
    return sum(hits) / len(hits), hits

def paired_acc_ci(a, b, B=5000, seed=0):
    """Bootstrap CI of accuracy(a) - accuracy(b) over the same examples."""
    import random; random.seed(seed); n = len(a); ds = []
    for _ in range(B):
        idx = [random.randrange(n) for _ in range(n)]; ds.append(sum(a[i] - b[i] for i in idx) / n)
    ds.sort(); return ds[int(.025 * B)], ds[int(.5 * B)], ds[int(.975 * B)]

if __name__ == "__main__":
    cal = torch.load(f"runs/policy_calib_{SHORT}.pt"); K, tr, ix, sc = cal["k"], cal["traces"], cal["indices"], cal["scale"]
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=".cache")
    rm = resolve_model(MODEL, cache_dir=".cache", allow_download=False); store = ExpertStore(rm); Eng = _engine_cls(rm.config)
    pols = [TopKPolicy(K), TopKPolicy(int(round(TARGET)))]; pols[1].name = f"top{int(round(TARGET))}(static)"
    q = ContributionPolicy(K, {l: torch.ones_like(sc[l]) for l in sc}, calibrate_taus(tr, None, ix, TARGET)); q.name = f"score_only@{TARGET}"; pols.append(q)
    c = ContributionPolicy(K, sc, calibrate_taus(tr, sc, ix, TARGET)); c.name = f"contribution@{TARGET}"; pols.append(c)
    bc = allocate_layer_budgets(tr, sc, ix, TARGET); cl = ContributionPolicy(K, sc, calibrate_taus_per_layer_target(tr, sc, ix, bc)); cl.name = f"contribution+layerbudget@{TARGET}"; pols.append(cl)
    import os
    CKPT = f"runs/policy_downstream_{SHORT}.ckpt.json"
    res = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    if res: print(f"resuming from {CKPT}: {sum(len(v) for v in res.values())} (policy, task) results", flush=True)
    for task in TASKS:
        try: ex = load_task(task, N_EX)
        except Exception as exc: print(f"  {task}: skipped ({type(exc).__name__}: {str(exc)[:80]})", flush=True); continue
        hits = {}
        for pol in pols:
            done = res.get(pol.name, {}).get(task)
            if done and len(done.get("hits", [])) == len(ex):          # resumed from checkpoint
                hits[pol.name] = done["hits"]; print(f"  {task:<10} {pol.name:<30} acc={done['acc']*100:5.1f}%  (n={len(ex)})  [checkpoint]", flush=True); continue
            eng = Eng(store, rm.config, policy=pol); acc, h = accuracy(eng, tok, ex); del eng; gc.collect()
            res.setdefault(pol.name, {})[task] = {"acc": acc, "hits": h}; hits[pol.name] = h
            json.dump(res, open(CKPT, "w"), indent=1)                    # checkpoint after every policy
            print(f"  {task:<10} {pol.name:<30} acc={acc*100:5.1f}%  (n={len(ex)})", flush=True)
        base = hits[pols[0].name]
        for pol in pols[1:]:
            lo, md, hi = paired_acc_ci(hits[pol.name], base)
            print(f"  {task:<10} {pol.name:<30} vs top-8: {md*100:+5.1f} pts [{lo*100:+5.1f}, {hi*100:+5.1f}]", flush=True)
        for a, b in ((f"contribution@{TARGET}", f"score_only@{TARGET}"), (f"contribution+layerbudget@{TARGET}", f"contribution@{TARGET}"), (f"contribution+layerbudget@{TARGET}", f"top{int(round(TARGET))}(static)")):
            if a in hits and b in hits:
                lo, md, hi = paired_acc_ci(hits[a], hits[b]); print(f"  {task:<10} {a} vs {b}: {md*100:+5.1f} pts [{lo*100:+5.1f}, {hi*100:+5.1f}]", flush=True)
    json.dump(res, open(f"runs/policy_downstream_{SHORT}.json", "w"), indent=1); print("DONE")
