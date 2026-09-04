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
TASKS = ARGS[2].split(",") if len(ARGS) > 2 else ["hellaswag", "arc_easy", "piqa"]
def _engine_cls(cfg):
    from moe_optimizer.runtime.stream import StreamingOLMoE, StreamingQwen3MoE
    return StreamingQwen3MoE if cfg.get("model_type", "").startswith("qwen") else StreamingOLMoE

def load_task(name, n):
    """-> list of (context, [choices], label)."""
    if name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation").select(range(n))
        return [(r["ctx"], [" " + e for e in r["endings"]], int(r["label"])) for r in ds]
    if name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(n))
        return [("Question: " + r["question"] + "\nAnswer:", [" " + t for t in r["choices"]["text"]], r["choices"]["label"].index(r["answerKey"])) for r in ds]
    if name == "piqa":
        ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True).select(range(n))
        return [("Question: " + r["goal"] + "\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], int(r["label"])) for r in ds]
    raise ValueError(name)

def loglik(eng, tok, ctx, cont):
    c = tok(ctx, return_tensors="pt").input_ids[0]; full = tok(ctx + cont, return_tensors="pt").input_ids[0]
    n_ctx = c.numel()
    lg, _ = eng.forward(full)
    lp = F.log_softmax(lg[n_ctx - 1:-1], -1)
    return float(lp.gather(1, full[n_ctx:].unsqueeze(1)).sum())

def accuracy(eng, tok, examples):
    correct = 0
    for ctx, choices, label in examples:
        scores = [loglik(eng, tok, ctx, ch) for ch in choices]
        correct += int(max(range(len(scores)), key=lambda j: scores[j]) == label)
    return correct / len(examples)

if __name__ == "__main__":
    cal = torch.load(f"runs/policy_calib_{SHORT}.pt"); K, tr, ix, sc = cal["k"], cal["traces"], cal["indices"], cal["scale"]
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=".cache")
    rm = resolve_model(MODEL, cache_dir=".cache", allow_download=False); store = ExpertStore(rm); Eng = _engine_cls(rm.config)
    pols = [TopKPolicy(K), TopKPolicy(int(round(TARGET)))]; pols[1].name = f"top{int(round(TARGET))}(static)"
    q = ContributionPolicy(K, {l: torch.ones_like(sc[l]) for l in sc}, calibrate_taus(tr, None, ix, TARGET)); q.name = f"score_only@{TARGET}"; pols.append(q)
    c = ContributionPolicy(K, sc, calibrate_taus(tr, sc, ix, TARGET)); c.name = f"contribution@{TARGET}"; pols.append(c)
    bc = allocate_layer_budgets(tr, sc, ix, TARGET); cl = ContributionPolicy(K, sc, calibrate_taus_per_layer_target(tr, sc, ix, bc)); cl.name = f"contribution+layerbudget@{TARGET}"; pols.append(cl)
    res = {}
    for task in TASKS:
        ex = load_task(task, N_EX)
        for pol in pols:
            eng = Eng(store, rm.config, policy=pol); acc = accuracy(eng, tok, ex); del eng; gc.collect()
            res.setdefault(pol.name, {})[task] = acc
            print(f"  {task:<10} {pol.name:<30} acc={acc*100:5.1f}%  (n={len(ex)})", flush=True)
    json.dump(res, open(f"runs/policy_downstream_{SHORT}.json", "w"), indent=1); print("DONE")
