"""Paired bootstrap CI for perplexity ratios between policies scored on the
same sequences.  Resamples sequences with replacement; reports the ratio of
mean-NLL-derived perplexities and its 95% interval.  Usage:
  python3 scripts/paired_bootstrap.py runs/policy_sweep_olmoe.json
"""
import sys, json, math, random
rows = {r["policy"]: r["per_seq_nll"] for r in json.load(open(sys.argv[1])) if r.get("per_seq_nll")}
if not rows: sys.exit("no per_seq_nll in file (run produced before logging was added)")
random.seed(0); B = 5000
def ci(a, b):
    n = len(a); rs = []
    for _ in range(B):
        idx = [random.randrange(n) for _ in range(n)]
        rs.append(math.exp(sum(a[i] for i in idx)/n) / math.exp(sum(b[i] for i in idx)/n) - 1)
    rs.sort(); return (rs[int(0.025*B)], rs[int(0.5*B)], rs[int(0.975*B)])
print(f"paired bootstrap over {len(next(iter(rows.values())))} sequences, B={B}")
for t in ("6.0", "5.0", "4.0"):
    c, q, s = rows.get(f"contribution@k'={t}"), rows.get(f"score_only@k'={t}"), rows.get(f"top{int(float(t))}(static)")
    cl, lt, ql = rows.get(f"contribution+layerbudget@k'={t}"), rows.get(f"layer_topk(static)@k'={t}"), rows.get(f"score_only+layerbudget@k'={t}")
    def lab(lo, hi): return "SIGNIFICANT (better)" if hi < 0 else "SIGNIFICANT (worse)" if lo > 0 else "n.s."
    for a, bb, name in ((cl, c, "contribution+layerbudget vs contribution"), (cl, lt, "contribution+layerbudget vs layer_topk(static)"),
                        (cl, s, "contribution+layerbudget vs static top-k"), (lt, s, "layer_topk(static) vs static top-k"), (cl, ql, "contribution+layerbudget vs score_only+layerbudget")):
        if a and bb: lo, md, hi = ci(a, bb); print(f"  k'~{t}: {name:<50} {md*100:+5.1f}%  [{lo*100:+5.1f}, {hi*100:+5.1f}]  {lab(lo, hi)}")
    def lab(lo, hi): return "SIGNIFICANT (better)" if hi < 0 else "SIGNIFICANT (worse)" if lo > 0 else "n.s."
    if c and q: lo, md, hi = ci(c, q); print(f"  k'~{t}: contribution vs score-only  {md*100:+5.1f}%  [{lo*100:+5.1f}, {hi*100:+5.1f}]  {lab(lo, hi)}")
    if c and s: lo, md, hi = ci(c, s); print(f"  k'~{t}: contribution vs static      {md*100:+5.1f}%  [{lo*100:+5.1f}, {hi*100:+5.1f}]  {lab(lo, hi)}")
