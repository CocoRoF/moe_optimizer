"""Exact loss-vs-savings report: for each policy, expert-load reduction, bytes/token
reduction, perplexity loss vs top-k with paired 95% CI, and -- when downstream
results exist -- accuracy loss in points with paired CI.  This is the table the
project is judged on.  Usage: python3 scripts/report_pareto.py [olmoe|qwen3]"""
import sys, json, math, random, os
SHORT = sys.argv[1] if len(sys.argv) > 1 else "olmoe"; K = 8
random.seed(0); B = 5000
def ci_ratio(a, b):
    n = len(a); rs = []
    for _ in range(B):
        idx = [random.randrange(n) for _ in range(n)]; rs.append(math.exp(sum(a[i] for i in idx)/n)/math.exp(sum(b[i] for i in idx)/n)-1)
    rs.sort(); return rs[int(.025*B)], rs[int(.5*B)], rs[int(.975*B)]
def ci_diff(a, b):
    n = len(a); ds = []
    for _ in range(B):
        idx = [random.randrange(n) for _ in range(n)]; ds.append(sum(a[i]-b[i] for i in idx)/n)
    ds.sort(); return ds[int(.025*B)], ds[int(.5*B)], ds[int(.975*B)]
files = [f for f in (f"results/{SHORT}/policy_sweep_{SHORT}_layerbudget.json", f"results/{SHORT}/policy_sweep_{SHORT}.json", f"runs/policy_sweep_{SHORT}.json") if os.path.exists(f)]
rows = json.load(open(files[0])); base = next(r for r in rows if r["policy"] == "top8")
n_tok = len(base["per_seq_nll"]) * 512
ds_path = next((f for f in (f"results/{SHORT}/policy_downstream_{SHORT}.json", f"runs/policy_downstream_{SHORT}.json", f"runs/policy_downstream_{SHORT}.ckpt.json") if os.path.exists(f)), None)
ds = json.load(open(ds_path)) if ds_path else {}
tasks = sorted({t for v in ds.values() for t in v}) if ds else []
print(f"# {SHORT}: source {files[0]} ({n_tok:,} test tokens, {len(base['per_seq_nll'])} seqs){'; downstream ' + ds_path if ds_path else '; downstream: NOT MEASURED'}")
hdr = f"{'policy':<34} {'k′':>5} {'loads −%':>9} {'bytes −%':>9} {'ppl':>8} {'Δppl% [95% CI]':>24}" + "".join(f" {'Δ'+t+' pts [CI]':>26}" for t in tasks)
print(hdr); print("-" * len(hdr))
for r in sorted(rows, key=lambda r: -r["mean_k"]):
    lo, md, hi = ci_ratio(r["per_seq_nll"], base["per_seq_nll"]) if r["policy"] != "top8" else (0, 0, 0)
    loads = (1 - r["mean_k"] / K) * 100; byt = (1 - r["MB_per_tok"] / base["MB_per_tok"]) * 100
    line = f"{r['policy']:<34} {r['mean_k']:>5.2f} {loads:>8.1f}% {byt:>8.1f}% {r['ppl']:>8.3f} {md*100:>+7.1f}% [{lo*100:+.1f},{hi*100:+.1f}]"
    for t in tasks:
        name = r["policy"].replace("@k'=", "@")
        a = ds.get(name, {}).get(t, {}).get("hits"); b = ds.get("top8", {}).get(t, {}).get("hits")
        if a and b and r["policy"] != "top8":
            l2, m2, h2 = ci_diff(a, b); line += f" {m2*100:>+7.1f} [{l2*100:+.1f},{h2*100:+.1f}]"
        elif r["policy"] == "top8" and b: line += f" {'acc %.1f%%' % (100*sum(b)/len(b)):>26}"
        else: line += f" {'—':>26}"
    print(line)
print("\nloads −% = expert loads avoided per token (k′ vs top-8); bytes −% is the prefill bytes/token which amortises across a sequence — batch-1 decode bytes scale with loads −% (F16).")
