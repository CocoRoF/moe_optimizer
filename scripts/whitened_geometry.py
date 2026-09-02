"""F9: re-measure F6 and F7 in activation space, using F8 calibration statistics.

Whitened cosine between two gate rows g1, g2 is exactly the correlation of
their pre-activations on the calibration distribution:

    corr(g1.x, g2.x) = g1^T C g2 / sqrt(g1^T C g1 * g2^T C g2),   C = E[x x^T]

so a high value means the two neurons *compute nearly the same thing on real
tokens*, whatever their raw-weight angle.  This is the metric every successful
training-free method optimises and the one every earlier finding here lacked.

Usage: python3 scripts/whitened_geometry.py runs/calib_olmoe_32k.pt
"""
import sys, math, time
sys.path.insert(0, "src")
import torch
from moe_optimizer.io.checkpoint import ExpertStore, resolve_model
from moe_optimizer.geometry.depth import layer_dictionary
from moe_optimizer.geometry.subspace import subspace_affinity
from moe_optimizer.types import Slot

torch.manual_seed(0)
calib = torch.load(sys.argv[1] if len(sys.argv) > 1 else "runs/calib_olmoe_32k.pt")
MODEL = sys.argv[2] if len(sys.argv) > 2 else calib.get("model", "allenai/OLMoE-1B-7B-0924")
stats = calib["layers"]
st = ExpertStore(resolve_model(MODEL, cache_dir=".cache", allow_download=False),
                 dtype=torch.float32)
A = st.arch
L, E, D, d = len(A.moe_layers), A.n_experts, A.d_expert, A.d_model
print(f"calibration: {calib['n_tokens']:,} tokens over {len(stats)} layers", flush=True)

# --- 0. how anisotropic is the residual stream?  (this is what makes F9 differ from F7)
print("\n=== residual-stream covariance anisotropy ===")
for l in sorted(stats)[:: max(1, len(stats) // 5)]:
    ev = torch.linalg.eigvalsh(stats[l]["input_cov"]).flip(0).clamp_min(0); p = ev / ev.sum()
    print(f"  layer {l:>2}: top-1 {p[0]:.3f}  top-16 {p[:16].sum():.3f}  top-128 {p[:128].sum():.3f}  "
          f"eff-rank {float(torch.exp(-(p*torch.log(p+1e-30)).sum())):5.0f} / {d}")

# --- whitening factor per layer: C = L L^T (ridge as in factorize.whiten)
def chol(l):
    C = stats[l]["input_cov"]; C = 0.5 * (C + C.T)
    C = C + 1e-4 * C.diagonal().mean() * torch.eye(d, dtype=C.dtype)
    return torch.linalg.cholesky(C).float()          # (d, d) lower
Ls = {l: chol(l) for l in A.moe_layers}

# --- 1. F7 in whitened space: NN cosine of gate rows, g -> L^T g
print("\n=== F7 whitened: neuron NN cosine (gate rows), per-layer whitening ===", flush=True)
N = L * E * D
rows = torch.empty((N, d), dtype=torch.float16); layer_of = torch.empty(N, dtype=torch.int16)
t0 = time.time()
for li, l in enumerate(A.moe_layers):
    Lt = Ls[l].T
    for e in range(E):
        w = st.expert(Slot(l, "gate"), e) @ Lt                     # rows now in whitened coords
        w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-8)
        i0 = (li * E + e) * D; rows[i0:i0+D] = w.half(); layer_of[i0:i0+D] = li
print(f"  whitened rows built in {time.time()-t0:.0f}s", flush=True)
Q = 8000; qidx = torch.randperm(N)[:Q]; q = rows[qidx].float()
best = torch.full((Q,), -2.0); best_x = torch.full((Q,), -2.0); CH = 65536
for s in range(0, N, CH):
    c = rows[s:s+CH].float(); sim = q @ c.T
    sim[qidx.unsqueeze(1) == torch.arange(s, s+c.shape[0]).unsqueeze(0)] = -2.0
    m = sim.max(1).values; best = torch.maximum(best, m)
    other = layer_of[s:s+c.shape[0]].unsqueeze(0) != layer_of[qidx].unsqueeze(1)
    best_x = torch.maximum(best_x, torch.where(other, sim, torch.tensor(-2.0)).max(1).values)
def qs(x): return " ".join(f"p{p}={x.quantile(p/100):.3f}" for p in (10,25,50,75,90,99))
print(f"  random baseline ~{math.sqrt(2*math.log(N)/d):.3f}   (raw-space F7 median was 0.191)")
print(f"  overall          : {qs(best)}")
print(f"  cross-layer only : {qs(best_x)}")
for th in (0.5, 0.7, 0.9):
    print(f"  frac NN > {th}: overall {(best>th).float().mean()*100:5.1f}%   cross-layer {(best_x>th).float().mean()*100:5.1f}%")

# --- 2. F6 in whitened space: adjacent-layer dictionary affinity + union rank, residual side of gate
print("\n=== F6 whitened: adjacent-layer residual-side dictionaries (gate, in) ===", flush=True)
def wdict(l, r):
    Lt = Ls[l].T; g = torch.zeros(d, d, dtype=torch.float64)
    for e in range(E):
        w = (st.expert(Slot(l, "gate"), e) @ Lt).double(); g += w.T @ w
    return torch.linalg.eigh(g)[1].flip(1)[:, :r].contiguous()
r = 64
mid = A.moe_layers[len(A.moe_layers) // 4]
pairs = [(mid + i, mid + i + 1) for i in range(3)]
B = {l: wdict(l, r) for l in sorted({x for pr in pairs for x in pr})}
for a, b in pairs:
    aff = subspace_affinity(B[a], B[b]); sv = torch.linalg.svdvals(torch.cat([B[a], B[b]], 1))
    print(f"  layers {a},{b}: affinity {aff:.3f} (raw was ~0.51)   union rank(>0.1) {int((sv>0.1).sum())}/128   angles<8deg: {128-int((sv>0.1).sum())}")
print("DONE")
