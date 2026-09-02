"""F7 probe: does every neuron have a near-duplicate somewhere in the model?

A neuron (one intermediate unit) is the finest unit with NO remaining symmetry:
its (gate_row, up_row, down_col) triple is fully identified.  If most neurons
have a close nearest neighbour across experts and layers, a shared prototype
pool -- "one block" -- can approximate every expert by indexing.  Measured on
gate rows first (cheapest, 2048-dim).  Random-vector baseline for NN cosine
among N vectors in d dims is ~sqrt(2 ln N / d).
"""
import sys, math, time
sys.path.insert(0, "src")
import torch
from moe_optimizer.io.checkpoint import ExpertStore, resolve_model
from moe_optimizer.types import Slot

torch.manual_seed(0)
st = ExpertStore(resolve_model("allenai/OLMoE-1B-7B-0924", cache_dir=".cache", allow_download=False),
                 dtype=torch.float32)
A = st.arch
L, E, D = len(A.moe_layers), A.n_experts, A.d_expert
N = L * E * D
print(f"{N:,} gate rows of dim {A.d_model}; random-NN baseline ~{math.sqrt(2*math.log(N)/A.d_model):.3f}", flush=True)

rows = torch.empty((N, A.d_model), dtype=torch.float16)
layer_of = torch.empty(N, dtype=torch.int16); expert_of = torch.empty(N, dtype=torch.int16)
t0 = time.time()
for li, l in enumerate(A.moe_layers):
    for e in range(E):
        w = st.expert(Slot(l, "gate"), e)
        w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-8)
        i0 = (li * E + e) * D
        rows[i0:i0 + D] = w.half(); layer_of[i0:i0 + D] = li; expert_of[i0:i0 + D] = e
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

Q = 8000
qidx = torch.randperm(N)[:Q]
q = rows[qidx].float()
best = torch.full((Q,), -2.0); best_j = torch.zeros(Q, dtype=torch.long)
best_xlayer = torch.full((Q,), -2.0)      # best NN restricted to OTHER layers
CH = 65536
t0 = time.time()
for s in range(0, N, CH):
    c = rows[s:s + CH].float()
    sim = q @ c.T                                       # (Q, CH)
    # exclude self
    self_mask = (qidx.unsqueeze(1) == torch.arange(s, s + c.shape[0]).unsqueeze(0))
    sim[self_mask] = -2.0
    m, j = sim.max(1)
    upd = m > best; best[upd] = m[upd]; best_j[upd] = j[upd] + s
    other = layer_of[s:s + c.shape[0]].unsqueeze(0) != layer_of[qidx].unsqueeze(1)
    sim_x = torch.where(other, sim, torch.tensor(-2.0))
    mx = sim_x.max(1).values
    updx = mx > best_xlayer; best_xlayer[updx] = mx[updx]
print(f"NN search {time.time()-t0:.0f}s", flush=True)

same_layer = layer_of[best_j] == layer_of[qidx]
same_expert = same_layer & (expert_of[best_j] == expert_of[qidx])
def qs(x): return " ".join(f"p{p}={x.quantile(p/100):.3f}" for p in (10, 25, 50, 75, 90, 99))
print("\n=== nearest-neighbour cosine, gate rows, OLMoE-1B-7B ===")
print(f"overall            : {qs(best)}")
print(f"cross-layer only   : {qs(best_xlayer)}")
print(f"NN is in same layer : {same_layer.float().mean()*100:.1f}%   same expert: {same_expert.float().mean()*100:.1f}%")
for th in (0.5, 0.7, 0.9):
    print(f"frac with NN cos > {th}: overall {(best>th).float().mean()*100:5.1f}%   cross-layer {(best_xlayer>th).float().mean()*100:5.1f}%")
print("DONE")
