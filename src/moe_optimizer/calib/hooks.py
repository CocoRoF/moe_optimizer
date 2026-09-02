"""Calibration statistics from a live forward pass.

Everything measured before this module existed was measured in raw weight
space, and arXiv:2606.03465 is explicit that this is the wrong metric: a
Frobenius-optimal approximation preserves aggregate mass while distorting the
directions activations actually depend on.  Findings F1b, F5, F6 and F7 in
docs/FINDINGS.md are all subject to that caveat.  These hooks are what lift it.

Per MoE layer we accumulate

  ``resid_cov``   E[x x^T] of the residual-stream input, (d_model, d_model) f64.
                  Whitens gate/up inputs and down outputs.  Its eigen-spectrum
                  is also the anisotropy that decides whether activation-space
                  similarity can differ from weight-space similarity at all.
  ``counts``      how often each expert was selected
  ``gate_mass``   summed router weight per expert
  ``coact``       (E, E) joint-selection counts, for NPMI affinity
  ``inter_sq``    per-expert, per-neuron E[h^2] of the intermediate activation
                  h = act(gate x) * (up x), (E, d_ff).  Diagonal only: the full
                  per-expert (d_ff, d_ff) matrix is 4 GB across a 16-layer model
                  and does not fit beside the model in 28 GB.

The experts module is fused (weights are 3-D tensors, one loop over hit
experts), so intermediate activations are recomputed inside the hook from the
same inputs rather than captured -- costs one extra expert pass per token, which
is negligible against the rest of the forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class SlotStats:
    layer: int
    n_experts: int
    d_model: int
    d_ff: int
    n_tokens: int = 0
    resid_cov: torch.Tensor = field(default=None)
    counts: torch.Tensor = field(default=None)
    gate_mass: torch.Tensor = field(default=None)
    coact: torch.Tensor = field(default=None)
    inter_sq: torch.Tensor = field(default=None)
    inter_n: torch.Tensor = field(default=None)

    def __post_init__(self):
        E, d, f = self.n_experts, self.d_model, self.d_ff
        self.resid_cov = torch.zeros(d, d, dtype=torch.float64)
        self.counts = torch.zeros(E, dtype=torch.int64)
        self.gate_mass = torch.zeros(E, dtype=torch.float64)
        self.coact = torch.zeros(E, E, dtype=torch.int64)
        self.inter_sq = torch.zeros(E, f, dtype=torch.float64)
        self.inter_n = torch.zeros(E, dtype=torch.int64)

    def finalize(self) -> dict:
        n = max(self.n_tokens, 1)
        return {
            "layer": self.layer, "n_tokens": self.n_tokens,
            "input_cov": self.resid_cov / n,
            "counts": self.counts, "gate_mass": self.gate_mass,
            "coactivation": self.coact,
            "importance": self.counts.double() / n,
            "inter_sq": self.inter_sq / self.inter_n.clamp_min(1).unsqueeze(1).double(),
        }


class CalibrationCollector:
    """Attach to a HF OLMoE-style model; call ``.stats()`` after the forward passes."""

    def __init__(self, model, layers: list[int]) -> None:
        cfg = model.config
        self.slots: dict[int, SlotStats] = {}
        self._handles = []
        for l in layers:
            blk = model.model.layers[l].mlp
            # Qwen-MoE keeps the dense FFN width in intermediate_size and the expert
            # width in moe_intermediate_size; OLMoE has only the former.
            d_ff = getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
            st = SlotStats(l, cfg.num_experts, cfg.hidden_size, d_ff)
            self.slots[l] = st
            self._handles.append(blk.experts.register_forward_pre_hook(self._make_hook(st, blk)))

    def _make_hook(self, st: SlotStats, blk):
        def hook(module, args):
            x, top_idx, top_w = args[0], args[1], args[2]
            x = x.reshape(-1, x.shape[-1])
            with torch.no_grad():
                xf = x.to(torch.float64)
                st.resid_cov += xf.T @ xf
                st.n_tokens += x.shape[0]

                E = st.n_experts
                onehot = F.one_hot(top_idx, num_classes=E)          # (T, k, E)
                sel = onehot.sum(1)                                  # (T, E) 0/1
                st.counts += sel.sum(0)
                st.gate_mass += (onehot * top_w.unsqueeze(-1).to(onehot.dtype if onehot.is_floating_point() else torch.float32)).sum((0, 1)).double() \
                    if False else torch.einsum("tk,tke->e", top_w.float(), onehot.float()).double()
                st.coact += (sel.T.float() @ sel.float()).long()

                hit = sel.any(0).nonzero(as_tuple=True)[0]
                for e in hit.tolist():
                    tok = sel[:, e].nonzero(as_tuple=True)[0]
                    gate, up = F.linear(x[tok], module.gate_up_proj[e]).chunk(2, dim=-1)
                    h = (module.act_fn(gate) * up).double()
                    st.inter_sq[e] += (h * h).sum(0)
                    st.inter_n[e] += h.shape[0]
        return hook

    def stats(self) -> dict[int, dict]:
        return {l: s.finalize() for l, s in self.slots.items()}

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
