"""A layer-streaming CPU decoder for OLMoE-class MoE models.

Why this exists
---------------
Two reasons, one practical and one scientific.

Practical: this machine cannot hold the 14 GB bf16 model under the 30 %-free
rule while another session is resident.  Streaming one layer's weights from the
safetensors mmap at a time peaks at ~2 GB.

Scientific: it is the deployment regime the mechanism targets.  On a
bandwidth-bound decoder every expert that is *not* selected is 3 matrices that
are *not* read.  ``bytes_read`` is therefore a first-class output of every
forward pass, and a dynamic-skipping policy is scored by (perplexity, mean
experts per token, bytes per token, wall time) -- not by FLOPs.

Fidelity to the reference implementation (transformers OlmoeForCausalLM)
-----------------------------------------------------------------------
* RMSNorm in fp32 with eps 1e-5, weight applied after casting back.
* q_norm / k_norm are RMSNorms over the *full* projected width (2048), applied
  before the head split.  MHA: 16 heads, 16 kv heads, head_dim 128.
* RoPE default, theta 1e4, rotate-half convention.
* Router: softmax over all experts in fp32, top-k, ``norm_topk_prob=False`` --
  the kept weights are the raw probabilities, NOT renormalised.
* Experts: down( silu(gate x) * up x ), weighted by the raw top-k prob, summed.
* Untied lm_head.
Validated against the HF model in ``tests``/``scripts/validate_stream.py``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..io.checkpoint import ExpertStore
from ..types import Slot


# --------------------------------------------------------------------------- #
#  expert-selection policies
# --------------------------------------------------------------------------- #
class ExpertPolicy:
    """Given the router's (T, E) probabilities, return (T, k') indices + weights.

    ``select`` may return fewer than k experts per token by setting weight 0 for
    dropped slots; the engine skips the load of any expert whose weight is 0 for
    every token in the batch.
    """

    name = "abstract"

    def select(self, probs: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TopKPolicy(ExpertPolicy):
    """The reference behaviour: static top-k, raw probabilities."""

    def __init__(self, k: int) -> None:
        self.k = k; self.name = f"top{k}"

    def select(self, probs, layer):
        w, i = probs.topk(self.k, dim=-1)
        return i, w


class MassRatioPolicy(ExpertPolicy):
    """The training-free baseline of arXiv:2512.21911, generalised from Lu et al.
    (ACL 2024): after top-k, drop the m lowest-weight experts when their share
    of the top-k gate mass is below beta[layer][m].  Thresholds are calibrated
    per layer; the paper uses the *median* over calibration tokens.
    """

    def __init__(self, k: int, beta: dict[int, list[float]]) -> None:
        self.k, self.beta = k, beta; self.name = "mass_ratio"

    def select(self, probs, layer):
        w, i = probs.topk(self.k, dim=-1)                    # descending
        total = w.sum(-1, keepdim=True).clamp_min(1e-12)
        tail = w.flip(-1).cumsum(-1).flip(-1) / total         # tail[:, j] = share of experts j..k-1
        keep = torch.ones_like(w, dtype=torch.bool)
        b = self.beta.get(layer)
        if b:
            for m in range(len(b), 0, -1):                    # largest allowed skip first
                skip = tail[:, self.k - m] < b[m - 1]
                keep[skip, self.k - m:] = False
                # tokens already decided keep their decision
                tail = torch.where(skip.unsqueeze(1), torch.full_like(tail, 2.0), tail)
        return i, w * keep


class ContributionPolicy(ExpertPolicy):
    """Ours: skip on *contribution* mass, not gate mass.

    An expert's effect on the residual is ~ w_e * ||E_e(x)||, and expert output
    norms differ by large factors within a layer (the router score is a dispatch
    signal trained under a load-balancing loss, not a magnitude signal).  With a
    calibrated per-expert scale s[layer][e] = E||E_e(x)||, the ranking and the
    tail-share test are done on w_e * s_e.  Thresholds tau[layer] are calibrated
    to a *target mean k'* rather than to a fixed skip rate.
    """

    def __init__(self, k: int, scale: dict[int, torch.Tensor], tau: dict[int, float],
                 min_keep: int = 1) -> None:
        self.k, self.scale, self.tau, self.min_keep = k, scale, tau, min_keep
        self.name = "contribution"

    def select(self, probs, layer):
        w, i = probs.topk(self.k, dim=-1)
        s = self.scale[layer][i]                              # (T, k) calibrated output scale
        c = w * s
        c_sorted, order = c.sort(-1, descending=True)
        share = c_sorted / c_sorted.sum(-1, keepdim=True).clamp_min(1e-12)
        cum = share.cumsum(-1)
        # keep the smallest prefix whose cumulative contribution share >= 1 - tau
        thr = 1.0 - self.tau.get(layer, 0.0)
        keep_sorted = torch.cat([torch.ones_like(cum[:, :1], dtype=torch.bool),
                                 cum[:, :-1] < thr], dim=1)
        keep_sorted[:, : self.min_keep] = True
        keep = torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
        return i, w * keep


# --------------------------------------------------------------------------- #
#  the engine
# --------------------------------------------------------------------------- #
@dataclass
class StepStats:
    tokens: int = 0
    bytes_read: int = 0
    expert_loads: int = 0            # (layer, expert) pairs loaded
    experts_per_token: float = 0.0   # mean k' over tokens and layers
    seconds: float = 0.0
    per_layer_k: list = field(default_factory=list)


class StreamingOLMoE:
    def __init__(self, store: ExpertStore, config: dict, policy: ExpertPolicy | None = None,
                 dtype: torch.dtype = torch.float32, threads: int = 11) -> None:
        torch.set_num_threads(threads)
        self.store, self.cfg, self.dtype = store, config, dtype
        self.L = config["num_hidden_layers"]; self.E = config["num_experts"]
        self.k = config["num_experts_per_tok"]; self.d = config["hidden_size"]
        self.H = config["num_attention_heads"]; self.hd = self.d // self.H
        self.eps = config.get("rms_norm_eps") or 1e-5
        self.theta = config.get("rope_theta", 10000.0)
        self.policy = policy or TopKPolicy(self.k)
        self.expert_bytes = 3 * config["intermediate_size"] * self.d * 2   # bf16 on disk
        self._g = lambda k: self.store.get(k).to(self.dtype)
        self.embed = self._g("model.embed_tokens.weight")
        self.final_norm = self._g("model.norm.weight")
        self.lm_head = self._g("lm_head.weight")
        self.record_output_norms: dict[int, torch.Tensor] | None = None

    # ---- primitives ------------------------------------------------------
    def _rms(self, x, w):
        xf = x.float(); v = xf.pow(2).mean(-1, keepdim=True)
        return w * (xf * torch.rsqrt(v + self.eps)).to(x.dtype)

    def _rope(self, T, offset):
        pos = torch.arange(offset, offset + T, dtype=torch.float32)
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        f = torch.outer(pos, inv); emb = torch.cat([f, f], -1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def _rot(x):
        a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-b, a], -1)

    # ---- one layer -------------------------------------------------------
    def _layer(self, l, h, cos, sin, cache, stats):
        p = f"model.layers.{l}."
        x = self._rms(h, self._g(p + "input_layernorm.weight"))
        q = self._rms(x @ self._g(p + "self_attn.q_proj.weight").T, self._g(p + "self_attn.q_norm.weight"))
        kk = self._rms(x @ self._g(p + "self_attn.k_proj.weight").T, self._g(p + "self_attn.k_norm.weight"))
        v = x @ self._g(p + "self_attn.v_proj.weight").T
        T = x.shape[0]
        q = q.view(T, self.H, self.hd).transpose(0, 1); kk = kk.view(T, self.H, self.hd).transpose(0, 1)
        v = v.view(T, self.H, self.hd).transpose(0, 1)
        c, s = cos.unsqueeze(0), sin.unsqueeze(0)
        q = q * c + self._rot(q) * s; kk = kk * c + self._rot(kk) * s
        if cache is not None:
            if l in cache:
                kk = torch.cat([cache[l][0], kk], 1); v = torch.cat([cache[l][1], v], 1)
            cache[l] = (kk, v)
        S = kk.shape[1]
        att = (q @ kk.transpose(1, 2)) * (self.hd ** -0.5)
        mask = torch.full((T, S), float("-inf")).triu(S - T + 1)
        att = F.softmax((att + mask).float(), -1).to(self.dtype) @ v
        att = att.transpose(0, 1).reshape(T, self.d) @ self._g(p + "self_attn.o_proj.weight").T
        stats.bytes_read += 4 * self.d * self.d * 2 + 4 * self.d * 2
        h = h + att

        x = self._rms(h, self._g(p + "post_attention_layernorm.weight"))
        logits = x @ self._g(p + "mlp.gate.weight").T
        probs = F.softmax(logits.float(), -1)
        idx, w = self.policy.select(probs, l)                  # (T, k), (T, k)
        w = w.to(self.dtype)
        out = torch.zeros_like(x)
        used = idx[w > 0].unique()
        stats.expert_loads += used.numel(); stats.bytes_read += used.numel() * self.expert_bytes
        stats.per_layer_k.append(float((w > 0).sum(1).float().mean()))
        for e in used.tolist():
            slot_tok, slot_pos = torch.where(idx == e)
            we = w[slot_tok, slot_pos]
            m = we > 0
            slot_tok, we = slot_tok[m], we[m]
            xe = x[slot_tok]
            g = xe @ self.store.expert(Slot(l, "gate"), e).to(self.dtype).T
            u = xe @ self.store.expert(Slot(l, "up"), e).to(self.dtype).T
            y = (F.silu(g) * u) @ self.store.expert(Slot(l, "down"), e).to(self.dtype).T
            if self.record_output_norms is not None:
                self.record_output_norms.setdefault(l, torch.zeros(self.E, 2, dtype=torch.float64))
                self.record_output_norms[l][e, 0] += y.norm(dim=-1).double().sum()
                self.record_output_norms[l][e, 1] += y.shape[0]
            out.index_add_(0, slot_tok, y * we.unsqueeze(1))
        return h + out

    # ---- public ----------------------------------------------------------
    @torch.no_grad()
    def forward(self, ids: torch.Tensor, cache: dict | None = None) -> tuple[torch.Tensor, StepStats]:
        """ids: (T,) token ids.  Returns (T, vocab) logits and stats."""
        st = StepStats(tokens=ids.numel()); t0 = time.perf_counter()
        offset = 0 if not cache or 0 not in cache else cache[0][0].shape[1]
        h = self.embed[ids]
        cos, sin = self._rope(ids.numel(), offset)
        for l in range(self.L):
            h = self._layer(l, h, cos, sin, cache, st)
        h = self._rms(h, self.final_norm)
        logits = h @ self.lm_head.T
        st.seconds = time.perf_counter() - t0
        st.experts_per_token = sum(st.per_layer_k) / max(len(st.per_layer_k), 1)
        return logits.float(), st

    @torch.no_grad()
    def perplexity(self, ids: torch.Tensor, seq_len: int = 512, verbose=True) -> tuple[float, StepStats]:
        n = ids.numel() // seq_len; ids = ids[: n * seq_len].view(n, seq_len)
        nll = 0.0; cnt = 0; tot = StepStats(); t0 = time.perf_counter()
        for i in range(n):
            lg, st = self.forward(ids[i])
            nll += F.cross_entropy(lg[:-1], ids[i, 1:], reduction="sum").item(); cnt += seq_len - 1
            tot.tokens += st.tokens; tot.bytes_read += st.bytes_read; tot.expert_loads += st.expert_loads
            tot.per_layer_k += st.per_layer_k
            if verbose:
                print(f"    [{i+1}/{n}] ppl {math.exp(nll/cnt):8.3f}  k'={sum(st.per_layer_k)/len(st.per_layer_k):.2f}  "
                      f"{st.bytes_read/st.tokens/1e6:.1f} MB/tok  {time.perf_counter()-t0:5.0f}s", flush=True)
        tot.seconds = time.perf_counter() - t0
        tot.experts_per_token = sum(tot.per_layer_k) / max(len(tot.per_layer_k), 1)
        return math.exp(nll / cnt), tot


class StreamingQwen3MoE(StreamingOLMoE):
    """Qwen3-MoE variant.  Differences from OLMoE, each verified against the HF
    source: grouped-query attention (32 q heads / 4 kv heads, head_dim 128);
    q_norm / k_norm are RMSNorms over *head_dim*, applied per head after the
    reshape; rope theta from config (1e6); ``norm_topk_prob=True`` so the kept
    top-k probabilities are renormalised to sum to one; experts are
    ``moe_intermediate_size`` wide.  Tensor names are identical in form.
    """

    def __init__(self, store, config, policy=None, dtype=torch.float32, threads: int = 11):
        super().__init__(store, config, policy, dtype, threads)
        self.hd = config.get("head_dim") or self.d // self.H
        self.KV = config["num_key_value_heads"]; self.groups = self.H // self.KV
        self.expert_bytes = 3 * config["moe_intermediate_size"] * self.d * 2

    def _layer(self, l, h, cos, sin, cache, stats):
        p = f"model.layers.{l}."
        x = self._rms(h, self._g(p + "input_layernorm.weight"))
        T = x.shape[0]
        q = (x @ self._g(p + "self_attn.q_proj.weight").T).view(T, self.H, self.hd)
        kk = (x @ self._g(p + "self_attn.k_proj.weight").T).view(T, self.KV, self.hd)
        v = (x @ self._g(p + "self_attn.v_proj.weight").T).view(T, self.KV, self.hd)
        q = self._rms(q, self._g(p + "self_attn.q_norm.weight")).transpose(0, 1)      # per-head norm
        kk = self._rms(kk, self._g(p + "self_attn.k_norm.weight")).transpose(0, 1)
        v = v.transpose(0, 1)
        c, s = cos.unsqueeze(0), sin.unsqueeze(0)
        q = q * c + self._rot(q) * s; kk = kk * c + self._rot(kk) * s
        if cache is not None:
            if l in cache:
                kk = torch.cat([cache[l][0], kk], 1); v = torch.cat([cache[l][1], v], 1)
            cache[l] = (kk, v)
        kk = kk.repeat_interleave(self.groups, 0); v = v.repeat_interleave(self.groups, 0)
        S = kk.shape[1]
        att = (q @ kk.transpose(1, 2)) * (self.hd ** -0.5)
        mask = torch.full((T, S), float("-inf")).triu(S - T + 1)
        att = F.softmax((att + mask).float(), -1).to(self.dtype) @ v
        att = att.transpose(0, 1).reshape(T, self.H * self.hd) @ self._g(p + "self_attn.o_proj.weight").T
        stats.bytes_read += (2 * self.H * self.hd * self.d + 2 * self.KV * self.hd * self.d) * 2
        h = h + att

        x = self._rms(h, self._g(p + "post_attention_layernorm.weight"))
        probs = F.softmax((x @ self._g(p + "mlp.gate.weight").T).float(), -1)
        idx, w = self.policy.select(probs, l)
        # norm_topk_prob=True: renormalise over the experts actually kept
        w = (w / w.sum(-1, keepdim=True).clamp_min(1e-12)).to(self.dtype)
        out = torch.zeros_like(x)
        used = idx[w > 0].unique()
        stats.expert_loads += used.numel(); stats.bytes_read += used.numel() * self.expert_bytes
        stats.per_layer_k.append(float((w > 0).sum(1).float().mean()))
        for e in used.tolist():
            slot_tok, slot_pos = torch.where(idx == e)
            we = w[slot_tok, slot_pos]; m = we > 0; slot_tok, we = slot_tok[m], we[m]
            xe = x[slot_tok]
            g = xe @ self.store.expert(Slot(l, "gate"), e).to(self.dtype).T
            u = xe @ self.store.expert(Slot(l, "up"), e).to(self.dtype).T
            y = (F.silu(g) * u) @ self.store.expert(Slot(l, "down"), e).to(self.dtype).T
            if self.record_output_norms is not None:
                self.record_output_norms.setdefault(l, torch.zeros(self.E, 2, dtype=torch.float64))
                self.record_output_norms[l][e, 0] += y.norm(dim=-1).double().sum()
                self.record_output_norms[l][e, 1] += y.shape[0]
            out.index_add_(0, slot_tok, y * we.unsqueeze(1))
        return h + out


@torch.no_grad()
def decode_benchmark(engine: StreamingOLMoE, ids: torch.Tensor, prefill: int = 32,
                     steps: int = 64) -> dict:
    """Batch-1 decode: one token per forward with the KV cache.

    This is the regime the mechanism targets and the one ``perplexity`` does not
    measure: in prefill an expert read once serves every token in the sequence
    that routes to it, so bytes/token is amortised; in decode every token's
    experts are read fresh.  Reports per-token bytes, expert loads and wall time
    over ``steps`` teacher-forced tokens after a ``prefill``-token prompt, plus a
    consistency check that the cached path reproduces the uncached logits.
    """
    ids = ids[: prefill + steps]
    cache: dict = {}
    lg_p, _ = engine.forward(ids[:prefill], cache)
    lg_full, _ = engine.forward(ids[: prefill + 1])          # uncached reference for token `prefill`
    lg_c, _ = engine.forward(ids[prefill: prefill + 1], cache)
    consistency = float((lg_c[-1] - lg_full[-1]).abs().max())
    cache = {}; engine.forward(ids[:prefill], cache)
    tot_bytes = tot_loads = 0; ks = []; t0 = time.perf_counter(); nll = 0.0
    for i in range(prefill, prefill + steps):
        lg, st = engine.forward(ids[i: i + 1], cache)
        tot_bytes += st.bytes_read; tot_loads += st.expert_loads; ks += st.per_layer_k
        if i + 1 < ids.numel():
            nll += F.cross_entropy(lg[-1:], ids[i + 1: i + 2]).item()
    sec = time.perf_counter() - t0
    return {"policy": engine.policy.name, "steps": steps, "tok_per_s": steps / sec,
            "MB_per_tok": tot_bytes / steps / 1e6, "expert_loads_per_tok": tot_loads / steps,
            "mean_k": sum(ks) / len(ks), "decode_ppl": math.exp(nll / max(steps - 1, 1)),
            "cache_consistency_max_dlogit": consistency}
