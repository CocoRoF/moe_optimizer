# Findings — Contribution-Calibrated Dynamic Expert Skipping

All numbers below are reproducible on CPU from this repository: `scripts/reproduce.sh` runs every step in order; raw outputs ship in `results/`; the environment, model revisions, data selection and seeds are in `results/ENV.md`. The chronological lab log (F1–F24, including the compression-era negative results this project began with) is `docs/LOG.md`.

**One-sentence result.** On a MoE whose router does *not* renormalise the kept top-k weights (OLMoE-1B-7B), ranking experts by `gate weight × calibrated output scale` instead of gate weight alone improves matched-budget perplexity by 3.0 % [2.0, 4.0] at k′≈5 and 7.0 % [5.3, 8.8] at k′≈4, matches an oracle that knows every expert's true output, and yields a 1.80× batch-1 decode speedup; on a router that *does* renormalise (Qwen3-30B-A3B) no norm-based rule — proxy, error model, or oracle — beats the score, and static top-k dominates.

---

## 1. Setup and hyperparameters

| item | value |
|---|---|
| **Models** | OLMoE-1B-7B (`allenai/OLMoE-1B-7B-0924` @ `6d84c48`), Qwen3-30B-A3B (`Qwen/Qwen3-30B-A3B` @ `ad44e77`) — details in `results/ENV.md` |
| **Engine** | `runtime/stream.py`: layer-streaming CPU decoder, weights read from the safetensors mmap one layer at a time, **fp32 arithmetic**; reference model bf16. Validated: top-1 agreement 100 % on both models; NLL 3.207 vs 3.212 (OLMoE), 3.508 vs 3.503 (Qwen3) (F13, F17) |
| **Routed experts** | top-k = 8 (both). OLMoE `norm_topk_prob=False` (raw softmax probs kept); Qwen3 `norm_topk_prob=True` (kept weights renormalised to 1) |
| **Calibration** | WikiText-2-raw-v1 **train**, first N tokens: N = 2,048 (OLMoE), 4,096 (Qwen3). Per layer: sorted top-8 router probs and expert indices per token; per-expert output scale `s[l,e] = mean ‖E_e(x)‖` over tokens routed to e (experts never routed → layer mean) |
| **Evaluation text** | WikiText-2-raw-v1 **test**, first N tokens, 512-token sequences; N per table |
| **Policies** | `top-k(static)`: top-k′ by score. `mass_ratio(median)`: arXiv:2512.21911 / Lu et al. 2024 — skip the m lowest-score experts when their share of top-8 mass < β_m, β_m = per-layer **median** over calibration tokens. `score_only@k′`: rank by score, keep smallest prefix whose cumulative share ≥ 1−τ_l. `contribution@k′`: same with c_e = w_e·s_e. `contribution_sq`: squared-share variant (renorm=False). `contribution_renorm`: error-model variant with amplification term. `oracle`: all experts computed, keep by true per-token w_e·‖E_e(x)‖ (diagnostic only) |
| **τ calibration** | per layer, bisection (40 iterations on [0,1]) on the calibration traces so the mean kept count equals the target k′; `min_keep=1` |
| **Statistics** | paired bootstrap over sequences on per-sequence mean NLL, B = 5000, `random.seed(0)`, 95 % percentile interval of the perplexity ratio |
| **Decode benchmark** | batch 1, KV cache, 32-token prompt, 64 teacher-forced steps (OLMoE) / 48 (Qwen3); bytes counted as fp32 bytes moved (2× the bf16 bytes on disk) |
| **Compute** | 11 threads (`OMP_NUM_THREADS=11`), no GPU; engine anonymous RSS 3.7–4.5 GB |

---

## 2. Main result — OLMoE-1B-7B, matched mean k′ (F20)

8,192 test tokens = 16 sequences. Calibration 2,048 tokens. Perplexity; Δ vs top-8 = 11.051.

| policy | mean k′ | ppl | Δ vs top-8 |
|---|---:|---:|---:|
| top-8 (reference) | 8.00 | 11.051 | — |
| mass_ratio (median) — published rule | 3.20 | 45.213 | +309 % |
| static top-5 | 5.00 | 12.299 | +11.3 % |
| score_only @5 | 4.94 | 12.579 | +13.8 % |
| **contribution @5** | 4.96 | **12.202** | **+10.4 %** |
| contribution_sq @5 | 4.92 | 12.345 | +11.7 % |
| static top-4 | 4.00 | 14.117 | +27.7 % |
| score_only @4 | 3.94 | 14.737 | +33.4 % |
| **contribution @4** | 3.96 | **13.701** | **+24.0 %** |
| contribution_sq @4 | 3.92 | 14.020 | +26.9 % |

Paired bootstrap (16 sequences), perplexity ratio, 95 % CI:

| comparison | k′≈5 | k′≈4 |
|---|---|---|
| contribution vs score_only | **−3.0 % [−4.0, −2.0]** | **−7.0 % [−8.8, −5.3]** |
| contribution vs static top-k | −0.8 % [−1.9, +0.3] | **−2.9 % [−4.9, −1.1]** |
| contribution_sq vs contribution | +1.2 % [+0.0, +2.3] | +2.3 % [+1.4, +3.4] |

Independent replications of the direction and size: 2,048 tokens (F15: −0.9 / −3.0 / −8.5 % at k′≈6/5/4) and the 1,024-token WikiText slice of the tail study (F19: −2.8 / −7.1 % at k′≈5/4).

**Reading.** Score-only dynamic skipping is *worse than a static cut* at every budget; the router score is near-orthogonal to output magnitude on this model (§4), so ranking by it ranks by noise. The calibrated scale recovers that and passes static at k′≈4. The published median rule has no quality target and skips 60 % of experts here; it destroys the model.

---

## 3. Bandwidth and speed — batch-1 decode, OLMoE (F16)

| policy | k′ | MB / token | expert loads / token | tok/s | speedup |
|---|---:|---:|---:|---:|---:|
| top-8 | 8.00 | 2148 | 128.0 | 1.86 | 1.00× |
| static top-6 | 6.00 | 1745 | 96.0 | 2.82 | 1.52× |
| contribution @6 | 5.90 | 1726 | 94.5 | 2.81 | 1.51× |
| static top-5 | 5.00 | 1544 | 80.0 | 3.32 | 1.78× |
| **contribution @5** | 4.89 | 1521 | 78.2 | **3.35** | **1.80×** |
| static top-4 | 4.00 | 1342 | 64.0 | 2.58* | 1.39× |
| contribution @4 | 3.87 | 1316 | 61.9 | 3.02 | 1.62× |

Bytes/token = 16 layers × k′ × 3 matrices × 2048×1024 fp32 + attention: **each skipped expert saves 16.8 MB per token, exactly linearly**. KV-cache path reproduces uncached logits to 0.000. *Row marked \* is system contention (its bytes are on the line). This table's 63-token perplexity column is not a quality measurement and is omitted.

---

## 4. Why the score is not enough — calibration statistics (F14, F18)

| | OLMoE-1B-7B | Qwen3-30B-A3B |
|---|---:|---:|
| within-layer CV of output scale s[l,e] (mean over layers) | 0.255 (0.17–0.39) | 0.341 (0.16–3.33) |
| Pearson r(s[e], mean gate weight of e), mean over layers | **+0.17** | **−0.05** |
| layers with r < −0.2 | 0 / 16 | 15 / 48 |
| sorted top-8 raw-prob medians w1 … w8 | .079 .064 .054 .046 .040 .034 .030 .027 | .077 .058 .048 .040 .034 .030 .027 .024 |
| tokens needing k′ ≥ 7 for 90 % of top-8 mass | — | 98 % |

Both routers are flat within the top-8 (head/tail ≈ 3×) and neither encodes output magnitude; on Qwen3 the correlation is negative in a third of the layers. Any rule that ranks by score alone is ranking by a quantity unrelated to what the expert adds to the residual.

---

## 5. Tail: math and code degrade less than text — OLMoE (F19)

1,024 tokens per corpus (WikiText-2 test / GSM8K q+a / HumanEval prompt+solution). Δ perplexity vs top-8.

| policy | wikitext | gsm8k | code | tail / wikitext |
|---|---:|---:|---:|---:|
| score_only @5 | +8.4 % | +6.4 % | +6.9 % | 1.00 |
| **contribution @5** | **+5.4 %** | **+5.3 %** | **+6.0 %** | 1.10 |
| score_only @4 | +24.8 % | +16.1 % | +13.5 % | 1.00 |
| **contribution @4** | **+15.9 %** | **+12.5 %** | **+11.3 %** | 1.00 |

Gate G4 (worst-domain < 3× mean): passes. Contribution wins every cell.

---

## 6. Oracle: is the 64-float proxy the limit? (F24)

4,096 tokens, k′≈5. The oracle computes every routed expert and keeps by the *true* per-token contribution; it is not deployable (bytes unchanged) and bounds what any contribution rule can do.

| | OLMoE | Qwen3 |
|---|---|---|
| score_only @5 | 11.643 (k′ 4.94) | 12.945 (k′ 5.00) |
| contribution (mean scale) @5 | 11.338 (k′ 4.97) | 12.987 (k′ 5.03) |
| **oracle** | 11.353 (k′ 4.63) | 13.626 (k′ 4.79) |
| oracle vs score_only | **−2.5 % [−4.2, −0.8]** | **+5.1 % [+1.7, +9.9]** |
| oracle vs contribution | +0.1 % [−1.1, +1.4] | +4.8 % [+2.6, +8.2] |

**OLMoE:** the mean-scale proxy ties the oracle at a smaller budget — the calibration is sufficient. **Qwen3:** the oracle is worse than the score; the contribution *signal* fails there, not the proxy.

---

## 7. Qwen3-30B-A3B: a characterised null (F21, F22)

4,096 test tokens (8 sequences), calibration 4,096 tokens, k′=5. Δ vs top-8 = 11.879.

| policy | k′ | ppl | Δ |
|---|---:|---:|---:|
| mass_ratio (median) | 4.44 | 184.0 | ×15 |
| **static top-5** | 5.00 | **12.642** | **+6.4 %** |
| score_only @5 | 5.00 | 12.945 | +9.0 % |
| contribution @5 | 5.03 | 12.987 | +9.3 % |
| contribution_renorm @5 (error model) | 5.02 | 13.123 | +10.5 % |

contribution vs score_only: +0.4 % [−1.7, +2.3] (tie). With 1,024 calibration tokens (F21) it had been a +5.3 % loss — calibration starvation (≈64 tokens per expert), fixed by 4×. Budget hogging by outlier experts: rejected (per-layer k′ flat at 4.9–5.2). Renormalisation-aware error model: worse.

**Interpretation.** Qwen3 renormalises the kept weights; dropping expert e also rescales every survivor by W_all/W_P, so the change in the layer output depends on the *directions* of kept and dropped outputs, which are only approximately orthogonal (whitened NN cosine median 0.40). No norm-only ranking sees that. On OLMoE removal is subtraction and the norm suffices. The counterfactual — Qwen3 with renormalisation switched off (F25) — is the measurement of this claim and is pending.

---

## 8. Reported numbers from the literature (not reproduced here; different models and protocols)

| method | training-free | model | reported | note |
|---|---|---|---|---|
| Lu et al. (ACL 2024) | yes | Mixtral-8x7B (top-2) | dynamic skip of 2nd expert, speedup with small loss | k = 2 only |
| arXiv:2512.21911 (generalised skip) | yes | DeepSeek-R1 | m=3 (16 % sparsity) lossless; m=4 (22 %) degrades math 82→75 | inside speculative verification only; **the median rule reproduced here collapses both models** |
| LExI (arXiv:2509.02753) | yes (data-free) | Qwen1.5-MoE | +10 % accuracy vs pruning at equal throughput | static per-layer k |
| Opportunistic Expert Activation (arXiv:2511.02237) | yes | Qwen3-30B / 235B | −39 % / −15 % MoE-layer latency at batch 16 | batch-aware, not batch-1 |
| SERE (ICLR 2026) | yes | — | up to 2.0× batch decode | re-routes tokens to similar experts |
| ZEDA (arXiv:2605.18643) | **no** (self-distillation) | Qwen3-30B, GLM-4.7-Flash | >50 % fewer expert ops, ~1.20× e2e, minimal loss | the ceiling training buys |
| AdapMoE (arXiv:2408.10284) | yes | Mixtral / Phi-3.5 | −25 % activations, 1.35× | layer-static sensitivity |

This work: OLMoE-1B-7B, batch-1, training-free — 38 % fewer expert loads (k′≈5) for +10.4 % perplexity and 1.80× decode; 51 % fewer (k′≈4) for +24 %. Roughly three times the quality-per-skip of the published training-free rule it replaces; not the ZEDA operating point.

---

## 9. Gates (pre-registered) and their outcomes

| gate | condition | outcome |
|---|---|---|
| G1 | within-layer CV of s < 0.15 → abandon | OLMoE 0.255, Qwen3 0.341 — **pass** |
| G2 | contribution not better than score-only at matched k′ | OLMoE **pass** (CI excludes 0); Qwen3 **fail** (tie) |
| G3 | bytes/token reduction does not become tok/s | **pass** (1.80×) |
| G4 | worst-domain > 3× mean degradation | **pass** (≤ 1.10) |
| E4 rule | run tail only if contribution wins ≥ 2 of 3 budgets | 3/3 |

---

## 10. Limitations

1. **Absolute cost.** +10 % perplexity for 38 % fewer expert loads. ZEDA reaches ~50 % with training at minimal loss; this closes none of that gap.
2. **One positive model.** The mechanism works on an unnormalised router and is a null on a renormalised one; generality is a claim about router type, supported by one model each and one pending counterfactual.
3. **Perplexity only.** No downstream accuracy without a GPU.
4. **Regime.** Batch-1 fp32 CPU decode. GPU batched serving converts bytes to latency sublinearly.
5. **The two principled variants lost.** The orthogonality-derived error model and the renormalisation-aware rule are both worse than the linear heuristic on both models (F22, F23).
6. **Routing profile.** Both base models are flat within the top-8; instruct/reasoning models with a "certain head" (arXiv:2602.02443) may leave less for any dynamic rule to fix — unmeasured.

## Pending

F25 — Qwen3 without top-k renormalisation (counterfactual for §7). F26 — Qwen3 batch-1 decode bandwidth (counterpart to §3). Both running via `scripts/reproduce.sh` steps 4 and 7.
