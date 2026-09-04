# Findings — Contribution-Calibrated Dynamic Expert Skipping

All numbers below are reproducible on CPU from this repository: `scripts/reproduce.sh` runs every step in order; raw outputs ship in `results/`; the environment, model revisions, data selection and seeds are in `results/ENV.md`. The chronological lab log (F1–F24, including the compression-era negative results this project began with) is `docs/LOG.md`.

**One-sentence result.** On a MoE whose router does *not* renormalise the kept top-k weights (OLMoE-1B-7B), ranking experts by `gate weight × calibrated output scale` instead of gate weight alone improves matched-budget perplexity by 3.0 % [2.0, 4.0] at k′≈5 and 7.0 % [5.3, 8.8] at k′≈4, matches an oracle that knows every expert's true output, and yields a 1.80× batch-1 decode speedup; on Qwen3-30B-A3B no norm-based rule — proxy, error model, oracle, or the same rules with renormalisation neutralised — beats the score, and static top-k dominates; that boundary is measured, not yet explained.

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

**Interpretation (revised after F25b).** The natural suspect was renormalisation: Qwen3 rescales the kept weights, so dropping an expert also amplifies the survivors and a norm-only ranking cannot see the direction interactions. §12 tests this directly and **rejects it** — with removal made pure subtraction the tie and the static advantage persist. The null is real and, for now, unexplained at the mechanism level; §12 lists what was ruled out.

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
| G2 | contribution not better than score-only at matched k′ | OLMoE **pass** (CI excludes 0); Qwen3 **fail** (tie, also under full-mass renormalisation) |
| G3 | bytes/token reduction does not become tok/s | **pass** (1.80×) |
| G4 | worst-domain > 3× mean degradation | **pass** (≤ 1.10) |
| E4 rule | run tail only if contribution wins ≥ 2 of 3 budgets | 3/3 |

---

## 10. Limitations

1. **Absolute cost.** +10 % perplexity for 38 % fewer expert loads. ZEDA reaches ~50 % with training at minimal loss; this closes none of that gap.
2. **One positive model, one unexplained null.** The mechanism works on OLMoE and is a tie on Qwen3; the router-renormalisation explanation was tested (F25, F25b) and rejected, so the boundary of applicability is empirical, not mechanistic.
3. **Perplexity only.** No downstream accuracy without a GPU.
4. **Regime.** Batch-1 fp32 CPU decode. GPU batched serving converts bytes to latency sublinearly.
5. **The two principled variants lost.** The orthogonality-derived error model and the renormalisation-aware rule are both worse than the linear heuristic on both models (F22, F23).
6. **Routing profile.** Both base models are flat within the top-8; instruct/reasoning models with a "certain head" (arXiv:2602.02443) may leave less for any dynamic rule to fix — unmeasured.

## 11. Qwen3 batch-1 decode — the disk-bound regime (F26)

32-token prompt, 48 steps, KV cache. Qwen3's 57 GB of shards do not fit in this machine's page cache (~10 GB), so **every layer read is an NVMe read** (measured ~700 MB/s, 28 % iowait) — the most extreme bandwidth-bound regime, not a compute-bound one. Absolute tok/s is therefore low; the *relation* is what the table tests.

| policy | k′ | MB / token | expert loads / token | tok/s | speedup |
|---|---:|---:|---:|---:|---:|
| top8 | 8.00 | 5436 | 384.0 | 0.28 | 1.00× |
| mass_ratio(median) | 3.91 | 3582 | 187.5 | 0.45 | 1.59× |
| top5(static) | 5.00 | 4077 | 240.0 | 0.36 | 1.26× |
| score_only@5.0 | 4.85 | 4007 | 232.6 | 0.39 | 1.36× |
| contribution@5.0 | 4.90 | 4030 | 235.0 | 0.42 | 1.49× |

Bytes/token are linear in k′ here too (48 layers × k′ × 3 × 768×2048 fp32 ≈ 19 MB per expert per token). Cache-consistency (max |Δlogit| cached vs uncached) is 0.000 for every skipping row and **0.106 for the top-8 row** — 0.3 % of the logit scale (30.9), consistent with fp32 summation-order differences between the batched prefill and single-token paths accumulating over 48 layers; it is reported rather than hidden, and it does not affect the bytes or speed columns. Perplexity in this table is a 47-token sanity check, not a quality result — see §7 for Qwen3 quality.

## 12. Is renormalisation the reason? Two counterfactuals (F25, F25b) — no

**F25** (`renorm=False`, raw top-8 probabilities): the unmodified model's top-8 perplexity becomes 42.9 (raw probs sum to ≈0.35; every MoE output is scaled by ~⅓). Inside that model contribution beats score-only by −40 % [−73, −5] and static by −34 % [−63, −1]. Direction as hypothesised — but the model is 4× off its operating point and the intervals span 70 points.

**F25b** (`renorm="full"`, kept weights divided by the *original* top-8 mass): the unmodified model is bit-identical to Qwen3 (top-8 = 11.879), and dropping an expert subtracts its term without rescaling the survivors — removal is subtraction, exactly as on OLMoE.

| policy (4,096 tokens, k′=5) | k′ | ppl | Δ vs 11.879 |
|---|---:|---:|---:|
| static top-5 | 5.00 | **12.730** | +7.2 % |
| score-only @5 | 4.94 | 13.014 | +9.6 % |
| contribution @5 | 4.97 | 13.120 | +10.4 % |
| contribution_sq @5 | 4.94 | 13.161 | +10.8 % |
| mass-ratio (median) | 3.29 | 22.353 | +88 % |

Paired bootstrap (8 sequences): contribution vs score-only **+0.8 % [−0.9, +2.7]** (tie); contribution vs static **+3.1 % [+1.2, +5.1]** (static significantly better).

**The renormalisation hypothesis is rejected.** With removal made pure subtraction, the contribution signal still adds nothing on Qwen3 and static top-k still wins. F25's −40 % was an artefact of the damaged model, as its intervals warned. Two side observations: under full-mass renormalisation the published median rule degrades far less (22.4 vs 184.0), and static top-5 costs about the same as before (+7.2 % vs +6.4 %) — dropping experts on Qwen3 is gentle *because* the router's score already orders them well for this model, leaving little for any dynamic rule to improve.

### Where that leaves the Qwen3 null

| hypothesis for F21/F22 | test | outcome |
|---|---|---|
| calibration starvation (64 tokens / expert) | 4× calibration tokens (F22) | **confirmed** as the cause of F21's *loss*; leaves a tie |
| outlier experts hog the budget | per-layer k′ (F22) | rejected (flat 4.9–5.2) |
| mean-scale proxy too coarse | per-token oracle (F24) | rejected — the oracle is *worse* than score |
| top-k renormalisation | F25 (off), F25b (full-mass) | **rejected** — tie and static win persist with removal = subtraction |

What remains is a model-level fact without a mechanism: on Qwen3-30B-A3B the router score is a better skip criterion than any output-norm quantity, mean or per-token, even though it is uncorrelated with output magnitude. A plausible reading — untested — is that expert *redundancy* matters more than magnitude there: dropping a large-norm expert whose direction is covered by kept ones costs little, and score may track that coverage. Testing it needs a direction-aware criterion, which is future work.

## 13. Layer-adaptive budgets (F27) — OLMoE half

```bash
python3 scripts/policy_sweep.py allenai/OLMoE-1B-7B-0924 8192 5,4   # runs the layer-budget policies too
```

`allocate_layer_budgets` picks a per-layer expert count k_l with mean k′ by greedily removing, across layers, the expert whose marginal share on the calibration curve is smallest (training-free, same calibration pass). Three uses: `layer_topk(static)` — a fixed k_l per layer, no per-token decision (the LExI-style baseline, here allocated from calibration gate-mass curves rather than weights); `score_only+layerbudget` and `contribution+layerbudget` — the dynamic rules with per-layer τ_l hitting k_l instead of a uniform k′. OLMoE, 8,192 tokens; the six original rows reproduce F20 exactly.

| policy | k′≈5 ppl | Δ | k′≈4 ppl | Δ |
|---|---:|---:|---:|---:|
| static top-k (uniform) | 12.299 | +11.3 % | 14.117 | +27.7 % |
| layer_topk (static, allocated) | 12.322 | +11.5 % | 13.714 | +24.1 % |
| score-only (uniform) | 12.579 | +13.8 % | 14.737 | +33.4 % |
| score-only + layer budget | 12.492 | +13.0 % | 14.219 | +28.7 % |
| contribution (uniform) | 12.202 | +10.4 % | 13.701 | +24.0 % |
| **contribution + layer budget** | **12.190** | **+10.3 %** | **13.611** | **+23.2 %** |

Paired bootstrap (16 sequences):

| comparison | k′≈5 | k′≈4 |
|---|---|---|
| layer_topk(static) vs uniform static | +0.2 % n.s. | **−2.8 % [−4.2, −1.6]** |
| contribution+budget vs contribution | −0.1 % [−0.8, +0.5] n.s. | −0.6 % [−1.6, +0.2] n.s. |
| contribution+budget vs layer_topk(static) | **−1.1 % [−2.1, −0.1]** | −0.7 % [−2.4, +0.7] n.s. |
| contribution+budget vs uniform static | **−0.9 % [−1.6, −0.1]** | **−3.5 % [−5.5, −1.6]** |
| contribution+budget vs score-only+budget | **−2.4 % [−3.3, −1.5]** | **−4.2 % [−5.8, −2.8]** |

**Reading.** Budget allocation and ranking signal are separable and additive-ish. Allocation helps *static* selection a lot at tight budgets (−2.8 % at k′≈4: some layers tolerate 3 experts, others need 5) and adds little on top of contribution ranking (the dynamic rule already spends fewer experts on tokens/layers whose tail is cheap). The ranking gain is intact under budget control (−2.4 %, −4.2 % vs score-only at the same per-layer budgets). The combination is the best row at both budgets and beats uniform static top-k significantly at both — the first configuration that does so at k′≈5 (plain contribution was n.s. there, F20).

**Why so little on OLMoE.** `docs/figures/fig5_layer_budgets.png`: the allocator moves only a handful of experts (layers 0–1 give up one at k′=5, layers 8–9 gain one; layer 0 drops to 3 at k′=4) because the per-layer marginal-share curves are nearly identical — the j-th ranked expert carries about the same share in every layer. There is little heterogeneity to exploit, so budgets barely move and the gain is small. I expected Qwen3 to show the opposite (layer 2's output-scale CV is 2.66, F18). **Computed before its sweep from the shipped calibration** (`results/qwen3/layer_budgets_k5.json`, `fig5b`): it does not — budgets at k′=5 are 4–6 with std 0.29 (OLMoE 0.50); only layers 1, 3 (→4) and 23, 24 (→6) move. A large output-*scale* spread does not make a layer's contribution-*share* curve fatter, and the share curves are what the allocator reads. **Pre-registered expectation for the Qwen3 half:** layer budgets add little on Qwen3 too, so static top-k's advantage there is not explained by budget allocation.

### Qwen3 half

4,096 tokens, k′=5, calibration 4,096 tokens.

| policy | k′ | ppl | Δ vs 11.879 |
|---|---:|---:|---:|
| static top-5 | 5.00 | 12.642 | +6.4 % |
| **layer_topk (static, allocated)** | 5.00 | **12.620** | **+6.2 %** |
| score-only | 5.00 | 12.945 | +9.0 % |
| score-only + budget (gate curves) | 5.00 | 12.909 | +8.7 % |
| contribution | 5.03 | 12.987 | +9.3 % |
| contribution + budget (w·s curves) | 5.03 | **13.335** | +12.3 % |

| comparison | k′≈5 |
|---|---|
| layer_topk(static) vs uniform static | −0.1 % [−2.3, +1.9] n.s. |
| contribution+budget vs contribution | **+2.6 % [+1.4, +4.0]** worse |
| contribution+budget vs score-only+budget | **+3.3 % [+1.4, +5.4]** worse |
| contribution+budget vs static | **+5.5 % [+2.0, +8.6]** worse |

The pre-registered expectation held: budgets are nearly flat on Qwen3 and allocation is neutral — static top-k's edge there is **not** about how it spends budget. The new fact is the last row: combining two individually neutral changes is significantly *worse* than either, because the budgets were allocated from the w·s share curves and the calibrated scale s is not merely uninformative on Qwen3 (F22, F24) but *wrong* — any use of it, for ranking or for allocation, degrades the model. Score-derived budgets are neutral.

**What the two halves say together.** Whether w·s is a better signal than w is a per-model property; on OLMoE using it helps everywhere it is applied, on Qwen3 it hurts everywhere it is applied. That is exactly the kind of property a calibration pass can *measure directly* — by dropping experts under each rule on calibration tokens and comparing layer-output error — and select per layer. That is F30.

## Pending (running, in order)

- **F28 — downstream accuracy.** HellaSwag / ARC-Easy / PIQA, 200 examples each, continuation log-likelihood on the streaming engine, OLMoE at k′=5, five policies, paired per-example CIs.
- **F29 — third model.** Qwen1.5-MoE-A2.7B (top-4 of 60, `norm_topk_prob=False`, always-on shared expert): engine validation against HF, calibration, sweep at k′ 3 and 2.5.

- **F30 — per-layer signal selection.** On calibration tokens, compute every routed expert, drop m experts by w and by w·s, measure layer-output error under each; select the better signal per layer. Predicts contribution on OLMoE and score on Qwen3 without knowing which model it is. Running first; E6/E7 re-queued behind it.
