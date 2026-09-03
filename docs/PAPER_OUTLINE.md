# 논문 골격 (초안) — Contribution-Calibrated Dynamic Expert Skipping

- **작성일:** 2026-09-03
- **상태:** F13–F21 반영. **F21이 부정 결과** — Qwen3에서 메커니즘이 진다. 논문 주장을 축소하거나(§6), renorm-aware 변형으로 회복하는지(진단 중)에 따라 제목·초록이 바뀐다. 아래 초록은 F21 전 버전이며 재작성 대상.
- **근거:** `docs/LOG.md` (chronological) / `docs/FINDINGS.md` (curated), `MOE_INFERENCE_SURVEY_2026-09-03.md`

---

## 제목 후보

1. *The Router Does Not Know How Loud Its Experts Are: Contribution-Calibrated Expert Skipping for Bandwidth-Bound MoE Decoding*
2. *Skip by Contribution, Not by Score: Training-Free Dynamic Expert Selection in Fine-Grained MoE*
3. (보수적) *Calibrated Output Scale as a Missing Signal in Training-Free MoE Expert Skipping*

## 초록 초안 (영문) — F22 이후 재작성

> Training-free dynamic expert skipping decides which routed experts to execute from the router score alone. We measure on two fine-grained MoE models that the score carries little information about an expert's output magnitude (r = +0.17 on OLMoE-1B-7B, −0.05 on Qwen3-30B-A3B). Ranking on w_e·s_e with a calibrated per-expert output scale s_e — one calibration pass, no training — improves matched-budget perplexity over the score-only rule on OLMoE by 3.0% [2.0, 4.0] and 7.0% [5.3, 8.8] at k′≈5 and 4 (8,192-token paired bootstrap), surpasses static top-k at k′≈4, degrades math and code less than general text, and yields a 1.80× batch-1 decode speedup on a bandwidth-bound CPU decoder. **On Qwen3-30B-A3B, whose router renormalises the kept weights, the same signal is a statistical tie with score-only (+0.4% [−1.7, +2.3]), and static top-k dominates both.** Two principled variants — an orthogonality-derived error model and a renormalisation-aware stopping rule — are worse than the heuristic on both models. An oracle that keeps by the true per-token contribution ties the 64-float proxy on OLMoE (+0.1%, n.s.) and is worse than score-only on Qwen3 (+5.1% [1.7, 9.9]): the calibrated scale is, up to the oracle, a sufficient correction on unnormalised routers, while on renormalised routers the contribution signal itself is not the right one. We identify the router's top-k renormalisation as the variable separating the two regimes.

## 초록 초안 (영문) — F22 이전 버전 (기록용)

> Dynamic expert skipping reduces the memory traffic of Mixture-of-Experts decoding by executing fewer than the routed top-k experts per token. Every training-free skipping rule to date decides on the router score alone. We measure, on two fine-grained MoE models (OLMoE-1B-7B, Qwen3-30B-A3B), that the router score carries almost no information about how much an expert actually contributes: the per-expert output magnitude varies by 26–34% within a layer and correlates with gate weight at r = +0.17 (OLMoE) and **r = −0.06** (Qwen3, negative in 15 of 48 layers). We propose ranking and thresholding on the *contribution* w_e · s_e, where s_e is a calibrated per-expert output scale (64–128 floats per layer, one calibration pass, no training), with per-layer thresholds fitted to a target mean expert count. On a bandwidth-bound batch-1 CPU decoder where every skipped expert is bytes not read, contribution skipping beats the fair score-only rule at matched budget by 3.0% and 8.5% perplexity at k′≈5 and 4, and — unlike score-only, which loses to a static top-k cut at every budget — surpasses static top-k. It degrades math and code *less* than general text, and at k′≈5 delivers a 1.80× decode speedup for +9.8% perplexity. The published median-threshold rule collapses the same model to 4.3× its perplexity.

## 섹션 골격 → 어느 finding이 채우는가

| § | 내용 | 근거 |
|---|---|---|
| 1 Intro | MoE 디코드는 대역폭 제한; 스킵 = 안 읽은 바이트; 기존 규칙은 score 전용 | 서베이 §2, F16 |
| 2 Related | 시간순 계보 3갈래 (어디서/몇 개/누구와). training-free 스킵 계열 넷의 약점 W1–W4 | 서베이 §1–3 (24편 ✓) |
| 3 Observation | **router score ≠ contribution.** CV·상관·층별 프로파일, 두 모델 | **F14, F18** |
| 4 Method | s_{ℓ,e} 캘리브레이션, 기여도 정렬, 목표 k′ 이분탐색, min_keep; 저장 E floats/층 | 서베이 §4.1 |
| 5.1 Setup | 스트리밍 엔진(검증 F13/F17), matched-k′ 프로토콜, fair score-only control, 정적 top-k, 중앙값 baseline | F13, F17 |
| 5.2 Quality | ppl vs k′ 표 + **paired bootstrap CI** | F15, **F20(대기)** |
| 5.3 Bandwidth | bytes/token 선형성, tok/s, 캐시 정확성 | F16 |
| 5.4 Tail | wikitext/gsm8k/code, tail/mean | F19 |
| 5.5 Generality | Qwen3 — **동률(null)**, oracle도 실패 → 신호 자체의 문제. renorm이 분리 변수. F25(renorm off): contribution −40 % — 방향 일치, 단 모델이 4× 손상되어 지지 증거에 그침. **F25b(full-mass renorm, top-8 동일): 깨끗한 측정, 실행 중** | F21, F22, F24, F25, **F25b(대기)** |
| 5.6 Ablation | contribution vs score-only vs static; linear vs squared share (F23); renorm 오차모델 (F22); **proxy vs oracle (F24: OLMoE 동률 — proxy 충분)** | F15, F22, F23, F24 |
| 6 Limitations | 절대 비용(+9.8% @ 37.5% 절감), ZEDA 대비, 모델 2개, CPU fp32 무대 | 아래 |
| 7 Conclusion | | |

## 그림/표 목록

- Fig 1: 층별 r(s, gate weight) — OLMoE vs Qwen3 (막대, 음수 층 강조)
- Fig 2: ppl vs k′ 곡선 4개 (static / score-only / contribution / median-rule 점)
- Fig 3: MB/token vs k′ (선형) + tok/s
- Tab 1: 계보표 축약 (training-free 스킵 4편의 W1–W4 체크)
- Tab 2: matched-k′ 결과 + CI
- Tab 3: 도메인별 tail
- Tab 4: Qwen3 복제

## 주장 ↔ 증거 대응 (리뷰어용)

| 주장 | 증거 | 강도 |
|---|---|---|
| score는 contribution 정보를 거의 안 담는다 | F14 r=+0.17, F18 r=−0.06 (15/48층 음) | 두 모델, 직접 측정 |
| contribution > score-only at matched k′ | F15 (2K), F19 wikitext (새 1K slice) 재현, F20 CI | 두 slice + CI 예정 |
| score-only < static top-k | F15 전 예산 | 한 모델 |
| 도메인 붕괴 없음 | F19 tail/mean ≤ 1.10 | 세 도메인, 1K씩 |
| 대역폭 선형·속도 1.8× | F16 | batch-1 CPU fp32 |
| 발표 규칙 붕괴 | F15 ppl 40.8 | 한 모델 |

## 한계 — 미리 쓴다

1. **절대 비용.** 최선의 training-free 규칙도 37.5% 로드 절감에 ppl +9.8%. ZEDA(학습)는 50%를 거의 무손실로. 본 논문은 training-free 규칙의 quality-per-skip을 ~3배 올린 것이지 학습과의 격차를 닫은 게 아니다.
2. **평가 규모.** 2K/8K token, seed 1, 모델 2개. 다운스트림 정확도(MMLU 등)는 GPU 없이 못 잰다 — perplexity만.
2b. **반사실의 해석.** F25는 renorm을 끄면서 모델을 운영점 밖으로 밀었다(ppl 11.9→42.9). F25b가 top-8을 그대로 두고 제거만 뺄셈으로 만드는 올바른 대조군이며, 논문의 §5.5 결론은 F25b에 걸려 있다.
3. **무대.** CPU fp32 스트리밍 디코더. GPU 배치 서빙에서 bytes→latency 전환은 다르다(sublinear). 주장은 "대역폭 제한 batch-1"로 한정.
4. **라우팅 프로파일.** 두 base 모델 모두 top-8 내에서 평탄. "certain head" 모델(Instruct, 추론)에서는 score-only의 손실이 작을 수 있다 — 미측정.
5. **s_e 안정성.** WikiText로 캘리브레이션해 gsm8k/code에 전이됨(F19)은 확인; 언어·도메인 대이동은 미측정.

---

# 문서 종료
