# MoE 추론 최적화: 시간순 문헌 정리, 약점 분석, 제안 메커니즘

- **작성일:** 2026-09-03
- **범위:** 이미 학습된 sparse MoE LLM의 **추론** 최적화 — 메모리 절약, 지연/처리량, 근사추론. weight 압축(어제 닫힘)은 제외.
- **근거 등급:** 원 리포트 §1.2와 동일 (P/T/O/I/H). arXiv ID는 본 세션에서 원문 대조한 것만 ✓ 표시.
- **목적:** 논문화. 따라서 순위표가 아니라 **계보**를 만든다.

---

# 1. 시간순 계보

## 1.1 2023 — 문제 정의: "sparse compute ≠ sparse memory"

| 시점 | 연구 | ID | 핵심 | 남긴 문제 |
|---|---|---|---|---|
| 2023-03 | Huang et al., *Towards MoE Deployment* ✓ | 2303.06182 | dynamic gating·expert buffering·load balancing. gating으로 처리량 6–11× | 메모리는 batch에서만 절약 |
| 2023-08 | **Pre-gated MoE** ✓ (ISCA'24) | 2308.12066 | 다음 층 expert를 *현재 층에서* 미리 선택 → 프리페치가 계산과 겹침 | pre-gate를 **학습**해야 함 |
| 2023-10 | QMoE ✓ | 2310.16795 | 1.6T SwitchTransformer를 <1 bit/param, 160 GB(20×)로 | 정밀도 축, 본 문서 범위 밖 |
| 2023-12 | Eliseev & Mazur, *MoE offloading* ✓ | 2312.17238 | LRU expert 캐시 + speculative loading으로 Mixtral을 소비자 GPU에서 | 캐시 적중률이 라우팅 예측 불가능성에 묶임 |

## 1.2 2024 — 오프로딩 스택의 성숙, 동적 스킵의 등장

| 시점 | 연구 | ID | 핵심 | 남긴 문제 |
|---|---|---|---|---|
| 2024-01 | **ExFlow** ✓ | 2401.08383 | **층간 expert affinity**로 all-to-all 통신 절반 | affinity를 통신에만 씀, 스킵/품질에는 안 씀 |
| 2024-01 | MoE-Infinity ✓ | 2401.14361 | sparsity-aware expert 캐시, 활성 트레이스로 교체·프리페치. 토큰 지연 3.1–16.7× 개선, **개인 장비** | 캐시 정책이 트레이스 대표성에 종속 |
| 2024-02 | Fiddler ✓ | 2402.07033 | CPU–GPU 협업: expert를 CPU에서 계산. 1.26–11.57× | CPU 연산이 병목 |
| 2024-06 | **Lu et al., ACL'24** ✓ | ACL 2024.334 | **training-free 동적 스킵**: top-2에서 2번째 expert의 router score 비율이 임계 미만이면 생략 | **k=2 전용**, Mixtral만 |
| 2024-08 | **AdapMoE** ✓ | 2408.10284 | 민감도 기반 adaptive gating + 프리페치·캐시. 활성 expert 25%↓, 1.35× | 민감도 = 층 단위 정적 |
| 2024-10 | ExpertFlow ✓ (DAC'26) | 2410.17954 | transformer 라우팅 경로 예측기(전 층 1회) + 토큰 스케줄링 + 예측 캐시. GPU 메모리 93.7%↓, 처리량 10× | 예측기 학습 필요 |
| 2024-11 | HOBBIT ✓ | 2411.01433 | 토큰 단위 동적 로딩 + 층 단위 프리페치 + 시퀀스 캐시, 덜 중요한 expert는 저정밀. 디코드 9.93× | 정밀도 축 |
| 2024-12 | **Survey** Liu et al. ✓ | 2412.14219 | model/system/hardware 3분류 | dynamic routing을 "model-level"의 한 항목으로만 |

## 1.3 2025 — fine-grained MoE(top-8/64–256)로 무대 이동, 예측기의 분화

| 시점 | 연구 | ID | 핵심 | 남긴 문제 |
|---|---|---|---|---|
| 2025-02 | **Fate** ✓ (Cross-Layer Gate) | 2502.12224 | 인접 층 gate 입력으로 expert 프리페치, edge. shallow-favoring 캐시로 hit 99% | 프리페치용; 스킵 없음 |
| 2025-05 | *Faster MoE for Extremely Large Models* ✓ | 2505.03531 | fine-grained MoE에서 **활성 expert 수를 줄여도 손실 작음**을 관측 | 규칙·품질 수치가 abstract에 없음 |
| 2025-09 | **LExI** ✓ | 2509.02753 | **data-free, 층별 정적 top-k**: 가중치만으로 층 중요도 추정 → 층마다 다른 k. Qwen1.5-MoE, pruning 대비 +10% | **정적** — 토큰마다 같은 k |
| 2025-09 | MoEpic ✓ (Adaptive Expert Split) | 2509.08342 | expert를 상/하 세그먼트로 나눠 hot expert의 상단만 캐시 → 캐시 적중↑, 지연 37–66%↓, GPU 비용 절반 | 캐시 정책 축 |
| 2025-09 | LayerScope ✓ | 2509.23638 | 예측 기반 층간 expert 스케줄링, multi-batch, 상용 서버(PCIe 병목). 처리량 +141%, 지연 −74.6% | 서버·배치 전제 |
| 2025-10 | SP-MoE ✓ | 2510.10302 | SD-aware 오프로딩 + compute/comm 파이프라이닝, TPOT 1.07–3.5× | SD 전제 |
| 2025-11 | **Opportunistic Expert Activation** ✓ (Dao 그룹) | 2511.02237 | **batch-aware**: 배치의 다른 토큰이 이미 로드한 expert에 편승. Qwen3-30B MoE-layer 지연 39%↓, training-free | **batch ≥ 16 전제**. batch 1 디코드에는 무력 |
| 2025-11 | *Caching & Prefetching Analysis* ✓ | 2511.05814 | LFU > LRU, speculative prefetch 트레이스 분석 | 캐시 정책은 라우팅 분포에 종속 |
| 2025-12 | OD-MoE ✓ | 2512.03927 | 캐시 없는 on-demand 로딩 + 다층 선행 emulative 예측기(정확도 99.94%), edge 분산. GPU 메모리 1/3로 완전캐시 성능의 75% | 분산 노드 전제 |
| 2025-12 | **Generalized skipping** ✓ | 2512.21911 | Lu et al.을 **k>2로 일반화**: 하위 m개의 gate-mass 비율 < β_m이면 스킵, β_m = 층별 **중앙값**. DeepSeek-R1, m=3 무손실, **m=4에서 수학 붕괴**(82→75) | ① 중앙값 = 고정 50% 스킵률, 품질 제어 없음 ② speculative **verification 안에서만** 평가 ③ 모델 1개 |

## 1.4 2026 — 학습 기반 동적화, 라우터 자체의 재고, 그리고 "확실한 머리 / 불확실한 꼬리"

| 시점 | 연구 | ID | 핵심 | 남긴 문제 |
|---|---|---|---|---|
| 2026-02 | **Certain Head, Uncertain Tail** ✓ | 2602.02443 | fine-grained MoE의 router score는 **고신뢰 머리 + 저신뢰 꼬리** 구조. 머리가 추론 능력, 꼬리는 다양성. Qwen3-30B에서 test-time scaling에 활용 | **계산 절감에는 쓰지 않음** — 같은 관측이 스킵의 근거인데 반대 방향으로만 사용 |
| 2026-02 | DALI ✓ | 2602.03495 | 로컬 PC: CPU/GPU greedy 배정 + **residual 기반 프리페치**(층간 정보) + workload-aware 캐시 | 수치 abstract에 없음 |
| 2026-02 | **SERE** ✓ (ICLR'26) | 2602.07616 | 배치 디코드에서 토큰을 *유사 expert로 재라우팅*해 활성 expert 수↓, training-free, vLLM plug-in, 최대 2.0× | **batch 전제**; 라우팅을 바꿈(router 출력 ≠ 실행) |
| 2026-02 | MELINOE ✓ | 2602.11192 | fine-tuning으로 *시퀀스당* 선호 expert 집합을 좁혀 캐시 적중↑, 1.2–3× | **학습 필요** |
| 2026-03 | TriMoE ✓ (DAC'26) | 2603.01058 | hot/warm/cold expert를 GPU / AMX-CPU / DIMM-NDP에 배치, 2.83× | 하드웨어 전제 |
| 2026-03 | Expert Threshold Routing ✓ | 2603.11535 | top-k 대신 EMA 임계 라우팅 — **사전학습 시** | 사후 적용 불가 |
| 2026-03 | **Speculating Experts** ✓ | 2603.19289 | 내부 표현으로 미래 expert 예측 → TPOT 14%↓ | 예측 정확도가 모델별로 편차 |
| 2026-04 | Routing-Free MoE ✓ | 2604.00801 | 라우터·softmax·top-k 제거, expert가 자기 활성을 연속 gradient로 학습 | 사전학습 구조; 사후 적용 불가 |
| 2026-05 | **ZEDA** ✓ | 2605.18643 | zero-expert 주입 + **self-distillation** → expert 연산 50%↓, 1.2×, Qwen3-30B·GLM-4.7-Flash. 동적 MoE 기법들보다 +4–6점 | **학습 필요** (teacher = 원본) |
| 2026-05 | Predictive Prefetch + Replication ✓ | 2605.11537 | 과부하 expert 예측·복제로 GPU 활용 ~100%, 3×, 성능 90–95% 유지 — **Switch-base**(구형)에서만 | 서버 규모; 현대 fine-grained MoE 미평가 |
| 2026-06 | Spatio-Temporal Prefetch ✓ | 2606.15453 | 라우팅 보존 경량 런타임 예측 + **재구성 가능 하드웨어** 공동설계 | 하드웨어 전제; training-free 여부는 본문 확인 필요 |
| 2026-07 | SpecPrefetch ✓ | 2607.24787 | 공유 경량 adapter가 다음 층 후보를 *전송용으로만* 예측, 실행은 원 router | adapter 학습 |
| 2026-07 | DraftExpert ✓ | 2607.24434 | 경량 draft expert **학습** + 프리페치, end-device self-speculative, 1.45× | 학습 필요 |
| 2026-08 | MoEXBench (S31) ✓ | 2608.21693 | 압축 기법 **조합** 평가; 평균이 tail을 숨김 | 동적 스킵은 평가 대상 아님 |

---

# 2. 계보에서 읽히는 구조

세 갈래가 있고, 서로 거의 만나지 않는다.

1. **어디서 가져올까** — 오프로딩·캐싱·프리페치 (2312 → 2401 → 2408 → 2511 → 2603 → 2606 → 2607). 라우팅을 *예측*해서 로드 지연을 숨긴다. **라우팅 자체는 건드리지 않는다.**
2. **몇 개를 쓸까** — 동적 스킵 (ACL'24 → 2509 LExI → 2512 일반화 → 2605 ZEDA). 로드해야 할 expert 수를 *줄인다*.
3. **누구와 나눌까** — batch-aware (2511.02237, 2602.07616). 배치 안에서 로드를 공유한다.

갈래 2가 메모리 대역폭에 **비례해서** 이득을 주는 유일한 갈래다 — 로드하지 않은 expert는 읽지 않은 바이트다. 그런데 갈래 2의 training-free 계열은 얇다: Lu(2024, k=2), LExI(정적), 2512.21911(중앙값, 검증 단계 전용), 그리고 갈래 3과 겹치는 SERE(재라우팅, batch 전제) 넷뿐이고, 가장 좋은 수치(ZEDA)는 학습이 필요하다.

---

# 3. 약점 — 정확히 어디가 비어 있는가

training-free 동적 스킵 계열(Lu → 2512.21911)의 결정 규칙은 전부 **router score 하나**에 걸려 있다.

\[
\text{skip } m \iff \frac{\sum_{j\in\text{tail}_m} w_j}{\sum_{j\le k} w_j} < \beta_m
\]

여기에 네 가지 약점이 겹친다.

**W1. router score는 dispatch 신호이지 contribution 신호가 아니다.**
score \(w_e\)는 load-balancing 보조 손실(OLMoE·Qwen3 모두 사용) 아래서 학습된다. 이 손실은 score 분포를 *균등하게* 밀어낸다. 토큰에 실제로 기여하는 양은 \(\|w_e E_e(x)\|\)이고, \(\|E_e(x)\|\)는 expert마다 다르다 — 얼마나 다른지는 **측정되어 있지 않다**. 다르다면 score-only 규칙은 큰-출력 expert를 버리고 작은-출력 expert를 남기는 실수를 체계적으로 한다. 2602.02443의 "머리/꼬리"도 score로만 정의했다.

**W2. 임계값이 품질이 아니라 분위수에 맞춰져 있다.**
β_m = 중앙값 → 각 m에서 토큰의 50%가 스킵된다. 층이 민감하든 아니든. 그래서 m=4에서 수학이 무너진다. 필요한 것은 *목표 평균 k′* 또는 *목표 출력 오차*에 맞춘 층별 임계값이다. LExI가 층별 k를 두지만 정적이다.

**W3. 평가 무대가 이득이 나는 무대가 아니다.**
2512.21911은 speculative verification 안에서만, Opportunistic은 batch≥16에서만, ZEDA는 GPU 처리량으로 잰다. **batch-1, 대역폭 제한 디코드** — 로컬/edge에서 MoE가 실제로 돌아가는 조건(S35, DALI, OD-MoE가 다루는 조건) — 에서 "스킵된 expert = 안 읽은 바이트"를 직접 잰 연구가 없다.

**W4. tail을 보고하지 않는다.**
MoEXBench의 경고 그대로다. 평균 정확도 옆에 worst-domain·worst-layer가 없다.

---

# 4. 제안 메커니즘 — Contribution-Calibrated Dynamic Expert Skipping

## 4.1 규칙

층 \(\ell\), expert \(e\)에 대해 calibration 통계 하나를 둔다.

\[
s_{\ell,e} = \mathbb{E}_{x \mid e \in \text{top-}k(x)} \left\|E_{\ell,e}(x)\right\|
\]

토큰의 top-k 후보를 **score가 아니라 기여도** \(c_e = w_e \, s_{\ell,e}\)로 정렬하고, 누적 기여 비율이 \(1-\tau_\ell\)에 도달하는 최소 접두사를 남긴다.

\[
\text{keep}(x) = \min\Big\{ j : \frac{\sum_{i \le j} c_{(i)}}{\sum_{i \le k} c_{(i)}} \ge 1-\tau_\ell \Big\}
\]

\(\tau_\ell\)은 **목표 평균 \(k'\)** (또는 층 출력 오차 예산)에 맞춰 calibration 토큰에서 이분탐색으로 정한다. 저장량: 층당 \(E\)개 float + 1개 float. 학습 없음. router 보존.

## 4.2 baseline 대비 delta — 정확히 무엇이 다른가

| | Lu'24 / 2512.21911 | LExI | Opportunistic | ZEDA | **제안** |
|---|---|---|---|---|---|
| 정렬 기준 | score | — | score(배치 공유) | 학습 | **score × 출력 스케일** |
| 임계 | 중앙값(고정 스킵률) | 정적 k | 배치 상태 | 학습 | **목표 k′ / 오차 예산** |
| 토큰 적응 | ○ | ✗ | ○ | ○ | ○ |
| 학습 | ✗ | ✗ | ✗ | **필요** | ✗ |
| 평가 무대 | spec-verify | GPU | batch≥16 GPU | GPU | **batch-1 대역폭 제한** |
| tail 보고 | ✗ | ✗ | ✗ | ✗ | ○ |

W1이 실재하지 않으면(출력 스케일이 expert 간에 균일하면) 제안은 2512.21911과 같아진다. **그래서 첫 측정은 \(s_{\ell,e}\)의 분산이다.**

## 4.3 반증 조건 — 미리 쓴다

- **G1.** \(s_{\ell,e}\)의 층 내 변동계수(CV) < 0.15 → W1은 실재하지 않고 제안은 baseline과 구분되지 않는다. 폐기.
- **G2.** 같은 평균 \(k'\)에서 제안의 perplexity가 score-only(중앙값 대신 같은 목표 k′로 맞춘 공정 버전)보다 낫지 않다 → 기여도 정렬은 무가치. 임계 교정(W2)만 남는다.
- **G3.** 대역폭 제한 디코드에서 bytes/token 감소가 tok/s로 이어지지 않는다 → 시스템 주장 철회, 알고리듬 주장만.
- **G4.** worst-domain 저하가 평균의 3배 이상 → MoEXBench 기준 실패.

## 4.4 실험 설계 (본 장비, 30% 여유 규칙 준수)

| 단계 | 내용 | 자원 |
|---|---|---|
| E0 | 스트리밍 엔진이 HF logits를 재현 | ≤ 7 GB (offload 참조) |
| E1 | calibration 1회: \(s_{\ell,e}\), top-k weight 분포, \(\beta_m\) 중앙값, \(\tau_\ell\) | ≤ 3 GB |
| E2 | perplexity vs 평균 k′ ∈ {8,7,6,5,4,3}: top-k 정적 / mass-ratio(중앙값) / mass-ratio(목표 k′) / **contribution** | ≤ 3 GB, 각 ~15분 |
| E3 | bytes/token, tok/s (batch 1, 스트리밍) | 동일 |
| E4 | domain별 tail: WikiText / code / math 소규모 | 동일 |

모델: OLMoE-1B-7B (top-8/64, fine-grained). Qwen3-30B는 offload 엔진으로 E1–E2의 축소판.

## 4.5 현재 상태

- 엔진 (`runtime/stream.py`), 정책 3종 (TopK / MassRatio / Contribution): **구현·단위테스트 완료**, 커밋 7690b61.
- **E0 검증 통과** (FINDINGS F13): top-1 일치 100%, NLL 3.207 vs 3.212, 34 token. 엔진 상주 ~3 GB.
- **E1 통과 (F14):** 출력 스케일 CV 0.26(층별 0.17–0.39), score와의 상관 +0.17 — W1 실측. OLMoE 라우팅은 평탄(head/tail 2.9×).
- **E2 완료 (F15).** matched k′에서 contribution vs score-only: **−0.9% / −3.0% / −8.5%** (k′≈6/5/4) — 스킵이 공격적일수록 격차 확대. score-only는 모든 예산에서 **정적 top-k보다 나쁨**; contribution은 k′≈5·4에서 정적을 앞섬. 발표된 중앙값 규칙은 ppl 9.55→40.8로 붕괴. **G2 통과, E4 규칙 3/3 통과.**
- **E3 완료 (F16).** batch-1 디코드에서 bytes/token이 k′에 **정확히 선형**(expert 하나당 ~16.8 MB/token), contribution@5에서 **1.80× tok/s**, KV 캐시 경로 오차 0. W3 해소. 단 E3의 63-token ppl 열은 잡음 — 품질은 E2와 8K 재확인이 결정.
- **Qwen3 엔진 검증 통과 (F17).** 복제 실행 중.
- **E4 완료 (F19).** gsm8k·code에서 도메인 붕괴 없음(tail/mean ≤ 1.10), contribution이 모든 셀에서 승. 새 slice의 wikitext 열이 F15를 독립 재현: −2.8% (k′≈5), −7.1% (k′≈4). G4 통과.
- **Qwen3 G1 통과 (F18):** CV 0.335, **score와의 상관 −0.06** — 48층 중 15층에서 음(−0.35까지). Qwen3에서는 score-only 규칙이 잡음이 아니라 *역방향*으로 정렬한다.
- **F20 (8K + paired bootstrap):** OLMoE 결론 확정 — contribution vs score-only −3.0% [−4.0,−2.0] (k′≈5), −7.0% [−8.8,−5.3] (k′≈4); vs static −2.9% [−4.9,−1.1] (k′≈4).
- **F21 (Qwen3): 부정 결과.** contribution이 score-only보다 +5.3%/+6.7%, static보다 +10%/+18% *나쁨*. 가설 3개 시험 완료 (F22): 캘리브레이션 4배 → 손실이 **통계적 동률**(+0.4% [−1.7, +2.3])로 축소, budget hogging 기각, renorm 오차모델은 오히려 악화. **Qwen3에서는 null.** F23: 제곱-share 변형도 OLMoE에서 열등. **최종 주장: OLMoE-class(unnormalised) router에서의 1-model positive + Qwen3 null.** **oracle 시험 (F24):** OLMoE에서는 oracle ≈ proxy(+0.1% n.s., 더 작은 k′에서) — 64-float 캘리브레이션이 달성 가능한 이득 전부를 회수. Qwen3에서는 oracle이 score-only보다 **+5.1% 나쁨** — proxy가 아니라 **신호 자체**가 실패. 두 체제를 가르는 변수는 router의 renormalisation(`norm_topk_prob`). 반사실 시험(F25, renorm off) 큐잉.

---

# 5. 참고: 검증 상태 요약

✓ = 본 세션 원문 대조. **표의 모든 항목이 arXiv 원문과 대조 완료**(2026-09-03). 인용 가능.

---

# 문서 종료
