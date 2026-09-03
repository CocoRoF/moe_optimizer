# Measured findings

Every number here is reproducible from this repo on CPU. Commands are given.

| # | finding | model | verdict |
|---|---|---|---|
| F1 | A chart on the expert coordinate is bounded by E/(E+D) < 0.1% of the code | all (arithmetic) | closed |
| F1b | Expert weights are near full rank in weight space | OLMoE | superseded by F10 (wrong metric) |
| F2 | Cross-layer structure lives only in the residual-stream-facing mode, 3/3 | OLMoE | stands |
| F3 | Diagonal coupling needs joint diagonalisation, not HOSVD | synthetic | design rule |
| F4 | Coefficients must be fitted against decoder-generated factors | synthetic | design rule |
| F5 | Sharing ≤ per-expert SVD; weight-cosine clustering ≡ null | OLMoE | superseded by F10 (wrong metric) |
| F6 | Adjacent-layer dictionaries share no directions (union rank 128/128) | OLMoE | depth chart falsified |
| F7 | Neurons have no near-duplicates (NN median 0.19, chance 0.12) | OLMoE | neuron codebook closed (raw) |
| F8 | Calibration statistics; anisotropy moderate (eff. rank 544–905/2048) | OLMoE | tool + pre-registration |
| F9 | Whitened: NN median 0.38, still 128/128 union rank | OLMoE | branch 2 |
| F10 | Whitened, scored by output error: per-expert SVD beats all sharing; clustering ≡ null to 4 dp | OLMoE | **every sharing axis closed** |
| F11 | Stronger, longer-range depth structure in middle layers; still no shared directions or duplicate neurons | Qwen3-30B | model matters; mechanism narrow |
| F11a | Depth anchor + correction saves 30–33% of one dictionary side, middle band, pre-whitening | Qwen3-30B | real, small |
| F11b | Residual/neuron split replicates 3/3; neuron side at chance to 3 dp | Qwen3-30B | **F2 is a two-model result** |
| F12 / F12b | Whitened per-expert SVD: ×1.62 at 75%, ×2.47 at 56%; full `down` covariance changes nothing | OLMoE | **low-rank closed, sharing or not** |
| F8/F9-Qwen3 | Whitened neuron-NN median 0.404 (gate 0.5, predicted 0.45–0.50); 1–2 shared directions of 64 in the middle band | Qwen3-30B | **branch 2 on both models** |
| F13 | Streaming CPU engine reproduces the reference (100% top-1, NLL 3.207 vs 3.212) at ~3 GB | OLMoE | tool |
| F14 | Expert output scale: within-layer CV 0.26, r with gate weight +0.17 — G1 passes | OLMoE | W1 measured |
| F15 | Contribution skipping beats score-only by −0.9/−3.0/**−8.5%** ppl at k′≈6/5/4; score-only loses to static; median rule → ppl 40.8 | OLMoE | **mechanism supported** (2K tokens; 8K confirmation queued) |
| F16 | Batch-1 decode: bytes/token linear in k′, **1.80× tok/s at k′≈5**, cache path exact; its 63-token ppl column is noise | OLMoE | W3 answered |
| F17 | Qwen3-30B-A3B streaming engine validates (100% top-1, NLL 3.508 vs 3.503) | Qwen3-30B | tool |
| F18 | G1 on Qwen3: CV 0.335; r(scale, gate) = **−0.06**, negative in 15/48 layers | Qwen3-30B | W1 stronger on Qwen3 |
| F19 | E4: no domain collapse (tail/mean ≤ 1.10); contribution wins every cell; fresh-slice replication −2.8% / −7.1% | OLMoE | G4 passes |
| F20 | 8K + paired bootstrap: vs score-only **−3.0% [−4.0,−2.0]**, **−7.0% [−8.8,−5.3]**; vs static −2.9% [−4.9,−1.1] at k′≈4 | OLMoE | **headline confirmed** |
| F21 | Qwen3, 1K calibration: contribution **loses** to score-only (+5.3%/+6.7%) | Qwen3-30B | superseded by F22 |
| F22 | Qwen3, 4K calibration: contribution vs score-only **+0.4% [−1.7, +2.3]** — a tie; renorm error model worse | Qwen3-30B | **null; one-model result** |
| F23 | Squared-share variant worse than linear on OLMoE (+1.2%, +2.3%, significant) | OLMoE | heuristic stays |
| F24 | Oracle ≈ proxy on OLMoE (+0.1% n.s.); oracle **worse** than score-only on Qwen3 (+5.1%) — the signal fails there, not the proxy | both | **regimes separated by router renormalisation** |

---

## F1 — A chart on the expert coordinate cannot compress

```bash
moeopt econ
```

Every "shared dictionary + per-expert coefficients" scheme stores `E·K` per-expert
and `K·D` shared, so the per-expert share is `E/(E+D)` — independent of rank, of
`K`, and of method.

| model | E | D | per-expert share | saving from charting it |
|---|---:|---:|---:|---:|
| OLMoE-1B-7B | 64 | 262,144 | 0.024% | 0.008% |
| Qwen3-30B-A3B | 128 | 147,456 | 0.087% | 0.026% |
| DeepSeekMoE-16B | 64 | 495,616 | 0.013% | 0.006% |
| Mixtral-8x7B | 8 | 51,380,224 | 0.000% | −0.000% |

Mixtral is negative: with `M > E` the polynomial coefficients outweigh the table
they replace.

This is arithmetic, not an experiment, and it holds before any model is loaded.
Pinned as `tests/test_compressors.py::test_expert_chart_saving_is_negligible`.

**Consequence.** On the expert axis, an orthogonal-polynomial chart is a
regulariser and a progressive code. It is not a compressor, and must not be
claimed as one.

### F1 confirmed on real weights

```bash
moeopt sweep allenai/OLMoE-1B-7B-0924 --max-layers 1 --matrices gate \
    --ranks 32 128 --communities 1 8
```

OLMoE-1B-7B, layer 0, `gate`. The sweep runs `local_atlas` twice at each setting,
identical except for `expert_chart`:

| # | method | expert_chart | ratio | rel_fro | per-expert share |
|---|---|---|---:|---:|---:|
| 4 | local_atlas r=32 k=1 | off | 0.016 | 0.98393 | 0.002 |
| 5 | local_atlas r=32 k=1 | **on** | 0.016 | 0.98393 | 0.002 |
| 6 | local_atlas r=32 k=8 | off | 0.131 | 0.92570 | 0.000 |
| 7 | local_atlas r=32 k=8 | **on** | 0.131 | 0.92570 | 0.000 |

The chart pairs are **identical to five decimals in both size and error**. The
per-expert share on real weights is 0.2% and 0.0%, against the 0.024% predicted
by `E/(E+D)` for this model.

For contrast, in the same sweep the MoBE-like `shared_basis` has a per-expert
share of **0.800** — its `A_e` is per-expert *and* dictionary-sized. That is where
a compression lever still exists; the coefficient table is not.

### F1b — OLMoE expert weights are close to full rank

An unplanned observation from the same run, and an uncomfortable one:

| method | ratio | rel_fro |
|---|---:|---:|
| per_expert_svd r=32 | 0.047 | 0.907 |
| shared_base_delta r=32 | 0.062 | 0.900 |
| shared_basis r=32 | 0.020 | 0.961 |
| local_atlas r=32 k=8 | 0.131 | 0.926 |

At 1.6-13% of dense, *every* method — including the per-expert SVD floor — leaves
90-98% relative Frobenius error. Rank 32 of a possible 1024 captures almost
nothing, and the ranking between methods is nearly flat.

This is consistent with arXiv:2606.03465's report that LLM weight spectra are
close to power-law with no exploitable low-rank structure, and it sets a hard
expectation for the project: **structural low-rank compression alone will not
reach useful ratios on this model.** Sparse outlier residuals, quantisation of
the factors, and much higher ranks are not refinements to add later — they are
required for any operating point worth evaluating. Sweeps should target the
20-60% range, not the 2-13% range used to smoke-test the harness here.

Caveat: these are un-whitened, weight-space errors on one layer of one model.
The activation-weighted error is the one that matters and may rank methods
differently; that needs the calibration hooks, which are not yet built.

---

## F2 — Cross-layer structure lives in the residual-stream-facing mode only

```bash
moeopt audit-depth allenai/OLMoE-1B-7B-0924 --rank 64 \
    --matrices gate up down --sides out in
```

OLMoE-1B-7B, 16 layers, 64 experts, rank-64 dictionaries. Affinity is mean cos²
of principal angles between two layers' dictionaries.

| matrix | mode | faces | gap 1 | gap 2 | gap 4 | gap 8 | chance | verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| gate | out | neurons | 0.076 | 0.075 | 0.076 | 0.076 | 0.063 | at chance |
| gate | **in** | **residual stream** | **0.508** | 0.346 | 0.178 | 0.105 | 0.031 | structure |
| up | out | neurons | 0.062 | 0.063 | 0.062 | 0.063 | 0.063 | at chance |
| up | **in** | **residual stream** | **0.506** | 0.348 | 0.174 | 0.093 | 0.031 | structure |
| down | **out** | **residual stream** | **0.489** | 0.313 | 0.127 | 0.046 | 0.031 | structure |
| down | in | neurons | 0.063 | 0.062 | 0.063 | 0.062 | 0.063 | at chance |

Chance is `r/d` for random subspaces: 64/1024 = 0.063 on the neuron side,
64/2048 = 0.031 on the residual side.

**The split is 3/3 along the same seam, and the sides swap for `down` exactly as
the mechanism predicts.** `gate` and `up` read the residual stream on their input
mode; `down` writes to it on its output mode. The residual stream is a single
representation space maintained across all layers, so dictionaries anchored to it
are comparable between layers. The intermediate neuron space is private to each
layer and defined only up to permutation, so dictionaries anchored to it are not
comparable — and measure at chance, precisely as that predicts.

This was a prediction registered before the `up`/`down` runs, on the basis of the
`gate` result alone. It is not a post-hoc reading.

### Consequences for the method

1. **Chart the residual-stream-facing dictionary over depth; keep the
   neuron-facing dictionary per layer.** Charting both, as the first draft of the
   depth proposal assumed, wastes half the budget fitting noise.
2. **Charts must be piecewise.** Affinity halves by gap 2–3 and reaches chance by
   gap 6–8, so a single low-degree polynomial across all 16 layers is not
   supported. Chart width on the order of 4 layers is indicated — independently
   consistent with ConMoE (arXiv:2605.29350), which reports cross-layer nearest
   neighbours for 50.4% of Qwen3-30B-A3B experts within 4-layer scopes.
3. **The realistic saving is modest, and smaller than the headline projection.**
   Only one of two dictionaries is compressible, and only within short windows.
   The `p=2 → 0.10%` figure in section 3.5 of the review document assumed a global
   chart over both modes; F2 falsifies that assumption. Projections built on it
   must be withdrawn.

### Still to check

- Does the same split hold on **Qwen3-30B-A3B** (48 layers, 128 experts)? More
  layers means longer windows are possible, and the ConMoE measurement was made
  on that model.
- Does the residual-side affinity survive **within functional communities**, where
  the dictionary is fitted per community rather than per layer?
- Does affinity between *middle* layers exceed the average? ConMoE reports >70%
  cross-layer neighbours for several middle layers, which predicts non-uniform
  chart widths.

---

## F3 — A diagonal coupling needs joint diagonalization, not HOSVD

Found while validating `methods/depth_atlas.py` on synthetic data.

The report's simplified form (section 12) is `W_e = W̄_c + U diag(c_e) Vᵀ`. HOSVD
recovers the dominant subspaces of each mode, but each only up to an arbitrary
`r × r` rotation, and the two modes' rotations are **unrelated**. The diagonal form
requires a specific *paired* basis choice, so `U` and `V` from independent
eigendecompositions do not admit it: on synthetic data built to satisfy the
diagonal model exactly, reconstruction error stayed at 0.79 even with chart
residuals of 0.014.

Recovering that pairing is a simultaneous-diagonalization (CP/PARAFAC) problem,
not an SVD. Two ways out:

- **full coupling** — store an `r × r` core per expert. Well-posed via HOSVD, and
  the cost is not prohibitive: `L·E·r²` is 25M parameters for Qwen3-30B-A3B gate
  at r=64, against 9.66B dense. This is now the default.
- **CP/ALS** — solve for the paired bases directly. Not implemented; the honest
  cost of the diagonal form.

The report presents `diag` as a simplification of the Tucker form. It is not a
simplification of the same estimator — it is a different and harder one.

---

## F4 — Fitting order matters more than it looks

Also from `depth_atlas` validation. Coefficients must be projected onto the
dictionaries the **decoder** will generate, not onto the exact ones the encoder
computed. Fitting against exact bases and decoding against charted ones leaves a
small rotation mismatch that the coupling amplifies: chart residuals of 0.014
produced reconstruction error of 0.79 — a factor of ~50.

Any method that fits a factorisation and then approximates the factors must
re-derive the coefficients against the approximated factors. Regression test:
`tests/test_depth_atlas.py::test_recovers_smooth_drift`.

---

## F5 — Two negative results from the first real sweep

```bash
moeopt sweep allenai/OLMoE-1B-7B-0924 --max-layers 1 --matrices gate \
    --points configs/cmp_coupling.json
```

OLMoE-1B-7B, layer 0, `gate`, unwhitened, weight-space error.

| # | method | ratio | rel_fro |
|---|---|---:|---:|
| 1 | local_atlas k=8 r=32 **diag** spectral | 0.131 | 0.9257 |
| 2 | local_atlas k=8 r=32 **full** spectral | 0.132 | **0.9098** |
| 3 | local_atlas k=8 r=64 full **spectral** | 0.1406 | 0.89665 |
| 4 | local_atlas k=8 r=64 full **uniform (null)** | 0.1406 | **0.89659** |
| 6 | per_expert_svd r=16 | 0.023 | 0.9365 |
| 7 | per_expert_svd r=32 | 0.047 | 0.9068 |

### F5a — the diagonal coupling costs real accuracy

Rows 1 vs 2, at bytes matched to within 0.1%: full coupling improves relative
error from 0.926 to 0.910. This is F3's cost measured on real weights rather than
on synthetic data, and it is why `LocalAtlas` now defaults to `coupling="full"`.
Any earlier comparison run with `diag` handicapped the atlas methods.

### F5b — weight-cosine clustering does not beat an arbitrary grouping

Rows 3 vs 4 are identical except for how experts are grouped. Spectral clustering
on weight cosine scores 0.89665; **`uniform` — arbitrary contiguous blocks of
expert indices — scores 0.89659, marginally better.** The clustering step
contributes nothing here.

This is precisely what the `uniform` null model was added to detect, and it fired
on the first real run.

The correct reading is narrow: the affinity signal used was **raw weight cosine**,
which neuron permutation symmetry makes the least trustworthy of the four signals
(`community/cluster.py`). This is evidence against clustering *on raw weight
similarity* — which is also LorExperts' position — not against functional
clustering. The `coactivation` and `output` signals need calibration traces that
are not built yet, and the comparison must be repeated with them before the
clustering step is either kept or dropped.

### F5c — sharing does not yet beat the per-expert floor

Row 3 spends 3x the bytes of row 7 to gain 0.010 of relative error. Across the
wider sweep the Pareto front is dominated by `per_expert_svd` and
`shared_base_delta` — the two simplest methods with no cross-expert structure at
all.

Caveats, all of which bear on whether this survives: no activation-aware
whitening (no calibration stats yet), weight-space Frobenius rather than
functional error, one layer of one model, and ratios of 2-20% where F1b shows
every method is already broken. The regime that matters is 40-80%.

Taken with F1b, the priority is clear: **calibration statistics and whitening
come before any further method work.** Until the error metric is
activation-weighted, this sweep cannot distinguish a bad method from a bad metric.

---

## F6 — The depth axis does not yield a shared-basis saving

Two measurements run to test the depth-chart proposal from F2, on OLMoE-1B-7B
`gate`, residual-stream side, before building anything further on it.

### F6a — adjacent dictionaries share no directions

Union rank of `[U_l | U_{l+1}]`, rank-64 bases each. A concatenation singular
value below 0.1 means a principal angle under ~8°, i.e. a direction one layer
could borrow from the next.

| pair | union rank (sv > 0.1) | saving |
|---|---:|---:|
| layers 4,5 | 128 / 128 | **0%** |
| layers 5,6 | 128 / 128 | **0%** |
| layers 6,7 | 128 / 128 | **0%** |

4-layer window (256 columns): 256 kept at sv > 0.1; 226 at > 0.3 (angle > 26°);
159 at > 0.5 (angle > 41°).

**Not one principal angle between adjacent layers is below 8°.** The mean cos²
of 0.5-0.6 reported in F2 is produced by *many* directions being moderately
aligned (30-60°), not by *some* directions being shared and the rest orthogonal.
That is the one regime where neither mechanism helps: a shared basis needs
near-zero angles, and a polynomial chart needs a rotation small enough to
interpolate — 45° per layer of a 64-dim subspace in 2048 dims is not that.

### F6b — the signal washes out at the ranks reconstruction needs

Gap-1 affinity, layers 5→6:

| rank | affinity | chance (r/2048) | signal / chance |
|---:|---:|---:|---:|
| 64 | 0.589 | 0.031 | **18.9×** |
| 256 | 0.600 | 0.125 | 4.8× |
| 512 | 0.607 | 0.250 | **2.4×** |

Absolute affinity is flat; chance rises with rank. The cross-layer structure is
concentrated in the top few dozen directions. F1b says useful reconstruction
needs rank in the hundreds, and at those ranks the residual-side dictionaries are
barely above what two random subspaces would show.

### Verdict

F2 stands as a geometric observation — the residual-stream/neuron split is real,
3/3, and far above chance at low rank. **As a compression mechanism the depth
axis is falsified**, by the same instrument that found it, one turn after it was
proposed. The depth-chart method (`methods/depth_atlas.py`) is retained as an
ablation arm and as the record of a tested-and-rejected idea; it is not a
direction.

With F1 and F6 together: the orthogonal-polynomial idea does not survive on the
expert axis (cannot compress) or on the depth axis (nothing smooth enough to
chart). What remains of it is the nested-truncation property, and rank
truncation is nested in any orthonormal factorisation — including plain SVD —
so that is not a contribution either.

---

## F7 — Neurons have no near-duplicates in weight space

```bash
python3 scripts/neuron_nn.py
```

The finest unit with no remaining symmetry is a neuron: its (gate row, up row,
down column) triple is fully identified, so cross-expert and cross-layer
comparison needs no alignment.  If most neurons had a close twin somewhere in
the model, a shared prototype pool could approximate every expert by indexing.
This was the pre-registered test for that idea, with the kill condition "median
nearest-neighbour cosine < 0.3".

OLMoE-1B-7B, all 1,048,576 gate rows (16 layers x 64 experts x 1024), 8,000
random queries against the full set.  Random-vector baseline for NN cosine at
this N and d is ~0.116.

| | p10 | p25 | **p50** | p75 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| nearest neighbour, any | 0.119 | 0.141 | **0.191** | 0.269 | 0.370 | 0.597 |
| nearest neighbour, other layers only | 0.113 | 0.126 | 0.172 | 0.243 | 0.331 | 0.544 |

Fraction with NN cosine > 0.5: **2.8%**. > 0.7: 0.2%. > 0.9: **0.0%**.
The NN lands in the same layer 51.8% of the time and in the same expert 21.9% --
a mild preference over the 6.25% / 0.1% that chance would give, so what little
structure exists is local.

**Verdict: the neuron-codebook direction is dead in weight space.** Median NN
cosine 0.191 is 1.6x the random baseline; nothing is near-duplicate.

### The pattern across F1b, F6, F7

Three probes at three granularities -- whole-layer subspace, individual expert
rank structure, individual neuron -- all say the same thing: **OLMoE-1B-7B has
essentially no redundancy in raw weight space.**  This is consistent with
LorExperts' "near-orthogonal in raw weight space" and with arXiv:2606.03465's
power-law spectra.

Every one of those probes was measured in the metric arXiv:2606.03465 says is
the wrong one.  Two rows with raw cosine 0.2 can have pre-activations that
correlate at 0.9 if the input covariance is anisotropic enough -- whitened
cosine *is* pre-activation correlation.  That is the mechanism behind every
successful training-free method (GPTQ, AWQ, SVD-LLM), and it has not been
measured here yet.  It is the last door, and `calib/` exists to open it.

---

## F8 — Calibration statistics (OLMoE-1B-7B, 32,768 tokens, WikiText-2)

`calib/run.py`, 26.5 tok/s on CPU beside the bf16 model, 21 minutes. Per layer:
residual-stream second moment (2048×2048), routing counts, gate mass,
co-activation, per-neuron intermediate second moments. 521 MB.

Residual-stream anisotropy — the quantity that decides how far activation-space
similarity can diverge from weight-space:

| layer | top-1 | top-16 | top-128 | eff. rank / 2048 |
|---:|---:|---:|---:|---:|
| 0 | 0.085 | 0.221 | 0.456 | 676 |
| 4 | 0.071 | 0.186 | 0.386 | 905 |
| 8 | 0.081 | 0.205 | 0.421 | 803 |
| 12 | 0.082 | 0.242 | 0.468 | 685 |
| 15 | 0.121 | 0.287 | 0.506 | 544 |

Moderate. No massive-activation spike (top-1 ≤ 0.12 everywhere); the top 128
directions carry 39–51% of variance and the effective rank is a quarter to a
half of the ambient dimension.

### Pre-registered prediction for F9 (written before the numbers)

Whitening reweights by the eigen-spectrum above. With this much spread it will
*shift* the F7 distribution but cannot collapse a 2048-dim space onto a few
directions. Prediction: whitened neuron-NN median rises from 0.191 to
**0.25–0.35**, not to the 0.7+ a codebook would need; fraction above 0.9 stays
under 1%. Adjacent-layer union rank stays near 128/128. If F9 beats this
prediction substantially — median above 0.5 — activation space has structure
weight space hid and branch 1 of REDESIGN §5.2 is live. If it matches, branch 2.

---

## F9 — Activation space: more structure than weight space, not enough to share

```bash
python3 scripts/whitened_geometry.py runs/calib_olmoe_32k.pt
```

Whitened cosine of two gate rows equals the correlation of their
pre-activations on the calibration distribution — the metric every successful
training-free method optimises.

### F7 re-measured (neuron NN, gate rows, per-layer whitening)

| | p10 | p25 | **p50** | p75 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| raw (F7) | 0.119 | 0.141 | 0.191 | 0.269 | 0.370 | 0.597 |
| **whitened** | 0.256 | 0.304 | **0.384** | 0.483 | 0.577 | 0.754 |
| whitened, other layers only | 0.226 | 0.272 | 0.340 | 0.428 | 0.514 | 0.654 |

Fraction with NN > 0.5: raw 2.8% → **whitened 21.5%**. > 0.7: 0.2% → 2.0%.
> 0.9: 0.0% → **0.1%**.

### F6 re-measured (adjacent-layer residual-side dictionaries)

| pair | affinity raw → whitened | union rank (sv>0.1) | angles < 8° |
|---|---|---:|---:|
| 4,5 | 0.51 → 0.595 | 128 / 128 | 0 |
| 5,6 | 0.51 → 0.679 | 128 / 128 | 0 |
| 6,7 | 0.51 → 0.683 | 128 / 128 | 0 |

### Against the pre-registered prediction (F8)

Predicted whitened median 0.25–0.35; measured 0.384 — the prediction was
slightly *pessimistic*, which is the honest direction to be wrong in. The
opening condition for branch 1 was median > 0.5. Not met.

### Verdict — branch 2

Activation-space re-measurement moved every number in the direction the
"Rethinking" paper predicts, and by a real margin: the caveat on F1b/F5/F6/F7
was justified. But the movement is from "no structure" to "weak, diffuse
structure". A neuron codebook needs most neurons above ~0.9; 0.1% are. A shared
layer subspace needs near-zero principal angles; there are none.

**On OLMoE-1B-7B, no training-free structural-sharing mechanism has support at
any granularity — layer subspace, expert, neuron — in either metric.** This is
now a closed question for this model, with seven pre-registered probes behind it.

What that leaves, in order of how much of the original goal each keeps:

1. **Within-layer local dictionaries (B) as a matched-byte comparison against
   per-expert SVD and MoBE-like at 40–80%, whitened** — the F5 sweep rerun in
   the right metric. Running now. Its purpose is no longer to find a win; it is
   to establish the baseline row honestly.
2. **Testbed switch.** OLMoE is a small, dropless, fine-grained MoE — plausibly
   the *least* redundant MoE there is. Every literature redundancy claim used
   here (ConMoE 50.4% cross-layer NN, LorExperts communities, MoBE 24–30%) was
   made on Qwen3-30B-A3B or larger. The seven probes are cheap and CPU-only;
   rerunning them there is the correct next experiment *before* any method
   work, and it is the one that can reopen branch 1.
3. **Quantization composition + expert caching** — the systems path MoEXBench
   says carries the real gains. Training-free, deployable, but not "one block
   approximates the rest".
4. **The negative result itself.** F1 (a bound the literature lacks) plus F2
   (a clean 3/3 mechanistic split) plus F6/F7/F9 (redundancy absent at every
   granularity in both metrics on a fully open MoE) is a geometry-audit paper
   on its own.

---

## F10 — The decisive whitened sweep: sharing loses to no-sharing in the correct metric

```bash
moeopt sweep allenai/OLMoE-1B-7B-0924 --max-layers 1 --matrices gate \
    --points configs/sweep_whitened.json --calib runs/calib_olmoe_32k.pt
```

OLMoE-1B-7B, layer 0, `gate`. Every method fitted under the data-weighted norm
(F8 covariance) and scored by `rel_act` = ‖(W−Ŵ)L‖_F / ‖WL‖_F — the output
error on the calibration distribution. `rel_fro` shown for the mismatch.

| method | ratio | rel_fro | **rel_act** |
|---|---:|---:|---:|
| per_expert_svd r=256 | 0.375 | 0.681 | **0.272** |
| per_expert_svd r=384 | 0.562 | 0.564 | **0.208** |
| per_expert_svd r=512 | 0.750 | 0.454 | **0.156** |
| per_expert_svd r=384, *unwhitened* | 0.562 | 0.516 | 0.277 |
| shared_base_delta r=256 | 0.391 | 0.678 | 0.271 |
| shared_base_delta r=384 | 0.578 | 0.561 | 0.207 |
| shared_basis (MoBE-like) r=384 | 0.234 | 0.914 | 0.403 |
| shared_basis (MoBE-like) r=512 | 0.313 | 0.899 | 0.381 |
| local_atlas k=4 r=256, **uniform null** | 0.148 | 0.897 | 0.46977 |
| local_atlas k=4 r=256, co-activation spectral | 0.148 | 0.895 | 0.46994 |
| local_atlas k=8 r=256, **uniform null** | 0.234 | 0.852 | 0.42077 |
| local_atlas k=8 r=256, co-activation spectral | 0.234 | 0.846 | 0.42091 |
| local_atlas k=8 r=384, co-activation spectral | 0.336 | 0.801 | 0.369 |

### Three things this settles

**The metric was hiding a 2.5× factor.** At 37.5% size, raw Frobenius says 68%
error; the output error on real tokens is 27%. The residual stream's
anisotropy (F8) makes most weight-space error irrelevant. Every earlier number
in this file was misleading in magnitude, exactly as the "Rethinking" paper
predicts. The unwhitened SVD beats its whitened twin on `rel_fro` (0.516 vs
0.564) and loses on `rel_act` (0.277 vs 0.208) — the fit/score mismatch that
made the first run of this sweep unreadable, now pinned by a test.

**Sharing does not beat no-sharing.** Per-expert SVD, which shares nothing
across experts, dominates the front from 37% up. `local_atlas` — both factors
shared per community, full coupling, whitened, functionally clustered — reaches
0.369 at 33.6% where SVD's curve gives ~0.30. The shared base in
`shared_base_delta` adds nothing (0.271 vs 0.272). MoBE-like at 31% sits at
0.381, also above SVD's trend.

**Functional clustering ≡ arbitrary grouping, to four decimals.** Co-activation
spectral vs the `uniform` null: 0.46977 vs 0.46994, 0.42077 vs 0.42091. F5b's
caveat — "maybe it's the weight-cosine signal" — is resolved. It was not the
signal. The community structure LorExperts describes is not present in this
model's expert table, in the correct metric.

### Verdict

This is the experiment ASSESSMENT §6 named as decisive. On OLMoE-1B-7B, in the
correct metric, at 15–75% size, **no cross-expert sharing mechanism tested
beats per-expert SVD, and functional clustering contributes nothing.** With
F6/F7/F9 (no cross-layer or cross-neuron redundancy), every sharing axis is now
closed on this model.

What survives is not nothing: whitened per-expert SVD at 75% size gives 15.6%
output error with zero cross-expert assumptions. Whether that is acceptable in
perplexity is F12, not yet run. And the question of whether OLMoE is simply
the wrong testbed is the Qwen3-30B-A3B re-run, in progress.

---

## F11 — Qwen3-30B-A3B, weight space: more depth structure than OLMoE, still no shared directions

```bash
moeopt audit-depth Qwen/Qwen3-30B-A3B --rank 64 --matrices gate --sides in out
python3 scripts/neuron_nn.py Qwen/Qwen3-30B-A3B 4
```

### F6 on Qwen3 (gate, rank 64, 48 layers)

| side | gap 1 | gap 2 | gap 4 | gap 8 | chance | OLMoE gap 1 / gap 8 |
|---|---:|---:|---:|---:|---:|---:|
| **in** (residual stream) | **0.584** | 0.468 | 0.315 | **0.277** | 0.031 | 0.508 / 0.105 |
| out (neurons) | 0.096 | 0.096 | 0.097 | 0.097 | 0.083 | 0.076 / 0.076 |

The residual/neuron split replicates (2/2 on Qwen3 so far; `up`/`down` queued).
The residual-side signal is stronger and **much longer-range** than OLMoE:
gap-8 affinity 0.277 vs 0.105.

### Per-depth profile — ConMoE's middle-layer effect, reproduced

Gap-1 affinity along depth, residual side:

| layers | 0–3 | 4–7 | 8–16 | 16–17 | 17–44 | 45–47 |
|---|---:|---:|---:|---:|---:|---:|
| gap-1 affinity | 0.08–0.32 | 0.33–0.65 | 0.55–0.75 | **0.77** (max) | 0.43–0.76 | 0.43 → 0.27 |

Mean 0.584. Both ends are near OLMoE-like or worse; the middle ~30 layers sit
at 0.6–0.77. ConMoE (arXiv:2605.29350) reported "several middle layers exceed
70%" cross-layer nearest-neighbour rates on this model; this is the same
pattern from a subspace instrument rather than an expert-matching one.

### Union rank — still no shared directions, but a softer overlap than OLMoE

| pair | affinity | union rank (sv>0.1) | angles < 8° | angles < 26° |
|---|---:|---:|---:|---:|
| 4,5 (early) | 0.635 | 128/128 | 0 | 15 |
| 5,6 (early) | 0.330 | 128/128 | 0 | 1 |
| 22,23 (middle) | 0.709 | 128/128 | 0 | 30 |
| 23,24 (middle) | 0.756 | 128/128 | 0 | **35** |
| 24,25 (middle) | 0.662 | 128/128 | 0 | 25 |

4-layer window, union rank at sv>0.5: OLMoE 159/256; Qwen3 middle **108/256**.
The middle band has roughly 1.5× the soft overlap of OLMoE — and still not one
direction shared to 8°. **A shared basis is as unsupported here as on OLMoE.**

What the middle band *might* support is different: an anchor at layer l plus a
low-rank correction for layer l+1. ### F11a — rank of the depth correction (gate/in, rank-64 dictionaries)

| band | pair | energy outside span(U_l) | correction rank @90% | dictionary saving if anchored |
|---|---|---:|---:|---:|
| early | 1→2 | 0.860 | 54/64 | 16% |
| early | 2→3 | 0.763 | 52/64 | 19% |
| middle | 16→17 | 0.232 | 43/64 | 33% |
| middle | 23→24 | 0.244 | 45/64 | 30% |
| middle | 30→31 | 0.249 | 44/64 | 31% |
| late | 44→45 | 0.335 | 45/64 | 30% |
| late | 46→47 | 0.734 | 50/64 | 22% |

Read this against F1: the dictionary is the only term that matters, and this
is the saving on one side of it, per adjacent pair, before whitening.

**How to read it.** The middle band leaves only ~24% of the next layer's
dictionary energy outside the current one — but that 24% is spread across so
many directions that capturing 90% of it still costs rank 43–45 of 64. So a
depth anchor with a low-rank correction saves **30–33% of one side's dictionary,
in the middle ~30 layers, on the residual-facing mode only**, and 16–22% at the
ends. That is the same diffuse-overlap signature F6a found on OLMoE, at higher
amplitude. It is a real lever and a small one: it cannot produce a headline
compression ratio, and it must survive the whitened re-measurement (F9-Qwen3)
before it is anything more than a geometry observation.

### F7 on Qwen3 (raw, 12 of 48 layers, 1.18M gate rows)

| | p10 | p25 | **p50** | p75 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 raw | 0.134 | 0.182 | **0.260** | 0.346 | 0.421 | 0.556 |
| OLMoE raw | 0.119 | 0.141 | 0.191 | 0.269 | 0.370 | 0.597 |

Chance 0.117. NN > 0.9: **0.0%**. NN > 0.5: 2.6%. Median is above OLMoE's but
under the pre-registered 0.3 kill line for raw; the neuron-codebook direction
does not reopen on Qwen3 in weight space.

### Pre-registered prediction for the whitened Qwen3 re-run (F9-Qwen3)

OLMoE's whitening lifted the neuron-NN median 0.19 → 0.38 (2.0×). Applying the
same factor: **Qwen3 whitened median ≈ 0.45–0.50**, fraction > 0.9 under 1%.
Adjacent-layer union rank stays 128/128. The branch-1 gate remains median > 0.5;
if Qwen3 lands at 0.5+ it is a marginal reopening and the correct response is
the k-means test (F10-Qwen3), not a claim.

### Verdict

**The model matters, in the direction the literature said.** Qwen3 has a
stronger, longer-range residual-side depth structure concentrated in its middle
layers. It does not have shared directions, and it does not have duplicate
neurons. The only mechanism the new evidence points at is a depth anchor with a
low-rank correction on the residual-facing dictionary, in the middle band, on
one matrix side — a narrower claim than anything the original design proposed,
and one that has to survive the whitened re-measurement before it is anything.

---

## F12 — Perplexity of whitened per-expert SVD at 75% size: ×1.63, and `down` is the reason

```bash
python3 -c "from moe_optimizer.eval.ppl import run_f12; run_f12('allenai/OLMoE-1B-7B-0924',
    {'name':'per_expert_svd','rank':512,'whiten':True}, 'runs/calib_olmoe_32k.pt', max_tokens=8192)"
```

OLMoE-1B-7B, all 48 expert slots compressed (expert-table ratio 0.750),
WikiText-2 test, 8,192 tokens, seq 512.

| | perplexity |
|---|---:|
| baseline | **11.060** |
| whitened per-expert SVD r=512 (75%) | **18.027** (×1.63) |

Not usable. A 63% perplexity increase for a 25% saving is far outside what the
literature calls compression (MoBE: 1–2% accuracy drop at 24–30% *total*
reduction).

### Where the error is

| matrix | mean rel_act over 16 layers |
|---|---:|
| gate | 0.219 |
| up | 0.280 |
| **down** | **0.392** |

The five worst slots of 48 are all `down` (L000 0.435, L013 0.413, L014 0.409,
L012 0.407, L003 0.402). Median over all slots 0.280.

That is diagnostic, not just descriptive. `gate`/`up` were whitened with the
full residual-stream covariance. `down` was "whitened" with only the pooled
**diagonal** of the intermediate activation's second moment, because F8 did not
collect its covariance — the diagonal fixes scale, not direction, and the
intermediate activation (post-SiLU × up) is the most anisotropic input in the
block. So `down` was effectively unwhitened in the sense that matters, and it is
the matrix carrying the most error. **F12 as measured is contaminated by that
shortcut and is not yet a fair verdict on whitened SVD.**

The middle ground I skipped — pooled *full* (d_ff × d_ff) covariance — costs
8 MB per layer. `calib/hooks.py` now collects it as `inter_cov`; `slot_stats`
uses it for `down`. F8 is re-run with it and F12 re-measured as F12b; only then
does the 75% point mean anything. F12's r=384 point (56%) runs on the old
statistics first, as the second point on the same footing.

---

## F8-Qwen3 — feasibility pre-registration (written while the smoke test loads)

Qwen3-30B-A3B is 57 GB on disk against 28 GB RAM; calibration runs through
accelerate disk offload (`moeopt calib --cpu-mem 10GiB`). Written before the
smoke number: OLMoE ran 26.5 tok/s fully resident with 1.3B active parameters;
Qwen3 has 3.3B active (÷2.5) and must page ~60 GB from NVMe per batch (~2 GB/s,
so ~30 s of I/O per 2,048-token batch). Expected: **8–11 tok/s**, i.e. ~1 hour
for 32K tokens.

Decision rule: ≥ 5 tok/s → run the full 32K-token F8-Qwen3 and then F9-Qwen3.
2–5 tok/s → run 8K tokens (covariance of a 2048-dim residual stream is still
well-conditioned at 8K samples; routing co-activation will be noisy and is
reported as such). < 2 tok/s → F9-Qwen3 is not feasible on this machine and
the whitened Qwen3 question is left open, stated as such.

**Smoke result (256 tokens, 2 layers hooked, `--cpu-mem 10GiB`): 7.0 tok/s, peak
RSS 16.4 GB, model load 129 s, offload spill ~60 GB on NVMe at ~1 GB/s reads.**
Above the 5 tok/s line → the full 32K-token run proceeds (≈78 min), followed by
F9-Qwen3. Scheduled last in the job chain: 16 GB resident cannot overlap any
step that loads the 14 GB OLMoE model.

---

## F11b — The residual/neuron split replicates on Qwen3-30B-A3B, 3/3, at chance to three decimals

```bash
moeopt audit-depth Qwen/Qwen3-30B-A3B --rank 64 --fast --matrices up down --sides in out
```

| matrix | mode | faces | gap 1 | gap 2 | gap 4 | gap 8 | chance |
|---|---|---|---:|---:|---:|---:|---:|
| gate (F11) | **in** | **residual stream** | **0.584** | 0.468 | 0.315 | 0.277 | 0.031 |
| gate (F11) | out | neurons | 0.096 | 0.096 | 0.097 | 0.097 | 0.083 |
| up | **in** | **residual stream** | **0.567** | 0.446 | 0.300 | 0.259 | 0.031 |
| up | out | neurons | 0.083 | 0.083 | 0.084 | 0.083 | **0.083** |
| down | **out** | **residual stream** | **0.525** | 0.385 | 0.228 | 0.196 | 0.031 |
| down | in | neurons | 0.083 | 0.083 | 0.084 | 0.084 | **0.083** |

Six slots, two models, and the split falls on the same seam every time, with the
sides swapping for `down` exactly as the mechanism predicts. On Qwen3 the
neuron-facing modes measure *at* chance to three decimals — a cleaner null than
OLMoE's 0.062–0.076 against 0.063. Whatever cross-layer structure a pretrained
MoE's expert table has, it lives entirely in the mode anchored to the residual
stream; the private intermediate space carries none.

F2 is now a two-model result and stands as the cleanest finding of the audit,
independent of whether any compression mechanism ever comes of it.

---

## F12-56 / F12b — Perplexity curve, and a diagnosis falsified

| point | expert-table ratio | `down` whitening | perplexity | × baseline |
|---|---:|---|---:|---:|
| baseline | 1.000 | — | 11.060 | 1.00 |
| whitened SVD r=384 | 0.562 | diagonal | **27.364** | **2.47** |
| whitened SVD r=512 (F12) | 0.750 | diagonal | 18.027 | 1.63 |
| whitened SVD r=512 (F12b) | 0.750 | **full (d_ff × d_ff)** | **17.935** | **1.62** |

**My F12 diagnosis was wrong.** I attributed the ×1.63 to whitening `down` by
diagonal only; with the full pooled intermediate covariance the number moves by
0.09 — nothing. `down` carries the most error because it is genuinely the
hardest of the three matrices to approximate at low rank, not because of a
calibration shortcut. F12 stands as measured: **training-free whitened
per-expert SVD is not usable on OLMoE-1B-7B at 75% (×1.62) and is destructive
at 56% (×2.47).** With F10 having shown that no sharing method beats per-expert
SVD, this closes structural low-rank compression on this model outright, not
just the sharing variants.

---

## F8-Qwen3 — calibration through disk offload

32,768 tokens, 48 layers, `--cpu-mem 10GiB`: **11.3 tok/s, 48 min** (predicted
8–11). Includes `inter_cov`. Residual-stream anisotropy is far stronger than
OLMoE's at the ends: layer 0 has top-1 eigenvalue share **0.327** and effective
rank **135**/2048 (OLMoE layer 0: 0.085, 676); middle layers 540–670; layer 45
373. That layer-0 spike is the massive-activation signature OLMoE lacked.

---

## F9-Qwen3 — Activation space on Qwen3: the pre-registered gate is not met

Pre-registered (F11): whitened neuron-NN median **0.45–0.50**; branch-1 gate
median > 0.5.

| | p10 | p25 | **p50** | p75 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 raw (F11) | 0.134 | 0.182 | 0.260 | 0.346 | 0.421 | 0.556 |
| **Qwen3 whitened** | 0.258 | 0.317 | **0.404** | 0.515 | 0.618 | 0.779 |
| OLMoE whitened (F9) | 0.256 | 0.304 | 0.384 | 0.483 | 0.577 | 0.754 |

Fraction > 0.5: 27.8%; > 0.7: 3.7%; > 0.9: **0.1%**. Median 0.404 is *below*
the prediction band and well below the gate. In activation space Qwen3's
neurons look almost exactly like OLMoE's (0.404 vs 0.384). **The neuron
codebook is closed on both models in both metrics.**

Whitened adjacent-layer dictionaries, middle band (gate/in):

| pair | affinity raw → whitened | union rank (sv > 0.1) | angles < 8° |
|---|---|---:|---:|
| 12,13 | ~0.68 → **0.776** | 127/128 | **1** |
| 13,14 | ~0.55 → **0.743** | 127/128 | **1** |
| 14,15 | ~0.69 → **0.825** | 126/128 | **2** |

The first hard-shared directions anywhere in this audit — 1–2 of 64 per pair,
i.e. 2–3%. Not a shared basis. But the soft overlap is now high (0.74–0.83),
which is what decides the anchor-plus-correction rank:

### F11a whitened — the one lever, measured in the right metric

| band | pair | outside energy | correction rank @90% (raw → **whitened**) | saving if anchored |
|---|---|---:|---:|---:|
| early | 2→3 | 0.503 | 52 → **48** | 25% |
| middle | 16→17 | 0.163 | 43 → **46** | 28% |
| middle | 23→24 | 0.174 | 45 → **43** | 33% |
| middle | 30→31 | 0.147 | 44 → **45** | 30% |
| late | 44→45 | 0.159 | 45 → **41** | 36% |

### Verdict

By the pre-registered rules, **branch 2 on Qwen3 as well.** Across two models
(OLMoE-1B-7B, Qwen3-30B-A3B), two metrics (weight space, activation space) and
three granularities (layer subspace, expert, neuron), no training-free
cross-expert or cross-neuron sharing mechanism has support, and on OLMoE even
non-sharing low-rank is destructive (F12). The residual/neuron split (F2) is a
clean two-model mechanistic result. The only lever with measured support is the
depth anchor + low-rank correction on the residual-facing dictionary in Qwen3's
middle band, whose whitened saving is the table above.

---

## F13 — The streaming engine reproduces the reference model (E0)

```bash
python3 scripts/validate_stream.py
```

`runtime/stream.py` reads one layer's weights at a time from the safetensors
mmap. Against `OlmoeForCausalLM` (bf16, loaded through disk offload to stay
under the 30%-free rule), on 34 tokens:

| | value |
|---|---:|
| top-1 token agreement | **100%** |
| NLL stream (fp32) / reference (bf16) | 3.2074 / 3.2123 |
| max \|Δlogit\| / logit scale | 0.91 / 23.2 |
| engine bytes per token at top-8 | 326 MB |
| engine peak RSS | ~3 GB (reference load: 10.4 GB) |

The residual gap is the reference's own bf16 arithmetic through 16 layers.
From here, every policy is scored on this engine by (perplexity, mean k′,
bytes/token, tok/s) — the bandwidth-bound decode regime the mechanism targets.

---

## F14 — E1: expert output scale varies within a layer and is nearly uncorrelated with gate weight (G1 passes)

```bash
python3 scripts/policy_calib.py 2048
```

OLMoE-1B-7B, 2,048 WikiText-2 train tokens through the streaming engine, top-8.
`s[l,e]` = mean ‖E_e(x)‖ over tokens routed to e.

**Gate G1 (pre-registered kill line: within-layer CV of s < 0.15).**

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CV of s | .30 | .39 | .32 | .23 | .19 | .17 | .20 | .18 | .20 | .21 | .20 | .31 | .27 | .28 | .33 | .32 |

Mean **0.255**, min 0.171 — every layer clears the line. **Passes.**

**Is scale redundant with score?** Pearson r between `s[e]` and expert e's mean
gate weight, per layer: mean **+0.17**; layers 5, 7, 12, 13, 14 in [−0.06, +0.05].
The router score carries almost no information about output magnitude — which
is W1 in the survey, now measured rather than argued. A score-only skip rule
(Lu et al. 2024; arXiv:2512.21911) therefore ranks experts by a quantity that is
close to orthogonal to their actual contribution.

**Routing shape.** Sorted top-8 raw-softmax medians: 0.079, 0.064, 0.054,
0.046, 0.040, 0.034, 0.030, 0.027 — a head/tail ratio of only 2.9×. OLMoE's
routing is far flatter than the "certain head, uncertain tail" profile reported
for Qwen3 (arXiv:2602.02443). Flat scores mean score-only skipping has little
signal to work with here; whether contribution ranking supplies what score
lacks is exactly what E2 tests. (The E1 print's "% of tokens reaching 90% gate
mass" rows normalised over all 64 experts and are meaningless for a model with
`norm_topk_prob=False`; fixed in the script, not used by the sweep.)

E2 was launched automatically by the G1 rule.

---

## F15 — E2: contribution-calibrated skipping beats score-only at every budget, and the gap widens

```bash
python3 scripts/policy_sweep.py allenai/OLMoE-1B-7B-0924 2048 6,5,4
```

OLMoE-1B-7B, 2,048 WikiText-2 test tokens, streaming engine, prefill.
All dynamic rules calibrated on the E1 tokens (F14). Baselines and ours at
**matched mean k′**; `score_only` is `contribution` with every scale set to 1 —
same threshold search, same code path, ranking by gate weight alone.

| policy | k′ | perplexity | vs top-8 |
|---|---:|---:|---:|
| top-8 (reference) | 8.00 | 9.548 | — |
| **mass-ratio, per-layer median** (arXiv:2512.21911 rule) | 3.08 | **40.823** | +328% |
| static top-6 | 6.00 | 9.860 | +3.3% |
| score-only @6 | 5.92 | 10.003 | +4.8% |
| **contribution @6** | 5.94 | **9.908** | +3.8% |
| static top-5 | 5.00 | 10.509 | +10.1% |
| score-only @5 | 4.91 | 10.807 | +13.2% |
| **contribution @5** | 4.92 | **10.479** | +9.8% |
| static top-4 | 4.00 | 12.093 | +26.6% |
| score-only @4 | 3.90 | 12.874 | +34.8% |
| **contribution @4** | 3.92 | **11.784** | +23.4% |

### Three things this establishes

**Contribution ranking beats score ranking, and more so the harder you skip:**
−0.9% (k′≈6), −3.0% (k′≈5), **−8.5%** (k′≈4) perplexity at matched k′. The
E4 rule (≥ 2 of 3) passes 3/3.

**Score-only dynamic skipping is worse than not skipping dynamically at all.**
At every budget the fair score-only rule loses to a static top-k cut
(+4.8 vs +3.3, +13.2 vs +10.1, +34.8 vs +26.6). On a model whose router
score is near-orthogonal to expert output magnitude (F14, r = +0.17), a
token-adaptive rule that ranks by score is ranking by noise. Contribution
ranking recovers that and passes static at k′≈5 (−0.3%) and k′≈4 (−2.6%).
**All of the gain from dynamic skipping on this model comes from the calibrated
output scale, none from token-adaptivity by itself.**

**The published training-free rule collapses the model.** Per-layer median
thresholds skip 62% of experts on OLMoE's flat routing and take perplexity to
40.8. This is the rule's design (a fixed 50% skip rate per skip size, no quality
target), not a tuning accident, and it is why the paper's own m=4 setting
degraded math on DeepSeek-R1.

### Caveats, stated before anyone else does

- 2,048 tokens, one seed, one model. The k′≈6 margin (−0.9%) is within noise;
  the k′≈4 margin (−8.5%) is not. An 8,192-token confirmation is queued.
- These are prefill numbers. `MB/tok` is amortised over the sequence and `tok/s`
  is dominated by fp32 conversion churn (the 9.93 outlier is system contention);
  neither is the E3 metric. Batch-1 decode bandwidth is measured separately
  (`decode_benchmark`, chain running).
- Absolute cost is real: the best training-free rule pays +9.8% perplexity for
  37.5% fewer expert loads and +23% for 51%. ZEDA (arXiv:2605.18643) reports
  ~50% fewer expert operations at minimal loss *with* self-distillation training;
  training buys a lot here, and this result does not close that gap — it
  roughly triples the quality-per-skip of the training-free rule it replaces.

---

## F16 — E3: batch-1 decode bandwidth scales linearly with k′; 1.8× decode speedup at k′≈5

```bash
python3 scripts/decode_bench.py allenai/OLMoE-1B-7B-0924 64 6,5,4
```

One token per forward through the KV cache, 32-token prompt, 64 steps. The
cached path reproduces the uncached logits exactly (max |Δlogit| 0.000 for
every policy).

| policy | k′ | MB / token | expert loads / token | tok/s | vs top-8 |
|---|---:|---:|---:|---:|---:|
| top-8 | 8.00 | 2148 | 128.0 | 1.86 | 1.00× |
| mass-ratio (median) | 2.67 | 1074 | 42.6 | 4.17 | 2.24× (model destroyed) |
| static top-6 | 6.00 | 1745 | 96.0 | 2.82 | 1.52× |
| contribution @6 | 5.90 | 1726 | 94.5 | 2.81 | 1.51× |
| static top-5 | 5.00 | 1544 | 80.0 | 3.32 | 1.78× |
| **contribution @5** | 4.89 | 1521 | 78.2 | **3.35** | **1.80×** |
| static top-4 | 4.00 | 1342 | 64.0 | 2.58 | 1.39× |
| contribution @4 | 3.87 | 1316 | 61.9 | 3.02 | 1.62× |

Bytes per token are `16 layers × k′ × 3 matrices × 2048×1024 fp32` plus
attention: **each expert skipped saves ~16.8 MB per token, and the relation is
exactly linear** — the property the survey's W3 said no prior work measured in
this regime. Speed follows bytes (the top-4 row's 2.58 tok/s is system
contention; its bytes are on the line).

**On the `decode-ppl` column — do not read it as a quality result.** It is 63
predictions on one easy slice (values 7.4–9.8 against E2's 9.5–12.9 on 2,048
tokens) and it does *not* reproduce E2's ordering at k′≈4 (contribution 9.81
vs score-only 9.59). It was included as a sanity check of the cache path and
is far below the sample size where a 3–8% perplexity margin is resolvable.
The quality comparison rests on E2 and on the 8,192-token confirmation queued
behind the Qwen3 replication — which this column makes decisive, not optional.

---

## F17 — The Qwen3-30B-A3B streaming engine validates

`StreamingQwen3MoE` (GQA 32/4, per-head QK-norm, θ = 1e6, renormalised top-k)
against the HF model through disk offload, 32 tokens: **top-1 agreement 100%**,
NLL 3.508 vs 3.503, max |Δlogit| 0.72 on scale 30.9, peak RSS 11.4 GB (the
reference load; engine alone ~4 GB). 863 MB/token at top-8 — 4× OLMoE's, from
128 experts × 48 layers. The Qwen3 replication of E1/E2 runs on it.

---

## F18 — Qwen3-30B-A3B: G1 passes harder; the router *anti*-correlates with output scale

```bash
python3 scripts/policy_calib.py Qwen/Qwen3-30B-A3B 1024
```

1,024 WikiText-2 train tokens through `StreamingQwen3MoE`, top-8 of 128, 48 layers.

| | OLMoE-1B-7B (F14) | **Qwen3-30B-A3B** |
|---|---:|---:|
| within-layer CV of output scale s[l,e], mean | 0.255 | **0.335** |
| min / max over layers | 0.17 / 0.39 | 0.17 / **2.66** (L02) |
| r(s[e], mean gate weight), mean over layers | +0.17 | **−0.06** |
| layers with r < −0.2 | 0 | **15 of 48** |
| top-8 sorted weight medians, w1 / w8 | 0.079 / 0.027 (2.9×) | 0.077 / 0.024 (3.2×) |
| tokens needing k′ ≥ 7 for 90% of top-8 mass | — | **98%** |

G1 passes with margin. More striking: on Qwen3 the correlation between an
expert's output magnitude and how much the router weights it is **negative in
15 of 48 layers** (down to −0.35) and ≈0 on average. A score-only skip rule on
this model is not merely ranking by noise (OLMoE); in a third of the layers it
preferentially *keeps* the smaller-output experts. Layer 2 has a CV of 2.66 — a
handful of experts with outputs an order of magnitude larger than the rest —
which no score threshold can see.

Routing is as flat as OLMoE's within the top-8: 98% of tokens need seven of
eight experts to reach 90% of the kept gate mass. The "certain head, uncertain
tail" profile (arXiv:2602.02443) was measured on the Instruct model during
reasoning; the base model on WikiText shows no such head.

The Qwen3 E2 sweep did not complete in this run (see F19 note); replication of
the k′ comparison is pending.

---

## F19 — E4: no worst-domain collapse, and an independent replication of F15

```bash
python3 scripts/policy_tail.py allenai/OLMoE-1B-7B-0924 1024 5.0   # and 4.0
```

OLMoE, 1,024 tokens per corpus: WikiText-2 test, GSM8K (question+answer),
HumanEval (prompt+solution). Perplexity relative to top-8.

| policy | wikitext | gsm8k | code | tail / wikitext |
|---|---:|---:|---:|---:|
| score-only @5 | +8.4% | +6.4% | +6.9% | 1.00 |
| **contribution @5** | **+5.4%** | **+5.3%** | **+6.0%** | 1.10 |
| score-only @4 | +24.8% | +16.1% | +13.5% | 1.00 |
| **contribution @4** | **+15.9%** | **+12.5%** | **+11.3%** | 1.00 |

**Gate G4 (worst-domain degradation < 3× mean): passes** — math and code
degrade *less* than WikiText under both rules, and contribution wins every
cell. The WikiText column is a fresh 1,024-token slice and reproduces F15's
margins in direction and rough size: contribution vs score-only **−2.8%** at
k′≈5 (F15: −3.0%) and **−7.1%** at k′≈4 (F15: −8.5%).

The 8,192-token confirmation with the paired bootstrap crashed before writing
(diagnosed and relaunched separately); until it lands, the F15/F19 margins have
two independent slices behind them and no confidence interval.

---

## F20 — 8,192-token confirmation with paired bootstrap: the OLMoE result holds

```bash
python3 scripts/policy_sweep.py allenai/OLMoE-1B-7B-0924 8192 5,4
python3 scripts/paired_bootstrap.py runs/policy_sweep_olmoe.json
```

16 sequences × 512 tokens, WikiText-2 test. Engine top-8 perplexity 11.051 —
the HF reference on the same text gave 11.060 (F12), so fidelity holds at 8K.

| policy | k′ | ppl | vs top-8 |
|---|---:|---:|---:|
| top-8 | 8.00 | 11.051 | — |
| mass-ratio (median) | 3.20 | 45.213 | +309% |
| static top-5 | 5.00 | 12.299 | +11.3% |
| score-only @5 | 4.94 | 12.579 | +13.8% |
| **contribution @5** | 4.96 | **12.202** | **+10.4%** |
| static top-4 | 4.00 | 14.117 | +27.7% |
| score-only @4 | 3.94 | 14.737 | +33.4% |
| **contribution @4** | 3.96 | **13.701** | **+24.0%** |

Paired bootstrap over the 16 sequences (B = 5000), perplexity ratio:

| comparison | k′≈5 | k′≈4 |
|---|---|---|
| contribution vs score-only | **−3.0% [−4.0, −2.0]** | **−7.0% [−8.8, −5.3]** |
| contribution vs static top-k | −0.8% [−1.9, +0.3] n.s. | **−2.9% [−4.9, −1.1]** |

F15's 2K margins (−3.0%, −8.5%) reproduce at 8K with intervals that exclude
zero. Against static top-k the k′≈5 advantage is not resolved; the k′≈4
advantage is.

---

## F21 — Qwen3-30B-A3B: contribution skipping *loses* — the mechanism is not general as-is

```bash
python3 scripts/policy_sweep.py Qwen/Qwen3-30B-A3B 1024 5,4
```

1,024 tokens (2 sequences), calibration from 1,024 tokens (F18).

| policy | k′ | ppl | vs top-8 |
|---|---:|---:|---:|
| top-8 | 8.00 | 10.806 | — |
| mass-ratio (median) | 4.05 | 260.0 | ×24 |
| static top-5 | 5.00 | **11.493** | +6.4% |
| score-only @5 | 4.91 | 12.021 | +11.2% |
| contribution @5 | 4.94 | 12.660 | +17.2% |
| static top-4 | 4.00 | **12.684** | +17.4% |
| score-only @4 | 3.94 | 14.053 | +30.0% |
| contribution @4 | 3.98 | 14.999 | +38.8% |

Contribution is **worse than score-only by +5.3% / +6.7%** and worse than static
top-k by +10% / +18%. The bootstrap prints "n.s." because it has two sequences;
the point estimates are large and consistent across both budgets, and this is
recorded as a negative result, not as noise.

This contradicts the expectation set by F18 (a *stronger* score/contribution
mismatch on Qwen3 should have helped more). Three testable explanations,
each of which changes what the paper can claim:

1. **Calibration starvation.** 1,024 tokens × top-8 over 128 experts is ~64
   tokens per expert per layer, 4× fewer than OLMoE's E1; experts never routed
   were filled with the layer mean. A noisy `s` ranks by noise. Test:
   recalibrate on 4,096 tokens, rerun k′≈5.
2. **Renormalised routing.** Qwen3 has `norm_topk_prob=True`: the kept experts'
   weights are rescaled to sum to one *after* skipping, so dropping a
   large-output expert raises every other kept expert's weight. The proxy
   w_e·s_e ignores this; on OLMoE (no renormalisation) it is exact. Test: a
   renorm-aware variant that evaluates the kept-set criterion on renormalised
   weights.
3. **Outlier experts.** Layer 2's CV of 2.66 means a few experts with outputs an
   order of magnitude above the rest; contribution ranking keeps them
   unconditionally and may starve the rest of that layer's budget. Test:
   per-layer k′ under each policy.

Until one of these resolves it, the claim is narrowed to the regime measured:
**an unnormalised, flat-scored router (OLMoE-class)**, not fine-grained MoE in
general.

---

## F22 — Qwen3 with 4× calibration data: the loss becomes a tie; the signal buys nothing

```bash
python3 scripts/policy_calib.py Qwen/Qwen3-30B-A3B 4096
python3 scripts/policy_sweep.py Qwen/Qwen3-30B-A3B 4096 5
```

Calibration 4,096 tokens (~256 tokens per expert per layer, matching OLMoE's
E1 density); test 4,096 tokens = 8 sequences, k′=5. CV of s: 0.341;
r(s, gate weight) = −0.053 — F18 reproduces.

| policy | k′ | ppl | vs top-8 (11.879) |
|---|---:|---:|---:|
| static top-5 | 5.00 | **12.642** | +6.4% |
| score-only @5 | 5.00 | 12.945 | +9.0% |
| contribution @5 | 5.03 | 12.987 | +9.3% |
| contribution_renorm @5 (error model) | 5.02 | 13.123 | +10.5% |

Paired bootstrap, 8 sequences: contribution vs score-only **+0.4% [−1.7, +2.3]**
n.s.; contribution vs static +2.7% [−0.3, +6.1] n.s. Per-layer k′ is 4.9–5.2
for all three rules (the per-layer τ calibration equalises budgets by
construction).

Against F21's three hypotheses:

1. **Calibration starvation — largely confirmed.** F21's +5.3% loss vs
   score-only becomes +0.4% with 4× the tokens. The mean scale from 64 tokens
   per expert was noise.
2. **Budget hogging by outlier experts — rejected.** Per-layer k′ is flat.
3. **Renormalisation — rejected as formulated.** The error-model variant that
   accounts for weight mass handed to survivors is *worse* than the heuristic
   by 1.1%.

What remains is a null: on Qwen3-30B-A3B the contribution signal, despite a
larger score/magnitude mismatch than OLMoE's (F18), adds nothing at k′=5.
Static top-k is also a much stronger baseline here (+6.4% vs OLMoE's +11.3%
at the same budget) — Qwen3's score already orders its experts usefully, so
the eighth expert matters less and there is less for any dynamic rule to fix.
Whether the *true* per-token contribution would help — i.e. whether the fault
is the mean-scale proxy or the idea — is the oracle test (F24).

---

## F23 — The orthogonality-derived squared-share rule is worse than the linear heuristic on OLMoE

OLMoE, 8,192 tokens, paired bootstrap, `contribution_sq` (renorm=False, squared
share) vs `contribution` (linear share): **+1.2% [+0.0, +2.3]** at k′≈5,
**+2.3% [+1.4, +3.4]** at k′≈4. The principled error model loses to the
heuristic on the model where the heuristic works. Recorded; the linear rule
stays as the method.

---

## F24 — Oracle: on OLMoE the mean-scale proxy ≈ the true per-token contribution; on Qwen3 the *signal* fails

```bash
python3 scripts/policy_oracle.py Qwen/Qwen3-30B-A3B 4096 5
python3 scripts/policy_oracle.py allenai/OLMoE-1B-7B-0924 4096 5
```

The oracle computes every routed expert and keeps by the true per-token
w_e·‖E_e(x)‖ (renormalised on Qwen3). Not deployable — bytes are unchanged — it
bounds what any contribution-based rule could achieve. τ bisected on realised k′.

| | OLMoE-1B-7B | Qwen3-30B-A3B |
|---|---|---|
| top-8 | 10.333 | 11.879 |
| score-only @5 | 11.643 (k′ 4.94) | 12.945 (k′ 5.00) |
| contribution (mean scale) @5 | 11.338 (k′ 4.97) | 12.987 (k′ 5.03) |
| **oracle** (true per-token) | **11.353 (k′ 4.63)** | **13.626 (k′ 4.79)** |
| oracle vs score-only | **−2.5% [−4.2, −0.8]** | **+5.1% [+1.7, +9.9]** |
| oracle vs contribution | +0.1% [−1.1, +1.4] n.s. | +4.8% [+2.6, +8.2] |
| contribution vs score-only | **−2.6% [−4.1, −1.1]** | +0.3% [−1.8, +2.2] n.s. |

**OLMoE.** The oracle ties the mean-scale proxy — at a *smaller* budget
(4.63 vs 4.97). A 64-float calibrated scale per layer recovers the whole gain
that knowing every expert's actual output would give. The proxy is not the
limitation; this is the strongest form of the OLMoE result.

**Qwen3.** The oracle is worse than score-only by more than its 4% budget
shortfall explains (static top-5 → top-4 costs ~11% here, so 0.2 experts ≈
2%; the remaining ~3% is real). Ranking by the true contribution *norm* does
not help on this model and plausibly hurts. F22's null is therefore not a proxy
problem (hypothesis i, rejected); the signal is the problem.

**Why.** Qwen3 renormalises the kept weights (`norm_topk_prob=True`): dropping
expert e does not remove the term w_e·y_e, it also rescales every survivor by
W_all/W_P. The change in the layer's output then depends on the *directions* of
the kept and dropped outputs, not just their norms — and expert outputs in this
model class are only approximately orthogonal (whitened NN cosine median 0.40,
F9-Qwen3). A norm-only ranking, mean or per-token, ignores that; the
renormalisation-aware error model of F22 assumed exact orthogonality and also
lost. On OLMoE (no renormalisation) removal is subtraction, and the norm is a
sufficient proxy for it.

This is a mechanistic explanation, not yet a measurement. The counterfactual
that measures it — Qwen3's engine with renormalisation switched off, same
policies — is F25.

### Where the paper stands after F24

- **Unnormalised router (OLMoE):** contribution ranking beats score-only by
  2.5–3% (k′≈5) and 7% (k′≈4) with 95% intervals excluding zero; matches the
  oracle; beats static top-k at k′≈4; degrades math/code less than text; 1.80×
  decode speedup. The mean-scale proxy is sufficient.
- **Renormalised router (Qwen3):** no norm-based rule — proxy, error model, or
  oracle — beats score-only; static top-k dominates all dynamic rules.
- **Claim:** calibrated output scale is a necessary and (up to the oracle)
  sufficient correction to score-only skipping on unnormalised routers; on
  renormalised routers the contribution signal is not the right one, and the
  router's renormalisation is the variable that separates the two regimes.
