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
