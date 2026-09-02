# Measured findings

Every number here is reproducible from this repo on CPU. Commands are given.

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
