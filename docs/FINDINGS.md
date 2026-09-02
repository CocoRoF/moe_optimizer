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
