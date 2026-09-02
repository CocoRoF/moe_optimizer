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
