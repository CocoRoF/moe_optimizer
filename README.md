# moe_optimizer

Training-free structural compression of pretrained Mixture-of-Experts LLMs.

The goal is to re-represent the expert weight table of an **already-trained** MoE
so that a serving system does not have to hold every expert in memory — without
retraining, without deleting or merging experts, and without touching the router.

Research context, prior-art survey and the falsification gates live in
[`../my_paper/`](../my_paper/). Read
`REVIEW_and_REDIRECTION_2026-09-02.md` first: it contains the finding that set
the current direction.

---

## The finding this repo is built around

The original design placed an orthogonal-polynomial (Legendre) chart on the
**expert-mode coordinate** — compressing the per-expert coefficient table.

That cannot work, and the reason is arithmetic rather than empirical. Every
"shared dictionary + per-expert coefficients" scheme stores

```
per-expert = E · K          shared = K · D          D ≫ E
```

so the per-expert share of the code is `E / (E + D)`, independent of rank and of
method. Measured on real configurations:

| model | E | D | per-expert share | saving from charting it |
|---|---:|---:|---:|---:|
| OLMoE-1B-7B | 64 | 262,144 | 0.024% | 0.008% |
| Qwen3-30B-A3B | 128 | 147,456 | 0.087% | 0.026% |
| DeepSeekMoE-16B | 64 | 495,616 | 0.013% | 0.006% |
| Mixtral-8x7B | 8 | 51,380,224 | 0.000% | −0.000% |

Reproduce with `moeopt econ`. The claim is also pinned as an executable
assertion in `tests/test_compressors.py::test_expert_chart_saving_is_negligible`.

**Consequence.** On the expert axis a polynomial chart is a *regulariser* and a
*progressive code*, never a compressor. The compression lever is the **depth
axis**, where the dictionary is replicated L times (16 for OLMoE, 48 for
Qwen3-30B-A3B) and where — unlike the expert index — a genuine total order exists.

---

## What has been measured so far

`moeopt audit-depth allenai/OLMoE-1B-7B-0924` — all three matrix types, rank 64.
Affinity is mean cos² of principal angles between two layers' dictionaries.

| matrix | mode | faces | gap 1 | gap 4 | gap 8 | chance |
|---|---|---|---:|---:|---:|---:|
| gate | out | neurons | 0.076 | 0.076 | 0.076 | 0.063 |
| gate | **in** | **residual stream** | **0.508** | 0.178 | 0.105 | 0.031 |
| up | out | neurons | 0.062 | 0.062 | 0.063 | 0.063 |
| up | **in** | **residual stream** | **0.506** | 0.174 | 0.093 | 0.031 |
| down | **out** | **residual stream** | **0.489** | 0.127 | 0.046 | 0.031 |
| down | in | neurons | 0.063 | 0.063 | 0.062 | 0.063 |

**The split is 3/3 along the same seam, and the sides swap for `down`.**
`gate` and `up` read the residual stream on their input mode; `down` writes to it
on its output mode. The residual stream is one representation space maintained
across all layers, so dictionaries anchored to it are comparable between layers.
The intermediate neuron space is private per layer and defined only up to
permutation — and those dictionaries measure at chance, exactly as that predicts.
The `up`/`down` runs were predictions registered from the `gate` result, not a
post-hoc reading.

Design consequences: the depth chart applies to the residual-stream-facing mode
only, the neuron-facing dictionary stays per-layer, and charts must be
**piecewise** — affinity halves by gap 2–3 and reaches chance by gap 6–8, so a
single low-degree polynomial across all layers is not supported. See
[`docs/FINDINGS.md`](docs/FINDINGS.md) for the full table and consequences,
including two findings (F3, F4) about estimator design that came out of
validation.

## Two negative results already on the board

From the first sweep on real weights (`docs/FINDINGS.md`, F5):

- **Weight-cosine clustering does not beat an arbitrary grouping.** Spectral
  clustering scores 0.89665 relative error; the `uniform` null model — contiguous
  blocks of expert indices — scores 0.89659 at identical bytes. The null model
  fired on its first run. Narrow reading: this indicts raw *weight* similarity,
  the signal permutation symmetry makes least trustworthy; the routing and output
  signals need calibration traces that are not built yet.
- **Sharing does not yet beat the per-expert floor.** The Pareto front is
  dominated by `per_expert_svd` and `shared_base_delta`, the two methods with no
  cross-expert structure at all.

Both are measured without activation-aware whitening, in weight space, at ratios
of 2-20% where F1b shows every method is already broken. That is the finding that
sets the next priority: **calibration statistics and whitening before any further
method work** — until the metric is activation-weighted, the sweep cannot
distinguish a bad method from a bad metric.

---

## Install

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install --python .venv/bin/python -e .
.venv/bin/python -m pytest tests/ -q
```

CPU-only by design: every algorithm streams one expert table at a time and the
whole pipeline runs in well under 28 GB with no GPU. Phases that genuinely need
a GPU (task benchmarks, latency, HBM) are deliberately not implemented here.

## Use

```bash
moeopt econ                                     # parameter-economics tables
moeopt audit-depth MODEL --rank 64              # gate G0: depth smoothness
moeopt sweep MODEL --ranks 16 32 64 --max-layers 2   # matched-budget Pareto
moeopt calib MODEL --out runs/calib.pt          # F8: calibration statistics
moeopt sweep MODEL --calib runs/calib.pt ...    # whitened fit, scored by rel_act
```

---

## Layout

```
src/moe_optimizer/
  param_economics.py    storage models for every method family; the C1 analysis
  registry.py           name -> factory, so every method is reachable uniformly
  types.py              MoEArch / Slot / RoutingStats
  io/
    checkpoint.py       shard-aware, memory-mapped, one-table-at-a-time access
    adapters/           Qwen3-MoE, OLMoE, Mixtral, DeepSeek naming + shapes
  geometry/
    spectrum.py         effective / stable rank, energy-at-rank
    subspace.py         principal angles, Grassmann + chordal distance
    alignment.py        neuron permutation alignment (exact, via LAP)
    depth.py            gate G0: does the dictionary rotate smoothly with depth?
  calib/
    hooks.py            forward-pass statistics: residual cov, routing, co-activation
    run.py              run a calibration corpus and save per-layer stats
  community/cluster.py  spectral / agglomerative / uniform-null clustering
  factorize/
    base.py             Compressor contract + byte-exact SlotCode accounting
    whiten.py           activation-aware whitening (default path, not optional)
    chart.py            Legendre + routing-weighted empirical orthogonal bases
  methods/
    baselines.py        per-expert SVD, shared base+delta, MoBE-like shared basis
    local_atlas.py      POEM-Atlas: communities + local dictionaries
  ablation/
    depth_atlas.py      depth chart -- falsified by F6, kept as the record
  eval/sweep.py         matched-budget Pareto sweeps
  cli.py
```

---

## Design rules

**Bytes, not parameters.** `SlotCode.nbytes` counts serialised bytes including
every scale, index and shape. Parameter counts hide quantisation; omitting
metadata hides sparse-residual index overhead. `component_bytes` always splits
shared / per-expert / residual, so the C1 ratio is visible on every result.

**One table in memory at a time.** A single expert table is 0.5–0.8 GB in float32;
the checkpoint is 14–60 GB. Every algorithm consumes `ExpertStore.stack(slot)`
and releases it. `geometry.depth` never even forms the table — it accumulates
Gram matrices one expert at a time.

**Whitening is the default path.** Plain factorisation minimises `‖W − Ŵ‖_F`;
deployment cares about `‖(W − Ŵ)x‖` under the real token distribution.
arXiv:2606.03465 traces the underperformance of tensor decompositions on LLMs to
exactly this mismatch, so the un-whitened variant is an ablation, not the default.

**Matrix factorisation, not Tucker.** The same paper reports TD-MoE
substantially underperforming the matrix baseline MoBE on the models we target.
A Tucker arm is kept for ablation only.

**A null model for every clustering claim.** `CLUSTERERS["uniform"]` groups
experts into arbitrary contiguous blocks. Any gain attributed to functional
clustering must be measured against it; if a method does no better, its
clustering step is decoration.

**Training-free, defined explicitly.**

| tier | permitted | status |
|---|---|---|
| T0 | closed form only — SVD, weighted QR, least squares | main result |
| T1 | T0 + alternating least squares on calibration statistics; no backprop, no labels | main result |
| T2 | gradient recovery, distillation, LM loss | ablation only |

Everything currently implemented is T0 or T1.

---

## Status

Working and tested: parameter economics, checkpoint I/O and adapters, geometry
(spectrum / subspace / alignment / depth), clustering, whitening, orthogonal
charts, four compression methods, byte accounting, Pareto sweeps, CLI.

Also built: `calib/` -- per-layer residual covariance, routing counts,
co-activation and per-neuron intermediate second moments from a live forward
pass (26 tok/s on CPU beside the bf16 model).

Not yet built: sparse outlier residuals, adaptive rank allocation, the
serialised artifact format, and the runtime modes.

Where the research stands (see `docs/FINDINGS.md`): on OLMoE-1B-7B every
sharing axis is closed, in both metrics. Cross-layer (F6, F9): no shared
directions. Cross-neuron (F7, F9): whitened NN median 0.38, 0.1% above 0.9.
Cross-expert (F10, whitened, scored by output error): per-expert SVD beats
every sharing method, and functional clustering ties an arbitrary grouping to
four decimals. Whitened per-expert SVD at 75% size gives 15.6% output error
with no sharing assumptions. Qwen3-30B-A3B (F11): the residual/neuron split
replicates; its middle ~30 layers show stronger, longer-range cross-layer
structure (gap-1 affinity 0.55-0.77, reproducing ConMoE's middle-layer effect)
but still zero shared directions and zero duplicate neurons. A depth anchor
with low-rank correction would save 30-33% of one side's dictionary in that
band, before whitening -- real, small, and the only lever with measured
support. F12: whitened per-expert SVD at 75% size takes OLMoE from 11.06 to 18.03
perplexity (x1.63) -- not usable -- with `down` carrying the most error
because its input was whitened by diagonal only; re-measured as F12b with the
full intermediate covariance. Open: the whitened Qwen3 re-run, which needs
disk-offload calibration. Task
benchmarks and system measurements need a GPU and are out of scope here.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
