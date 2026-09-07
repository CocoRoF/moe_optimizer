# moe_optimizer

**Contribution-calibrated dynamic expert skipping for bandwidth-bound MoE decoding — training-free, router-preserving, CPU-reproducible.**

Every routed expert that is *not* executed is three weight matrices not read from memory. Training-free skipping rules to date decide which experts to drop from the router score alone. We measure that the score carries almost no information about how much an expert contributes (r = +0.17 on OLMoE-1B-7B, −0.05 on Qwen3-30B-A3B), and that ranking by `score × calibrated output scale` — 64 floats per layer from one calibration pass — fixes this on an unnormalised router and does not on a renormalised one.

📄 **Results with hyperparameters:** [`docs/FINDINGS.md`](docs/FINDINGS.md) · 🧪 **Raw outputs:** [`results/`](results/) · 🔁 **Reproduce:** `scripts/reproduce.sh` · 🖥 **Environment:** [`results/ENV.md`](results/ENV.md) · 📓 **Lab log (incl. the compression-era negatives):** [`docs/LOG.md`](docs/LOG.md) · 📚 **Literature survey (24 papers, verified):** [`docs/SURVEY.md`](docs/SURVEY.md) · ✍️ **Paper outline:** [`docs/PAPER_OUTLINE.md`](docs/PAPER_OUTLINE.md) · 📊 **Slides:** [`docs/slides/`](docs/slides/)

## What is measured (OLMoE-1B-7B; exact numbers, paired 95 % CIs)

**Perplexity frontier** (8,192 test tokens, best rule at each budget): **+0.6 %** at 12.8 % fewer expert loads · **+3.3 %** at 25 % · **+10.3 %** at 38 % · **+23 %** at 50 %. The contribution-ranking signal beats the score-only rule at every budget (−0.5 % → −7 %, growing with the cut) and static top-k at 25 %+ (−1.0 % [−1.7, −0.3] at 25 %; −2.9 % [−4.9, −1.1] at 50 %). The published median-threshold rule collapses the model (+309 %).

**Downstream accuracy** (ARC-Easy, ARC-Challenge, OpenBookQA, HellaSwag; n = 200 each): every rule loses **≈ 1 point** on average at 25 % and ≈ 1–2 at 38 % fewer loads; per-task intervals are ±3–4 points and **no rule is distinguishable from another**. The two comparisons that reach significance are adverse for the contribution rule (ARC, −4.5 [−8.0, −1.5] at each budget). The perplexity advantages above are below what this sample size resolves; n = 1,000 is queued.

**Batch-1 decode** (CPU, bytes counted): 1.80× tok/s at 38 % fewer loads; bytes/token linear in loads.

**Other models.** Qwen1.5-MoE-A2.7B (shared expert): **+1.1 % ppl at 25 %** and +1.6 % at 37.5 % fewer *routed* loads — the lowest loss of the three, because the always-on shared expert carries much of each layer; the ranking signal is irrelevant there and per-layer budgets dominate. Qwen3-30B-A3B: static top-k wins (+6.2 % at 37.5 %); no norm-based signal helps, and calibration alone detects this (F30: 42/48 layers choose score).

Full tables, hyperparameters and CIs: [`docs/FINDINGS.md`](docs/FINDINGS.md) §16–19; regenerate with `scripts/report_pareto.py`.

## Install and run

```bash
scripts/setup_env.sh                          # .venv with the exact pinned versions (torch CPU wheel from PyTorch's index, rest from PyPI)
.venv/bin/python -m pytest tests/ -q          # 37 tests
scripts/reproduce.sh                          # everything, in order; ~8 h CPU total, one job at a time
```

Individual steps (each writes to `runs/`; compare with `results/`):

```bash
PYTHONPATH=src OMP_NUM_THREADS=11 .venv/bin/python scripts/validate_stream.py allenai/OLMoE-1B-7B-0924 6GiB   # F13
PYTHONPATH=src OMP_NUM_THREADS=11 .venv/bin/python scripts/policy_calib.py    allenai/OLMoE-1B-7B-0924 2048   # F14
PYTHONPATH=src OMP_NUM_THREADS=11 .venv/bin/python scripts/policy_sweep.py    allenai/OLMoE-1B-7B-0924 8192 5,4 && .venv/bin/python scripts/paired_bootstrap.py runs/policy_sweep_olmoe.json   # F20
PYTHONPATH=src OMP_NUM_THREADS=11 .venv/bin/python scripts/decode_bench.py    allenai/OLMoE-1B-7B-0924 64 6,5,4   # F16
PYTHONPATH=src OMP_NUM_THREADS=11 .venv/bin/python scripts/policy_tail.py     allenai/OLMoE-1B-7B-0924 1024 5.0   # F19
PYTHONPATH=src OMP_NUM_THREADS=11 .venv/bin/python scripts/policy_oracle.py   allenai/OLMoE-1B-7B-0924 4096 5     # F24
```

Models are fetched from the HF Hub on first use (13 GB / 57 GB) into `.cache/`. Qwen3 validation additionally spills ~60 GB to `runs/offload_*`. No GPU; 28 GB RAM suffices (engine anonymous RSS ≤ 4.5 GB; the reference-model validation peaks at ~11 GB).

## The method in one paragraph

For each layer `l` and expert `e`, one calibration pass records `s[l,e] = mean ‖E_e(x)‖` over tokens routed to `e`. At inference the router's top-k candidates are ranked by `c_e = w_e · s[l,e]` and the smallest prefix whose cumulative share of `Σ c` reaches `1 − τ_l` is executed; `τ_l` is set per layer by bisection on the calibration traces so that the mean executed count equals a target `k′`. Nothing is trained; the router and expert identities are untouched; storage is `E` floats per layer. `runtime/stream.py` implements it as `ContributionPolicy`, next to the reference `TopKPolicy`, the published `MassRatioPolicy` baseline, and the diagnostic variants.

## Layout

```
src/moe_optimizer/
  runtime/stream.py      layer-streaming CPU decoders (OLMoE, Qwen3-MoE), expert policies, oracle, decode benchmark
  runtime/calibrate.py   median betas (baseline), per-layer tau for a target k', error-model variants
  calib/                 forward-pass statistics (residual covariance, routing, co-activation)
  io/                    shard-aware mmap checkpoint access; model adapters
  geometry/, community/, factorize/, methods/, ablation/   compression-era code (LOG.md F1–F12)
scripts/                 validate_stream, policy_calib, policy_sweep, decode_bench, policy_tail, policy_oracle,
                         paired_bootstrap, reproduce.sh
results/                 raw JSON/log outputs, calibration artefacts, ENV.md
docs/                    FINDINGS.md (results), LOG.md (chronological), slides/, figures/
tests/                   37 tests: policy exactness, engine round-trips, bootstrap, undefined-name check
```

## Design rules

- **Matched budget, same code path.** Every dynamic rule is scored at the same mean k′; the score-only control is `ContributionPolicy` with all scales set to 1 — identical threshold search, so any difference is the ranking signal.
- **Pre-registered gates.** Kill conditions (G1–G4) were written before the runs; outcomes are in FINDINGS §9, including the failures.
- **Bytes, not FLOPs.** The engine counts bytes moved; a skipped expert is a measured saving, not a modelled one.
- **Error bars.** Per-sequence NLL is logged and every headline comparison is a paired bootstrap on identical text.
- **Sequence long jobs in one shell; never guard on `pgrep -f` or git state; never kill by marker.** Five chained runs deadlocked or killed themselves on self-matching text: `pgrep -f "<pattern>"` from a script whose own command line contained the pattern (via the guard line, a relaunch line, and `pkill`), a `git log` guard a later commit made unsatisfiable, and `pkill -f "[E]7 DONE"` from a chain whose own marker was `E6E7 DONE`. The bracket trick protects only its own line. Working pattern: one background shell runs each step in the foreground, sequentially; the only gate is free memory; completion markers are unique strings that are not substrings of one another; a chain never kills processes.
- **Static check.** `pyflakes` over `scripts/` and `src/` runs in the test suite after two runs died on a `NameError` that `ast.parse` cannot see.

## Limitations

+10 % perplexity for 38 % fewer expert loads is real cost; ZEDA (with self-distillation) reports ~50 % at minimal loss. One positive model and one characterised null; perplexity only (no GPU for downstream tasks); batch-1 fp32 CPU regime. Both principled variants we derived lost to the heuristic. Full list in FINDINGS §10.

## History

This repository began as a study of training-free *structural* compression of MoE expert tables (Legendre charts, local atlases, neuron codebooks). Twelve pre-registered probes across two models and two metrics found no exploitable redundancy at any granularity, and whitened per-expert SVD at 75 % size raised perplexity ×1.62. That work is preserved in `docs/LOG.md` F1–F12 and `src/moe_optimizer/{geometry,community,factorize,methods,ablation}`; the inference-optimisation line above is what survived.

## License

Apache-2.0.
