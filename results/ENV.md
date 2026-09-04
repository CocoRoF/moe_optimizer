# Environment used for every number in this repository

| item | value |
|---|---|
| date of runs | 2026-09-02 – 2026-09-03 |
| CPU | AMD Ryzen 9 6900HX with Radeon Graphics, 16 threads; runs capped at 11 (`OMP_NUM_THREADS=11`, `torch.set_num_threads(11)`) |
| RAM | 28 GB; a 30 %-free rule was enforced (engine anonymous RSS 3.7–4.5 GB) |
| GPU | none |
| page cache | the streaming engine reads expert weights from the safetensors mmap; it runs at the reported speeds only while a model's shards stay resident in page cache (OLMoE 13 GB, Qwen3 57 GB — the latter never fits here, so its runs are NVMe-bound, F26). Any concurrent bulk disk write (e.g. a model download) evicts the shards and slows OLMoE runs ~3× (measured: 660–770 MB/s process reads, 28 % iowait). Run one model at a time with nothing else touching the disk. |
| OS / kernel | Linux 6.8.0-124-generic |
| Python | 3.12.3 |
| torch | 2.14.0+cpu |
| transformers | 5.16.1 |
| safetensors / datasets / numpy | 0.8.0 / 5.0.1 / 2.5.2 |
| full lock | `requirements-lock.txt`; install with `scripts/setup_env.sh` (torch from the PyTorch CPU index, the rest from PyPI — a single mixed-index resolve fails). Verified on a fresh clone: 36/36 tests pass. |

## Model checkpoints (HF Hub snapshot commit)

| model | repo id | snapshot revision | config facts used |
|---|---|---|---|
| OLMoE-1B-7B | `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | 16 layers, 64 experts, top-8, d=2048, d_ff=1024, `norm_topk_prob=False`, MHA 16 heads, RoPE θ=1e4, RMSNorm eps 1e-5 |
| Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B` | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | 48 layers, 128 experts, top-8, d=2048, moe_d_ff=768, `norm_topk_prob=True`, GQA 32/4, head_dim 128, RoPE θ=1e6, eps 1e-6 |

## Data

| use | dataset | split | selection |
|---|---|---|---|
| calibration | `Salesforce/wikitext` · `wikitext-2-raw-v1` | train | non-empty lines joined by blank line, tokenised with the model's tokenizer, **first N tokens** (N in each table) |
| perplexity | same | test | same construction, first N tokens, cut into 512-token sequences |
| tail: math | `openai/gsm8k` · main | test | first 400 rows, question + " " + answer |
| tail: code | `openai/openai_humaneval` | test | prompt + canonical_solution, all 164 |

## Randomness

The only stochastic steps are the paired bootstrap (`random.seed(0)`, B = 5000) and the k-means/spectral clustering in the compression-era code (`seed=0`). Expert-skipping calibration, τ bisection, sweeps and the oracle are deterministic given the token selection above and the thread count (fp32 reductions can differ in the last bits across thread counts).
