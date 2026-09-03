#!/usr/bin/env bash
# Reproduce every expert-skipping number in docs/FINDINGS.md, in dependency order.
# CPU only. Wall-clock on a 16-core AVX2 box at 11 threads is given per step.
# All runs read the model from ./.cache (downloaded on first use) and write to ./runs.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=11 PYTHONPATH=src
PY=.venv/bin/python
OLMOE=allenai/OLMoE-1B-7B-0924; QWEN=Qwen/Qwen3-30B-A3B

echo "== 0. environment =="; $PY -c "import torch,transformers;print(torch.__version__,transformers.__version__)"
echo "== 1. engine validation against HF (F13, F17)  ~5 min / ~8 min =="
$PY scripts/validate_stream.py $OLMOE 6GiB
$PY scripts/validate_stream.py $QWEN  6GiB          # needs ~60 GB disk for the offload spill

echo "== 2. calibration (F14, F18)  ~2 min / ~45 min =="
$PY scripts/policy_calib.py $OLMOE 2048             # -> runs/policy_calib_olmoe.pt
$PY scripts/policy_calib.py $QWEN  4096             # -> runs/policy_calib_qwen3.pt

echo "== 3. matched-k' sweeps + paired bootstrap (F20, F23, F22)  ~50 min / ~2 h =="
$PY scripts/policy_sweep.py $OLMOE 8192 5,4 && $PY scripts/paired_bootstrap.py runs/policy_sweep_olmoe.json
$PY scripts/policy_sweep.py $QWEN  4096 5   && $PY scripts/paired_bootstrap.py runs/policy_sweep_qwen3.json

echo "== 4. batch-1 decode bandwidth (F16)  ~15 min =="
$PY scripts/decode_bench.py $OLMOE 64 6,5,4

echo "== 5. worst-domain tail (F19)  ~20 min =="
$PY scripts/policy_tail.py $OLMOE 1024 5.0; $PY scripts/policy_tail.py $OLMOE 1024 4.0

echo "== 6. oracle (F24)  ~1 h / ~2 h =="
$PY scripts/policy_oracle.py $OLMOE 4096 5; $PY scripts/policy_oracle.py $QWEN 4096 5

echo "== 7. counterfactual: Qwen3 without top-k renormalisation (F25)  ~1.5 h =="
$PY scripts/policy_sweep.py $QWEN 4096 5 --no-renorm && $PY scripts/paired_bootstrap.py runs/policy_sweep_qwen3_norenorm.json
echo "done. Compare runs/*.json against results/."
