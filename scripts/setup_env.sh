#!/usr/bin/env bash
# Create .venv exactly as used for every number in results/.  Two steps because
# torch==2.14.0+cpu exists only on PyTorch's CPU index, and mixing indexes in one
# resolve makes uv pick unrelated packages from the wrong index.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0+cpu"
grep -vE "^torch==" requirements-lock.txt > /tmp/req_no_torch.txt
VIRTUAL_ENV=.venv uv pip install --python .venv/bin/python -r /tmp/req_no_torch.txt -e .
.venv/bin/python -c "import torch, transformers, moe_optimizer; print('ok: torch', torch.__version__, 'transformers', transformers.__version__)"
