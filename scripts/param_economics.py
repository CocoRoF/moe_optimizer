#!/usr/bin/env python3
"""Where do the parameters of a compressed MoE expert table actually live?

Thin wrapper -- the implementation lives in the package so that
``moeopt econ`` and this script cannot drift apart.

Run:  python3 scripts/param_economics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_optimizer.param_economics import report  # noqa: E402

if __name__ == "__main__":
    report()
