"""Per-slot views of a saved calibration file."""

from __future__ import annotations

import torch


def slot_stats(calib: dict, layer: int, matrix: str) -> dict:
    """Per-slot statistics for the compressors.

    gate/up read the residual stream, so their input covariance is the layer's
    residual second moment.  down reads the intermediate activation; it gets the
    pooled full covariance when the calibration file has it, and falls back to
    the pooled diagonal for older files (scale only, not direction -- the
    setting under which F12's first run was measured).
    """
    st = calib[layer]
    if matrix in ("gate", "up"):
        cov = st["input_cov"]
    elif "inter_cov" in st:
        cov = st["inter_cov"]
    else:
        cov = torch.diag(st["inter_sq"].mean(0))
    return {"input_cov": cov, "importance": st["importance"],
            "counts": st["counts"], "coactivation": st["coactivation"],
            "n_tokens": st["n_tokens"]}
