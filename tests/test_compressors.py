"""Every compressor must reconstruct what it claims to reconstruct.

A silently wrong reconstructor would poison every downstream number, so these
tests check exactness on inputs whose structure each method is supposed to
represent perfectly, not merely that the code runs.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

import moe_optimizer.methods  # noqa: F401  (registration)
from moe_optimizer.factorize.base import reconstruction_report
from moe_optimizer.registry import COMPRESSORS

E, D_OUT, D_IN = 16, 48, 64


def make_stack(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(E, D_OUT, D_IN, generator=g, dtype=torch.float32)


def make_community_stack(n_comm=4, rank=6, seed=0):
    """Experts that exactly match the local-atlas model: anchor + U diag(m) V^T."""
    g = torch.Generator().manual_seed(seed)
    stack = torch.empty(E, D_OUT, D_IN, dtype=torch.float64)
    per = E // n_comm
    for c in range(n_comm):
        anchor = torch.randn(D_OUT, D_IN, generator=g, dtype=torch.float64)
        U, _ = torch.linalg.qr(torch.randn(D_OUT, rank, generator=g, dtype=torch.float64))
        V, _ = torch.linalg.qr(torch.randn(D_IN, rank, generator=g, dtype=torch.float64))
        for j in range(per):
            m = torch.randn(rank, generator=g, dtype=torch.float64)
            stack[c * per + j] = anchor + U @ torch.diag(m) @ V.T
    return stack.float()


@pytest.mark.parametrize("name,kw", [
    ("per_expert_svd", {"rank": min(D_OUT, D_IN), "whiten": False, "dtype": "float32"}),
    ("shared_base_delta", {"rank": min(D_OUT, D_IN), "whiten": False, "dtype": "float32"}),
])
def test_full_rank_round_trip_is_exact(name, kw):
    stack = make_stack()
    code = COMPRESSORS.get(name)(**kw).fit(stack)
    rep = reconstruction_report(stack, code)
    assert rep["rel_fro"] < 1e-9, rep


def test_local_atlas_recovers_exact_community_structure():
    """Given data that exactly satisfies its model, the error must be ~0."""
    stack = make_community_stack(n_comm=4, rank=6)
    comp = COMPRESSORS.get("local_atlas")(
        n_communities=4, rank=6, coupling="diag", clusterer="uniform",
        whiten=False, dtype="float32")
    rep = reconstruction_report(stack, comp.fit(stack))
    # Bounded by float32 epsilon (~1.2e-7): the stack is stored in float32 as a
    # real checkpoint would be, so this is exactness up to input precision.
    assert rep["rel_fro"] < 1e-6, rep


def test_local_atlas_full_coupling_is_at_least_as_good_as_diag():
    stack = make_stack(seed=3)
    kw = dict(n_communities=4, rank=8, clusterer="uniform", whiten=False, dtype="float32")
    diag = reconstruction_report(stack, COMPRESSORS.get("local_atlas")(coupling="diag", **kw).fit(stack))
    full = reconstruction_report(stack, COMPRESSORS.get("local_atlas")(coupling="full", **kw).fit(stack))
    assert full["rel_fro"] <= diag["rel_fro"] + 1e-12


def test_expert_chart_reconstructs_consistently():
    """The chart path must reproduce the no-chart path when the residual is kept."""
    stack = make_community_stack(n_comm=4, rank=6)
    kw = dict(n_communities=4, rank=6, coupling="diag", clusterer="uniform",
              whiten=False, dtype="float32")
    plain = COMPRESSORS.get("local_atlas")(**kw).fit(stack)
    charted = COMPRESSORS.get("local_atlas")(expert_chart=True, chart_q=2,
                                             chart_degree=3, **kw).fit(stack)
    torch.testing.assert_close(plain.reconstruct(), charted.reconstruct(),
                               atol=1e-8, rtol=1e-6)


def test_expert_chart_saving_is_negligible():
    """The C1 finding, as an executable assertion.

    Re-coding the per-expert coefficient table cannot move the total, because the
    per-expert share of the code is bounded by E/(E+D).  If this test ever fails,
    the analysis in my_paper/REVIEW_and_REDIRECTION_2026-09-02.md is wrong.
    """
    stack = make_community_stack(n_comm=4, rank=6)
    kw = dict(n_communities=4, rank=6, coupling="diag", clusterer="uniform",
              whiten=False, dtype="float16")
    plain = COMPRESSORS.get("local_atlas")(**kw).fit(stack)
    assert plain.per_expert_share < 0.05, plain.summary()


def test_shared_basis_per_expert_share_is_dominant():
    """The mirror-image fact: in a MoBE-like code, A_e dominates the bytes."""
    stack = make_stack(seed=7)
    code = COMPRESSORS.get("shared_basis")(rank=8, n_basis=4, iters=3,
                                           whiten=False, dtype="float16").fit(stack)
    assert code.per_expert_share > 0.5, code.summary()


def test_byte_accounting_is_consistent():
    stack = make_stack()
    code = COMPRESSORS.get("local_atlas")(n_communities=2, rank=8, clusterer="uniform",
                                          whiten=False, dtype="float16").fit(stack)
    parts = code.component_bytes
    assert abs(sum(parts.values()) - code.nbytes) < 1e-6
    assert code.dense_bytes("bfloat16") == E * D_OUT * D_IN * 2
    assert 0 < code.ratio() < 10
