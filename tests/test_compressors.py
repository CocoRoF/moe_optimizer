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


def test_rel_act_rewards_whitened_fit_on_anisotropic_input():
    """A whitened fit must win on rel_act and may lose on rel_fro.

    This is the metric mismatch that made the first whitened sweep unreadable:
    fits optimised for the data-weighted norm were scored in the raw one.
    """
    g = torch.Generator().manual_seed(11)
    stack = torch.randn(E, D_OUT, D_IN, generator=g)
    x = torch.randn(4000, D_IN, generator=g) @ torch.diag(torch.linspace(4.0, 0.05, D_IN))
    cov = (x.T @ x / 4000).double()
    stats = {"input_cov": cov}
    raw = COMPRESSORS.get("per_expert_svd")(rank=6, whiten=False, dtype="float32").fit(stack)
    wht = COMPRESSORS.get("per_expert_svd")(rank=6, whiten=True, dtype="float32").fit(stack, stats)
    r_raw = reconstruction_report(stack, raw, input_cov=cov)
    r_wht = reconstruction_report(stack, wht, input_cov=cov)
    assert r_raw["rel_fro"] <= r_wht["rel_fro"] + 1e-9        # raw wins its own metric
    assert r_wht["rel_act"] < r_raw["rel_act"]                 # whitened wins the real one


def test_write_slot_targets_the_right_half_of_fused_gate_up():
    """gate -> first d_ff rows of gate_up_proj, up -> second, down -> down_proj."""
    import torch.nn as nn
    from moe_optimizer.eval.ppl import write_slot

    class Fused(nn.Module):
        def __init__(self, E=3, d_ff=4, d=5):
            super().__init__()
            self.intermediate_dim = d_ff
            self.gate_up_proj = nn.Parameter(torch.zeros(E, 2 * d_ff, d))
            self.down_proj = nn.Parameter(torch.zeros(E, d, d_ff))

    f = Fused()
    g, u = torch.full((3, 4, 5), 1.0), torch.full((3, 4, 5), 2.0)
    d = torch.full((3, 5, 4), 3.0)
    write_slot(f, "gate", g); write_slot(f, "up", u); write_slot(f, "down", d)
    assert torch.equal(f.gate_up_proj[:, :4], g) and torch.equal(f.gate_up_proj[:, 4:], u)
    assert torch.equal(f.down_proj, d)


def test_compressed_dir_round_trips_into_a_fused_module(tmp_path):
    """Phase 1 writes what phase 2 reads, and the tensor lands in the right slice."""
    import json
    import torch.nn as nn
    from safetensors.torch import save_file
    from moe_optimizer.eval.ppl import load_compressed_into

    class Fused(nn.Module):
        def __init__(self, E=2, d_ff=3, d=4):
            super().__init__()
            self.intermediate_dim = d_ff
            self.gate_up_proj = nn.Parameter(torch.zeros(E, 2 * d_ff, d))
            self.down_proj = nn.Parameter(torch.zeros(E, d, d_ff))

    class MLP(nn.Module):
        def __init__(self): super().__init__(); self.experts = Fused()
    class Layer(nn.Module):
        def __init__(self): super().__init__(); self.mlp = MLP()
    class Inner(nn.Module):
        def __init__(self): super().__init__(); self.layers = nn.ModuleList([Layer()])
    class Model(nn.Module):
        def __init__(self): super().__init__(); self.model = Inner()

    up = torch.full((2, 3, 4), 7.0)
    save_file({"w": up.half()}, str(tmp_path / "L000.up.safetensors"))
    (tmp_path / "rows.json").write_text(json.dumps(
        {"spec": {}, "rows": [{"layer": 0, "matrix": "up", "slot": "L000.up"}], "expert_ratio": 0.5}))
    m = Model()
    load_compressed_into(m, str(tmp_path), verbose=False)
    assert torch.equal(m.model.layers[0].mlp.experts.gate_up_proj[:, 3:], up)
    assert torch.equal(m.model.layers[0].mlp.experts.gate_up_proj[:, :3], torch.zeros(2, 3, 4))
