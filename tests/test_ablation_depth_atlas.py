"""The depth chart must recover smooth drift and must refuse random drift.

An instrument that fits everything measures nothing, so both directions are
tested: a false positive here would manufacture support for the depth-axis
proposal out of noise.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from moe_optimizer.ablation.depth_atlas import (DepthAtlas, depth_coordinates,
                                               gauge_align, procrustes_rotation)

L, E, D_OUT, D_IN, R = 16, 8, 40, 56, 6


def _stacks(smooth: bool, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    qr = lambda d: torch.linalg.qr(torch.randn(d, R, generator=g, dtype=torch.float64))[0]
    U0, U1, V0, V1 = qr(D_OUT), qr(D_OUT), qr(D_IN), qr(D_IN)
    A0 = torch.randn(D_OUT, D_IN, generator=g, dtype=torch.float64)
    A1 = torch.randn(D_OUT, D_IN, generator=g, dtype=torch.float64)
    out = {}
    for l in range(L):
        t = l / (L - 1)
        if smooth:
            U = torch.linalg.qr((1 - t) * U0 + t * U1)[0]
            V = torch.linalg.qr((1 - t) * V0 + t * V1)[0]
            anc = (1 - t) * A0 + t * A1
        else:
            U, V = qr(D_OUT), qr(D_IN)
            anc = torch.randn(D_OUT, D_IN, generator=g, dtype=torch.float64)
        core = torch.randn(E, R, R, generator=g, dtype=torch.float64)
        out[l] = (anc.unsqueeze(0) + torch.einsum("or,ers,is->eoi", U, core, V)).float()
    return out


def _rel_err(stacks, code, **kw):
    orig = torch.stack([stacks[l].double() for l in sorted(stacks)])
    return float((orig - code.reconstruct(**kw)).norm() / orig.norm())


def test_procrustes_is_orthogonal_and_optimal():
    g = torch.Generator().manual_seed(0)
    a = torch.linalg.qr(torch.randn(20, 5, generator=g, dtype=torch.float64))[0]
    q = torch.linalg.qr(torch.randn(5, 5, generator=g, dtype=torch.float64))[0]
    r = procrustes_rotation(a @ q, a)
    torch.testing.assert_close(r @ r.T, torch.eye(5, dtype=torch.float64), atol=1e-10, rtol=0)
    torch.testing.assert_close(a @ q @ r, a, atol=1e-10, rtol=0)


def test_gauge_alignment_preserves_span():
    g = torch.Generator().manual_seed(1)
    bases = [torch.linalg.qr(torch.randn(20, 5, generator=g, dtype=torch.float64))[0]
             for _ in range(6)]
    aligned, rots = gauge_align(bases)
    for b, a, r in zip(bases, aligned, rots):
        torch.testing.assert_close(b @ r, a, atol=1e-12, rtol=0)
        torch.testing.assert_close(a.T @ a, torch.eye(5, dtype=torch.float64),
                                   atol=1e-10, rtol=0)


def test_depth_coordinates_span_unit_interval():
    z = depth_coordinates(list(range(10)))
    assert float(z.min()) == -1.0 and float(z.max()) == 1.0
    assert float(depth_coordinates([3])[0]) == 0.0


def test_recovers_smooth_drift():
    st = _stacks(smooth=True)
    code = DepthAtlas(rank=R, degree=4).fit(st)
    assert code.meta["chart_rel_resid_U"] < 0.05
    assert code.meta["chart_rel_resid_V"] < 0.05
    assert _rel_err(st, code) < 0.10


def test_refuses_random_drift():
    st = _stacks(smooth=False)
    code = DepthAtlas(rank=R, degree=4).fit(st)
    assert code.meta["chart_rel_resid_U"] > 0.4
    assert _rel_err(st, code) > 0.5


def test_rank_truncation_is_monotone():
    """Rank truncation is exactly nested, so error must not improve as rank drops."""
    st = _stacks(smooth=True)
    code = DepthAtlas(rank=R, degree=4).fit(st)
    errs = [_rel_err(st, code, rank=k) for k in (2, 4, 6)]
    assert errs[0] >= errs[1] >= errs[2] - 1e-12, errs


def test_byte_accounting_beats_dense():
    st = _stacks(smooth=True)
    code = DepthAtlas(rank=R, degree=4).fit(st)
    assert code.ratio() < 1.0
    assert abs(sum(code.component_bytes.values()) - code.nbytes) < 1e-6


def test_float32_gram_matches_float64_subspace():
    """--fast must not change the G0 verdict: same top subspace to ~1e-5.

    The planted spectrum has a clear gap after rank 6.  With a near-degenerate
    spectrum (i.i.d. random rows) the top-6 subspace is ill-defined and rotates
    between precisions, which would test the data rather than the code.
    """
    from moe_optimizer.geometry.subspace import subspace_affinity
    g = torch.Generator().manual_seed(3)
    basis = torch.linalg.qr(torch.randn(56, 6, generator=g, dtype=torch.float64))[0]
    ws = [torch.randn(40, 6, generator=g, dtype=torch.float64) * 10.0 @ basis.T
          + 0.1 * torch.randn(40, 56, generator=g, dtype=torch.float64)
          for _ in range(8)]

    def dict_(dtype):
        gram = torch.zeros(56, 56, dtype=dtype)
        for w in ws:
            w = w.to(dtype); gram += w.T @ w
        return torch.linalg.eigh(gram.to(torch.float64))[1].flip(1)[:, :6]

    assert abs(1.0 - subspace_affinity(dict_(torch.float64), dict_(torch.float32))) < 1e-5
