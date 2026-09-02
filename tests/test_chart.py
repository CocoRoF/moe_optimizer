"""The chart machinery must be exactly orthogonal and exactly nested."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from moe_optimizer.factorize.chart import (fit_chart, legendre_vandermonde, multi_indices,
                                           multivariate_vandermonde, truncate,
                                           weighted_orthogonal_basis)


def test_legendre_recurrence_matches_closed_form():
    z = torch.linspace(-1, 1, 33, dtype=torch.float64)
    V = legendre_vandermonde(z, 4)
    torch.testing.assert_close(V[:, 0], torch.ones_like(z))
    torch.testing.assert_close(V[:, 1], z)
    torch.testing.assert_close(V[:, 2], 0.5 * (3 * z**2 - 1))
    torch.testing.assert_close(V[:, 3], 0.5 * (5 * z**3 - 3 * z))
    torch.testing.assert_close(V[:, 4], (35 * z**4 - 30 * z**2 + 3) / 8)


def test_multi_index_count_matches_binomial():
    from math import comb
    for q in (1, 2, 3, 4):
        for p in (1, 2, 3, 4):
            assert len(multi_indices(q, p)) == comb(q + p, p)


def test_basis_is_orthonormal_under_its_weight_measure():
    for weights in (None, torch.rand(40, dtype=torch.float64) + 0.05):
        b = weighted_orthogonal_basis(torch.linspace(-1, 1, 40), 5, weights)
        assert b.orthogonality_error() < 1e-10


def test_chart_fits_smooth_and_rejects_noise():
    z = torch.linspace(-1, 1, 48, dtype=torch.float64)
    b = weighted_orthogonal_basis(z, 6)
    smooth = torch.stack([torch.sin(1.3 * z + k) for k in range(8)], 1)
    _, fit = fit_chart(smooth, b)
    assert float((smooth - fit).norm() / smooth.norm()) < 1e-3

    noise = torch.randn(48, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    _, fit_n = fit_chart(noise, b)
    assert float((noise - fit_n).norm() / noise.norm()) > 0.7


def test_truncation_is_nested_and_optimal():
    """Dropping high-degree rows must equal refitting at the lower degree.

    This is the property the progressive-code claim rests on: one artifact must
    yield the optimal lower-degree approximation with no refit.
    """
    z = torch.linspace(-1, 1, 30, dtype=torch.float64)
    y = torch.stack([torch.exp(-((z - 0.3 * k) ** 2)) for k in range(5)], 1)

    hi = weighted_orthogonal_basis(z, 6)
    Theta_hi, _ = fit_chart(y, hi)
    truncated_fit = hi.Psi[:, : len(truncate(Theta_hi, hi, 3))] @ truncate(Theta_hi, hi, 3)

    lo = weighted_orthogonal_basis(z, 3)
    Theta_lo, direct_fit = fit_chart(y, lo)
    torch.testing.assert_close(truncated_fit, direct_fit, atol=1e-9, rtol=1e-7)


def test_multivariate_shape():
    z = torch.rand(20, 3, dtype=torch.float64) * 2 - 1
    assert multivariate_vandermonde(z, 3).shape == (20, 20)  # C(3+3,3) = 20
