

def test_float32_gram_matches_float64_subspace():
    """--fast must not change the G0 verdict: same top subspace to ~1e-5."""
    from moe_optimizer.geometry.subspace import subspace_affinity
    g = torch.Generator().manual_seed(3)
    ws = [torch.randn(40, 56, generator=g) for _ in range(8)]

    def dict_(dtype):
        gram = torch.zeros(56, 56, dtype=dtype)
        for w in ws:
            w = w.to(dtype); gram += w.T @ w
        return torch.linalg.eigh(gram.to(torch.float64))[1].flip(1)[:, :6]

    assert abs(1.0 - subspace_affinity(dict_(torch.float64), dict_(torch.float32))) < 1e-5
