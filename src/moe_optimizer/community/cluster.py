"""Grouping experts into functional communities.

Which similarity signal to use is itself a research question (report section 24),
so all of them are exposed and selected by name rather than hard-coded:

``weight``          raw weight cosine -- the naive choice, and the one that
                    permutation symmetry makes least trustworthy
``aligned_weight``  cosine after neuron permutation alignment
``coactivation``    router NPMI from calibration traces (needs stats)
``output``          expert output cosine on calibration tokens (needs stats)

The report's position, following LorExperts, is that functional signals should
dominate.  That is a hypothesis; this module makes it cheap to falsify by
swapping one config field.
"""

from __future__ import annotations

import torch

from ..registry import CLUSTERERS


def _normalise_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def weight_affinity(stack: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between vectorised experts, (E, E)."""
    flat = _normalise_rows(stack.flatten(1).to(torch.float64))
    return (flat @ flat.T).clamp(-1, 1)


def coactivation_npmi(coactivation: torch.Tensor, counts: torch.Tensor,
                      n_tokens: int) -> torch.Tensor:
    """Normalised pointwise mutual information of joint expert selection.

    NPMI rather than raw co-occurrence because expert usage is heavily skewed:
    two hot experts co-occur often merely by being hot, and raw counts would
    cluster by popularity rather than by function.
    """
    n = max(int(n_tokens), 1)
    p = (counts.to(torch.float64) / n).clamp(1e-12, 1.0)
    pij = (coactivation.to(torch.float64) / n).clamp_min(1e-12)
    pmi = torch.log(pij / (p.unsqueeze(1) * p.unsqueeze(0)))
    npmi = pmi / (-torch.log(pij))
    return torch.nan_to_num(npmi, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)


def expert_affinity(
    stack: torch.Tensor, signal: str = "weight", stats: dict | None = None
) -> torch.Tensor:
    if signal == "weight":
        return weight_affinity(stack)
    if signal == "coactivation":
        if not stats or "coactivation" not in stats:
            raise ValueError("coactivation affinity needs calibration stats")
        return coactivation_npmi(
            torch.as_tensor(stats["coactivation"]),
            torch.as_tensor(stats["counts"]),
            stats.get("n_tokens", 1),
        )
    if signal == "output":
        if not stats or "expert_outputs" not in stats:
            raise ValueError("output affinity needs calibration stats")
        o = _normalise_rows(torch.as_tensor(stats["expert_outputs"]).flatten(1).double())
        return (o @ o.T).clamp(-1, 1)
    raise ValueError(f"unknown affinity signal {signal!r}")


@CLUSTERERS.register("spectral")
def spectral_clusters(affinity: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """Normalised-cut spectral clustering on a dense affinity matrix.

    Small E (8-384) means we can afford the exact dense eigendecomposition; no
    approximation is warranted at this scale.
    """
    a = affinity.to(torch.float64).clamp_min(0)
    a.fill_diagonal_(0)
    deg = a.sum(1).clamp_min(1e-12)
    dinv = deg.rsqrt()
    lap = torch.eye(a.shape[0], dtype=torch.float64) - dinv.unsqueeze(1) * a * dinv.unsqueeze(0)
    _, evecs = torch.linalg.eigh(lap)
    emb = _normalise_rows(evecs[:, :k])
    return _kmeans(emb, k, seed)


@CLUSTERERS.register("agglomerative")
def agglomerative_clusters(affinity: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """Average-linkage agglomerative clustering on 1 - affinity."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    d = (1.0 - affinity.to(torch.float64)).clamp_min(0)
    d.fill_diagonal_(0)
    d = 0.5 * (d + d.T)
    z = linkage(squareform(d.numpy(), checks=False), method="average")
    lab = fcluster(z, t=k, criterion="maxclust") - 1
    return torch.from_numpy(lab).long()


@CLUSTERERS.register("uniform")
def uniform_clusters(affinity: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """Contiguous equal-size blocks by expert index -- a deliberate null model.

    Any gain from functional clustering must be measured against this.  If a
    method does no better than arbitrary grouping, its clustering step is
    decoration.
    """
    e = affinity.shape[0]
    return (torch.arange(e) * k // e).long()


def _kmeans(x: torch.Tensor, k: int, seed: int = 0, iters: int = 100) -> torch.Tensor:
    """k-means++ initialisation followed by Lloyd iterations."""
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    k = min(k, n)
    centres = x[torch.randint(n, (1,), generator=g)]
    for _ in range(1, k):
        d2 = torch.cdist(x, centres).pow(2).min(1).values.clamp_min(0)
        if float(d2.sum()) <= 0:
            centres = torch.cat([centres, x[torch.randint(n, (1,), generator=g)]])
            continue
        idx = torch.multinomial(d2 / d2.sum(), 1, generator=g)
        centres = torch.cat([centres, x[idx]])

    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        new = torch.cdist(x, centres).argmin(1)
        if torch.equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centres[c] = x[m].mean(0)
    return labels


def cluster_experts(
    stack: torch.Tensor, k: int, signal: str = "weight",
    algorithm: str = "spectral", stats: dict | None = None, seed: int = 0,
) -> torch.Tensor:
    """Community label per expert, shape (E,)."""
    if k <= 1:
        return torch.zeros(stack.shape[0], dtype=torch.long)
    aff = expert_affinity(stack, signal=signal, stats=stats)
    return CLUSTERERS.get(algorithm)(aff, k, seed)
