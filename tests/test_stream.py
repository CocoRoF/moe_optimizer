"""Policies must be exact on constructed inputs; the engine's shapes must hold."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from moe_optimizer.runtime.stream import TopKPolicy, MassRatioPolicy, ContributionPolicy


def test_topk_returns_raw_probs_unnormalised():
    p = torch.tensor([[0.5, 0.3, 0.1, 0.1]])
    i, w = TopKPolicy(2).select(p, 0)
    assert i.tolist() == [[0, 1]] and torch.allclose(w, torch.tensor([[0.5, 0.3]]))


def test_mass_ratio_skips_only_below_threshold():
    p = torch.tensor([[0.6, 0.3, 0.05, 0.05],     # tail share of last 2 = 0.1 -> skip if beta1.. 
                      [0.3, 0.3, 0.2, 0.2]])      # tail share of last 2 = 0.4 -> keep
    pol = MassRatioPolicy(k=4, beta={0: [0.2, 0.2, 0.2]})   # skip m if share(last m) < 0.2
    i, w = pol.select(p, 0)
    assert (w[0] > 0).tolist() == [True, True, False, False]
    assert (w[1] > 0).tolist() == [True, True, True, True]


def test_contribution_uses_scale_not_gate_weight():
    """A low-weight expert with a large calibrated output scale must be kept over
    a high-weight expert with a tiny scale."""
    p = torch.tensor([[0.4, 0.35, 0.15, 0.10]])
    scale = {0: torch.tensor([0.1, 0.1, 10.0, 0.1])}       # expert 2 has huge outputs
    pol = ContributionPolicy(k=4, scale=scale, tau={0: 0.5}, min_keep=1)
    i, w = pol.select(p, 0)
    kept = i[w > 0].tolist()
    assert 2 in kept, (i, w)                                 # kept despite 3rd-lowest weight
    assert len(kept) < 4                                     # and something was skipped


def test_contribution_tau_zero_keeps_all_and_min_keep_holds():
    p = torch.rand(5, 8); p = p / p.sum(1, keepdim=True)
    scale = {0: torch.ones(8)}
    _, w = ContributionPolicy(8, scale, {0: 0.0}).select(p, 0)
    assert (w > 0).all()
    _, w = ContributionPolicy(8, scale, {0: 0.999}, min_keep=2).select(p, 0)
    assert ((w > 0).sum(1) >= 2).all()


def test_paired_bootstrap_flags_a_real_difference_and_not_noise():
    import json, subprocess, sys, tempfile, random
    random.seed(1)
    base = [random.gauss(2.3, 0.2) for _ in range(16)]
    rows = [{"policy": "contribution@k'=4.0", "per_seq_nll": [x - 0.08 for x in base]},
            {"policy": "score_only@k'=4.0", "per_seq_nll": base},
            {"policy": "top4(static)", "per_seq_nll": [x + random.gauss(0, 0.01) for x in base]},
            {"policy": "contribution@k'=6.0", "per_seq_nll": [x + random.gauss(0, 0.05) for x in base]},
            {"policy": "score_only@k'=6.0", "per_seq_nll": base}]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f: json.dump(rows, f)
    out = subprocess.run([sys.executable, "scripts/paired_bootstrap.py", f.name], capture_output=True, text=True).stdout
    assert "k'~4.0: contribution vs score-only" in out and "SIGNIFICANT" in out.split("k'~4.0: contribution vs score-only")[1].split("\n")[0]
    assert "n.s." in out.split("k'~6.0: contribution vs score-only")[1].split("\n")[0]


def test_scripts_and_package_have_no_undefined_names():
    """The 8K confirmation and the Qwen3 sweep both crashed on a NameError that
    ast.parse cannot see.  pyflakes can."""
    import subprocess, sys, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    files = [str(p) for p in (root / "scripts").glob("*.py")] + [str(p) for p in (root / "src").rglob("*.py")]
    out = subprocess.run([sys.executable, "-m", "pyflakes", *files], capture_output=True, text=True).stdout
    bad = [l for l in out.splitlines() if "undefined name" in l]
    assert not bad, "\n".join(bad)


def test_renorm_policy_reduces_to_squared_share_without_renorm_and_penalises_dropping_under_renorm():
    from moe_optimizer.runtime.stream import ContributionRenormPolicy
    p = torch.tensor([[0.30, 0.25, 0.20, 0.15, 0.10]]); scale = {0: torch.ones(5)}
    # no renorm: squared-share; dropping the two smallest leaves err = (0.15^2+0.10^2)/sum = 0.1327
    tot = (p**2).sum(); err2 = (0.15**2 + 0.10**2) / tot
    _, w = ContributionRenormPolicy(5, scale, {0: float(err2) + 1e-6}, renorm=False).select(p, 0)
    assert (w > 0).sum() == 3
    # with renorm the same tau must keep MORE (amplification adds error)
    _, w2 = ContributionRenormPolicy(5, scale, {0: float(err2) + 1e-6}, renorm=True).select(p, 0)
    assert (w2 > 0).sum() >= 3 and (w2 > 0).sum() > (w > 0).sum() - 1
    # tau=0 keeps everything; tau=1 keeps min_keep
    assert (ContributionRenormPolicy(5, scale, {0: 0.0}).select(p, 0)[1] > 0).all()
    assert (ContributionRenormPolicy(5, scale, {0: 1.0}, min_keep=2).select(p, 0)[1] > 0).sum() == 2


def test_error_model_tau_calibration_hits_target_k():
    from moe_optimizer.runtime.calibrate import calibrate_taus_err, error_curve
    g = torch.Generator().manual_seed(5)
    tr = {0: torch.rand(400, 8, generator=g).sort(1, descending=True).values}
    ix = {0: torch.randint(0, 64, (400, 8), generator=g)}; sc = {0: torch.rand(64, generator=g) + 0.5}
    for renorm in (False, True):
        tau = calibrate_taus_err(tr, sc, ix, 5.0, renorm)
        err = error_curve(tr, sc, ix, renorm)[0]; ok = err <= tau[0]; ok[:, -1] = True
        mean_k = float((ok.float().argmax(1) + 1).float().mean())
        assert abs(mean_k - 5.0) < 0.15, (renorm, mean_k)


def test_oracle_keep_rule_matches_contribution_rule_on_known_norms():
    """With the engine's oracle share rule fed known norms, the keep-set equals
    ContributionPolicy's on scale=norm -- same criterion, exact signal."""
    import torch
    from moe_optimizer.runtime.stream import ContributionPolicy
    w = torch.tensor([[0.3, 0.25, 0.2, 0.15, 0.1]]); norms = torch.tensor([[1.0, 0.2, 3.0, 0.5, 0.1]])
    c = w * norms; c_sorted, order = c.sort(-1, descending=True)
    share = (c_sorted / c_sorted.sum()).cumsum(-1); thr = 1 - 0.3
    keep_sorted = torch.cat([torch.ones_like(share[:, :1], dtype=torch.bool), share[:, :-1] < thr], 1)
    oracle_keep = torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
    idx = torch.arange(5).unsqueeze(0); probs = torch.zeros(1, 5).scatter(1, idx, w)
    _, wk = ContributionPolicy(5, {0: norms[0]}, {0: 0.3}).select(probs, 0)
    assert torch.equal(wk > 0, oracle_keep)
