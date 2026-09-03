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
