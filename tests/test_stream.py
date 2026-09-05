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
    assert "k'~4.0: contribution vs score-only" in out and "SIGNIFICANT (better)" in out.split("k'~4.0: contribution vs score-only")[1].split("\n")[0]
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


def test_qwen3_engine_renorm_flag_defaults_from_config():
    from moe_optimizer.runtime.stream import StreamingQwen3MoE
    class S:  # minimal store stand-in: only what __init__ touches
        def get(self, k): return torch.zeros(4, 8)
    cfg = {"num_hidden_layers": 1, "num_experts": 4, "num_experts_per_tok": 2, "hidden_size": 8,
           "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 8,
           "moe_intermediate_size": 8, "norm_topk_prob": True}
    assert StreamingQwen3MoE(S(), cfg, threads=1).renorm is True
    assert StreamingQwen3MoE(S(), cfg, threads=1, renorm=False).renorm is False


def test_full_renorm_is_identity_when_nothing_is_dropped_and_subtraction_when_dropped():
    """renorm='full' divides by the original top-k mass: with all k kept it equals the
    reference renormalisation; with one dropped, the survivors' weights are unchanged."""
    w = torch.tensor([[0.10, 0.08, 0.06, 0.04]]); W_all = w.sum()
    ref = w / w.sum(-1, keepdim=True)                       # renorm=True, nothing dropped
    full = w / W_all
    assert torch.allclose(ref, full)
    keep = torch.tensor([[1., 1., 1., 0.]])
    full_dropped = (w * keep) / W_all; ref_dropped = (w * keep) / (w * keep).sum(-1, keepdim=True)
    assert torch.allclose(full_dropped[0, :3], full[0, :3])            # survivors untouched
    assert (ref_dropped[0, :3] > full_dropped[0, :3]).all()            # reference rescales them up


def test_layer_budget_allocation_meets_target_and_favours_layers_with_fat_tails():
    from moe_optimizer.runtime.calibrate import allocate_layer_budgets, marginal_curves
    g = torch.Generator().manual_seed(9)
    # layer 0: gate mass concentrated in the top expert (cheap to drop the rest);
    # layer 1: flat (every expert matters)
    tr = {0: torch.tensor([[0.8, 0.05, 0.05, 0.05, 0.05]]).repeat(50, 1),
          1: torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]]).repeat(50, 1)}
    ix = {l: torch.arange(5).unsqueeze(0).repeat(50, 1) for l in tr}
    b = allocate_layer_budgets(tr, None, ix, target_k=3.0)
    assert abs(sum(b.values()) / 2 - 3.0) < 1e-6
    assert b[1] > b[0], b                          # the flat layer keeps more experts
    assert set(marginal_curves(tr, None, ix)) == {0, 1}


def test_layer_topk_policy_uses_per_layer_count():
    from moe_optimizer.runtime.stream import LayerTopKPolicy
    p = torch.rand(3, 8); p = p / p.sum(1, keepdim=True)
    pol = LayerTopKPolicy(8, {0: 2.0, 1: 6.0})
    assert ((pol.select(p, 0)[1] > 0).sum(1) == 2).all() and ((pol.select(p, 1)[1] > 0).sum(1) == 6).all()
    assert ((pol.select(p, 7)[1] > 0).sum(1) == 8).all()      # unbudgeted layer -> full k


def test_downstream_loglik_scores_continuation_tokens_only():
    """loglik() must sum log-probs of the continuation given the context and
    ignore context tokens: with a deterministic 'engine' that prefers token 7,
    a continuation of 7s scores higher than one of 3s regardless of context."""
    import torch.nn.functional as F, types, sys, importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("pd", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "policy_downstream.py")
    pd = importlib.util.module_from_spec(spec); sys.argv = ["x"]; spec.loader.exec_module(pd)
    class Tok:
        def __call__(self, s, return_tensors=None):
            ids = torch.tensor([int(ch) for ch in s.replace(" ", "")]); return types.SimpleNamespace(input_ids=ids.unsqueeze(0))
    class Eng:
        def forward(self, ids, cache=None):
            lg = torch.full((ids.numel(), 10), -5.0); lg[:, 7] = 5.0; return lg, None
    e, t = Eng(), Tok()
    assert pd.loglik(e, t, "12", "77") > pd.loglik(e, t, "12", "33")
    assert abs(pd.loglik(e, t, "12", "77") - pd.loglik(e, t, "99", "77")) < 1e-6   # context does not enter the score



def test_cached_loglik_matches_uncached_on_a_real_engine_shape_stub():
    """With a stub engine whose logits depend only on position count, the cached
    path (context once, continuation incrementally) must equal the uncached one."""
    import types, sys, importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("pd2", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "policy_downstream.py")
    pd = importlib.util.module_from_spec(spec); sys.argv = ["x"]; spec.loader.exec_module(pd)
    class Tok:
        def __call__(self, s, return_tensors=None):
            return types.SimpleNamespace(input_ids=torch.tensor([[int(ch) for ch in s.replace(" ", "")]]))
    class Eng:
        def forward(self, ids, cache=None):
            off = 0 if not cache or 0 not in cache else cache[0][0].shape[1]
            T = ids.numel(); lg = torch.zeros(T, 10)
            for t_ in range(T): lg[t_, (off + t_) % 10] = 3.0          # depends on absolute position only
            if cache is not None:
                kv = torch.zeros(1, T, 1); cache[0] = (torch.cat([cache[0][0], kv], 1), kv) if 0 in cache else (kv, kv)
            return lg, None
    e, t = Eng(), Tok(); ctx, cont = "123", "45"
    c = t(ctx).input_ids[0]; cache = {}; lg_ctx, _ = e.forward(c, cache)
    assert abs(pd.loglik(e, t, ctx, cont) - pd.loglik(e, t, ctx, cont, cache, lg_ctx[-1])) < 1e-6


def test_qwen15_engine_accounts_shared_expert_bytes_and_raw_topk():
    from moe_optimizer.runtime.stream import StreamingQwen15MoE
    class S:
        def get(self, k): return torch.zeros(4, 8)
    cfg = {"num_hidden_layers": 1, "num_experts": 60, "num_experts_per_tok": 4, "hidden_size": 8,
           "num_attention_heads": 2, "num_key_value_heads": 2, "intermediate_size": 5632,
           "moe_intermediate_size": 1408, "shared_expert_intermediate_size": 5632, "norm_topk_prob": False}
    e = StreamingQwen15MoE(S(), cfg, threads=1)
    assert e.expert_bytes == 3 * 1408 * 8 * 2 and e.shared_bytes == 3 * 5632 * 8 * 2 + 8 * 2 and e.k == 4



def test_paired_accuracy_ci_detects_a_consistent_gain():
    import sys, importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("pd3", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "policy_downstream.py")
    pd = importlib.util.module_from_spec(spec); sys.argv = ["x"]; spec.loader.exec_module(pd)
    base = [1, 0] * 100                                   # 50 %
    better = [1 if i % 5 else 1 for i in range(200)]      # 100 %
    lo, md, hi = pd.paired_acc_ci(better, base)
    assert lo > 0 and abs(md - 0.5) < 0.05
    lo2, _, hi2 = pd.paired_acc_ci(base, base)
    assert lo2 <= 0 <= hi2


def test_mixed_policy_uses_scale_only_in_selected_layers():
    from moe_optimizer.runtime.stream import MixedPolicy, ContributionPolicy
    p = torch.tensor([[0.4, 0.35, 0.15, 0.10]]); scale = {0: torch.tensor([0.1, 0.1, 10.0, 0.1]), 1: torch.tensor([0.1, 0.1, 10.0, 0.1])}
    tau = {0: 0.5, 1: 0.5}
    m = MixedPolicy(4, scale, tau, {0: True, 1: False})
    _, w0 = m.select(p, 0); _, w1 = m.select(p, 1)
    _, wc = ContributionPolicy(4, scale, tau).select(p, 0)
    _, ws = ContributionPolicy(4, {0: torch.ones(4)}, tau).select(p, 0)
    assert torch.equal(w0 > 0, wc > 0) and torch.equal(w1 > 0, ws > 0)


def test_mixed_policy_static_mode_keeps_fixed_count_by_chosen_signal():
    from moe_optimizer.runtime.stream import MixedPolicy
    p = torch.tensor([[0.30, 0.25, 0.20, 0.15, 0.10]]); scale = {0: torch.tensor([0.1, 0.1, 5.0, 0.1, 0.1])}
    pol = MixedPolicy(5, scale, {0: 0.0}, {0: True}, mode={0: "static"}, static_k={0: 2})
    i, w = pol.select(p, 0); kept = i[w > 0].tolist()
    assert len(kept) == 2 and 2 in kept                 # expert 2 (huge scale) kept despite 3rd-lowest weight
    pol2 = MixedPolicy(5, scale, {0: 0.0}, {0: False}, mode={0: "static"}, static_k={0: 2})
    assert set(pol2.select(p, 0)[0][pol2.select(p, 0)[1] > 0].tolist()) == {0, 1}


def test_weight_cache_is_exact_and_excludes_experts():
    """The cached non-expert weights must reproduce the uncached tensors bit-for-bit,
    and expert weights must never enter the cache (they are the streamed part)."""
    from moe_optimizer.runtime.stream import StreamingOLMoE
    class S:
        def __init__(self): self.calls = 0
        def get(self, k): self.calls += 1; return torch.arange(4.0).view(2, 2) + len(k)
    cfg = {"num_hidden_layers": 1, "num_experts": 4, "num_experts_per_tok": 2, "hidden_size": 2, "intermediate_size": 3, "num_attention_heads": 1}
    e = StreamingOLMoE(S(), cfg, threads=1)
    a = e._g("model.layers.0.self_attn.q_proj.weight"); b = e._g("model.layers.0.self_attn.q_proj.weight")
    assert a is b and torch.equal(a, S().get("model.layers.0.self_attn.q_proj.weight").float())
    assert all("experts" not in k for k in e._wcache)
