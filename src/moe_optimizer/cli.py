"""Command line entry point.

    moeopt econ                             parameter-economics tables
    moeopt audit-depth  MODEL [options]     gate G0: depth smoothness
    moeopt sweep        MODEL [options]     matched-budget Pareto sweep
    moeopt calib        MODEL [options]     calibration statistics (F8)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .calib.stats import slot_stats


def _store(args):
    from .io.checkpoint import ExpertStore, resolve_model

    model = resolve_model(args.model, cache_dir=args.cache_dir,
                          allow_download=not args.no_download)
    print(f"model      : {model.arch.model_id}")
    print(f"path       : {model.path}")
    print(f"arch       : {model.arch.n_layers} layers, {model.arch.n_experts} experts, "
          f"top-{model.arch.top_k}, d_model={model.arch.d_model}, "
          f"d_expert={model.arch.d_expert}")
    print(f"moe layers : {len(model.arch.moe_layers)}")
    print(f"expert par : {model.arch.total_expert_params / 1e9:.2f}B")
    return ExpertStore(model)


def cmd_econ(args) -> int:
    from .param_economics import report

    report()
    return 0


def cmd_audit_depth(args) -> int:
    from .geometry.depth import depth_profile

    store = _store(args)
    layers = list(store.arch.moe_layers)[: args.max_layers] if args.max_layers else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for matrix in args.matrices:
        for side in args.sides:
            print(f"\n=== depth profile: {matrix} / {side} (rank {args.rank}) ===")
            prof = depth_profile(store, matrix=matrix, rank=args.rank, side=side,
                                 layers=layers,
                                 gram_dtype=torch.float32 if args.fast else torch.float64)
            curve = prof.decay_curve()
            verdict = prof.verdict()
            print(f"\n  affinity vs layer gap (mean cos^2 of principal angles):")
            for g in range(min(9, curve.numel())):
                bar = "#" * int(60 * float(curve[g]))
                print(f"    gap {g:>2}: {float(curve[g]):.4f} {bar}")
            print(f"\n  VERDICT: {verdict}")

            key = f"{matrix}_{side}"
            summary[key] = {"verdict": verdict, "decay_curve": curve.tolist(),
                            "layers": prof.layers, "rank": prof.rank}
            torch.save({"affinity": prof.affinity, "energy": prof.energy,
                        "layers": prof.layers},
                       out_dir / f"depth_{key}.pt")

    (out_dir / "depth_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/depth_summary.json")
    return 0


def cmd_calib(args) -> int:
    from .calib.run import run_calibration

    run_calibration(
        args.model, args.out, n_tokens=args.tokens, seq_len=args.seq_len, batch=args.batch,
        cache_dir=args.cache_dir, layers=args.layers or None, max_batches=args.max_batches,
        cpu_mem=args.cpu_mem or None, offload_dir=args.offload_dir or None,
    )
    return 0


def cmd_sweep(args) -> int:
    from .eval.sweep import pareto_front, sweep_slot, write_results
    from .types import Slot
    import moe_optimizer.methods  # noqa: F401

    store = _store(args)
    points = json.loads(Path(args.points).read_text()) if args.points else _default_points(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = torch.load(args.calib)["layers"] if args.calib else None
    if calib is not None:
        print(f"calibration : {args.calib} ({sum(v['n_tokens'] for v in calib.values()) // len(calib):,} tokens/layer)")

    all_results = []
    layers = list(store.arch.moe_layers)[: args.max_layers] if args.max_layers else \
        list(store.arch.moe_layers)
    for layer in layers:
        for matrix in args.matrices:
            slot = Slot(layer, matrix)
            print(f"\n=== {slot} ===")
            stack = store.stack(slot)
            stats = slot_stats(calib, layer, matrix) if calib is not None else None
            imp = stats["importance"] if stats else None
            all_results += sweep_slot(stack, str(slot), points, stats=stats, importance=imp)
            del stack

    write_results(all_results, str(out_dir / "sweep.jsonl"))
    print(f"\nPareto front ({len(all_results)} points):")
    metric = "rel_act" if calib is not None else "rel_fro"
    print(f"  (ranked by {metric})")
    for r in pareto_front(all_results, metric):
        print(f"  {r.method:<20} ratio={r.ratio:.4f}  {metric}={r.error[metric]:.5f}  {r.params}")
    print(f"\nwrote {out_dir}/sweep.jsonl")
    return 0


def _default_points(args) -> list[dict]:
    pts: list[dict] = []
    for r in args.ranks:
        pts.append({"name": "per_expert_svd", "rank": r, "whiten": False})
        pts.append({"name": "shared_base_delta", "rank": r, "whiten": False})
        pts.append({"name": "shared_basis", "rank": r, "n_basis": 8, "iters": 4,
                    "whiten": False})
        for k in args.communities:
            pts.append({"name": "local_atlas", "n_communities": k, "rank": r,
                        "coupling": "diag", "whiten": False})
            pts.append({"name": "local_atlas", "n_communities": k, "rank": r,
                        "coupling": "diag", "whiten": False, "expert_chart": True})
    return pts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="moeopt", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("econ", help="parameter economics tables").set_defaults(fn=cmd_econ)

    def common(sp):
        sp.add_argument("model")
        sp.add_argument("--cache-dir", default=".cache")
        sp.add_argument("--no-download", action="store_true")
        sp.add_argument("--out", default="runs/latest")
        sp.add_argument("--max-layers", type=int, default=0)
        sp.add_argument("--matrices", nargs="+", default=["gate", "up", "down"])

    a = sub.add_parser("audit-depth", help="gate G0: does the dictionary rotate smoothly?")
    common(a)
    a.add_argument("--rank", type=int, default=64)
    a.add_argument("--sides", nargs="+", default=["out", "in"])
    a.add_argument("--fast", action="store_true", help="float32 Gram accumulation (~4x faster)")
    a.set_defaults(fn=cmd_audit_depth)

    s = sub.add_parser("sweep", help="matched-budget Pareto sweep")
    common(s)
    s.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64, 128])
    s.add_argument("--communities", type=int, nargs="+", default=[1, 4, 8])
    s.add_argument("--points", default="", help="JSON file of explicit sweep points")
    s.add_argument("--calib", default="", help="calibration .pt from `calib/run.py`; enables whitening")
    s.set_defaults(fn=cmd_sweep)

    k = sub.add_parser("calib", help="collect calibration statistics from a forward pass")
    k.add_argument("model")
    k.add_argument("--cache-dir", default=".cache")
    k.add_argument("--out", required=True, help="output .pt")
    k.add_argument("--tokens", type=int, default=32768)
    k.add_argument("--seq-len", type=int, default=512)
    k.add_argument("--batch", type=int, default=4)
    k.add_argument("--layers", type=int, nargs="*", default=[])
    k.add_argument("--max-batches", type=int, default=None)
    k.add_argument("--cpu-mem", default="", help='e.g. "10GiB": enable disk offload for models > RAM')
    k.add_argument("--offload-dir", default="")
    k.set_defaults(fn=cmd_calib)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
