# Raw results

Every table in `docs/FINDINGS.md` and the README is generated from the files here.
`runs/` (git-ignored) is the working directory; these are the copies that ship.

| file | experiment | produces |
|---|---|---|
| `olmoe/policy_sweep_olmoe.json` | E2 confirmation, 8,192 test tokens, k′ 5 and 4, all policies, per-sequence NLL | F20, F23 tables; paired bootstrap |
| `olmoe/decode_bench_olmoe.json` | E3 batch-1 decode, 32 prefill + 64 steps | F16 |
| `olmoe/policy_tail_olmoe.json` | E4 worst-domain at k′=5 and 4 (`e4_tail_k*.log` hold both) | F19 |
| `olmoe/policy_oracle_olmoe.json` | F24 oracle, 4,096 tokens, k′≈5 | F24 |
| `qwen3/policy_sweep_qwen3.json` | Qwen3 E2, 4,096 test tokens, k′=5, 4,096-token calibration | F22 |
| `qwen3/policy_oracle_qwen3.json` | F24 oracle on Qwen3 | F24 |
| `calib/policy_calib_olmoe_2048tok.pt` | E1: per-layer sorted top-k traces, expert indices, output scales, median betas | all OLMoE policies |
| `calib/policy_calib_qwen3_4096tok.pt` | same for Qwen3 (the 1,024-token version that produced F21 was overwritten; F21 is superseded by F22) | all Qwen3 policies |
| `*/e*_*.log`, `*/f24_*.log` | console logs of the runs above, warnings stripped | provenance |
| `olmoe/policy_sweep_olmoe_layerbudget.json`, `qwen3/policy_sweep_qwen3_layerbudget.json` | E5: matched-k′ sweeps including `layer_topk(static)`, `score_only+layerbudget`, `contribution+layerbudget` | F27 |
| `olmoe/policy_downstream_olmoe.json` | E6: HellaSwag / ARC-Easy / PIQA accuracy at k′=5, per-example hits for paired CIs | F28 |
| `qwen15/` | E7: third model (Qwen1.5-MoE-A2.7B, top-4 of 60, shared expert) — validation, calibration, sweep at k′ 3 and 2.5 | F29 |
| `qwen3/policy_sweep_qwen3_norenorm.json`, `qwen3/policy_sweep_qwen3_renormfull.json` | F25 / F25b counterfactuals | §12 |
| `*/depth_summary_g0.json` | compression-era gate G0 (F2/F11b) | F2 |

Reproduce any of them with `scripts/reproduce.sh` (see its header for the order and runtimes).
