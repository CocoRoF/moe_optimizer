# Sweep point files

Pass with `moeopt sweep MODEL --points configs/<file>.json`.

Each entry is `{"name": <registered compressor>, ...constructor kwargs}`.
`moeopt sweep` without `--points` builds a default grid from `--ranks` and
`--communities`.

## `sweep_baselines.json`

The comparison set for the matched-budget Pareto table. Two pairings in it are
deliberate and should be kept whenever the file is edited:

- **`clusterer: "uniform"` alongside `clusterer: "spectral"` at every community
  count.** `uniform` groups experts into arbitrary contiguous blocks and is the
  null model. Any quality attributed to functional clustering has to beat it.
- **`expert_chart: true` against the otherwise identical entry without it.**
  The pair isolates what the polynomial chart on the expert coordinate actually
  does. Expect the byte difference to be under 0.1% — that is finding F1 in
  `docs/FINDINGS.md`, and this pair is how it gets re-measured on real weights
  rather than assumed.
