# GPQA Diamond 2026.06.1: client implementation

How the pipette client runs the `eval_gpqa_diamond_2026.06.1` benchmark. For the
**methodology** (what it measures, the dataset, scoring rules, and reported
metrics) see the canonical article in pipette-mgmt:
[gpqa_diamond-2026.06.1](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/gpqa_diamond-2026.06.1.md).
This page covers only the client side: generation and the repeat plumbing.

## Benchmark identity

- `parameter_eval_id`: `gpqa_diamond`
- `parameter_dataset_name`: `2026.06.1`
- `parameter_max_tokens`: `8192`

The served prompt already contains the shuffled A–D options and the
`Answer: A/B/C/D` instruction; the client forwards it verbatim and does not parse
or alter it.

## Generation

This eval does **not** use greedy decoding. The client samples at
`temperature: 0.6`, assigned by `BenchmarkDefinition::eval_temperature()` keyed
on `parameter_eval_id` (`gpqa_diamond` → `0.6`). No fixed seed is sent, so each
repeated attempt is an independent draw. See
[Eval Methodology § sampling](evals.md#measurement-protocol) for the per-runtime
plumbing and the **iOS caveat** (iOS cannot sample and runs this eval greedy,
which collapses the repeats to a single answer). `max_tokens` is `8192`.

## Repeats

The `2026.06.1` dataset sets `metadata.repeats = 5`, so the score server serves
the 198 questions as 990 salted `#k` ids. The client has **no repeat-specific
code**: it completes each `#k` id like any other sample (one completion per id,
appended to the [resumable checkpoint](../pipette-cli/eval-checkpoint.md) under that id) and
submits a flat `completions` array. An interrupted run resumes per id, drawing
fresh completions only for the `#k` ids not yet done.

## Code references

- Client temperature policy: `BenchmarkDefinition::eval_temperature` in
  [`crates/pipette-plan-types/src/benchmark/mod.rs`](../../crates/pipette-plan-types/src/benchmark/mod.rs).
- Shared eval run protocol: [evals.md](evals.md).
- Methodology, dataset, and scoring (canonical):
  [pipette-mgmt gpqa_diamond-2026.06.1](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/gpqa_diamond-2026.06.1.md).
- Scorer:
  [`pipette_scores.scoring.gpqa_diamond`](https://github.com/Liquid4All/pipette-scores/blob/main/packages/pipette-scores/pipette_scores/scoring/gpqa_diamond.py).
