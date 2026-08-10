# IFStruct release_v1_0: client implementation

How the pipette client runs the `eval_ifstruct_release_v1_0` benchmark. For the
**methodology** (what it measures, the dataset, scoring rules, and reported
metrics) see the canonical article in pipette-mgmt:
[ifstruct-release_v1_0](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/ifstruct-release_v1_0.md).
This page covers only the client side: generation.

## Benchmark identity

- `parameter_eval_id`: `ifstruct`
- `parameter_dataset_name`: `release_v1_0`
- `parameter_max_tokens`: `8192`

The full 2000-task IFStruct set, served verbatim. Each prompt asks for JSON or
YAML satisfying an explicit schema and structural constraints; the client
forwards the prompt as a single user message and does not pre-validate the
output (the scorer does).

## Generation

This eval does **not** use greedy decoding. The client samples at
`temperature: 0.6`, assigned by `BenchmarkDefinition::eval_temperature()` keyed
on `parameter_eval_id` (`ifstruct` → `0.6`). No fixed seed is sent. See
[Eval Methodology § sampling](evals.md#measurement-protocol) for the per-runtime
plumbing and the **iOS caveat** (iOS cannot sample and runs this eval greedy).
`max_tokens` is `8192`.

## No repeats

Unlike ifbench/gpqa/math, this dataset has no `metadata.repeats` salting. Each
prompt is served and completed **once**, so the result is a single sampled draw
per prompt (expect small run-to-run variation). The client completes each id
like any other sample, appending to the
[resumable checkpoint](../pipette-cli/eval-checkpoint.md), and submits a flat `completions`
array.

## Code references

- Client temperature policy: `BenchmarkDefinition::eval_temperature` in
  [`crates/pipette-plan-types/src/benchmark/mod.rs`](../../crates/pipette-plan-types/src/benchmark/mod.rs).
- Shared eval run protocol: [evals.md](evals.md).
- Methodology, dataset, and scoring (canonical):
  [pipette-mgmt ifstruct-release_v1_0](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/ifstruct-release_v1_0.md).
- Scorer:
  [`pipette_scores.scoring.ifstruct`](https://github.com/Liquid4All/pipette-scores/blob/main/packages/pipette-scores/pipette_scores/scoring/ifstruct.py).
- Validator:
  [`ifstruct.validator`](https://github.com/Liquid4All/ifstruct/blob/4a35ac44a2e532554907eee51a38267782a9a742/ifstruct/validator.py).
