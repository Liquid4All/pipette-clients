# Benchmark Methodology

These methodology notes describe how pipette measures performance and what each
reported metric means. They address readers who want to reproduce, compare, or
audit the measurements, including readers who are new to pipette internals.

## Published results

Results are published at [pipette.liquid.ai](https://pipette.liquid.ai). Every
published number carries the full configuration it was produced under (device,
OS and version, chip, runtime and version, model, and quantization) recorded
with the result (see
[Recording and Comparability](selection-policies.md#recording-and-comparability)).
Two numbers are comparable only when those recorded fields match.

The harness is open: the runner CLIs in this repository consume runtimes and
weights from their public upstream or community sources at pinned versions (see
[Selection policies](selection-policies.md)), so a published result is traceable
to the public components that produced it.

**Independence.** Liquid both publishes these benchmarks and ships models that
appear in them. We do not ask readers to take the numbers on trust: the runner
is this open-source harness, the runtimes and weights are public artifacts at
pinned versions, and every published result records the exact model,
quantization, runtime, version, and flags it ran under. A third party can take a
published result's recorded selection, run the same CLI against the same public
artifacts, and reproduce it: for our own models and competitors' alike, which
are benchmarked under the identical selection rules.

## Pipeline overview

The articles refer to three components by role:

- **Clients** (this repository): run the benchmarks on the target device or
  host and emit raw result payloads (elapsed times, token counts, peak bytes,
  eval completions).
- **Management service**: the server the clients submit results to. It stores
  each run with its full device/model/runtime context and derives the
  presentation metrics that the clients do not store directly, such as the
  tokens-per-second rate from `prefill_time_ms` and the `ttft` surfaced on
  end-to-end latency rows.
- **Score service** (`pipette-scores`, separate repository): scores eval
  completions. It is model-independent: it receives only the completions and
  never sees the model or device.

A run therefore flows clients → management service (store + derive) and, for
evals, completions → score service (score) → management service.

Each article follows the same style:

- Start with the measurement concept in plain language.
- Introduce benchmark IDs, reported fields, and implementation names after the
  concept is clear.
- Explain the main sources of bias before describing the control mechanism.
- Present the general measurement protocol before runtime-specific details.
- Use short noun-phrase headers instead of questions.
- Keep platform and implementation references at the end of the article.

This section covers non-vision-language performance benchmarks:

- [End-to-end latency](end-to-end-latency.md)
- [Prefill throughput](prefill-throughput.md)
- [Decode throughput](decode-throughput.md)
- [Peak memory usage](peak-memory-usage.md)

This section also covers quality eval benchmarks:

- [Eval methodology](evals.md)
- [IFBench 2026.06.1](ifbench-2026.06.1.md): client generation + repeat plumbing for the instruction-following eval (methodology in pipette-mgmt)
- [IFStruct release_v1_0](ifstruct-release_v1_0.md): client generation for the structured-output eval (methodology in pipette-mgmt)
- [GPQA Diamond 2026.06.1](gpqa_diamond-2026.06.1.md): client generation + repeat plumbing for the science MCQ eval (methodology in pipette-mgmt)
- [MATH-500 2026.06.1](math_500-2026.06.1.md): client generation + repeat plumbing for the competition-math eval (methodology in pipette-mgmt)

Model, weights, and runtime selection (the fairness policy that applies across
all of these benchmarks) is described separately:

- [Selection policies](selection-policies.md)

The device environment (thermal state, power, and cooling) that every timing
benchmark is measured under, and the per-platform specifics, are described
separately:

- [Device conditions](device-conditions.md)
- [MacBook Neo thermal behavior](macbook-neo-thermal-behavior.md): the detailed
  characterization behind the macOS row: why the thermal-pressure enum is a
  timer rather than a temperature signal, what the die-temperature noise floor
  rules out, and what the external cooler does and does not change
- [MacBook Pro (M5 Max) thermal behavior](macbook-m5-thermal-behavior.md): why
  the same enum lets a batch heat from 35 °C to 72 °C without leaving `nominal`,
  what that cost the measurements, and the die-temperature criterion added for it

`vl_throughput` has image-token and multimodal-projector concerns, so it should
be documented separately from the text-only performance benchmarks.

The shared goals across the non-vision-language performance benchmarks are:

- Use explicit benchmark definitions for token counts and result fields.
- Use exact-token inputs or benchmark-native token-count controls.
- Keep warmup and readiness waits outside reported measurements.
- Reject runs with token-count mismatches before reporting ambiguous data.
- Report enough per-run output detail to audit what was actually measured.
