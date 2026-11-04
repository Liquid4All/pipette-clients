# Benchmark Methodology

These methodology notes describe how pipette measures performance. They address
readers who want to reproduce, compare, or audit the measurements, including
readers who are new to pipette internals. The article list is in the
[docs index](../README.md).

## Published results

Results are published at [pipette.liquid.ai](https://pipette.liquid.ai). Each
published number records the full configuration that produced it: the device,
the OS and its version, the chip, the runtime and its version, the model, the
weight source, the quantization, and the effective runtime flags (see
[Recording and Comparability](selection-policies.md#recording-and-comparability)).
Two numbers are comparable only when those recorded fields match.

The harness is open: the runner CLIs in this repository consume runtimes and
weights from their public upstream or community sources at pinned versions (see
[Selection policies](selection-policies.md)), so a published result is traceable
to the public components that produced it.

**Independence.** Liquid both publishes these benchmarks and ships models that
appear in them. Readers can check every number:

- The runner is this open-source harness.
- The runtimes and weights are public artifacts at pinned versions.
- Every published result records the exact model, quantization, runtime,
  version, and flags that it ran under.

A third party can take the recorded selection of a published result, run the
same CLI against the same public artifacts, and reproduce the result. The same
selection rules apply to our own models and to the models of other vendors.
Published iOS results come from the private-API build of this app, which anyone
can make from this repository (see
[iOS results and the private-API build](#ios-results-and-the-private-api-build)).

### iOS results and the private-API build

Published iOS results come from a build of this app with the private-API flags
on: `PIPETTE_PRIVATE_THERMAL` and `PIPETTE_PRIVATE_MEMORY`. The flags give two
readings that the public iOS API does not expose:

- The exact die temperature of the SoC (system on a chip). The readiness gate
  waits on it before each measured repetition
  ([System Readiness Control](end-to-end-latency.md#system-readiness-control)).
- The kernel's footprint peak over a measured bracket. The iOS peak-memory
  figure uses it
  ([Peak memory: iOS](peak-memory-ios.md#private-api-and-public-builds)).

App Review rejects a build that uses private APIs. The private-API build
therefore runs only side-loaded on registered development devices, and it is
not on the App Store or TestFlight. The distributable build has both flags
off: its gate reads `ProcessInfo.thermalState` and a calibrated IMU estimate,
and its peak memory is the sampled `phys_footprint` alone.

Anyone can make the private-API build from this repository: run
`ios/build.sh` with `PIPETTE_PRIVATE_THERMAL=1`. The app reports which flags a
build has (`headlessrun version`), and the runtime descriptor of each result
records them (`private_thermal`, `private_memory`). The build steps and the
rules for these flags are in
[Private-API builds](../pipette-ios/private-thermal-release-build.md).

## Pipeline overview

The articles refer to three components by role:

- **Clients** (this repository): run the benchmarks on the device or host and
  submit raw results (elapsed times, token counts, peak bytes, eval
  completions).
- **Management service** (pipette-mgmt): stores each run with its
  device, model and runtime context and derives the reported metrics. The
  formulas are in pipette-mgmt
  [docs/benchmarks.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md).
- **Score service** (`pipette-scores`): scores eval completions. It receives
  the completions, how each one ended, whether the job asked for thinking, and,
  for a vision-language (VLM) or speech-recognition (ASR) eval, the prompt
  profile, which names the model family. It never receives the model
  identifier or the device. The contract is in pipette-mgmt
  [docs/scoring-service.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/scoring-service.md).

## How an article is organized

Most articles use the same order:

1. The measurement concept, in plain language.
2. The benchmark IDs, the reported fields, and the implementation names.
3. The main sources of bias, then the controls for them.
4. The general measurement protocol, then the details for each runtime.
5. The platform and code references, at the end.

## Shared goals

The non-vision-language performance benchmarks share these goals:

- Use explicit benchmark definitions for token counts and result fields.
- Use exact-token inputs or benchmark-native token-count controls.
- Keep warmup and readiness waits outside reported measurements.
- Reject runs with token-count mismatches before reporting ambiguous data.
- Report enough per-run output detail to audit what was actually measured.
