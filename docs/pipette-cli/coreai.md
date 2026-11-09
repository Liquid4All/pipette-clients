# Apple Core AI runtime

Backend library: `pipette-coreai`, driven by the unified `pipette` CLI.

The runtime runs benchmarks on **Apple Silicon with macOS 27 or later**. A
bundled Swift sidecar drives Apple's Core AI framework (`LanguageBundle` +
`EngineFactory`). The sidecar serves a short-lived local HTTP server. Shared
operator flow: [usage.md](usage.md).
Notation: [models-and-runtimes.md](models-and-runtimes.md). Specialization
cache: [coreai-specialization.md](../methodology/coreai-specialization.md).

## Differences from llama.cpp / MLX

| | Core AI | MLX | llama.cpp |
|---|---------|-----|-----------|
| Engine | Ships with macOS 27 (Core AI) | `mlx-lm` (`mlx-vlm` on vision-language rows) in the runtime venv | llama.cpp tools in the release archive |
| Runtime artifact | Swift sidecar, compiled against an immutable `apple/coreai-models` commit | Python venv with locked `mlx-lm` (and `mlx-vlm` on vision-language rows) | Upstream release archive |
| Install | `runtimes pull` builds the sidecar into the runtime store; `runtimes remove` deletes that entry | `runtimes pull` fetches a venv, or the first `benchmarks run` installs it | `runtimes pull` fetches a release, or the first `benchmarks run` installs it |
| Models | `.aimodel` bundle (`metadata.json` + `*.aimodel/` + tokenizer/) | Hugging Face (HF) repo snapshot in the shared `models/` store | GGUF file(s) in the shared `models/` store |
| Host | Apple Silicon, macOS 27+ only | Apple Silicon only | Cross-platform |

## Runtime

The Core AI *engine* ships with macOS 27. The URI names the
`apple/coreai-models` commit (`CoreAILM` product):
`27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60`. The build compiles the sidecar
against this commit. The commit contains
[Apple PR #227](https://github.com/apple/coreai-models/pull/227), which fixes
prefill with a sequence length of 1 and the logits allocation. The engine reads
chunking from the bundle descriptors.

The Swift stack also holds `swift-transformers` 1.2.0, `xgrammar` 0.2.2, and
`swift-jinja` 2.3.2. These versions match the lockfile of the Apple commit. All
four packages are part of the runtime identity.

```bash
pipette runtimes pull --runtime 'core-ai-macos-pipette://version=27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60'
```

`runtimes pull` builds the embedded Swift package into the runtime store and
publishes a manifest. The store key covers all four Swift packages and the
sidecar sources. The initial build needs a Swift toolchain (Xcode or the
Command Line Tools) and access to the package dependencies. Later runs reuse
the installed executable. The build finishes before dispatch and the readiness
check. `runtimes list` shows the manifest. `runtimes remove` removes that entry.
`runtimes list` and the worker capabilities skip entries from an older bundled
build. Storage garbage collection can remove them. After an upgrade that changes
the sidecar sources or the Swift stack, run `runtimes pull` again.

Pipette does not read or change `~/Library/Caches/pipette-coreai`. Delete this
directory if no older Pipette client uses it. This directory is separate from
Apple's model-specialization cache.

Pipette rejects a `version=` value or a JSON `--runtime` with a different Swift
stack. The sidecar is compiled against the bundled Swift stack only. A result
for any other Swift stack would name a runtime that did not produce it.

### Capability flags

A worker reports the Core AI flags when its runtime store contains a valid
Core AI manifest. Run `runtimes pull` on a new worker before the server assigns
Core AI jobs to it. Each flag names the bundled Swift stack:

| Flag | Pins |
|------|------|
| `runtime:core_ai` | any Core AI host |
| `runtime:core_ai:<coreai-models commit>` | the `coreai-models` commit |
| `runtime:core_ai:coreai-models=<commit>` | the `coreai-models` commit, named |
| `runtime:core_ai:swift-transformers=<version>` | the `swift-transformers` version |
| `runtime:core_ai:xgrammar=<version>` | the `xgrammar` version |
| `runtime:core_ai:swift-jinja=<version>` | the `swift-jinja` version |

A plan that requires all four package flags runs only on a client with that
Swift stack.

### Installation location

Core AI uses the standard workspace runtime store. Pipette ignores the
environment variables `PIPETTE_COREAI_SIDECAR`, `PIPETTE_COREAI_SWIFT`, and
`PIPETTE_COREAI_CACHE`. Pipette embeds the Swift sources, so the build runs on a
host without the source checkout. To change the sidecar, rebuild Pipette with
the changed sources.

## Model

```bash
--model 'core-ai://repo=mlboydaisuke/Qwen3.8-27B-CoreAI&prefix=gpu-pipelined/qwen3_8_27b_decode_int4lin'
pipette models pull --model 'core-ai://repo=…&prefix=…'
```

Pipette downloads the Hugging Face snapshot into the shared `models/` store.
The sidecar loads it with `LanguageBundle(at:)`. For a gated repository, set
`PIPETTE_HF_TOKEN`.

## Benchmarks

Supported: `prefill_throughput`, `decode_throughput`, `end_to_end_latency`,
`max_memory_usage`.

Pipette refuses every other benchmark type on Core AI.

Each run starts the sidecar on the bundle directory. The run then sends the task
over HTTP with deterministic token-count prompts and no chat template. Last, the
run stops the sidecar. Specialization occurs while the sidecar loads, before
the sidecar prints `PIPETTE_COREAI_READY`. That whole start, the load included
and on a warm cache too, does not count against `attempt_timeout_seconds`
([usage.md](usage.md#5c-the-attempt-limit)). See
[coreai-specialization.md](../methodology/coreai-specialization.md).

Core AI has no runtime flags. Pipette rejects a non-empty `--runtime-flags`
object. Pipette accepts an empty `{}`.

## Setup sketch

```bash
pipette init
pipette runtimes pull --runtime 'core-ai-macos-pipette://version=27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60'
pipette benchmarks init-local   # optional
pipette benchmarks run \
  --benchmark local/decode_throughput_512_100 \
  --model 'core-ai://repo=mlboydaisuke/Qwen3.8-27B-CoreAI&prefix=gpu-pipelined/qwen3_8_27b_decode_int4lin' \
  --runtime 'core-ai-macos-pipette://version=27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60'
```
