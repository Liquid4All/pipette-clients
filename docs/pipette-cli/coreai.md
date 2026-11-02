# Apple Core AI runtime

Backend library: `pipette-coreai`, driven by the unified `pipette` CLI.

Runs **Apple Silicon (macOS 27+)** benchmarks via a bundled Swift sidecar that
drives Apple's Core AI framework (`LanguageBundle` + `EngineFactory`) over a
short-lived local HTTP server. Shared operator flow: [usage.md](usage.md).
Notation: [models-and-runtimes.md](models-and-runtimes.md). Specialization
cache: [coreai-specialization.md](../methodology/coreai-specialization.md).

## Differences from llama.cpp / MLX

| | Core AI | MLX | llama.cpp |
|---|---------|-----|-----------|
| Engine | Ships with macOS 27 (Core AI) | Python venv + `mlx-lm` | Upstream release archive |
| Runtime artifact | Swift sidecar, compiled against an immutable `apple/coreai-models` commit | Python venv + locked `mlx-lm` | GGUF release archive |
| Install | `runtimes pull` builds the pin-keyed sidecar; `runtimes remove` clears it | `runtimes pull` (fetches a venv) | `runtimes pull` (fetches a release) |
| Models | `.aimodel` bundle (`metadata.json` + `*.aimodel/` + tokenizer/) | HF repo snapshot | GGUF file(s) |
| Host | Apple Silicon, macOS 27+ only | Apple Silicon only | Cross-platform |

## Runtime

The Core AI *engine* ships with macOS 27. The URI names the Swift-package pin
the sidecar is compiled against: official `apple/coreai-models` (`CoreAILM`
product), pinned at `27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60`. This commit
contains [Apple PR #227](https://github.com/apple/coreai-models/pull/227), which
fixes S=1 prefill and logits allocation. The sidecar no longer sets
`COREAI_CHUNK_THRESHOLD`; the engine uses the bundle descriptors.

The resolved stack uses `swift-transformers` 1.2.0, `xgrammar` 0.2.2, and
`swift-jinja` 2.3.2, matching the Apple commit's lockfile. All four pins are
part of the runtime identity; this changes the cache key from the zoo stack.

```bash
pipette runtimes pull --runtime 'core-ai-macos-pipette://version=27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60'
```

`runtimes pull` succeeds without writing a store entry, and **builds** the
sidecar (or reuses the cache, keyed on the full bundled Swift stack, at
`~/Library/Caches/pipette-coreai/<stack-key>/`). That build needs `swift` on
`PATH`. `runtimes remove` deletes the cache. A `version=` (URI) or a JSON
`--runtime` whose Swift stack differs from the bundled pin is rejected: the
sidecar is compiled against that stack only, so any other pin would record a
number the running binary did not produce.

### Environment overrides

- `PIPETTE_COREAI_SIDECAR=/path/to/binary`: skip the Swift build and use a
  prebuilt sidecar. Optional, not a requirement.
- `PIPETTE_COREAI_SWIFT=/path/to/swift`: point at the crate's `swift/`
  package directory when the `pipette` binary was shipped away from its
  build tree (so `CARGO_MANIFEST_DIR` no longer resolves).
- `PIPETTE_COREAI_CACHE=/path`: override the cache root
  (`~/Library/Caches/pipette-coreai` by default).

## Model

```bash
--model 'core-ai://repo=mlboydaisuke/Qwen3.8-27B-CoreAI&prefix=gpu-pipelined/qwen3_8_27b_decode_int4lin'
pipette models pull --model 'core-ai://repo=…&prefix=…'
```

`pipette` materializes the HF snapshot into the shared `models/` store and the
sidecar loads it with `LanguageBundle(at:)`. Gated repos: `PIPETTE_HF_TOKEN`.

## Benchmarks

Supported: `prefill_throughput`, `decode_throughput`, `end_to_end_latency`,
`max_memory_usage`.

Not supported (yet): `eval`, `vl_throughput`.

Each run starts the sidecar on the materialized bundle dir, drives the task over
HTTP (deterministic token-count prompts, no chat template), then tears it down.
Specialization happens during sidecar load, before READY. See
[coreai-specialization.md](../methodology/coreai-specialization.md).

`--runtime-flags` is **refused** for Core AI: the runtime has no flag cell, so
any non-empty object fails (an empty `{}` is accepted).

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
