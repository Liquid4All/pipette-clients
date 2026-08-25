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
| Runtime artifact | Swift sidecar, compiled against a pinned `john-rocky/coreai-models` (`0.2.2-zoo` fork) | Python venv + locked `mlx-lm` | GGUF release archive |
| Install | `runtimes pull` is a no-op (engine is OS-bundled); first run builds the sidecar | `runtimes pull` (fetches a venv) | `runtimes pull` (fetches a release) |
| Models | `.aimodel` bundle (`metadata.json` + `*.aimodel/` + tokenizer/) | HF repo snapshot | GGUF file(s) |
| Host | Apple Silicon, macOS 27+ only | Apple Silicon only | Cross-platform |

## Runtime

The Core AI *engine* ships with macOS 27. The URI names the Swift-package pin
the sidecar is compiled against: the `john-rocky/coreai-models` zoo fork of
Apple's `coreai-models` (`CoreAILM` product), pinned at `0.2.2-zoo`. See
`Package.swift` for why the fork is required (Apple upstream cannot yet chunk a
multi-token prefill into the S=1 decode bundles pipette benchmarks):

```bash
pipette runtimes pull --runtime 'core-ai-macos-pipette://version=0.2.2-zoo'
```

`runtimes pull` succeeds without writing a store entry. The sidecar is compiled
from the crate's bundled `swift/` package on first use (cached under
`~/Library/Caches/pipette-coreai/`). That build needs `swift` on `PATH`.

`PIPETTE_COREAI_SIDECAR=/path/to/binary` skips the Swift build and uses a
prebuilt sidecar. It is an optional override, not a requirement, once
`swift build --show-bin-path` can find the product.

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
pipette runtimes pull --runtime 'core-ai-macos-pipette://version=0.2.2-zoo'
pipette benchmarks init-local   # optional
pipette benchmarks run \
  --benchmark local/decode_throughput_512_100 \
  --model 'core-ai://repo=mlboydaisuke/Qwen3.8-27B-CoreAI&prefix=gpu-pipelined/qwen3_8_27b_decode_int4lin' \
  --runtime 'core-ai-macos-pipette://version=0.2.2-zoo'
```
