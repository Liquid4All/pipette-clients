# Apple Core AI runtime

Backend library: `pipette-coreai`, driven by the unified `pipette` CLI.

Runs **Apple Silicon (macOS 27+)** benchmarks via a bundled Swift sidecar that
drives Apple's Core AI framework (`LanguageBundle` + `EngineFactory`) over a
short-lived local HTTP server. Shared operator flow: [usage.md](usage.md).
Notation: [models-and-runtimes.md](models-and-runtimes.md).

## Differences from llama.cpp / MLX

| | Core AI | MLX | llama.cpp |
|---|---------|-----|-----------|
| Engine | Ships with macOS 27 (FoundationModels / Core AI) | Python venv + `mlx-lm` | Upstream release archive |
| Runtime artifact | The Swift sidecar (built from source on first run) | Python venv + locked `mlx-lm` | GGUF release archive |
| Install | `runtimes pull 'core-ai-macos-pipette://'` (marker; engine is OS-bundled) | `runtimes pull` (fetches a venv) | `runtimes pull` (fetches a release) |
| Models | `.aimodel` bundle (metadata.json + `*.aimodel/` + tokenizer/) | HF repo snapshot (quantized safetensors) | GGUF file(s) |
| Host | Apple Silicon, macOS 27+ only | Apple Silicon only | Cross-platform |

## Runtime

The Core AI engine ships with macOS 27, so there is nothing to fetch — the URI
is a marker with no keys, and the Swift sidecar is compiled from the crate's
bundled `swift/` package on first use (cached under `~/Library/Caches/pipette-coreai/`):

```bash
pipette runtimes pull --runtime 'core-ai-macos-pipette://'
pipette runtimes list
```

Requires `swift` on `PATH` for the first run (the sidecar build). Set
`PIPETTE_COREAI_SIDECAR=/path/to/binary` to use a prebuilt sidecar and skip the
build.

## Model

Core AI models are `.aimodel` bundles. The `core-ai://` scheme reuses the shared
directory-shaped grammar (like `mlx://`), where `prefix` selects a variant
subdirectory inside the repo:

```bash
--model 'core-ai://repo=mlboydaisuke/Qwen3.8-27B-CoreAI&prefix=gpu-pipelined/qwen3_8_27b_decode_int4lin'
# optional: &rev=…
pipette models pull --model 'core-ai://repo=…&prefix=…'   # optional pre-fetch
```

`pipette` materializes the HF snapshot into the shared `models/` store and the
Swift sidecar loads it with `LanguageBundle(at:)`. Gated repos:
`PIPETTE_HF_TOKEN`.

## Benchmarks

Supported: `prefill_throughput`, `decode_throughput`, `end_to_end_latency`,
`max_memory_usage`.

Not supported (yet): `eval`, `vl_throughput`.

Each run starts the sidecar on the materialized bundle dir, drives the task over
HTTP (deterministic token-count prompts, no chat template — the same contract as
`pipette-mlx`), then tears it down.

`--runtime-flags` is **refused** for Core AI, like MLX: the runtime has no flag
cell, so any non-empty object fails before the runtime is even fetched (an empty
`{}` is accepted).

## Setup sketch

```bash
pipette init
pipette runtimes pull --runtime 'core-ai-macos-pipette://'   # marker; builds the sidecar on first run
pipette benchmarks init-local   # optional
# first run ensures the runtime + fetches the model
pipette benchmarks run \
  --benchmark local/decode_throughput_512_100 \
  --model 'core-ai://repo=mlboydaisuke/Qwen3.8-27B-CoreAI&prefix=gpu-pipelined/qwen3_8_27b_decode_int4lin' \
  --runtime 'core-ai-macos-pipette://'
```
