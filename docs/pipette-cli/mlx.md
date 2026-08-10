# MLX runtime

Backend library: `pipette-mlx`, driven by the unified `pipette` CLI.

Runs **Apple Silicon** benchmarks via a uv-managed `mlx-lm` venv and a short-lived
local HTTP server. Shared operator flow: [usage.md](usage.md). Notation:
[models-and-runtimes.md](models-and-runtimes.md).

## Differences from llama.cpp

| | MLX | llama.cpp |
|---|-----|-----------|
| Runtime artifact | Python venv + locked `mlx-lm` | Upstream release archive |
| Install | `runtimes pull` or auto-fetch on first `benchmarks run` (shared store) | `runtimes pull` or auto-fetch |
| Models | HF repo snapshot in shared `models/` | GGUF file(s) in shared `models/` |
| Host | Apple Silicon only | Cross-platform |

## Runtime

MLX installs through the same shared runtime store as llama.cpp / UV:

```bash
pipette runtimes catalog mlx_macos_pipette
#  REF                                 | FLAVOR
# -------------------------------------+--------------
#  mlx-macos-pipette://version=0.31.3  | macos-arm64
# `flavor` defaults to macos-arm64, so the listed ref is complete as printed.

pipette runtimes pull --runtime 'mlx-macos-pipette://version=0.31.3&flavor=macos-arm64'
pipette runtimes list
```

```bash
pipette benchmarks run \
  --benchmark local/prefill_throughput_smoke \
  --model 'mlx://repo=mlx-community/LFM2-350M-4bit' \
  --runtime 'mlx-macos-pipette://version=0.31.3&flavor=macos-arm64'
```

Requires `uv` on `PATH`. The relocatable venv lands under
`.pipette/runtimes/<key>/blobs/venv/`. Catalog pins also auto-install on the
first `benchmarks run` when not already pulled. Custom (non-catalog) requirement
sets need an explicit `runtimes pull` (or a JSON `--runtime` with a full source).

## Model

```bash
--model 'mlx://repo=mlx-community/LFM2-350M-4bit'
# optional: &prefix=… &rev=…
pipette models pull --model 'mlx://repo=…'   # optional pre-fetch
```

`pipette` materializes the HF snapshot into the shared `models/` store and
starts the server with that **local directory** (not an ambient
`~/.cache/huggingface` path). Gated repos: `PIPETTE_HF_TOKEN`.

## Benchmarks

Supported: `prefill_throughput`, `decode_throughput`, `end_to_end_latency`,
`max_memory_usage`, `eval`.

Not supported: `vl_throughput`.

Each run starts an `mlx-lm` HTTP server on the materialized model dir, drives
the task over HTTP, then tears the server down.

`--runtime-flags` is **refused** for MLX, not ignored: the `mlx_macos_pipette`
runtime has no flag cell at all, so any non-empty object fails before the runtime
is even fetched (an empty `{}` means "no flags" and is accepted).

```text
no runtime flags defined for <benchmark> × MlxMacosPipette × Mlx
```

`--model-enable-thinking` / `--model-flags` do apply on eval, and
`--http-timeout-seconds` sets the SSE idle deadline there (default 1800s).

## Eval resume

Eval checkpoints each completed sample. See
[eval-checkpoint.md](eval-checkpoint.md).

## Setup sketch

```bash
pipette init
pipette auth register --organization <org> --contact-email <email> \
  --client-details "<what this box is>"   # only needed to submit results
pipette runtimes catalog mlx_macos_pipette
pipette runtimes pull --runtime 'mlx-macos-pipette://version=0.31.3&flavor=macos-arm64'   # optional warm-up
pipette benchmarks init-local   # optional
# first run ensures the runtime + fetches the model
```
