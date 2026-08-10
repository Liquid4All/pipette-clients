# llama.cpp runtime

Backend library: `pipette-llamacpp`, driven by the unified `pipette` CLI.

Rust owns control plane (workspace, auth, stores, measurement). Upstream
**llama.cpp release binaries** do the inference:

| Tool | Benchmark types |
|------|-----------------|
| `llama-bench` | `prefill_throughput`, `decode_throughput`, `max_memory_usage` |
| `llama-server` (HTTP) | `end_to_end_latency`, `eval`, `vl_throughput` |

Shared operator flow (init, auth, sync, results): [usage.md](usage.md).
Notation for `--model` / `--runtime` / `--runtime-flags`:
[models-and-runtimes.md](models-and-runtimes.md).

## Runtime

Install with `runtimes pull`, or let the first `benchmarks run` fetch via the
shared runtime store:

```bash
pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'
pipette runtimes flavors                           # the --flavor vocabulary
pipette runtimes catalog llamacpp_cli_stock_tools --flavor macos-arm64
```

URI body:

| Keys | Meaning |
|------|---------|
| `version` + `flavor` | Upstream release tag + host package (required pair unless `url=`) |
| `repo` | Optional GitHub `org/repo` (default `ggml-org/llama.cpp`) |
| `url` | Optional prebuilt archive URL (mutually exclusive with version/repo) |

Examples:

```bash
pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'
pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://repo=github.com/acme/llama.cpp&version=b9305&flavor=linux-x64-cpu'
pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://url=https://example.com/llama-b9305.tar.gz&flavor=macos-arm64'
```

`flavor` selects the real upstream asset (e.g. `macos-arm64`, `linux-x64-cpu`).
It is not decorative: wrong flavor ⇒ wrong binary for the host.

## Model

GGUF weights in the shared `models/` store:

```bash
# text
--model 'gguf-text://repo=LiquidAI/LFM2-350M-GGUF&path=LFM2-350M-Q4_K_M.gguf'

# vision (weights + mmproj in one URI — no separate --mmproj flag)
--model 'gguf-vision://repo=org/repo&model=a.gguf&mmproj=mm.gguf'
```

Also: `url=https://…` for a direct GGUF fetch; JSON `Model` for forms the URI
cannot express. Gated repos: `PIPETTE_HF_TOKEN`.

## Quickstart

```bash
pipette init
pipette auth register --organization <org> --contact-email <email> \
  --client-details "<what this box is>"   # only needed to submit results
pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://version=<ver>&flavor=<flavor>'
pipette benchmarks init-local   # optional local-only set

pipette benchmarks run \
  --benchmark local/prefill_throughput_smoke \
  --model 'gguf-text://repo=LiquidAI/LFM2-350M-GGUF&path=LFM2-350M-Q4_K_M.gguf' \
  --runtime 'llamacpp-cli-stock-tools://version=<ver>&flavor=<flavor>'

# VL
pipette benchmarks run \
  --benchmark local/vl_throughput_smoke \
  --model 'gguf-vision://repo=LiquidAI/LFM2.5-VL-450M-GGUF&model=LFM2.5-VL-450M-Q4_0.gguf&mmproj=mmproj-LFM2.5-VL-450m-F16.gguf' \
  --runtime 'llamacpp-cli-stock-tools://version=<ver>&flavor=<flavor>'
```

For remote benchmarks: `pipette sync`, then `benchmarks run … --sync` (or sync
afterward). Only results from **remote** definitions are submitted.

## Execution notes

- **llama-server path:** the client starts the server, waits until ready, drives
  HTTP, measures outside the process, then tears the server down.
- **Runtime flags:** on the CLI, `--runtime-flags` takes a JSON **object** of
  settings for this cell (no axis keys; those come from `--benchmark`,
  `--runtime`, `--model`). llama-bench cells (prefill / decode / max-memory)
  accept `threads`, `number_gpu_layers`, `mmap`, `flash_attention` and `raw`;
  llama-server cells (end-to-end / eval, and vl on a vision model) add
  `ctx_size` and `no_cache`. `raw` refuses tokens that alias a typed setting,
  plus the ones the harness owns *for that cell*: llama-bench cells reserve
  `--output`, `--model`, `--n-prompt`, `--n-gen`, `--n-depth` and
  `--repetitions` (and `--ctx-size` on max-memory), while llama-server cells
  reserve `--model`, `--mmproj`, `--host`, `--port` and `--no-warmup`.
  The cell as it ran, plus what the client derived for the benchmark, is
  recorded on the result. Full tables:
  [models-and-runtimes.md](models-and-runtimes.md#per-cell-flags).
  Measurement rules: [methodology docs](../methodology/README.md).
- **Eval resume:** sample checkpoints under `state/evals/`.  
  [eval-checkpoint.md](eval-checkpoint.md)

Exact flags for the shared CLI: `pipette <group> --help` (see [usage.md](usage.md)).
