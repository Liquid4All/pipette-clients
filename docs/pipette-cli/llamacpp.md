# llama.cpp runtime

Backend library: `pipette-llamacpp`, driven by the unified `pipette` CLI.

Rust owns control plane (workspace, auth, stores, measurement). Upstream
**llama.cpp release binaries** do the inference:

| Tool | Benchmark types |
|------|-----------------|
| `llama-bench` | `prefill_throughput`, `decode_throughput`, `max_memory_usage` |
| `llama-batched-bench` | `prefill_throughput`, `decode_throughput` above `parameter_batch_size` 1 |
| `llama-server` (HTTP) | `end_to_end_latency`, `eval`, `vl_throughput`, `vl_max_memory`, `vlm_eval`, `asr_eval`, `vl_spec_decode`, `asr_throughput`, `asr_max_memory` |

Shared operator flow (init, auth, sync, results): [usage.md](usage.md).
Notation for `--model` / `--runtime` / `--runtime-flags`:
[models-and-runtimes.md](models-and-runtimes.md).

## Runtime

Install with `runtimes pull`, or let the first `benchmarks run` fetch via the
shared runtime store:

```bash
pipette runtimes flavors                           # the --flavor vocabulary
pipette runtimes catalog llamacpp_cli_stock_tools --flavor macos-arm64
pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'
```

The URI keys (`version`, `flavor`, `repo`, `asset`, `url`), their defaults
and more examples are in
[models-and-runtimes.md](models-and-runtimes.md#runtime-schemes).

How `flavor` works for llama.cpp:

- **Renamed files.** Upstream renames some release files when a toolkit
  changes version (ROCm 7.2 → 7.14 → 10.0, and each OpenVINO release).
  Pipette picks the file name that was in use at the tag.
- **Files that pipette does not know.** Name the file in `asset` (`"asset"` in
  the JSON form). Keep the flavor that matches the build, because the flavor
  also decides how the memory benchmarks read the build's memory.
- **Unlisted flavors.** A flavor that `pipette runtimes flavors` does not list
  is an operator's build. Its cells run if it installs from `url=` or a named
  `asset`. Pipette refuses its memory cells, because it does not know which
  devices to read.
- **Ascend flavors.** Pipette also refuses the memory cells of the four
  `linux-openeuler-*-ascend-*` flavors, because nothing reads an Ascend NPU's
  memory.
- **CUDA 12.8 on Linux x86_64.** An upstream archive (from `b10969`) or a
  prebuilt one under `url=` takes `linux-cuda-12-8`. Its memory cells read the
  GPU through NVML (NVIDIA Management Library).
- **CUDA 13 on Linux.** An archive takes `linux-x64-cuda-13`, or
  `linux-arm64-cuda-13` on arm64. Its memory cells read the GPU through NVML,
  except on a Jetson. A Jetson GPU allocates from the tool's own resident set,
  so the memory cells read only that
  ([peak-memory-linux](../methodology/peak-memory-linux.md#llamacpp-gpu-counter)).

## Model

GGUF weights in the shared `models/` store:

```bash
# text
--model 'gguf-text://repo=LiquidAI/LFM2-350M-GGUF&path=LFM2-350M-Q4_K_M.gguf'

# vision (weights + mmproj in one URI, no separate --mmproj flag)
--model 'gguf-vision://repo=org/repo&model=a.gguf&mmproj=mm.gguf'
```

Also: `url=https://…` for a direct GGUF fetch; JSON `Model` for forms the URI
cannot express. Gated repos: `PIPETTE_HF_TOKEN`, see [usage.md](usage.md).

## Quickstart

```bash
pipette init
pipette benchmarks init-local   # seeds the local/ set
pipette benchmarks run \
  --benchmark local/prefill_throughput_smoke \
  --model 'gguf-text://repo=LiquidAI/LFM2-350M-GGUF&path=LFM2-350M-Q4_K_M.gguf' \
  --runtime 'llamacpp-cli-stock-tools://version=<ver>&flavor=<flavor>'
```

The full workflow (init, register, seed local benchmarks, sync, submit) is in
[usage.md](usage.md).

## Execution notes

- **llama-server path:** the client starts the server, waits until ready, drives
  HTTP, measures outside the process, then stops the server.
- **`parameter_batch_size`:** above 1, `end_to_end_latency` opens that many
  llama-server slots, and `prefill_throughput` / `decode_throughput` move from
  `llama-bench` to `llama-batched-bench`. Rules:
  [end-to-end-latency.md](../methodology/end-to-end-latency.md#batched).
- **`vl_spec_decode`:** one server, one corpus traversal, one row per run. The
  runtime flag `speculative_enabled` follows the model when unset (on for
  `gguf_vision_speculative`, off for `gguf_vision`). With speculation on, the
  client adds the draft length (and a separate drafter, when the model has one)
  to the cell's `envs` and `raw`, and refuses the launch when the drafter's
  announced block is shorter than the benchmark's draft length, since the
  server would clamp it. Above `parameter_batch_size` 1 the client opens that
  many slots, 8192 tokens each unless the cell sets `ctx_size`. DSpark is a
  speculative-decoding drafter (for example `LiquidAI/LFM2.5-VL-3B-DSpark`).
  Its speculation flags exist only on a llama.cpp fork that publishes no release:
  pin a build of it as a `remote_archive`, as
  [`examples/plans/llamacpp-macos-vl-spec-decode.toml`](../../examples/plans/llamacpp-macos-vl-spec-decode.toml)
  does. Measurement rules:
  [vl-spec-decode.md](../methodology/vl-spec-decode.md).
- **Runtime flags:** `--runtime-flags` takes a JSON object of settings for this
  cell. The keys each cell accepts and the `raw` flags the harness reserves:
  [models-and-runtimes.md](models-and-runtimes.md#per-cell-flags).
- **Load mode:** `load_mode` picks how weights reach memory: `none` (anonymous
  RAM), `mmap`, `mlock`, `mmap_mlock` or `direct_io`. It renders as upstream's
  `--load-mode`, which needs llama.cpp `b10105` or newer. Left unset, every cell
  but `eval` pins the weights in RAM (`--load-mode none`), so a model that only
  fits file-backed fails rather than producing rows nothing can compare
  against. The deprecated `mmap` boolean is still accepted, with a warning,
  because `mmap = true` renders no `--load-mode` (nothing on llama-server,
  `--mmap 1` on llama-bench) and so is how a cell runs a build older than
  `b10105`; a cell that sets both is refused.
- **No cache:**
  - `no_cache = true` renders `--no-cache-prompt` and `--ctx-checkpoints 0`.
    It also renders `--parallel 1`, unless the cell sets a slot count above
    one. Left unset or `false`, `no_cache` adds nothing.
  - Some stored `no_cache` rows come from client versions whose `no_cache`
    rendered no `--parallel 1`. Those rows ran four slots, and their
    `client_version` tells them apart.
  - Why: with the prompt cache off, llama-server still saves checkpoints of a
    hybrid or sliding-window model's state while it prefills. It reads one
    back only on a cache hit, so the checkpoints cost time and memory for no
    gain. Each of its four default slots also keeps its own recurrent and
    sliding-window state.
  - The `end_to_end_latency`, `vl_throughput`, `vl_spec_decode`,
    `vl_max_memory`, `asr_throughput` and `asr_max_memory` requests always
    send `cache_prompt: false`. On those cells, `no_cache` turns the
    checkpoints off and runs one slot.
  - A count in `raw` replaces either default, in any spelling
    (`--ctx-checkpoints`, `-ctxcp`, `--swa-checkpoints`; `--parallel`, `-np`).
  - `--ctx-checkpoints` requires llama.cpp `b6685` or newer. On an older build
    that knows `--swa-checkpoints`, put `--swa-checkpoints 0` in `raw`, which
    replaces it.
  - llama-server honors an explicit `--parallel 1` from `b7433`. Older builds
    open four slots.
  - Apart from the `vlm_eval` prompt caches (below), the client renders no
    other server defaults per benchmark, so each row's `runtime_flags` records
    the setting.
- **`vlm_eval` without prompt caches:** a `vlm_eval` server always gets
  `--no-cache-prompt`, `--ctx-checkpoints 0` and `--cache-ram 0`, each unless
  the cell's `raw` sets it, at the cell's own slot count; `runtime_flags` does
  not record them. Every sample is a new prompt, so the caches can only cost:
  [vlm-eval.md](../methodology/vlm-eval.md#prompt-caches).
- **Like-for-like with `llama-bench`:** a plan that wants the server cells
  comparable with `llama-bench` sets `no_cache = true` on `end_to_end_latency`,
  `vl_throughput`, `vl_spec_decode` and `asr_throughput`, and `no_cache = true`
  with `raw = ["--fit", "off"]` on `vl_max_memory` and `asr_max_memory`
  (`--fit` requires `b7410`). On
  llama.cpp b10216 with Apple Silicon, no checkpoints cut `end_to_end_latency`
  by 2.0 to 2.7% on hybrid and sliding-window models, and one slot lowered
  Qwen3.5-2B's recorded memory by about 60 MiB and moved the timings by at most
  about 3 ms. On `vl_spec_decode` with a hybrid Qwen3.5-2B target, no
  checkpoints left the output unchanged and lowered the recorded memory by 26%.
  The shifts per cell:
  [end-to-end-latency.md](../methodology/end-to-end-latency.md#llamacpp),
  [vl-throughput.md](../methodology/vl-throughput.md#context-checkpoints),
  [vl-spec-decode.md](../methodology/vl-spec-decode.md#context-checkpoints),
  [vl-max-memory.md](../methodology/vl-max-memory.md#server-flags).
- **Deadlines:** the cells that own a subprocess run under a wall-clock
  deadline. A driver bug or a platform hang then fails the cell and does not
  block the runner.
  - Each prefill and decode throughput process gets 1 hour.
  - The peak-memory cells get 600 s, unless `bench_timeout_seconds` sets
    another value. On `max_memory_usage` the deadline bounds the `llama-bench`
    subprocess. On `vl_max_memory` and `asr_max_memory` it bounds the server
    run from launch to the peak read.
  - The warm-up run (text) or warm-up server (vision and audio cells) gets the
    same budget, so a memory cell can take up to twice the deadline.
  - `vl_spec_decode` checks `bench_timeout_seconds` between samples, because
    no child process ends the HTTP loop. Left unset, it has no deadline: the
    corpus sets the length of a traversal, and a fixed deadline would stop a
    healthy three-hour run.
  - The other server cells (`eval`, `vlm_eval`, `end_to_end_latency`,
    `vl_throughput`, `asr_throughput`) are bounded per HTTP request, through
    `http_timeout_seconds`.
- **ASR (automatic speech recognition):** `asr_throughput` and `asr_max_memory` run every server with
  `--reasoning off`, so they need llama.cpp `b8287` or newer, where `-rea` /
  `--reasoning` first appears; an older build's server exits on the unknown
  flag. They refuse a `raw` reasoning or chat-template flag, an env forward
  or host-configuration env of one, and an audio encoder no recipe asks
  ([asr-throughput.md](../methodology/asr-throughput.md#the-transcription-prompt)).
  A recipe that serves its own template reads the server's `/props` to check
  it is the one rendered.
- **Eval resume:** sample checkpoints under `state/evals/`:
  [eval-checkpoint.md](eval-checkpoint.md).

Exact flags for the shared CLI: `pipette <group> --help` (see [usage.md](usage.md)).
