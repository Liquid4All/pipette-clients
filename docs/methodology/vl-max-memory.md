# Vision-Language Peak Memory Methodology

## Measured Quantity

`vl_max_memory` measures the **peak memory** used while running a single
image-plus-text workload through a vision-language model. It reports the same
fields as the text [peak memory benchmark](peak-memory-usage.md):
`max_host_bytes`, `max_gpu_bytes` and `max_npu_bytes`, each set or left null per
platform as [Platform status](#platform-status) lists. The metrics derived from
them are management's (pipette-mgmt
[docs/benchmarks.md §2.7](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#27-vl-peak-memory)).

The counters, the probing flow and the unified-vs-split rule are the text
benchmark's ([peak-memory-usage.md](peak-memory-usage.md)). This doc covers only
what differs for the vision-language case.

## Why we test it

On edge devices memory is usually the limit that decides whether a model runs
at all. A vision-language model adds two memory costs on top of the
language weights:

1. **The vision encoder + multimodal projector** (a separate model file, often
   0.5–1 GB), loaded resident alongside the weights.
2. **A much larger prefill.** Images expand into hundreds to thousands of tokens
   (see [vl-image-tokens.md](vl-image-tokens.md)), and those tokens occupy the
   KV cache and prefill activation buffers. This dominates and scales with the
   visual workload.

Testing peak memory across image size and frame count shows how far a model's
footprint grows with the image workload: the number that determines device fit.

## What is reused, and why there is a separate runner

**We reuse the peak-memory *probes*; we do not reuse the llama-bench
*driver*.** The text `max_memory_usage` runs `llama-bench -pg N,1`, and
`llama-bench` has **no multimodal support**: it prefills a synthetic *text*-token
count and has no way to feed an image through the vision encoder / mmproj.

Instead `vl_max_memory` drives **`llama-server`'s chat endpoint**
(`/v1/chat/completions` with an `image_url` part), the only llama.cpp path that
runs the image through the vision tower, the same pattern as the MLX memory
runner. The sampler (`pipette-memprobe`) is the text cell's, attached to the
server from its spawn until the workload ends. On Linux and Android it reports
the same swap-aware figure and `client_extras` as the text cell, so a run
that swaps shows in `max_swap_kib` and `major_faults` rather than in a
lower peak. Like the text cell, it has no model-file floor
([why](peak-memory-android.md#no-model-file-floor)).

**Workload.** One (or `num_images`) synthetic solid-gray image(s) go through the
vision tower with `parameter_text_tokens` of text, to a reply of
`parameter_decode_tokens` tokens (`max_tokens: N`, `ignore_eos`,
`cache_prompt: false`), as `vl_throughput` sends it. The parameter is optional:
unset, the reply is one token, and the row's `benchmark_params` carry no
`decode_tokens`. The server samples the first token from the prefill's logits,
so a one-token reply runs no decode step and the KV cache holds the prompt
alone, and N tokens run N-1 decode steps. The peak captures the vision-encode
buffers, the image-token KV cache, the prefill's buffers and the decode.
`llama-bench -pg N,1` decodes one token after its prompt, a step the one-token
VL request skips and one that moved no text peak measurably
([peak memory](peak-memory-usage.md#the-llama-bench-shape)).
The reply must run to its length (`finish_reason: length` and N completion
tokens), as the text cell's must: a shorter one would label the row with an
output length that never ran, so the cell fails instead.

**Projector.** The benchmark requires an mmproj (`--mmproj`); its quantization
is part of the run condition and must match across compared runs, since the
vision encoder's precision changes its resident size.

## Server flags

Beyond the [reserved flags](#deviations-from-the-peak-memory-methodology-and-why),
the client adds no server flags for this cell: a plan chooses them through the
cell's `no_cache` and `raw`, so each row's `runtime_flags` records what its
server ran. For a server that holds what `llama-bench` holds for the text cell
(one sequence, no context checkpoints, no fit pass), a plan sets:

```toml
no_cache = true
raw = ["--fit", "off"]
```

What `no_cache` renders, why each flag matters, the `raw` spellings that replace
its counts and the build floors are in [llamacpp.md](../pipette-cli/llamacpp.md)
(**No cache**). Two points are specific to this cell:

- `--fit off`. The fit pass runs a trial load, then places what the cell leaves
  unset (all GPU layers when `number_gpu_layers` is unset, MoE experts, the split
  across GPUs) from the free memory it reads at launch, less a 1 GiB margin per
  device. On a CUDA, HIP or Vulkan device other processes' memory counts, so with
  the pass on a row can depend on what else holds the device. It cannot shrink
  this cell's context, which the cell always sets. `llama-bench` runs no such
  pass by default. The trial frees what it allocates, but it moves the server's
  peak (figures below). With the pass off and `number_gpu_layers` unset, every
  layer goes to the GPU, so a model too large for a discrete GPU fails to load;
  on Apple Silicon a model past the recommended working set loads, and llama.cpp
  warns only when the weights are memory-mapped. `--fit` needs b7410; a plan
  pinning an older build leaves it out of `raw`.
- A slot count in `raw`. An explicit count turns off the unified cache that
  llama-server's own four-slot default uses, so each slot gets `--ctx-size`
  divided by the count. The cell sizes the context for one request, so its
  request then overflows a slot unless the cell also pins a larger `ctx_size` or
  adds `--kv-unified`; `--parallel 4 --kv-unified` reproduces the default layout.

Each flag's effect on the peak, against the server without it, on
`vl_max_memory_512x512_text1024_img1`, llama.cpp b10216, three interleaved
rounds each. The effects roughly add up.

On Apple Silicon (unified memory, `phys_footprint`):

- `--parallel 1` lowers the peak by 61.05 MiB on Qwen3.5-2B, whose recurrent
  state drops from 77.06 to 19.27 MiB; by 6.6 MiB on gemma-4-E2B, whose
  sliding-window cache drops from 1280 to 1024 cells; by 2.5 MiB on
  InternVL3.5-2B; and by 1.4 MiB on LFM2.5-VL-450M.
- `--ctx-checkpoints 0` moves it by about 0.
- `--fit off` raises it by 25 MiB on Qwen3.5-2B, 58 MiB on gemma-4-E2B and
  23 MiB on InternVL3.5-2B, and lowers it by 2.7 MiB on LFM2.5-VL-450M. The
  difference is the allocator's page state before the tensors load, not a buffer
  either server holds.

On Linux with an NVIDIA GPU (a CUDA build of the same release), Qwen3.5-2B and
gemma-4-E2B:

- `--parallel 1` lowers the GPU figure (NVML) by 60 and 6 MiB, and the host
  figure (`VmHWM`) by about 2.4 MiB.
- `--ctx-checkpoints 0` lowers the host figure by 35.5 and 17.3 MiB. On
  Qwen3.5-2B, two 19.3 MiB checkpoints are held at the peak.
- `--fit off` lowers the host figure by 16.9 and 8.3 MiB.

With all three against none, on Apple Silicon Qwen3.5-2B reads 35.9 MiB lower,
gemma-4-E2B 51.6 MiB higher and InternVL3.5-2B 22.9 MiB higher, while
LFM2.5-VL-450M reads 3.4 to 10.3 MiB lower, as its reading without the flags
moves by about 7 MiB between sessions. On the NVIDIA host, Qwen3.5-2B and
gemma-4-E2B read about 55 and 28 MiB lower in the host figure, and 60 and 6 MiB
lower in the GPU figure.

These figures are the peak of a one-token reply. A `vl_throughput` row records
the server's peak over its whole run, and the same flags move that reading by
different amounts ([vl-throughput.md](vl-throughput.md#context-checkpoints)).

`no_cache = true` runs one slot. Rows split on their recorded `command` and on
`runtime_flags`, which carries `no_cache` and `raw`. How to tell rows that ran
four slots under `no_cache` is in
[llamacpp.md](../pipette-cli/llamacpp.md#execution-notes) (No cache); a
Qwen3.5-class model reads about 60 MiB higher there.

## Deviations from the peak-memory methodology (and why)

Everything in [peak-memory-usage.md](peak-memory-usage.md) applies unchanged
except the following, which are forced by the multimodal workload:

| Aspect | Text `max_memory_usage` | `vl_max_memory` | Why it deviates |
| --- | --- | --- | --- |
| Workload driver | `llama-bench` (one-shot CLI) | `llama-server` chat endpoint | `llama-bench` can't run images through the vision tower; the chat endpoint is the only path that does (same as MLX). |
| Shape params | `parameter_prefill_tokens` | `parameter_image_width` / `_height`, `parameter_text_tokens`, `parameter_num_images`, optionally `parameter_decode_tokens` | VL memory is shaped by image geometry + frame count, not a single text-token count. |
| Context size | `--ctx-size` reserved as a llama-bench flag | sized from the calibrated image-token count ([vl-calibration.md](vl-calibration.md#deriving-n_ctx)), **no floor**; a cell that pins `ctx_size` keeps it and skips the calibration | Same intent: a padded context inflates the KV peak. There is no `--n-prompt` to size from, and one rule must serve all models. |
| Reserved flags | llama-bench `--n-prompt/--n-gen/--n-depth/--repetitions/--ctx-size/--model/--output/--no-warmup` | the server flags in pipette-mgmt [`reserved_flags.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/cell/reserved_flags.rs) (`--model`, `--mmproj`, `--host`, `--port`, `--ctx-size`, `--no-warmup` and others); the runner owns the workload shape (`--no-warmup` in particular: the single measured request must be the first); the plan sets the rest ([server flags](#server-flags)) | Different tool: a server, not a bench binary. |
| Platform coverage | macOS, Linux, Windows, Android, iOS | see [Platform status](#platform-status) | iOS has no `llama-server` and runs the workload in process; MLX drives its own server. |

Without the calibration helper, the context comes from the patch-grid estimate
([Fallback & safety](vl-calibration.md#fallback--safety)). That estimate is
often 5× or more above the real count, so `max_host_bytes` then carries a
bounded KV over-allocation.

Everything else is identical: one unrepeated run with no warmup request inside
the measured server, a server before it whose reading is dropped (below),
**no thermal/CPU readiness loop** (memory is thermal-independent), one reply
token unless the cell sets `parameter_decode_tokens`, byte units, independent
per-counter peaks with no cross-subtraction, and `max_gpu_bytes` null on
unified memory.

### Why the context size moves the peak

llama.cpp allocates the whole KV cache for `n_ctx` when it creates the context;
it does not grow it with the prompt, as paged-KV engines such as vLLM do. The
buffer is `n_ctx` cells × the attention layers × `n_kv_head · head_dim · 2 ·`
the K/V dtype's bytes, so every position the context holds past the prompt is
memory the peak counts.

On b8814, LFM2-VL-3B at 1280x1280 with no text, K/V f16, varying only
`--ctx-size`:

| `--ctx-size` | KV cells | KV buffer | Peak `phys_footprint` |
| --- | ---: | ---: | ---: |
| 2633 (calibrated) | 2816 | 44 MiB | 4.288 GB |
| 8354 (`(w/14)²` fallback) | 8448 | 132 MiB | 4.381 GB |

The 88 MiB KV difference accounts for the 93 MB difference in the peak. The
saving scales with the attention layers times the KV bytes per token. LFM2 is
hybrid, with 8 attention layers, so it saves little; full-attention models such
as Qwen and gemma save proportionally more, and every model saves more at high
frame counts.

## Warm-up server

On every flavor the runner first starts a `llama-server` on the same command,
sends it the cell's request, checks its reading and drops it, and stops it, then
starts the measured server; why is in
[peak memory: compile caches](peak-memory-usage.md#compile-caches). If that first
server fails to start or to answer, the cell fails. Each server gets the cell's
`bench_timeout_seconds` for itself. The response and the peak are the measured
server's; the observation covers both servers.

`vl_max_memory_smoke` on `LiquidAI/LFM2.5-VL-450M-GGUF` (Q4_0, F16 mmproj),
llama.cpp b11109, all layers offloaded, before the warm-up server, on an empty
cache and on one an earlier run filled:

| OS, GPU | `max_host_bytes`, empty cache | `max_host_bytes`, filled cache | `max_gpu_bytes`, either |
| --- | ---: | ---: | ---: |
| Linux, Intel Iris Xe, Vulkan | 241 MiB | 149 MiB | 493 MiB |
| Windows, AMD Radeon 890M, Vulkan | 211 to 213 MiB | 152 MiB | 519 MiB |
| macOS, Apple M3 Ultra, Metal | 958 to 960 MiB | 948 to 951 MiB | `null` |

The CPU flavors read 457 MiB on Linux and 452 to 453 MiB on Windows either way.
With the warm-up server, an empty cache reads 149 MiB on Linux, 152 MiB on
Windows and 949 MiB on macOS, two runs each, as a filled one does. On Linux the
493 MiB GPU figure equals a 2 ms read of the server's fdinfo.

## Frame count (`num_images`)

More frames mean more image tokens (≈ `num_images ×` the single-image count),
which grows the KV cache and activation buffers, so `num_images` is a main
axis of the memory sweep. See [vl-image-tokens.md](vl-image-tokens.md#multiple-images-num_images-and-the-prompt)
for how multiple images are packed into the prompt. `t0` (no text) cells isolate
the image's memory cost from text prefill.

## Reported fields and comparability

Compare only across the same model, **projector + its quantization**, runtime,
and benchmark shape (image size, frames, text).

### Consistency with the text (llama-bench) benchmark

The peak is sampled with the same counter and the same mechanic as the text
`max_memory_usage`. Only the driven process differs: text waits for
`llama-bench` to exit, while VL drives `llama-server` and stops it once its
request is done. With the plan's [server flags](#server-flags), the server holds
what `llama-bench` holds for the workload: one sequence, no checkpoints, no fit
pass. What it holds beyond that is its own: the HTTP stack and its threads, plus
the projector and the image path, which `llama-bench` cannot load. That share
moves with the model and the platform, so compare VL rows with VL rows, and read
a VL peak as the footprint of serving that model and image through
`llama-server`.

## Platform status

The runner dispatches on `LlamaCppFlavor`, as the text `max_memory_usage` does,
so only flavors with a matching probe run:

- **macOS**: `phys_footprint` from a poller. Unified memory, so `max_gpu_bytes`
  is null.
- **Linux and Android CPU**: the swap-aware `/proc/<pid>/status` figure
  (`max(VmHWM, max(VmRSS + VmSwap))`); `max_gpu_bytes` is null.
- **Linux GPU**: the same host figure, plus the server's DRM clients' resident
  bytes from its fdinfo and, on an NVIDIA GPU, NVML's bytes for it, as
  `max_gpu_bytes` ([peak memory: Linux](peak-memory-linux.md#llamacpp-linux)).
  A GPU showing none fails the run.
- **Windows**: PSAPI `PeakWorkingSetSize` (host), plus on GPU flavors the PDH
  per-process GPU-committed counter. A probe failure or a zero GPU reading fails
  the run rather than reporting a mislabeled number.
- **Android Snapdragon**: the host figure, plus its Adreno and Hexagon memory in
  `max_gpu_bytes` and `max_npu_bytes` from the KGSL mappings and dmabufs
  ([peak memory: Android](peak-memory-android.md#gpu-and-npu-counters)). Each is
  null when the run held none there, and an unreadable `maps` fails the run.
- **Refused before anything starts**: the Linux Ascend flavors, since nothing
  reads the Ascend NPU's memory and the host figure alone would miss it; the
  OpenVINO backend on any device but the CPU, since it falls back to the CPU on
  a device name OpenVINO does not list, which a memory reading cannot tell from a
  GPU run; the Linux Snapdragon build (`linux-arm64-snapdragon`), whose Adreno
  and Hexagon memory nothing reads on Linux; and custom flavors, whose devices
  pipette does not know.
- **iOS** measures in process and **MLX** through `mlx_vlm.server`, both below.
  The Android app does not run this cell.

## iOS

The phone has no server to drive, so the app runs the workload in its own process,
through the same `mtmd` path as `vl_throughput`, and reads the
[iOS peak-memory](peak-memory-ios.md) counter around it. It runs no warm-up
first, since a warm-up in the app's own process would raise the measured peak:

1. Settle the footprint to a floor, then start the `phys_footprint` sampler. The
   [private-API build](../pipette-ios/private-thermal-release-build.md) also
   resets the kernel's interval peak.
2. Load the weights, then the projector against them. The projector goes on the
   GPU, as `llama-server` places it.
3. Count one gray frame's image tokens with the projector, as the desktop's
   calibration helper counts them, and create the context from that count (see
   below).
4. Build `vl_throughput`'s prompt: `parameter_text_tokens` of text and one media
   marker per frame, in one user turn of the chat template. Preprocess the gray
   frames, encode them, prefill the prompt and decode `parameter_decode_tokens`
   tokens, one when unset.
5. Free the projector and the model, stop the sampler, and report the peak as
   `max_host_bytes`. `max_gpu_bytes` is null: the memory is unified.

Three details differ from the desktop runner:

- **Where the count comes from.** The phone has no `llama-mtmd-cli`, but mtmd
  needs only the weights, so the projector counts the frame before the context
  exists. The context gets the desktop's calibrated size, with no floor
  ([vl-calibration.md](vl-calibration.md#deriving-n_ctx)). A cell that pins
  `ctx_size` keeps it. The count is not cached: it costs one image preprocess
  per cell.
- **The image check.** The run fails when the prompt holds no image chunk,
  because then the markers did not expand and the peak is a text prompt's.
- **One decode step more.** The loop masks every end-of-generation token and
  decodes every token it samples, the last one included, a step the server
  skips. Its count is not checked: the loop runs every step it is asked for.

Measured for PR #1533 (merged 2026-09-28) on an iPhone 17 Pro ([private-API build](../pipette-ios/private-thermal-release-build.md), iOS 27.0), the six
`vl_max_memory` cells in the catalog then (every entry except
`256x256_text1024_decode256_img5`) read
1761 to 1875 MiB on LFM2.5-VL-1.6B Q4_K_M and 2057 to 2464 MiB on InternVL3.5-2B
Q4_K_M, both with F16 projectors. Two passes agreed within 3 MiB, and the sampler
matched the kernel's peak on every run. Sizing from the estimate instead would have
put InternVL's `1280x1280_text1024` cell at 3096 MiB: a 9472-position context
where 3840 hold the prompt, 616 MiB more KV cache than the calibrated row.

## MLX (macOS)

`pipette-mlx` drives `mlx_vlm.server`, the runtime's own server, from a
runtime row that ships `mlx-vlm`:

1. Build the request before the server starts: `parameter_text_tokens` of text
   sized with the server's tokenizer, then one gray PNG per image,
   `parameter_decode_tokens` reply tokens (one when unset), and the EOS ids
   biased away, as `vl_throughput` sends it.
2. Start the server with its prefix cache off and the Metal probe attached, and
   wait for `/health`, which answers only once the model is loaded.
3. Start the `phys_footprint` sampler, send the request and stop the sampler.
   It reads the kernel's lifetime maximum, so the model load's peak counts too,
   as in the MLX text runner. The reply must run to its length, as on
   `llama-server`: the bias covers only the tokenizer's EOS ids, so a reply that
   stops early fails the cell.
4. Do steps 2 and 3 twice: first on a warm-up server whose reading is checked
   and dropped and which stops before the measured one starts, so the measured
   server finds its Metal kernels compiled
   ([peak memory: macOS](peak-memory-macos.md#mlx)). The observation covers
   both servers.
5. Report the measured server's peak as `max_host_bytes`; `max_gpu_bytes` is
   null. The Metal allocator peak is a logged diagnostic, and a missing
   snapshot loses only that log line.

It differs from the `llama-server` runner in two ways:

- **No context to size.** `mlx_vlm`'s KV cache grows with the prompt in
  256-token steps, so nothing is reserved ahead of the request.
- **No plan flags.** Like the MLX text memory cell, it carries no
  `http_timeout_seconds` or `bench_timeout_seconds`. The request waits up to
  3600 s.

The figure is the whole Python server: the interpreter, the processor, the
weights and the request's buffers, which the server frees afterwards. After
loading, the two servers below idled at 692 and 970 MiB.

Measured on an M3 Max (macOS 27.0), three runs per cell, before the warm-up
server, in MiB:

| Cell | LFM2-VL-450M-8bit | Qwen3.5-0.8B-MLX-4bit |
| --- | ---: | ---: |
| `256x256_text0_img1` | 1221 to 1222 | 1436 to 1475 |
| `512x512_text0_img1` | 1316 to 1319 | 2034 to 2087 |
| `1280x1280_text1024_img1` | 2759 to 2776 | 3930 to 3936 |

Some workloads vary run to run: two 512x512 images with 16 text tokens on
LFM2-VL-450M-8bit read 1383 to 1773 MiB over seven runs, and a server driven by
hand varied as widely. Compare several runs of one cell, as for the MLX text
memory cell.

## Code References

The runner is
[`pipette-llamacpp vl_max_memory`](../../crates/pipette-llamacpp/src/execute/vl_max_memory.rs);
it picks the flavor's probe in
[`memory::peak`](../../crates/pipette-llamacpp/src/memory.rs), takes the
Linux and Android swap-aware figure from the probe's
[`Peak::counting_swap`](../../crates/pipette-memprobe/src/lib.rs) and the
`client_extras` from
[`swap_aware`](../../crates/pipette-llamacpp/src/swap_aware.rs), and sizes the
context in
[`mtmd`](../../crates/pipette-llamacpp/src/mtmd.rs), and `no_cache` renders in
[`ServerCommandBuilder::args`](../../crates/pipette-llamacpp/src/server.rs).
On iOS the runner is
[`LlamaExecute.vlMaxMemory`](../../ios/Pipette/Pipette/Runtimes/Llama/Execute/LlamaVlMaxMemory.swift),
which measures through
[`ProcessMemory.maxMemoryBracket`](../../ios/Pipette/Pipette/Runtimes/ProcessMemory.swift)
and sizes the context in
[`LlamaMtmd.CtxInputs`](../../ios/Pipette/Pipette/Runtimes/Llama/LlamaMtmd.swift).
On MLX the runner is
[`pipette-mlx vl_max_memory`](../../crates/pipette-mlx/src/execute/vl_max_memory.rs),
with the server launch in [`server`](../../crates/pipette-mlx/src/server.rs).
