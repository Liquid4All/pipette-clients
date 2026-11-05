# Peak Memory Usage Methodology

## Measured Quantity

Peak memory usage is the highest memory pressure observed by the selected
platform counters while a runtime loads a model and runs one controlled
generation workload. The benchmark reports byte counts because model fit is a
capacity question: the device either had enough resident host memory and
accelerator memory at the peak, or the workload failed to fit.

In pipette, this metric is reported by the `max_memory_usage` benchmark family.
The submission fields, derived metrics and the rules for reading them belong to
pipette-mgmt:
[docs/benchmarks.md §2.4](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#24-peak-memory-usage).

This document describes the `max_memory_usage` **metric**: a scored figure
produced by a benchmark whose whole job is to measure memory. It also describes
the memory **observations** that the other benchmarks record
([Observations vs metrics](#observations-vs-metrics)).

## The metric

The methodology uses three result fields:

- `max_host_bytes`: peak host/process memory attributed to the workload.
- `max_gpu_bytes`: peak GPU memory attributed to the workload, when a GPU
  probe exists for the runtime and platform.
- `max_npu_bytes`: peak NPU memory attributed to the workload, when an NPU
  probe exists for the runtime and platform.

On the wire `max_host_bytes` and `max_gpu_bytes` go out as `max_ram_bytes` and
`max_vram_bytes`
([pipette-mgmt `pipette-plan-types` submission](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/submission/mod.rs)).

Each reported field is the peak of its own counter. Treat the fields as
independent readings; no field is the residual of another, and the benchmark
preserves the raw peaks without cross-subtraction. On unified-memory systems,
host memory can already include accelerator allocations. On systems with
separate physical pools, host and GPU peaks are separate pressure sources.
Consumers must interpret the fields through the runtime and platform that
produced the result.

## Observations vs metrics

Every benchmark except the eval benchmarks also records the memory that its
runtime held while it ran, as **observations**:

| Column | Meaning |
|--------|---------|
| `observation_max_host_bytes` | peak **resident** memory the run held (`VmHWM`, `PeakWorkingSetSize`; on macOS `phys_footprint`, which also counts compressed pages; on torch-oai its cgroup's own memory, sampled) |
| `observation_max_swap_bytes` | most the kernel held of the run in swap at once |

These are measured workload facts rather than results, in the same sense as
`observation_vl_throughput_prefill_tokens`. They exist because a timing number
means something different depending on the memory conditions it was measured
under. Take a decode-throughput row that hit its figure while zram (the Linux
kernel's compressed swap in RAM) held part of the model. That row is not
comparable to one measured resident, and nothing else in the row says which
happened. They carry no benchmark prefix because they are not specific to one
benchmark type.

An observation spans the whole cell, not only its measured reps. It covers the
runtime process that served the cell from its start, so the model load counts,
and so does what the process served before the reps (prompt sizing, the
warm-up). A cell that starts the tool afresh for every rep (`llama-bench`,
`llama-batched-bench`) reports the largest of those processes. Most host
counters are high-water marks, and not every platform can reset one
mid-process, so a figure for the reps alone cannot be read. The warm-up repeats
a rep's request, so beyond the reps the span adds only the load. That load can
be all of a row's swap: on the Galaxy S26 Ultra a cell's first process can swap
while its reps run resident
([Galaxy S26 Ultra memory pressure](galaxy-s26-ultra-memory-pressure.md)). A
memory cell that warms up in a process of its own covers that process too,
compile included ([compile caches](#compile-caches)). OVMS sizes its prompt on a
server of its own that no cell watches. So on an empty cache the model's compile
is in no OVMS observation, though its warm-up's GPU kernel compiles are
([OVMS](../ovms.md#max_memory_usage)).

| Runtime | Counter | Cells |
|---------|---------|-------|
| llama.cpp | the per-OS sampler, on the tool's process | all but the eval benchmarks |
| MLX | the macOS sampler, on `mlx_lm.server` or `mlx_vlm.server` | all but `eval` |
| OVMS, OpenVINO, ONNX GenAI | the per-OS sampler, on the server's process | all (these engines run no eval benchmark) |
| torch-oai | the server's cgroup's own memory, on uv less its other processes', sampled ([Linux](peak-memory-linux.md#host-counter)) | all but the eval benchmarks on Docker, and a uv `max_memory_usage` cell; on a Linux host only |

The Core AI cells record no observation.

The eval benchmarks (`eval`, `vlm_eval`, `asr_eval`) are the deliberate
exception on every engine and report neither term. A run resumed from its
checkpoint starts its server for the samples left. llama.cpp and MLX also
restart a server that crashes mid-run. So the server that ends the cell may have
served only part of it, and any single figure would describe part of the run
while it looks like a figure for all of it. An absent observation says "not
observed", which is true, where a partial one would read as complete.

Two properties matter when reading them:

- **The two terms are independent, and must not be added.** The host term is a
  resident watermark; the swap term is the largest swap reading sampled. On
  torch-oai the host term is the largest of its reads and, on Docker, the swap
  term is the cgroup's watermark. The two terms need not fall at the same
  instant, so their sum describes no moment that happened. A non-zero swap term
  tells you that reclaim suppressed the resident peak beside it, and roughly by
  how much.
- **`observation_max_host_bytes` is resident-only, so it is not always the same
  quantity as `max_host_bytes` on the same row.** On Android and Linux,
  llama.cpp's text, VL and ASR memory metrics use the swap-aware peak (`max(VmHWM, max(VmRSS + VmSwap))`).
  On a row that swapped, that metric reads *higher* than the observation beside
  it. The difference is the metric's swap uplift, shown rather than hidden.
  Every other metric reads the same counter as its observation.

A worked example, measured on a Galaxy S26 Ultra loading a 5417 MiB model with
`--mmap 0`, which reads the weights into anonymous memory as
`--load-mode none` does:

| | |
|---|---|
| `observation_max_host_bytes` | 5004 MiB (what stayed resident) |
| `observation_max_swap_bytes` | 4105 MiB (what zram held at its worst) |
| Android `max_host_bytes` (metric) | 6138 MiB (the swap-aware peak) |

The resident figure alone is *below the model file*, which this model's
unpressured runs never read. The swap term beside it says why, and the metric
scores the real requirement.

### The observation must not move the number it rides along with

A benchmark that reports time cannot pay for being watched. So the observer
polls every 100 ms, where the memory benchmark polls every 10 ms
(Android/Linux) or 20 ms (macOS). That is two small reads per tenth of a
second, roughly 0.1% of one core. On Windows the sampler reads once, when it
stops, so it costs the reps nothing. torch-oai reads its cgroup's `memory.stat`
every 100 ms.

The host peak counters are kernel marks (`VmHWM`, the macOS lifetime maximum,
`PeakWorkingSetSize`), so the slow cadence rarely costs the host term. On
Linux, Android and macOS the mark must be read before the process is reaped
([Android host counter](peak-memory-android.md#host-counter)). Windows holds a
handle from the launch and reads the mark after the reap
([Windows host counter](peak-memory-windows.md#host-counter),
[`pipette-memprobe`](../../crates/pipette-memprobe/README.md)). torch-oai's host
term is the largest of its reads, so a rise and fall between two reads is
missed. The swap term has no kernel watermark and is always a sampled maximum.
An observation asks "did this run swap, and roughly how much", and a coarse
maximum answers that. Precision there belongs to the memory benchmark, which
keeps its fast cadence because it reports bytes rather than time.

Android and Linux report both terms. macOS reports the host term from
`phys_footprint`, which already bills compressed pages, so it has no separate
swap term. Windows reports the host term and never the swap one. A platform
that samples nothing contributes no keys, so absence means "not observed"
rather than zero, while a `0` means the sampler looked and found the run
resident.

## Measurement Challenges and Controls

Memory measurements have a different failure mode from latency and throughput
measurements. A short spike can decide whether a model fits, even when the
steady state looks smaller. The benchmark therefore reports peaks instead of
means. Each runtime path is arranged to cover the largest useful portion of
model load, prefill, and decode that its platform counters can observe.
Lifetime high-water counters can include startup and model-load peaks. Sampled
current counters see allocations that are still present while sampling is
active.

Runtime allocator logs are useful diagnostics, but they are smaller than the
process-level cost a device must actually carry. The OS or driver also bills
loaded libraries, runtime heaps, stacks, dynamic linker state, command queues,
driver metadata, staging buffers, and other bookkeeping. The benchmark reports
OS-attributed or allocator-attributed peak counters from the running process or
cgroup, and captures runtime output separately so allocator-level breakdowns
can still be audited.

Host and accelerator counters answer different questions. A host resident-set
counter answers how large the process became according to the kernel. A GPU
counter answers how much memory the GPU runtime or OS attributed to that
process. On Apple unified memory those counters overlap. On discrete GPU
systems they usually represent separate physical pressure. On Windows, an
integrated GPU with a unified memory architecture (UMA) can bill shared GPU
memory that overlaps with the host working set. The result keeps both readings
visible and avoids a cross-platform arithmetic rule.

Sampling is used where the platform exposes a current counter instead of a
lifetime peak counter, or where the process must still be alive for the counter
read. The sampled counters target allocations that persist for the benchmark
process, such as model buffers, KV cache, compute scratch, command buffers, and
driver state. For those workloads, a short polling interval observes the peak
after allocation and before process exit. Kernel-maintained high-water counters
are read directly where the platform has them.

## Measurement Protocol

The memory benchmark has a shared measurement contract, with platform-specific
probe placement:

1. Resolve the requested prompt-token count.
2. Start the runtime process, server, or container and enable the memory
   counters needed for that platform. Some counters are attached before process
   launch; lifetime high-water counters can be read after startup and still
   include earlier peaks.
3. Run a workload with the requested prompt-token count and a small decode.
4. Validate the prompt and decode token counts when the runtime reports them.
5. Stop or read the probes after the workload completes.
6. Report the observed host, GPU, and NPU peaks.

The benchmark measures one workload per cell and has no measured repetitions.
A second run inside the same process would carry allocator caches and kernel
high-water marks across from the first, so it would not represent a fresh model
workload. Most engines first run the same workload once in a process of their
own and drop its reading ([compile caches](#compile-caches)). Runtime or server
readiness still happens before the measured workload when a server is involved.
Memory runners do not run the thermal and CPU readiness gate used by the timing
benchmarks, because a peak does not move with the device's temperature.

### Compile caches

A GPU runtime compiles its kernels the first time a machine runs them. It keeps
the result in an on-disk cache that its driver owns. pipette does not set or
record that cache, and the compile runs inside the measured process. So on an
empty cache the host peak counts the compiler as well as the model, and the same
cell reads differently depending on what ran on the machine before.

To keep the compile out of the peak, these memory cells run the cell's workload
once first, in a process of their own, and drop that reading:

- **llama.cpp (CLI), every build:** `max_memory_usage` runs the same
  `llama-bench` command once. For `vl_max_memory` and `asr_max_memory`, a
  `llama-server` on the same command serves the cell's request once and stops
  before the measured server starts.
- **MLX:** an `mlx_lm.server` (or `mlx_vlm.server` for `vl_max_memory`) on the
  same command serves the cell's request once and stops before the measured
  server starts.
- **OpenVINO:** a warm-up server compiles the model. The measured server then
  loads a compiled blob and, on a GPU, the kernels that the warm-up's request
  compiled for the prompt's shape. That cache is the `CACHE_DIR` that pipette
  sets, not a driver's ([OpenVINO](../openvino-measurement.md)).
- **OVMS:** as OpenVINO, after the launch that sizes its prompt has compiled the
  model ([OVMS](../ovms.md#max_memory_usage)).
- **ONNX GenAI:** the first import after a runtime install writes the venv's
  Python bytecode inside the server, which read 6 to 8 MB higher on macOS
  ([ONNX GenAI](../onnx-genai-measurement.md#host-memory)).

The warm-up is watched and checked as the measured run is, and its reading is
dropped. If it fails, the cell fails. It gets the cell's deadline for itself, so
it does not shorten the measured run's deadline. The result is the measured
run's. The observation spans the whole cell, so it covers the warm-up, compile
included.

MLX ships most of its kernels prebuilt, but Metal still builds their GPU code
into the `com.apple.metal` cache. The MLX warm-up cannot run in the measured
server: the macOS sampler reads the process's lifetime maximum, so it would
count the warm-up.

The llama.cpp CPU builds warm up too. The CPU builds that were measured compile
nothing that outlives the run, and they read the same on an empty cache. A
choice per build would need to know which backends keep a compile cache, and
most were not measured for it: SYCL, ROCm, HIP, CUDA and OpenCL, and llama.cpp's
OpenVINO backend, which keeps its compiled model when `GGML_OPENVINO_CACHE_DIR`
or `GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR` is set. A build wrongly left cold
would read differently with the machine's history, and nothing on the row would
show it. A build that needs no warm-up pays one more model load.

These memory cells run no warm-up, so they measure their first run:

- **torch-oai.**
- **Core AI:** on a cold specialization cache the lifetime peak also holds the
  specialization ([Core AI specialization](coreai-specialization.md)).
- **The iOS app's llama.cpp cell:** it runs in the app's own process and
  carries its Metal shaders as source, as the macOS build does. So its figure
  can also depend on a cold cache. That was not measured.

What an empty cache cost each GPU, and what it reads with the warm-up, is
measured on the per-OS pages: [Linux](peak-memory-linux.md#shader-cache),
[Windows](peak-memory-windows.md#shader-cache) and
[macOS](peak-memory-macos.md#shader-cache). On
[Android](peak-memory-android.md#measured) only the empty cache's cost is
measured; the warm-up run was not checked on a phone.

## Benchmark Shape and Reported Fields

Concrete benchmark definitions use one token-count field:

- `parameter_prefill_tokens`: exact number of prompt tokens to prefill.

The standard ladder uses IDs such as `max_memory_usage_512`, meaning a
512-token prefill. The local smoke benchmark, `max_memory_usage_smoke`, uses 8
prompt tokens.

The submitted result reports:

- `max_host_bytes`: required host memory signal, always populated by supported
  implementations.
- `max_gpu_bytes`: optional GPU memory signal. A missing value means the
  path reads no GPU (a CPU cell, or unified memory the host figure holds), or,
  on Windows and on Android's Snapdragon build, that the run held none on that
  kind of device. A path that reads a GPU and sees none there fails the cell.
- `max_npu_bytes`: optional NPU memory signal. Android's Snapdragon build sets
  it from the Hexagon NPU's buffers. On Windows, every cell that the Performance
  Data Helper (PDH) watches sets it from what the Windows Display Driver Model
  (WDDM) bills the process on a compute-only adapter. Those cells are the
  llama.cpp GPU flavors, and OVMS, OpenVINO and ONNX GenAI off the CPU.

The unit is bytes because memory capacity and allocation APIs are byte-counted.
Storing bytes keeps results lossless and comparable.

No standard-deviation field is reported: the benchmark submits one observed
peak per cell, for the reason in [Measurement Protocol](#measurement-protocol).

## Workload Construction and Token Control

A complete memory cell needs visibility into model load, prefill, and decode.
Model load covers weights, runtime metadata, libraries, and initial driver
state. Prefill covers prompt processing, batched attention, and
context-dependent compute buffers. The first decode step can allocate additional
kernels, sampling buffers, command queues, or driver state that prefill alone
would miss.

The exact model-load coverage is counter dependent. Process-lifetime counters
can carry model-load peaks forward. GPU samplers run from the launch and read
each process's current commitment, so an allocation that rises and falls within
one tick can be missed. The controlled workload always exercises prefill and
decode, and the per-OS pages state how model-load memory is covered.

The workload has to produce a precise prompt-token count, but memory is a
capacity benchmark and tokenizer latency is outside the measured signal.
Runtimes can therefore use the most direct token-control path that preserves
the requested model work: llama.cpp uses `llama-bench -pg`, which takes the
token counts; MLX and torch-oai send exact-token text through the serving
endpoint. The invariant is the prefill-token count that shapes model memory.

For llama.cpp and MLX, the workload uses the requested prompt-token count and
one generated token, decoded on top of the prompt. One decode token is enough
to run the decode path on that context. For torch-oai runtimes (Docker and
uv), the benchmark sends a text completion request with the requested prompt
token count and one generated token, which comes off the prefill; vLLM and
SGLang reserve their KV cache and decode graphs at startup, so a decode step
would not move the peak ([Linux](peak-memory-linux.md#workload)).

Because the peak is dominated by model weights and prefill scratch, the decode
length moves it only marginally. Compare peak figures within a runtime, not
across runtimes, since the underlying counters differ (see the per-OS pages).

The exact-token invariant is enforced at the most reliable point for each
runtime. llama.cpp checks the row `llama-bench` prints for the counts. MLX and
torch-oai validate the OpenAI-compatible `usage.prompt_tokens` and
`usage.completion_tokens` fields after the request, and require
`finish_reason = "length"`.

### The llama-bench shape

The CLI's llama.cpp `max_memory_usage` cell runs
`llama-bench --output json --model <gguf> --repetitions 1 --n-prompt 0 --n-gen 0 -pg <N>,1`.
`-pg <N>,1` is one test: the prompt, then one token decoded on its KV cache,
in a context of `<N> + 1` tokens, which llama.cpp rounds up to a multiple of
256. The zeros turn off the `pp512` and `tg128` tests `llama-bench` adds when
either flag is left unset. The benchmark captures `llama-bench` stdout and
stderr for audit. The submitted metric comes from the memory counters. The row
in the JSON output is checked for its token counts, and its timing is audit
context only. The same command runs once before, its reading dropped
([compile caches](#compile-caches)).

`--n-prompt <N> --n-gen 1` would instead run two tests, the token in a fresh
context of its own, so the decode would not see the prompt's KV cache. The
decode step itself moved no peak measurably on `LiquidAI/LFM2-350M-GGUF`
(Q4_K_M, llama.cpp `b11192`, Linux CPU and macOS Metal): `llama-bench` decodes
without a sampler, and llama.cpp reserves its prompt and generation compute
buffers when the context is created. What the one-test shape adds is the larger
context. Mean `max_host_bytes` over six alternating runs, two-test shape first,
on `linux-x64-cpu` (8 threads, Xeon Platinum 8358) and `macos-arm64` (M3 Max):

| `<N>` | Context tokens | Linux (`VmHWM`) | macOS (`phys_footprint`) |
| ---: | ---: | ---: | ---: |
| 100 | 256 → 256 | 263.2 → 263.2 MiB | 540.5 → 540.0 MiB |
| 512 | 512 → 768 | 294.2 → 297.2 MiB | 547.8 → 551.0 MiB |
| 4096 | 4096 → 4352 | 339.9 → 342.9 MiB | 597.5 → 599.9 MiB |

Linux repeats within 0.1 MiB; macOS varies by about 3 MiB from run to run.

The memory benchmark reserves the workload-shaping flags (`--n-prompt`,
`--n-gen`, `--n-depth`, `--repetitions`, `--ctx-size`, `--model`, `--output`,
`--no-warmup` and their short forms;
[pipette-mgmt `reserved_flags.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/cell/reserved_flags.rs)),
so user flags cannot silently change the benchmarked memory shape. A raw flag
that adds tests to the run, such as a second `-pg` or a comma list of values,
fails the cell instead: `llama-bench` gives each test its own context, and all
of them would count toward the one peak.

## Counter Interpretation

`max_host_bytes` is process or cgroup memory attributed by the OS. Depending
on the platform, this can be a lifetime resident-set peak, the sampled peak of
a cgroup's own memory, or a platform-specific footprint counter.

`max_gpu_bytes` is accelerator memory attributed either by an in-process runtime
probe or by an OS GPU-memory counter. The source is platform-specific. A value
of `null` means GPU memory was outside that path's measurement coverage, or, on
Windows and on Android's Snapdragon build, that the run held none on that kind of
device, since no field is ever zero.

`max_npu_bytes` is NPU memory attributed by an OS counter: the dmabufs Android's
Snapdragon build hands its Hexagon NPU, and on Windows what WDDM bills the
process on a compute-only adapter. A value of `null` reads like the GPU's.

The fields are independent peaks and can occur at different moments. A host
peak during model load and a GPU peak during the benchmark request are both
real. Subtracting one from the other would imply a temporal relationship the
benchmark left unmeasured.

### Unified vs. split memory

A platform reports memory by **how the OS accounts for it**, not by the physical
layout of the silicon:

- **No split (truly unified).** The OS bills GPU allocations to the single
  whole-process resident counter and exposes no separate per-process GPU-memory
  accounting. The entire cost is reported as `max_host_bytes`, and `max_gpu_bytes`
  is `null`. There is no second pool to fit into, so the host footprint *is* the
  device-fit number. This is the Apple Silicon case (iOS and macOS Metal): Metal
  allocations live inside `phys_footprint`, and the driver keeps no carve-out
  outside it.
- **Split (virtual or physical).** The platform accounts host and GPU memory
  separately: a discrete GPU's own VRAM (physical split), or a driver/OS virtual
  carve-out on shared silicon that is tracked as its own pool (virtual split, e.g.
  Windows WDDM "GPU Process Memory", or the pools of Android's Kernel Graphics
  Support Layer (KGSL) driver and its dmabufs). Each is
  reported in its own field. Whether the two overlap (a virtual split on shared
  silicon) or are disjoint (a discrete GPU with its own VRAM) is a property of
  that platform and, on Windows, of the GPU driver: documented on its
  per-platform page and applied by the consumer from the result's
  platform/runtime.

`max_gpu_bytes` is therefore meaningful only when the platform exposes a real
host/GPU accounting split. On a unified device with no split, a GPU-allocator
high-water mark (Metal `currentAllocatedSize`, MLX `peakMemory`) is a useful
*diagnostic* of how much of the footprint the GPU runtime holds. It is not a
separate capacity dimension, so it is not reported as `max_gpu_bytes`; it
belongs in captured runtime logs. Both Apple platforms follow this rule.

## Per-platform pages

Each OS page holds its counters, cadences, reference measurements and caveats.

| OS | Host counter | Device counters |
| --- | --- | --- |
| [macOS](peak-memory-macos.md) | `phys_footprint` lifetime maximum (llama.cpp, MLX, ONNX GenAI, Core AI) | none: unified memory; MLX's Metal allocator peak is a diagnostic |
| [Windows](peak-memory-windows.md) | Process Status API (PSAPI) `PeakWorkingSetSize` | PDH `Total Committed` per adapter, GPU and NPU |
| [Android](peak-memory-android.md) | `max(VmHWM, max(VmRSS + VmSwap))` from `/proc` | Snapdragon build: KGSL mappings (GPU) and dmabufs (NPU) |
| [Linux](peak-memory-linux.md) | llama.cpp: Android's swap-aware `/proc` figure; OVMS, OpenVINO, ONNX GenAI: `VmHWM`; torch-oai: its cgroup's own memory | the Direct Rendering Manager (DRM) drivers' fdinfo, and the NVIDIA Management Library (NVML); torch-oai's AMD flavors the job's whole cards |
| [iOS](peak-memory-ios.md) | in-process `phys_footprint` poller, plus the kernel's interval maximum on the [private-API build](../pipette-ios/private-thermal-release-build.md) | none: unified memory |

## Result Validity and Comparability

A run is invalid if the workload fails, the required host memory read fails, or
the runtime reports token counts that differ from the benchmark shape. A
llama.cpp run is also invalid if `llama-bench` prints anything but one row. A
missing GPU field is valid only for paths that read no GPU, or on Windows and
Android's Snapdragon build for a device the run held no memory on. A zero in any
field is refused rather than submitted, since a peak reads zero only when no
read landed: the cell fails instead.

A run that made the device swap is still collected and stored, but it is held back
from the published results, together with every other run of that model and
quantization on that device. That policy covers all benchmarks, not only peak
memory, so it is stated with the selection rules: see
[Selection policies → Swap exclusion](selection-policies.md#swap-exclusion).

To compare `max_memory_usage` results:

- Use the same benchmark ID, model, quantization, runtime, and relevant runtime
  flags.
- Compare llama.cpp rows of one `llama-bench` shape. A row whose command preview
  shows `--n-prompt <N> --n-gen 1` ran two tests. It reads lower where `<N>` is a
  multiple of 256, by the KV cache of the 256 tokens that the one-test context
  adds.
- Compare llama.cpp GPU rows and MLX rows that ran a warm-up only with each
  other. Without a warm-up, on an empty driver cache, llama.cpp rows read up to
  about 100 MiB higher, and MLX rows read from about 180 MiB lower to 55 MiB
  higher ([macOS](peak-memory-macos.md#mlx)). The command preview is the same
  either way, so only the row's `client_version` tells them apart. On
  llama.cpp, rows whose `client_version` is dated 2026-10-03 or earlier, and
  some dated 2026-10-04, ran no warm-up. On MLX, rows dated 2026-10-04 or
  earlier, and some dated 2026-10-05, ran no warm-up. CPU rows read the same
  either way.
- Compare torch-oai rows of one probe, and a torch-oai `max_host_bytes` with
  care against another engine's: it is a cgroup's own memory, not a `VmHWM`. A
  row whose command preview reads `memprobe: cgroup memory.peak` recorded the
  cgroup's `memory.peak`, which counts the page cache, and reads higher
  ([Linux](peak-memory-linux.md#host-counter)).
- On torch-oai's vLLM and SGLang, `max_gpu_bytes` is the engine's up-front
  reservation, a fraction of the card's memory, so it follows the card and that
  setting, not the model, and is not comparable across cards or with other
  engines ([Linux](peak-memory-linux.md#gpu-counter)).
- Read the fields against their own limits, never summed or subtracted:
  pipette-mgmt's
  [Reading the memory fields](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#reading-the-memory-fields),
  with each pool's overlap on the per-OS pages
  ([Windows](peak-memory-windows.md#host-and-device-pools),
  [Linux](peak-memory-linux.md#host-and-gpu-pools-of-the-servers),
  [Android](peak-memory-android.md#reading-the-three-together)).
- On Windows, treat `Total Committed` as the memory WDDM billed the process on
  that kind of adapter, committed rather than resident.
- Inspect captured runtime output when auditing allocator-level details such as
  model buffers, KV cache, compute scratch, or driver logs.

The result artifacts preserve enough context to audit the run later:

- Benchmark ID and resolved benchmark parameters.
- Model name, quantization, runtime name, and runtime version.
- Effective runtime flags when the runtime exposes configurable flags.
- Command preview for the runtime request or server command.
- Captured benchmark stdout or stderr when the runner records it.

## Code References

The benchmark definition and result fields are implemented in
[`pipette_plan_types::benchmark`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/benchmark/mod.rs) and
[`pipette_cli::benchmarks`](../../crates/pipette-cli/src/benchmarks).

The shared measurement (warm-up, zero refusal) is
[`pipette-ops max_memory`](../../crates/pipette-ops/src/measurement/max_memory.rs).

The per-OS samplers (`/proc`, DRM fdinfo, NVML, `phys_footprint`, PSAPI and PDH),
the cgroup sampler, the AMD whole-card reader and the macOS Metal probe are
implemented in [`pipette-memprobe`](../../crates/pipette-memprobe).

The runners:

- llama.cpp: [`max_memory_usage.rs`](../../crates/pipette-llamacpp/src/execute/max_memory_usage.rs)
- MLX: [`max_memory_usage.rs`](../../crates/pipette-mlx/src/execute/max_memory_usage.rs)
- torch-oai: [`max_memory_usage.rs`](../../crates/pipette-torch-oai/src/execute/max_memory_usage.rs),
  which finds the server's cgroup in [`memprobe.rs`](../../crates/pipette-torch-oai/src/memprobe.rs)
- OVMS: [`max_memory_usage.rs`](../../crates/pipette-ovms/src/execute/max_memory_usage.rs)
- OpenVINO: [`max_memory_usage.rs`](../../crates/pipette-openvino/src/execute/max_memory_usage.rs)
- ONNX GenAI: [`max_memory_usage.rs`](../../crates/pipette-onnx-genai/src/execute/max_memory_usage.rs)
- Core AI: [`max_memory_usage.rs`](../../crates/pipette-coreai/src/execute/max_memory_usage.rs)
