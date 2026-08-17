# Peak Memory Usage Methodology

## Measured Quantity

Peak memory usage is the highest memory pressure observed by the selected
platform counters while a runtime loads a model and runs one controlled
generation workload. The benchmark reports byte counts because model fit is a
capacity question: the device either had enough resident host memory and
accelerator memory at the peak, or the workload failed to fit.

In pipette, this metric is reported by the `max_memory_usage` benchmark family.

The methodology uses three conceptual result fields:

- `max_host_bytes`: peak host/process memory attributed to the workload.
- `max_gpu_bytes`: peak GPU memory attributed to the workload, when a GPU
  probe exists for the runtime and platform.
- `max_npu_bytes`: peak NPU memory, reserved for future backends.

The current client wire format still serializes `max_host_bytes` as
`max_ram_bytes` and `max_gpu_bytes` as `max_vram_bytes` for management-server
compatibility. The methodology names describe the intended meaning of those
fields.

Each reported field is the peak of its own counter. Treat the fields as
independent readings; no field is the residual of another. The benchmark
preserves the raw peaks without cross-subtraction. On unified-memory systems,
host memory can already include accelerator allocations. On systems with
separate physical pools, host and GPU peaks are separate pressure sources.
Consumers must interpret the fields through the runtime and platform that
produced the result.

The exact coverage depends on the counter source and when that counter is
attached. Lifetime high-water counters can include earlier startup peaks.
Sampled current counters see allocations that are still present while sampling
is active.

## Measurement Challenges and Controls

Memory measurements have a different failure mode from latency and throughput
measurements. A short spike can decide whether a model fits, even when the
steady state looks smaller. The benchmark therefore reports peaks instead of
means. Each runtime path is arranged to cover the largest useful portion of
model load, prefill, and decode that its platform counters can observe.
Lifetime high-water counters can include startup and model-load peaks. Current
sampled counters primarily see allocations that remain visible while the
controlled request is running.

Runtime allocator logs are useful diagnostics, but they are smaller than the
process-level cost a device must actually carry. The OS or driver also bills
loaded libraries, runtime heaps, stacks, dynamic linker state, command queues,
driver metadata, staging buffers, and other bookkeeping. The benchmark reports
OS-attributed or allocator-attributed peak counters from the running process or
container, and captures runtime output separately so allocator-level
breakdowns can still be audited.

Host and accelerator counters answer different questions. A host resident-set
counter answers how large the process became according to the kernel. A GPU
counter answers how much memory the GPU runtime or OS attributed to that
process. On Apple unified memory those counters overlap. On discrete GPU
systems they usually represent separate physical pressure. Windows UMA can also
include shared GPU memory that overlaps with the host working set. The result
keeps both readings visible and avoids a cross-platform arithmetic rule.

Sampling is used where the platform exposes a current counter instead of a
lifetime peak counter, or where the process must still be alive for the counter
read. The sampled counters target allocations that persist for the benchmark
process, such as model buffers, KV cache, compute scratch, command buffers, and
driver state. For those workloads, a short polling interval observes the peak
after allocation and before process exit. Kernel-maintained high-water counters
are read as high-water marks directly when the platform supports that.

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

The benchmark runs one measured workload per cell. Repeating a memory cell
inside the same process can make allocator caches and kernel high-water marks
carry across attempts, which makes the later runs less representative of a
fresh model workload. The reported value is therefore the peak from one
controlled run.

The protocol has no measured repetitions and no warmup request. Runtime or
server readiness still happens before the measured workload when a server is
involved. Current memory runners do not run the platform thermal or CPU
readiness loop used by end-to-end latency because there are no repeated
measured requests.

## Benchmark Shape and Reported Fields

Concrete benchmark definitions use one token-count field:

- `parameter_prefill_tokens`: exact number of prompt tokens to prefill.

The standard ladder uses IDs such as `max_memory_usage_512`, meaning a
512-token prefill. The local smoke benchmark uses 8 prompt tokens.

The submitted result reports:

- `max_host_bytes`: required host memory signal, always populated by supported
  implementations.
- `max_gpu_bytes`: optional GPU memory signal. A missing value means the
  path had no GPU probe, the runtime flavor was host-only, or no matching GPU
  sample was observed.
- `max_npu_bytes`: optional NPU memory signal. Current implementations leave it
  unset.

The unit is bytes because memory capacity and allocation APIs are byte-counted.
Human-readable MiB or GiB conversions are useful for reports, but storing bytes
keeps results lossless and comparable.

No standard-deviation field is reported for `max_memory_usage`. The benchmark
submits one observed peak per cell because allocator caches and high-water
counters can carry state across repeated attempts inside the same process.

## Workload Construction and Token Control

A complete memory cell needs visibility into model load, prefill, and decode.
Model load covers weights, runtime metadata, libraries, and initial driver
state. Prefill covers prompt processing, batched attention, and
context-dependent compute buffers. The first decode step can allocate additional
kernels, sampling buffers, command queues, or driver state that prefill alone
would miss.

The exact model-load coverage is counter dependent. Process-lifetime counters
can carry model-load peaks forward. Request-window GPU samplers see allocations
that are still present during the controlled request, but can miss a transient
startup-only GPU spike. The controlled workload always exercises prefill and
decode, and the platform sections below state how model-load memory is covered.

The workload has to produce a precise prompt-token count, but memory is a
capacity benchmark and tokenizer latency is outside the measured signal.
Runtimes can therefore use the most direct token-control path that preserves
the requested model work. Some backends expose benchmark-native token-count
controls; others need exact-token text because the serving API only accepts
text prompts.

For llama.cpp and MLX, the workload uses the requested prompt-token count and
one generated token. One decode token is enough to create the decode-path
allocations that persist for the process. For torch-oai Docker runtimes, the
current benchmark sends a text completion request with the requested prompt
token count and 16 generated tokens. That small decode flows a normal
completion path while keeping runtime cost low; token usage is validated from
the OpenAI-compatible response.

The decode length therefore differs by runtime (one token for llama.cpp and
MLX, sixteen for torch-oai). Because the peak is dominated by model weights and
prefill scratch, a few extra decode tokens move it only marginally, but it is
one more reason peak figures are compared within a runtime, not across them
(the underlying counters differ too; see the per-OS pages).

Tokenizer latency is outside this benchmark's goal. Runtime paths use the
prompt mechanism that is practical for that backend: llama.cpp uses
`llama-bench --n-prompt`, MLX uses a deterministic token sequence in the Python
sidecar, and torch-oai sends exact-token text through the serving endpoint. The
invariant is the prefill-token count that shapes model memory.

The exact-token invariant is enforced at the most reliable point for each
runtime. Llama.cpp relies on `llama-bench`'s native `--n-prompt` and `--n-gen`
controls. MLX returns prompt and completion token counts from the sidecar and
the Rust runner validates them. Torch-oai validates the OpenAI-compatible
`usage.prompt_tokens` and `usage.completion_tokens` fields after the request.

For llama.cpp, the memory benchmark reserves the workload-shaping flags:
`--n-prompt`, `--n-gen`, `--repetitions`, `--ctx-size`, `--model`, and
`--output`. Runtime flag overrides for those flags are rejected, so user flags
cannot silently change the benchmarked memory shape.

## Counter Interpretation

`max_host_bytes` is process or container memory attributed by the OS. Depending
on the platform, this can be a lifetime resident-set peak, a cgroup memory peak,
or a platform-specific footprint counter.

`max_gpu_bytes` is accelerator memory attributed either by an in-process runtime
probe or by an OS GPU-memory counter. The source is platform-specific. A value
of `null` means GPU memory was outside that path's measurement coverage. Read
it as an absent measurement, not as zero GPU memory use.

`max_npu_bytes` is reserved for future NPU backends. It remains unset in the
current clients.

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
  device-fit number. This is the Apple-Silicon case (iOS and macOS Metal): Metal
  allocations live inside `phys_footprint`, with no driver-managed carve-out the
  host counter cannot see.
- **Split (virtual or physical).** The platform accounts host and GPU memory
  separately: a discrete GPU's own VRAM (physical split), or a driver/OS virtual
  carve-out on shared silicon that is tracked as its own pool (virtual split, e.g.
  Windows WDDM "GPU Process Memory"). Each is reported in its own field. Whether
  the two overlap (a virtual split on shared silicon) or are disjoint (a discrete
  GPU with its own VRAM) is a property of that platform: documented on its
  per-platform page and applied by the consumer from the result's platform/runtime.

`max_gpu_bytes` is therefore meaningful only when the platform exposes a real
host/GPU accounting split. On a unified device with no split, a GPU-allocator
high-water mark (Metal `currentAllocatedSize`, MLX `peakMemory`) is a useful
*diagnostic* of how much of the footprint the GPU runtime holds, but it is not a
separate capacity dimension and is not reported as `max_gpu_bytes`: it belongs in
captured runtime logs.

Both Apple platforms follow this rule: iOS and macOS (Apple Silicon, M1–M5) are
unified with no split, so both report the whole cost as `max_host_bytes` and
leave `max_gpu_bytes = null`, keeping the Metal allocator peak only as a
diagnostic.

## Platform Counter Semantics

The shared fields have platform-specific coverage because each platform exposes
different counters and process-lifetime rules.

| Path | Probe placement | Host coverage | GPU coverage | Interpretation caveat |
| --- | --- | --- | --- | --- |
| macOS llama.cpp | Metal probe attached before `llama-bench` launch; host poller runs during child lifetime. | `phys_footprint` plus kernel lifetime maximum. | `null`: Apple Silicon unified memory, no host/GPU split. | Unified memory: reported entirely as host. The Metal `currentAllocatedSize` peak is captured as a diagnostic, not a field. |
| macOS MLX | Metal probe attached before Python sidecar launch; host poller starts after sidecar readiness. | `phys_footprint` plus kernel lifetime maximum, so earlier model-load peaks remain visible while the process lives. | `null`: Apple Silicon unified memory, no host/GPU split. | Unified memory: reported entirely as host. Metal `currentAllocatedSize` peak captured as a diagnostic. |
| Android llama.cpp CPU | `llama-bench` launched directly; `/proc/<pid>/status` sampler runs during child lifetime, seeded by one synchronous read at spawn. | `max(VmHWM, max(VmRSS + VmSwap))`, so pages zram compressed out of the resident set still count. | Unset. | Current Android arm64-v8a runtime is host-only. Devices carry more zram than RAM, so a resident-only figure under-reports by as much as 966 MB; a peak below the model file size is refused. |
| Linux llama.cpp CPU | `llama-bench` launched directly; `/proc/<pid>/status` `VmHWM` poller runs during child lifetime, seeded by one synchronous read at spawn. | `VmHWM` peak resident-set high-water mark. | Unset. | CPU flavors only; GPU/accelerator flavors are rejected pending a DRM-fdinfo probe. |
| Windows llama.cpp | PSAPI handle kept across child exit; PDH poller starts after spawn for GPU flavors. | `PeakWorkingSetSize` lifetime peak. | PDH `Total Committed` sampled every 20 ms for GPU flavors. | CPU flavors report `null`; PDH setup or read failure currently reports `0` and requires log audit. |
| Linux torch-oai Docker | Server starts and reaches readiness before the benchmark memory probe starts. | cgroup v2 `memory.peak`, which covers the container lifetime. | Host `nvidia-smi` samples matching container PIDs every 500 ms during the benchmark request. | GPU sampling sees persistent allocations during the request; transient GPU-only startup spikes can fall outside the sampled window. |
| iOS (llama.cpp + MLX) | In-process `phys_footprint` poller across a fresh model load, prefill, and one decode step. | `phys_footprint` peak, reported as `max_ram_bytes`. The whole-process cost of running that one model on that one runtime. | `null`: unified memory, no host/GPU accounting split. | Per-(model, runtime, quant) measure; the GPU-allocator high-water mark is kept as a log diagnostic, not a reported field. Stickiness is bounded by sampling across the per-cell fresh-load bracket, not the kernel lifetime-max. |

## Per-Platform Details

For all implemented llama.cpp memory paths, the runner executes
`llama-bench --output json --n-prompt <N> --n-gen 1 -r 1`. The benchmark
captures `llama-bench` stdout and stderr for audit; the submitted metric comes
from the memory counters, and timing rows in the JSON output are audit context
only.

Each platform exposes different counters and process-lifetime rules. The counter
mechanics, reference measurements, and caveats for each are documented
separately:

- [macOS](peak-memory-macos.md): `phys_footprint` host counter, shared by
  llama.cpp and MLX (Apple Silicon, unified memory); reported entirely as
  `max_host_bytes`, `max_gpu_bytes = null`. The injected Metal
  `currentAllocatedSize` probe is kept as a diagnostic.
- [Windows](peak-memory-windows.md): PSAPI `PeakWorkingSetSize` host counter and
  a PDH `Total Committed` GPU poller (a separate pool from the host working set).
- [Android](peak-memory-android.md): `max(VmHWM, VmRSS + VmSwap)` host counter,
  because zram can hold part of the run; CPU-only, no GPU counter.
- [Linux](peak-memory-linux.md): torch-oai Docker uses cgroup v2 `memory.peak`
  and `nvidia-smi`; llama.cpp CPU flavors use `/proc` `VmHWM` (GPU/accelerator
  flavors not implemented).
- [iOS](peak-memory-ios.md): in-process `phys_footprint` poller (unified memory,
  no host/GPU accounting split); reported entirely as `max_host_bytes`, with
  `max_gpu_bytes = null`.

## Result Validity and Comparability

A run is invalid if the workload fails, the required host memory read fails, or
the runtime reports token counts that differ from the benchmark shape. A missing
GPU field is valid only for paths where no GPU probe exists or no matching GPU
sample was observed.

A run that made the device swap is still collected and stored, but it is held back
from the published results, together with every other run of that model and
quantization on that device. That policy covers all benchmarks, not only peak
memory, so it is stated with the selection rules: see
[Selection policies → Swap exclusion](selection-policies.md#swap-exclusion).

To compare `max_memory_usage` results:

- Use the same benchmark ID, model, quantization, runtime, and relevant runtime
  flags.
- Compare byte counts by field, not by a derived cross-platform total.
- Treat `max_gpu_bytes = null` as "unmeasured or unused"; a zero value means a
  probe returned zero and should be audited in platform context.
- On Apple unified memory (iOS, macOS) there is no GPU field to add:
  `max_gpu_bytes` is `null` and `max_host_bytes` (the `phys_footprint`) is the
  whole device-fit number, since it already includes Metal allocations.
- For device fit, read the fields against their own limits, not as a sum. On
  unified-memory systems (Apple Silicon, and Windows UMA where GPU pressure
  overlaps the host) `max_host_bytes` alone is the footprint to fit, since it
  already subsumes the GPU allocations. On systems with a discrete GPU, check
  `max_host_bytes` against system RAM and `max_gpu_bytes` against VRAM
  separately. They are independent limits.
- On Windows UMA, treat `Total Committed` as the WDDM-attributed GPU pressure;
  shared portions can overlap with the host working set.
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
[`pipette_plan_types::benchmark`](../../crates/pipette-plan-types/src/benchmark/mod.rs) and
[`pipette_cli::benchmarks`](../../crates/pipette-cli/src/benchmarks).

The macOS Metal probe and host footprint poller are implemented in
[`pipette-memprobe-metal`](../../crates/pipette-memprobe-metal).

The llama.cpp platform-specific memory runners are implemented under
[`pipette-llamacpp max_memory_usage`](../../crates/pipette-llamacpp/src/execute/max_memory_usage).

The MLX memory runner is implemented in
[`pipette-mlx max_memory_usage`](../../crates/pipette-mlx/src/execute/max_memory_usage.rs).

The torch-oai Docker memory runner and probe are implemented in
[`pipette-torch-oai max_memory_usage`](../../crates/pipette-torch-oai/src/execute/max_memory_usage.rs)
and [`pipette-torch-oai memprobe`](../../crates/pipette-torch-oai/src/memprobe.rs).
