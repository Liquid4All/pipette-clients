# Peak Memory: Windows

This is the per-platform companion to the high-level
[peak memory methodology](peak-memory-usage.md). It documents how the Windows
`pipette-llamacpp` path produces `max_host_bytes` and `max_gpu_bytes`.

## Host Counter

The host counter reads `PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize` through
PSAPI, from a process handle held open across the child's exit. This is the
kernel's lifetime peak working set for the process, so it is read once after the
child returns rather than polled. It covers the binary, DLLs, heap, stack,
host-visible mirror pages, model load, prefill, and the single decode step.

## GPU Counter

GPU flavors run a PDH poller that samples
`\GPU Process Memory(pid_<PID>_*)\Total Committed` every 20 ms and tracks the
maximum. `Total Committed` is the per-sample joint GPU total reported by WDDM
(dedicated plus shared). It is used instead of summing the independent peaks of
`Dedicated Usage` and `Shared Usage`, which would overstate the peak when those
two counters reach their maxima at different moments.

PDH is chosen over an injected GPU-API layer because it is GPU-API agnostic: one
counter path covers Vulkan, HIP, SYCL, CUDA, and OpenCL without per-runtime
injection. The trade-off is that PDH is an OS-attribution counter and includes
WDDM driver state the GPU API never exposes (see Reference Measurements).

The PDH instance for the process does not exist until the child makes its first
GPU allocation, so the earliest poll samples return no data and are skipped.
CPU flavors never start the PDH poller and report `max_gpu_bytes` as `null`.

## Host and GPU are separate pools

PSAPI `PeakWorkingSetSize` does not count GPU-driver-managed memory, so
`max_host_bytes` and `max_gpu_bytes` measure different memory here. On a discrete
GPU they are disjoint pools. Check host against system RAM and GPU against VRAM
separately. On AMD UMA the shared component of `Total Committed` can overlap host
working-set pages; an operator wanting a precise unified total should parse the
runtime's announced breakdown from captured stderr rather than adding the two
fields.

## Workload

`llama-bench --output json --n-prompt <N> --n-gen 1 -r 1`, the same shape as the
other llama.cpp paths.

## Reference Measurements

`LiquidAI/LFM2-350M-GGUF` (Q4_K_M), llama.cpp `b9058` win-vulkan-x64, on an
AMD Ryzen AI 9 HX 370 with a Radeon 890M iGPU (AMDVLK driver 32.0.23002.1006),
all layers offloaded to `Vulkan0`. The announced-buffers column is the runtime's
own reported Vulkan buffer sum.

| Ctx  | `max_host_bytes` (`PeakWorkingSetSize`) | `max_gpu_bytes` (PDH `Total Committed`) | Announced Vulkan buffers |
| ---: | ---: | ---: | ---: |
| 256  | 152.85 MiB | 375.95 MiB | 342.75 MiB |
| 1024 | 159.51 MiB | 458.27 MiB | 423.24 MiB |
| 2048 | 158.04 MiB | 481.68 MiB | 437.24 MiB |

`max_gpu_bytes` runs roughly +33 to +44 MiB above the announced Vulkan sum
across this range, and the gap grows with the number and size of allocations.
That gap is WDDM driver state the OS bills to the process but the Vulkan API
does not expose: command-buffer pools, descriptor heaps, paging-table entries,
and residency scratch. Because the gap grows with allocation count, track a
regression by diffing `max_gpu_bytes` against itself, not against the announced
sum.

## Caveats

- The WDDM driver-state gap is strictly per-process. When two processes share
  the GPU, each pays its full driver-state overhead independently; there is no
  cross-process amortization.
- PDH setup or data-collection failures are logged and report a measured `0`
  for the GPU channel. Treat a Windows result with `max_gpu_bytes = 0` as an
  audit case: inspect the captured logs before reading it as a real
  no-allocation result.

## Code References

- [`pipette-llamacpp max_memory_usage::windows`](../../crates/pipette-llamacpp/src/execute/max_memory_usage/windows.rs)
- [`pipette-llamacpp max_memory_usage::pdh_poller`](../../crates/pipette-llamacpp/src/execute/max_memory_usage/pdh_poller.rs)
