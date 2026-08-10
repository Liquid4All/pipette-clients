# Peak Memory: macOS

This is the per-platform companion to the high-level
[peak memory methodology](peak-memory-usage.md). It documents how the macOS
paths produce `max_host_bytes`. The same probes serve `pipette-llamacpp` and
`pipette-mlx`.

macOS targets are Apple Silicon (M1–M5) only: **unified memory with no
host/GPU accounting split**. Per the
[unified-vs-split rule](peak-memory-usage.md#unified-vs-split-memory), the whole
cost is reported as `max_host_bytes` and `max_gpu_bytes` is `null`; the Metal
allocator peak is captured as a diagnostic, not a reported field.

## Host Counter

The host counter polls `proc_pid_rusage(pid, RUSAGE_INFO_V4)` and tracks the
peak `ri_phys_footprint`. `phys_footprint` is the kernel's accounting of the
physical memory a process is responsible for, including Metal allocations on
unified-memory hardware. The poller samples every 20 ms and keeps a running
maximum; the kernel also maintains a lifetime high-water mark, so a model-load
peak that occurred before the request stays visible while the process is alive.

For `pipette-llamacpp` the poller runs across the whole `llama-bench` child
lifetime. For `pipette-mlx` the poller starts after the Python sidecar reports
readiness, but because it reads the kernel lifetime maximum it still captures
the earlier model-load peak.

## GPU Allocator (diagnostic only)

A small Metal shim (`peakmtl`) is injected into the child process with
`DYLD_INSERT_LIBRARIES`. It samples `[MTLDevice currentAllocatedSize]` every
20 ms, summed across all Metal devices, and writes its latest peak to a
parent-owned file whenever the peak grows. On unified memory this allocator peak
is **a subset of `phys_footprint`**, not a separate pool, so it is **not**
reported as `max_gpu_bytes` (which stays `null`). It is logged as a diagnostic:
useful for seeing how much of the host footprint the Metal allocator holds, and
for the cross-check against the runtime's announced buffers below.

The shim additionally reads `recommendedMaxWorkingSetSize` for diagnostics.

## Why no `max_gpu_bytes`

Apple Silicon is unified memory: Metal allocations are billed to the process
`phys_footprint`, so there is no separate GPU pool to fit into. Reporting the
allocator peak as `max_gpu_bytes` would label a subset of the host footprint as
if it were a second capacity dimension. Per the unified-vs-split rule, the host
footprint is the single device-fit number and `max_gpu_bytes = null`.

## Workload

`llama-bench --output json --n-prompt <N> --n-gen 1 -r 1`. The `--n-gen 1`
forces one decode step so the measurement covers the kernels, sampling buffers,
and command queues allocated on the first decode, not just model load and
prefill. The MLX path drives the equivalent work through the sidecar's
`/max_memory_usage` endpoint with `prompt_tokens = <N>` and `decode_tokens = 1`.

## Reference Measurements

`LiquidAI/LFM2-350M-GGUF` (Q4_K_M, 350M parameters), llama.cpp `b9058`
macos-arm64, Metal backend with all 17 layers offloaded, on a production
benchmark host. Only `max_host_bytes` is reported; the Metal allocator peak and
the runtime's announced `MTL0` buffer sum are shown as diagnostics.

| Ctx  | `max_host_bytes` (`phys_footprint`, reported) | Metal allocator peak (diagnostic) | Announced `MTL0` buffers |
| ---: | ---: | ---: | ---: |
| 256  | 530.69 MiB | 290.71 MiB | 289.74 MiB |
| 1024 | 536.66 MiB | 371.47 MiB | 370.50 MiB |
| 2048 | 552.71 MiB | 375.47 MiB | 374.50 MiB |

The Metal allocator peak runs a constant ~1.0 MiB above the runtime's announced
buffer sum (Metal driver state (command queue, residency-set scratch,
pipeline-state objects) outside `MTLBuffer` accounting but inside the
allocator's reach), and sits well below `phys_footprint` (which also carries the
process's non-GPU memory). `max_gpu_bytes` itself is `null`; these columns are
diagnostics for auditing where the host footprint goes.

## Caveats

- Injection can be blocked. Hardened Runtime, Library Validation, or SIP on the
  target binary each suppress `DYLD_INSERT_LIBRARIES`, and the shim then produces
  no output. The reported result is unaffected (`max_host_bytes` comes from the
  host poller and `max_gpu_bytes` is `null` regardless) only the GPU-allocator
  diagnostic is lost.
- The probe sums `currentAllocatedSize` across all Metal devices; a multi-GPU
  Mac reports the combined peak.

## Code References

- [`pipette-llamacpp max_memory_usage::macos`](../../crates/pipette-llamacpp/src/execute/max_memory_usage/macos.rs)
- [`pipette-memprobe-metal`](../../crates/pipette-memprobe-metal) (the `peakmtl` shim and host poller)
- [`pipette-mlx max_memory_usage`](../../crates/pipette-mlx/src/execute/max_memory_usage.rs)
