# Peak Memory: macOS

This is the per-platform companion to the
[peak memory methodology](peak-memory-usage.md). It documents how the macOS
paths produce `max_host_bytes`. The same probes serve `pipette-llamacpp`,
`pipette-mlx`, `pipette-onnx-genai` and `pipette-coreai`.

Apple Silicon is unified memory with no host/GPU accounting split, so the whole
cost is `max_host_bytes` and `max_gpu_bytes` is `null`
([unified vs. split memory](peak-memory-usage.md#unified-vs-split-memory)). The
`macos-x64` llama.cpp flavor has no Metal backend and runs on the CPU, so its
host footprint is also its whole cost.

## Host Counter

The host counter polls `proc_pid_rusage(pid, RUSAGE_INFO_V4)` every 20 ms. Each
read takes the larger of `ri_phys_footprint` and
`ri_lifetime_max_phys_footprint`, and the poller keeps the largest read.
`phys_footprint` is the kernel's accounting of the physical memory a process is
responsible for, including Metal allocations on unified-memory hardware. The
lifetime maximum is the kernel's high-water mark, so a model-load peak before
the request stays visible while the process is alive.

The poller reads once more when it is stopped, so a peak after its last tick
still counts. That works for a process that is still alive, such as the MLX
server. It also works for a process that has exited but is not yet reaped: the
kernel keeps an exited process's usage, lifetime maximum included, for
`proc_pid_rusage` until the reap. So `llama-bench`'s runner stops the poller
between the exit and the reap. A peak whose read at the stop did not land fails,
rather than report reads that could have missed the end of the run. The poller
pins the process start time (`ri_proc_start_abstime`, which an exited process
keeps until the reap) and skips a read whose start time differs. So a pid
recycled after the reap never counts as the process.

For `pipette-llamacpp` the poller runs across the whole `llama-bench` child
lifetime. For `pipette-mlx` it starts after the measured server
(`mlx_lm.server`, or `mlx_vlm.server` for `vl_max_memory`) reports healthy, and
still captures the model-load peak because it reads the kernel lifetime
maximum. For `pipette-onnx-genai` it runs from the measured server's launch. For
`pipette-coreai` it starts after the sidecar reports ready, and the lifetime
maximum holds the load. llama.cpp, MLX and ONNX GenAI run a warm-up first; Core
AI does not ([compile caches](peak-memory-usage.md#compile-caches)).

A cell whose engine asks for a GPU's or an NPU's own memory, such as an ONNX
GenAI CUDA or DirectML provider, is refused before it starts: the host
footprint holds what Metal allocates, so it cannot tell whether the run reached
the device. A cell on a device its engine does not name, such as an ONNX GenAI
custom provider, records the host footprint as its whole cost.

## GPU Allocator (diagnostic only, MLX)

On the MLX path a small Metal shim (`peakmtl`) is injected into each memory
server, warm-up and measured, with `DYLD_INSERT_LIBRARIES`. It samples
`[MTLDevice currentAllocatedSize]` every 20 ms, summed across all Metal
devices, and writes its latest peak to a parent-owned file whenever the peak
grows. It also reads `recommendedMaxWorkingSetSize`. On unified memory this
allocator peak is a subset of `phys_footprint`, not a separate pool, so it is
logged as a diagnostic of how much of the footprint the Metal allocator holds
and never reported as `max_gpu_bytes`. The llama.cpp cells do not inject it.

For scale, one llama.cpp measurement injected the shim into `llama-bench`
(`LiquidAI/LFM2-350M-GGUF` Q4_K_M, `b9058` `macos-arm64`, all 17 layers on
Metal). It read a `phys_footprint` peak of 531 to 553 MiB at contexts 256 to 2048,
against an allocator peak of 291 to 375 MiB. The allocator peak ran well below
the footprint, which also holds the process's non-GPU memory, and a constant
0.97 MiB above the runtime's announced `MTL0` buffer sum (Metal driver state
outside `MTLBuffer` accounting) at contexts 256, 1024 and 2048. Another such
run read a constant 1.30 MiB at prompt sizes 256, 2048 and 8192. That constant
is why the 20 ms cadence is enough: a missed peak would show as a delta that
grows with prompt size.

## Workload

llama.cpp runs the shared `llama-bench ... -pg <N>,1` command
([the llama-bench shape](peak-memory-usage.md#the-llama-bench-shape)). The MLX
path drives the equivalent work through a `/v1/completions` call with an
`<N>`-token prompt and `max_tokens = 1`.

## Shader Cache

The warm-up rule and its reasoning live in
[compile caches](peak-memory-usage.md#compile-caches). This section records
the macOS cache and what it measured.

llama.cpp's Metal builds (`macos-arm64`, `macos-arm64-kleidiai`) carry their
shaders as source and compile them when the model loads, and Metal keeps the
compiled code in the user's cache (`com.apple.metal` under
`getconf DARWIN_USER_CACHE_DIR`; a tool with no bundle id shares the top-level
one). On an empty cache the compile adds to the tool's footprint and takes
about 17 s. `macos-x64` ships no Metal backend and compiles nothing, but warms
up too, as every flavor does.

On an Apple M3 Ultra with llama.cpp b11109 `macos-arm64`, without the warm-up
run, with the cache moved aside and in place:

| Cell, `-ngl` | `max_host_bytes`, empty cache | `max_host_bytes`, filled cache |
| --- | ---: | ---: |
| `max_memory_usage_512`, 99 | 776 to 777 MiB | 755 to 757 MiB |
| `max_memory_usage_512`, 0 | 768 to 770 MiB | 747 to 749 MiB |
| `vl_max_memory_smoke`, 99 | 958 to 960 MiB | 948 to 951 MiB |

Three runs of each, with `LiquidAI/LFM2.5-350M-GGUF` Q4_K_M and
`LiquidAI/LFM2.5-VL-450M-GGUF` Q4_0 with its F16 projector. The gap is smaller
than on Linux or Windows, about 21 MiB for the text cell and 9 MiB for the VL
one, and every run on an empty cache read above every run on a filled one. With
the warm-up run, an empty cache reads 753 to 755 MiB for the text cell and
949 MiB for the VL one, two runs each, as a filled one does.

### MLX

MLX ships most of its kernels prebuilt in `mlx.metallib`, and Metal builds
their GPU code into the same top-level `com.apple.metal` cache. Its servers are
processes with no bundle id, like `llama-bench`. Each MLX memory cell warms up
on a server of its own ([compile caches](peak-memory-usage.md#compile-caches)).

With MLX 0.32.2 (text, `LiquidAI/LFM2.5-350M-MLX-4bit`) and `mlx-vlm` 0.7.4
(VL, `mlx-community/LFM2-VL-450M-8bit`), without the warm-up server, with
`com.apple.metal` and `com.apple.metalfe` removed and with them filled by an
earlier run:

| Chip | Cell | `max_host_bytes`, kernels to compile | `max_host_bytes`, all cached |
| --- | --- | ---: | ---: |
| M3 Max | `max_memory_usage_512` | 894 to 1129 MiB | 1074 to 1075 MiB |
| M3 Max | `vl_max_memory_smoke` | 1218 to 1222 MiB | 1232 to 1234 MiB |
| M3 Ultra | `max_memory_usage_512` | 1084 to 1120 MiB | 1198 to 1244 MiB |
| M3 Ultra | `vl_max_memory_smoke` | 1404 to 1409 MiB | 1430 to 1431 MiB |

Unlike llama.cpp's, MLX's figure with kernels to compile is not simply higher.
On the M3 Max the text cell's six runs read 894, 912, 996, 999, 1031 and
1129 MiB, three on an empty cache and three on a partly filled one, and its
measured request took 0.5 to 2.4 s against 0.1 to 0.2 s once cached. The
compile itself runs in Metal's compiler service, outside the measured process.

With the warm-up server, an empty cache reads 1074 to 1076 MiB for the text
cell and 1231 to 1232 MiB for the VL one on the M3 Max, three runs each, as a
filled one does. On the M3 Ultra it reads 1236 to 1277 MiB and 1417 to
1432 MiB, against 1211 to 1241 MiB and 1431 to 1435 MiB filled. Every measured
request there took 0.05 to 0.07 s after a warm-up request of 1.5 to 2.4 s, so
none compiled, and warm-ups on a filled cache spread as widely (1230 to
1241 MiB text, 1412 to 1431 MiB VL). The M3 Ultra's figure moves by up to about
45 MiB run to run with every kernel cached, so compare several runs there.
Each server logs its request time (`request` on `peak-memory server
measured`): seconds rather than a tenth of one mean it still compiled kernels.

macOS MLX figures read hundreds of MiB above the iPhone's for the same models,
for reasons [memory carry-over](../pipette-ios/memory-carry-over.md#mlx-on-macos-is-not-a-reference-for-the-phone)
accounts for, so they are no reference for iOS.

## Caveats

- Injection can be blocked. Hardened Runtime, Library Validation, or SIP on the
  target binary each suppress `DYLD_INSERT_LIBRARIES`, and the shim then
  produces no output. The reported result is unaffected, since
  `max_host_bytes` comes from the host poller; only the GPU-allocator
  diagnostic is lost.
- The probe sums `currentAllocatedSize` across all Metal devices; a multi-GPU
  Mac reports the combined peak.

## Code References

- [`pipette-llamacpp max_memory_usage`](../../crates/pipette-llamacpp/src/execute/max_memory_usage.rs)
- [`pipette-memprobe`](../../crates/pipette-memprobe) (the `peakmtl` shim and the `phys_footprint` sampler)
- [`pipette-mlx max_memory_usage`](../../crates/pipette-mlx/src/execute/max_memory_usage.rs)
- [`pipette-mlx vl_max_memory`](../../crates/pipette-mlx/src/execute/vl_max_memory.rs)
- [`pipette-onnx-genai max_memory_usage`](../../crates/pipette-onnx-genai/src/execute/max_memory_usage.rs)
- [`pipette-coreai max_memory_usage`](../../crates/pipette-coreai/src/execute/max_memory_usage.rs)
