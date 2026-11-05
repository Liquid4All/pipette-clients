# Peak Memory: Windows

This is the per-platform companion to the high-level
[peak memory methodology](peak-memory-usage.md). It documents how the Windows
paths produce `max_host_bytes`, `max_gpu_bytes` and `max_npu_bytes`: llama.cpp's
`llama-bench`, and the OVMS, OpenVINO and ONNX GenAI servers.

## Host Counter

The host counter reads `PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize` through
the Process Status API (PSAPI), from a process handle held open from the launch. This is the kernel's
lifetime peak working set for the process, so it is read once rather than
polled: after a `llama-bench` child exits, or before a server is stopped. It
covers the binary, DLLs, heap, stack, host-visible mirror pages, model load and
the cell's workload.

There is no swap term (`observation_max_swap_bytes`). The working set leaves out
the pages Windows trimmed, compressed or wrote to the page file, but no
per-process counter reads those pages apart. The nearest one, private commit
less the private working set (`PROCESS_MEMORY_COUNTERS_EX2`), also counts
memory the process committed and never touched and, on an integrated GPU, the
driver's allocations. On an unpressured Ryzen AI 9 HX 370 with 28 GB of RAM
and 22 GB free (Windows 11 build 26200, llama.cpp b10516 Vulkan, a 2.4 GiB
Q4_0 model, a 512-token prompt), it read 1455 MiB at `-ngl 0` and 3520 MiB at
`-ngl 99`, against peak working sets of 2685 and 652 MiB, so a run that swapped
could not be told from one that did not.

## GPU and NPU Counter

A run on a GPU or an NPU starts a Performance Data Helper (PDH) poller that samples
`\GPU Process Memory(pid_<PID>_*)\Total Committed` every 20 ms. `Total
Committed` is the per-sample total that the Windows Display Driver Model (WDDM)
bills the process on one adapter
(dedicated plus shared). It is used instead of summing the independent peaks of
`Dedicated Usage` and `Shared Usage`, which would overstate the peak when those
two counters reach their maxima at different moments.

The counter has one instance per process and adapter,
`pid_<PID>_luid_0x<HIGH>_0x<LOW>_phys_<N>`, and an NPU is one of those adapters:
WDDM drives it as a compute-only adapter and bills its memory to the same
counter. So each sample sums the instances by the kind of adapter the LUID
names, read once per adapter from `D3DKMT_ADAPTERTYPE`:

- `ComputeOnly` set: an NPU, into `max_npu_bytes`. Intel's NPU also sets
  `RenderSupported`, so that bit cannot tell it apart.
- `SoftwareDevice` set: the Basic Render Driver, a CPU rasterizer, left out.
- Anything else, or an adapter that cannot be asked: a GPU, into
  `max_gpu_bytes`.

Each field is the largest per-sample sum of its kind, so the two can peak at
different moments.

pipette uses PDH rather than an injected GPU-API layer, because PDH is GPU-API agnostic: one
counter path covers Vulkan, HIP, SYCL, CUDA, OpenCL and an NPU's driver without
per-runtime injection. The trade-off is that PDH is an OS-attribution counter and
includes WDDM driver state the GPU API never exposes (see Reference
Measurements).

The PDH instance for the process does not exist until the child makes its first
device allocation, so the earliest poll samples return no data and are skipped.
The poller samples once more when it is stopped, so a server's last interval
counts; a program that has exited by then has no instance left to sample.

Which device a run must show memory on follows from the cell:

| Path | GPU and NPU |
| --- | --- |
| llama.cpp GPU flavors (Vulkan, HIP, SYCL, CUDA, OpenCL on Adreno) | the GPU must show memory |
| llama.cpp's OpenVINO backend | `GGML_OPENVINO_DEVICE` unset or `CPU`: not read, both fields `null`; any other device is refused before the tool starts |
| OVMS and OpenVINO, `gpu` / `npu` | the GPU / the NPU must show memory |
| OVMS and OpenVINO, a custom device (`AUTO`, `HETERO:...`, `GPU.1`) | whatever it held, none required |
| ONNX GenAI, CUDA and DirectML | the GPU must show memory |
| ONNX GenAI, WinML or a custom provider | whatever it held, none required |
| CPU flavors, devices and providers | not read; both fields `null` |

A device that must show memory and shows none fails the cell, as does a PDH
query that cannot be set up: the result would otherwise be a host figure filed
under a device run. Any field the run held nothing in is `null`, never zero.
The memory a run holds on the other kind of device is recorded too: OVMS keeps
about 22 MiB on the iGPU whichever device it serves from.

## Host and device pools

Whether a device's bytes also sit in the host working set is the driver's
choice, so the fields are reported apart and never summed:

- On an AMD Radeon 890M under Vulkan, most of the GPU bytes are outside the
  working set (below: 153 MiB working set against 376 MiB of GPU).
- On the Intel Arc B390 iGPU and NPU measured, the device bytes are inside it:
  during each run the box's free memory fell by about the working set (within
  12 MiB), far less than the working set plus the device bytes (below).

Check `max_host_bytes` against system RAM for fit. On a discrete GPU, check
`max_gpu_bytes` against VRAM separately. On an AMD iGPU most of `max_gpu_bytes`
needs room on top of `max_host_bytes`. Adding the fields overstates an Intel
run.

## Workload

llama.cpp runs [the llama-bench shape](peak-memory-usage.md#the-llama-bench-shape).
The servers (OVMS, OpenVINO and ONNX GenAI) take one request at the cell's
prompt length after the model load. Every engine on this page runs a warm-up
first ([compile caches](peak-memory-usage.md#compile-caches)).

## Shader cache

A llama.cpp GPU flavor compiles its kernels the first time a machine runs them,
into a cache of the run's user: the AMD driver's Vulkan pipeline cache is
`%LOCALAPPDATA%\AMD\VkCache`. The warm-up fills that cache before the measured
run ([compile caches](peak-memory-usage.md#compile-caches)). The VL cell's
numbers are in [VL peak memory](vl-max-memory.md#warm-up-server).

`max_memory_usage_512` on an AMD Radeon 890M (driver 32.0.31036.15) with
llama.cpp b11109 and `LiquidAI/LFM2.5-350M-GGUF` Q4_K_M, without the warm-up run,
with `VkCache` empty and with it filled by an earlier run of the same cell, three
runs each:

| Flavor, `-ngl` | `max_host_bytes`, empty cache | `max_host_bytes`, filled cache | `max_gpu_bytes`, either |
| --- | ---: | ---: | ---: |
| `windows-x64-vulkan`, 99 | 187 to 189 MiB | 146 MiB | 489 MiB |
| `windows-x64-vulkan`, 0 | 347 to 348 MiB | 304 MiB | 440 MiB |
| `windows-x64-cpu` | 294 MiB | 294 MiB | `null` |

The GPU figure did not move. An empty cache took 5 s against 3 s, and left one
512 KiB file in `VkCache`. With the run first, an empty cache reads as a filled
one: 146 and 304 MiB for the two Vulkan rows, two runs each, with the same GPU
figures.

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

`OpenVINO/LFM2.5-350M-int8-ov`, `max_memory_usage_512`, on an Intel Core Ultra
X7 358H (Arc B390 iGPU, Intel NPU driver 32.0.100.4509), OpenVINO 2026.2.1. The
last column is how far the box's free physical memory fell during the run,
sampled every 100 ms. OVMS's figures on the same box are in
[OVMS](../ovms.md#max_memory_usage).

| Device | `max_host_bytes` | `max_gpu_bytes` | `max_npu_bytes` | Free memory fell by |
| --- | ---: | ---: | ---: | ---: |
| `cpu` | 1,006 MiB | `null` | `null` | 1,003 MiB |
| `gpu` | 1,017 MiB | 428 MiB | `null` | 1,029 MiB |
| `npu` | 1,037 MiB | `null` | 769 MiB | 929 MiB |

On both engines the NPU's host figure moves between runs with no change in code
or cache, while its NPU figure is the same in every run
([OpenVINO](../openvino-measurement.md), [OVMS](../ovms.md#max_memory_usage)).

## Caveats

- The WDDM driver-state gap is strictly per-process. When two processes share
  the GPU, each pays its full driver-state overhead independently; there is no
  cross-process amortization. Two concurrent `llama-bench` runs on the Radeon
  890M each read a 375.96 MiB peak, and the two summed exactly to the adapter's
  rise ([pdh-multiproc-exp](../../tools/pdh-multiproc-exp/README.md)).
- `Total Committed` is committed memory, not resident memory. On the Intel NPU
  a compiled model is committed before its first inference makes it resident,
  so the NPU figure is the most the process reserved there.
- Telling an NPU by `ComputeOnly` also files a GPU running as a compute-only
  (MCDM) adapter, such as a datacenter card, under `max_npu_bytes`, so a GPU
  cell on one fails with an error that names this case. No such device runs
  these cells.
- An NPU's memory reaches PDH only if its driver allocates through WDDM. Intel's
  does. AMD's XDNA and Qualcomm's Hexagon are unchecked; their cells run through
  an ONNX GenAI custom provider, which requires no device memory, so if such a
  driver bills nothing the cell records the host figure with `max_npu_bytes`
  `null`.

## Code References

- [`pipette-llamacpp max_memory_usage`](../../crates/pipette-llamacpp/src/execute/max_memory_usage.rs)
- [`pipette-ovms max_memory_usage`](../../crates/pipette-ovms/src/execute/max_memory_usage.rs),
  [`pipette-openvino max_memory_usage`](../../crates/pipette-openvino/src/execute/max_memory_usage.rs),
  [`pipette-onnx-genai max_memory_usage`](../../crates/pipette-onnx-genai/src/execute/max_memory_usage.rs)
- [`pipette-memprobe`'s Windows sampler](../../crates/pipette-memprobe/src/windows.rs) and its [PDH poller](../../crates/pipette-memprobe/src/pdh.rs)
