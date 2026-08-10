# Peak Memory: iOS

This is the per-platform companion to the high-level
[peak memory methodology](peak-memory-usage.md). The iOS path runs the benchmark
in process rather than launching a child, which shapes how memory is measured.

## What the number means

iOS `max_memory_usage` answers a single, concrete question: **how much memory does
it take to run this one model on this one runtime, on this device?** It is a
per-(model, runtime, quantization) measure, not a device constant, so the result
is recorded with `model_name`, `model_quant`, `runtime_name`, and `runtime_version`,
and is only comparable against another result with the same identity.

Because the question is "how much to run it," the figure is the **whole-process
physical footprint**, which includes three things:

1. the **model**: weights + KV/activation buffers;
2. the **runtime**: llama.cpp or MLX engine state, Metal driver/command buffers,
   compute scratch;
3. the **harness**: the Pipette app itself, meaning the Swift/SwiftUI runtime, networking,
   logging/telemetry, and framework working set.

**No baseline is subtracted and no separate baseline is recorded.** The reported
figure is the absolute peak footprint (the model, the runtime, and the harness
together), because you cannot run the model without the process it runs in, so that
cost is part of the answer. We deliberately do **not** capture a "harness baseline"
and subtract it, for two reasons:

1. **It's negligible.** Measured on device, the harness floor (the app's footprint
   before any model/runtime work) is **~8 MB**: about 0.1% of a multi-GB run and
   only ~2% even of a 230M model. Subtracting ~8 MB would change nothing meaningful
   while adding a field and a failure mode.
2. **It's part of "how much to run it."** The app is the process the model runs in;
   its cost is real memory the device must hold.

The harness is kept small and is identical across runs, so it's a constant ~8 MB
offset, not a per-model variable, which is what keeps results comparable. (Note:
the larger gap between the reported figure and the runtime's *own* buffer
self-report (tens of MiB to ~190 MiB depending on the model) is **Metal driver /
pipeline / residency state**, i.e. part of the runtime, **not** the harness. See
On-device validation.)

## Counter

iOS is **unified memory with no host/GPU accounting split** (Apple Silicon: Metal
allocations are billed to the process footprint, with no driver carve-out the host
counter cannot see). Per the
[unified-vs-split rule](peak-memory-usage.md#unified-vs-split-memory), the entire
cost is therefore reported as the host figure and there is no separate GPU field:

- `max_host_bytes` = peak process **physical footprint** (`task_vm_info.phys_footprint`),
  the whole-process resident + compressed memory iOS jetsam kills on. This is the
  same class of counter the other host-reporting platforms use (macOS
  `phys_footprint`, Linux `VmHWM`, Android `Max RSS`, Windows `PeakWorkingSetSize`),
  so iOS host numbers are comparable across runtimes and platforms.
- `max_gpu_bytes` = `null`. There is no second pool to fit into; the host footprint
  already subsumes the Metal/MLX allocations.

An in-process `MemoryPeakSampler` polls `phys_footprint` on a background thread
every **20 ms** across the bracket of one model's load, prefill, and single decode
step, and keeps the running max (plus a guaranteed sample at each end of the
bracket). It samples the *current* footprint, **not** the kernel's own exact
high-water mark (`task_vm_info.ledger_phys_footprint_peak`): that counter is a
monotonic, process-lifetime value with no reset API, so in a long-lived in-process
app it would stick at the largest model ever loaded. The peak of a `max_memory` run
is a sustained plateau (the weights stay resident for the whole run), not a
sub-20 ms transient, so polling lands on it reliably. Before the bracket opens, the
footprint is first settled to a clean floor. See
[Cross-cell isolation](#cross-cell-isolation-settle-to-floor).

### The GPU allocator peak is a diagnostic, not a field

The runtime's own GPU-allocator high-water mark (Metal `currentAllocatedSize` for
llama.cpp, `MLX.Memory.peakMemory` for MLX) is a useful breakdown of *how much of
the footprint the GPU runtime holds*, but on unified memory it is a subset of the
host footprint, not a separate capacity dimension, so it is **not** reported as
`max_gpu_bytes`. It remains visible in captured runtime output (llama.cpp's own
`MTL0 ... buffer size` stderr lines; MLX's `[MLXMEM]` `mlxPeak` log) for auditing.

An earlier revision reported that allocator counter *as* the single `max_ram_bytes`.
On-device cross-checks against llama.cpp's own buffer log showed why that
under-counts. The Metal counter matches the sum of llama's Metal (`MTL0`) buffers
to ~1 MiB, so it omits everything else the process holds: llama's own CPU-side
buffers (~210 MiB on an 8B) plus non-buffer Metal driver / pipeline / residency
state (~190 MiB on an 8B). Net, the allocator counter ran **~7% below
`phys_footprint` on the 8B and ~17% below on the 230M** (a larger fraction on
small models, where the fixed overhead is a bigger share), and with no GPU offload
it collapses entirely (see On-device validation). The footprint is the figure a
device must actually fit, so it is the reported `max_host_bytes`.

## Cross-cell isolation (settle-to-floor)

The iOS app runs **every benchmark cell in one long-lived process** (the desktop
paths fork a fresh child per cell and are immune to this). `phys_footprint` is
process-wide, and freed memory is not returned to the OS promptly: in particular
MLX parks dropped weight buffers in its **Metal buffer cache**, and dirty pages
linger until the OS reclaims them. So without care, a small model measured right
after a large one inherits the large one's un-reclaimed footprint as its "peak."

Before it opens the sampling bracket, `max_memory` therefore settles the process to
a clean floor (`ProcessMemory.settleToFloor`):

1. **Drain caches**: `MLX.GPU.clearCache()`, called regardless of the current
   runtime, so a prior MLX cell's buffer cache can't inflate a following llama
   measurement.
2. **Poll to a plateau**: read `phys_footprint` until it stops falling (three
   consecutive ~50 ms samples within 1 MiB) or a 4 s timeout elapses.

The reported figure is **not** reduced by this floor. The absolute peak is still
what jetsam counts. The floor is *logged alongside* the peak (`AppLog.memory`:
`enter=… floor=… peak=…`) so a run whose footprint never fell back to the harness
level (contamination the platform couldn't reclaim in time) is detectable after
the fact.

**On-device demonstration** (iPhone 17 Pro, one process, MLX): running the 8B then
the 230M model back-to-back, the 230M cell **entered at 5756 MB** (the 8B's
un-reclaimed pages), but the gate scrubbed it and it **reported 462 MB**; matching
the 457 MB the same model reports from a clean process. Without the gate it would
have reported ~5.7 GB, ~12× too high. The `memseq` headless diagnostic
(`HeadlessRunner`) drives this sequence for verification:
`headlessrun memseq models=<big>,<small>`.

## On-device validation

Measured on an iPhone 17 Pro (LFM2.5-8B and LFM2.5-230M; MLX 4-bit and llama.cpp
Q4_K_M), sampling every candidate counter across the run:

- **`phys_footprint` is literally the jetsam counter.** `phys_footprint +
  os_proc_available_memory()` was constant at **~6.98 GB** across every run. That
  is the device's per-app kill limit, so `available = limit − phys_footprint`. No
  other counter has this property.
- **The old Metal counter collapses without GPU offload.** With `n_gpu_layers = 0`
  (CPU inference) the weights live in CPU buffers, so Metal `currentAllocatedSize`
  read **0.18 GB for an 8B model that occupied 5.57 GB** (−97%), and **0.46 MB**
  for the 230M (−99.8%). `phys_footprint` reported the true 5.57 GB / 0.27 GB. The
  allocator counter is only ever close when the whole model is GPU-resident; it is
  not a reliable host figure.
- **`resident_size` is not reliable either.** For MLX it *under*-counts (GPU/IOKit
  buffers are not fully resident: 230M `resident` 0.27 GB vs `phys_footprint` 0.48
  GB) and collapses further under memory pressure (the OS compresses pages, which
  `resident_size` excludes but `phys_footprint` still counts); for llama it slightly
  *over*-shoots. `phys_footprint` is the only counter stable across both runtimes,
  GPU/CPU placement, and compression.
- Across runtimes `phys_footprint` sat **above** llama.cpp's *full* self-reported
  buffer total (Metal + CPU); exactly the process-level cost the methodology
  requires `max_host_bytes` to include. The size of that gap is **model-specific**.
  Two measured points, reconciled line-for-line against llama.cpp's own buffer log:
  - **LFM2.5-8B-A1B (MoE), Q4_K_M, llama.cpp:** reported **5473 MiB** vs a buffer
    total of **~5425 MiB** (weights 5114 + KV/RS 24 + compute 286 + output/cache 2)
    → **~48 MiB** of driver/residency + harness overhead.
  - **A dense 8B:** ~190 MiB over the buffer total.
  A staged trace (footprint at cold launch → runtime init → weights → peak)
  attributes the overhead: the **harness floor is only ~8 MB** (7.7–8.5 MB across
  all runs), runtime/Metal init is ~25–40 MB, and the rest is Metal driver /
  pipeline / residency state. So the gap is essentially all **runtime/Metal**, not
  the app: the harness contribution to the reported figure is ~8 MB (0.1–2%), which
  is why it is left in rather than baselined out.

## Workload

The app loads the model fresh, tokenizes a prompt of the requested prefill length,
prefills it, and runs one greedy ignore-end-of-generation decode step. The single
decode step exercises the same first-decode allocations the `--n-gen 1` flag covers
on the CLI paths.

The fresh load is mandatory: unlike the latency and throughput benchmarks, which can
attach to an already-loaded model, max-memory must observe the model-load
allocations, so the reuse-an-open-model entry point rejects `max_memory_usage` and
the caller uses the fresh-load path (`LlamaBenchmark.maxMemory` /
`MLXBenchmark.maxMemory`), which settles the footprint to a clean floor (see
[Cross-cell isolation](#cross-cell-isolation-settle-to-floor)) and then brackets the
entire load + drive with the sampler. One observation is reported, matching the
single-peak rule for `max_memory_usage`.

## Caveats

- This assumes the model is loaded without mmap (the llama path sets
  `use_mmap = false`). With mmap, clean file-backed GGUF pages can be evicted without
  cost and may not all count toward `phys_footprint`, which would understate the
  peak. The counter assumes copied-resident weights.
- The in-process path does not emit the `llama-bench` JSON the CLI paths do, but the
  runtime's stderr (llama.cpp's `model buffer size` / `KV` / `compute buffer` lines)
  is captured and can be audited against the reported figure.

## Code References

- [`LlamaBenchmark.maxMemory`](../../ios/Pipette/Pipette/Runtimes/Llama/LlamaBenchmark.swift)
- [`MLXBenchmark.maxMemory`](../../ios/Pipette/Pipette/Runtimes/MLX/MLXBenchmark.swift)
- [`ProcessMemory` / `MemoryPeakSampler` / `settleToFloor`](../../ios/Pipette/Pipette/Runtimes/ProcessMemory.swift)
- [`HeadlessRunner` `memseq` diagnostic](../../ios/Pipette/Pipette/Headless/HeadlessRunner.swift)
