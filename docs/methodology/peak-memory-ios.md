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
2. the **runtime**: llama.cpp, MLX or ONNX Runtime GenAI engine state, Metal
   driver and command buffers, compute scratch;
3. the **harness**: the Pipette app itself, meaning the Swift/SwiftUI runtime, networking,
   logging/telemetry, and framework working set.

The reported figure is the absolute peak footprint of the model, the runtime and
the harness together. The app records no harness baseline and subtracts none,
for two reasons:

1. **The harness is small.** Measured on the device, the harness floor (the
   app's footprint before any model or runtime work) is about 8 MB. That is
   about 0.1% of a multi-GB run and about 2% of a 230M model.
2. **The harness is part of the cost.** The app is the process that the model
   runs in, so its memory is real memory that the device must hold.

The harness is the same in every run, so it is a constant offset of about 8 MB,
not a per-model variable. The larger gap between the reported figure and the
runtime's own buffer report (tens of MiB to about 190 MiB, by model) is Metal
driver, pipeline and residency state. That state is part of the runtime, not
the harness ([On-device validation](#on-device-validation)).

## Counter

iOS is **unified memory with no host/GPU accounting split** (Apple Silicon: Metal
allocations are billed to the process footprint, with no driver carve-out the host
counter cannot see). Per the
[unified-vs-split rule](peak-memory-usage.md#unified-vs-split-memory), the entire
cost is therefore reported as the host figure and there is no separate GPU field:

- `max_host_bytes` = peak process **physical footprint** (`task_vm_info.phys_footprint`),
  the whole-process resident and compressed memory that iOS jetsam kills on.
  macOS reports the same counter. Compare an iOS figure only with another iOS
  figure of the same model, runtime and quantization.
- `max_gpu_bytes` = `null`. There is no second pool to fit into; the host footprint
  already subsumes the Metal/MLX allocations.

An in-process `MemoryPeakSampler` polls `phys_footprint` on a background thread
every **20 ms** across the bracket of one model's load, prefill, and single decode
step, and keeps the running max (plus a guaranteed sample at each end of the
bracket). It samples the *current* footprint, **not** the kernel's own exact
high-water mark (`task_vm_info.ledger_phys_footprint_peak`): that counter is a
monotonic, process-lifetime value, so in a long-lived in-process app it would stick
at the largest model ever loaded. Before the bracket opens, the footprint is first
settled to a clean floor. See
[Cross-cell isolation](#cross-cell-isolation-settle-to-floor).

The kernel also keeps an *interval* high-water mark that can be reset, but only
through private libproc calls, so only the private-API build (`PIPETTE_PRIVATE_MEMORY`, see
[private-API builds](../pipette-ios/private-thermal-release-build.md)) uses it: it
reset it (`proc_reset_footprint_interval`) as the sampler starts, read
`ri_interval_max_phys_footprint` as it stops, and report the larger of the two peaks.
A failed reset or read leaves the sampled figure. The kernel figure is a guard against
a spike shorter than the poll. A llama.cpp or ONNX peak is a plateau (the weights stay
resident) that polling lands on. On the MLX cells the two agree within 1 MiB at 512
tokens on six MLX models ([memory carry-over](../pipette-ios/memory-carry-over.md)
describes a spike that the kernel figure caught). A public build reports the sampled figure alone (see
[Private-API and public builds](#private-api-and-public-builds)).

ONNX Runtime GenAI is the exception to the footprint-only rule: it maps a package's
external-data file and reads the 4-bit scales from the mapping, pages `phys_footprint`
does not bill. Its cell polls the footprint plus the file-backed pages that became
resident after the bracket opened (see
[the ONNX runtime](../pipette-ios/onnx-genai-runtime.md)).

### The GPU allocator peak is a diagnostic, not a field

The runtime's own GPU-allocator high-water mark (Metal `currentAllocatedSize` for
llama.cpp, `MLX.Memory.peakMemory` for MLX) is a diagnostic, not `max_gpu_bytes`
([unified vs. split memory](peak-memory-usage.md#unified-vs-split-memory)). It stays
visible in captured runtime output for audit: llama.cpp's `MTL0 ... buffer size`
stderr lines, and the `mlxPeak=` field of MLX's memory log line.

As a host figure the allocator counter under-counts. On-device cross-checks against
llama.cpp's own buffer log show why: the Metal counter matches the sum of llama's Metal (`MTL0`) buffers
to ~1 MiB, so it omits everything else the process holds: llama's own CPU-side
buffers (~210 MiB on an 8B) plus non-buffer Metal driver / pipeline / residency
state (~190 MiB on an 8B). Net, the allocator counter runs **~7% below
`phys_footprint` on the 8B and ~17% below on the 230M** (a larger fraction on
small models, where the fixed overhead is a bigger share), and with no GPU offload
it collapses entirely (see On-device validation). The footprint is the figure that a
device must fit, so it is the reported `max_host_bytes`.

## Cross-cell isolation (settle-to-floor)

The iOS app runs **every benchmark cell in one long-lived process** (the desktop
paths fork a fresh child per cell and are immune to this). `phys_footprint` is
process-wide, and freed memory is not returned to the OS promptly: in particular
MLX parks dropped weight buffers in its **Metal buffer cache**, and dirty pages
linger until the OS reclaims them. So without care, a small model measured right
after a large one inherits the large one's un-reclaimed footprint as its "peak."

Before it opens the sampling bracket, `max_memory` therefore settles the process to
a clean floor (`ProcessMemory.settleToFloor`):

1. **Drain caches**: `MLX.Memory.clearCache()`, called regardless of the current
   runtime, so a prior MLX cell's buffer cache can't inflate a following llama
   measurement. The private-API build also flushes libmalloc's reclaim ring (below).
2. **Poll to a plateau**: read `phys_footprint` until it stops falling (three
   consecutive ~50 ms samples within 1 MiB) or a 4 s timeout elapses.

The reported figure is **not** reduced by this floor. The absolute peak is still
what jetsam counts. The floor is *logged alongside* the peak (`AppLog.memory`:
`enter=… floor=… peak=… sampled=… kernel=… fresh=…`) so a run whose footprint never fell
back to the harness level (contamination the platform couldn't reclaim in time) is
detectable after the fact. `fresh=no` means the app had already run a cell since launch
(`ProcessMemory.hasRunEngineWork`).

Three things can still stay counted after a cell ends. The causes, the evidence, and how
to repeat every experiment are in [memory carry-over](../pipette-ios/memory-carry-over.md).

- **Memory llama.cpp kept.** Patches 002 and 003 under `ios/patches/` fix it on every
  build.
- **Freed memory iOS still counts.** iOS 27's allocator parks freed blocks in the task's
  reclaim ring, still counted in `phys_footprint`, until the kernel trims it, and the
  plateau wait cannot tell it from live memory. The private-API build flushes the ring in the
  drain step (`mach_vm_reclaim_ring_flush`, 3 ms at most); a public build cannot.
- **State an engine keeps while the app runs** (Metal pipelines, MLX's device). Nothing
  removes it, so only a cell that runs first in its process is exact.

What a reading is worth after other cells, on each build, is in
[Private-API and public builds](#private-api-and-public-builds). The `diag memseq` headless
diagnostic runs models one after another in one process to check it:
`headlessrun diag memseq models=<big>,<small> prefill=<tokens>` (see
[usage](../pipette-ios/usage.md)).

## Private-API and public builds

The private-API build is this app compiled with `PIPETTE_PRIVATE_MEMORY` (and
`PIPETTE_PRIVATE_THERMAL`); [private-API builds](../pipette-ios/private-thermal-release-build.md)
says how to make it. Apple's App Review rejects such a build, so it runs only
side-loaded on registered development devices. Published iOS results come from
it ([methodology overview](README.md) gives the full explanation). Every other
build is a public build, the App Store and TestFlight builds included. Memory
cells run on both. Two steps need private Apple calls and exist only in the
private-API build:

| What | Private-API build | Public build |
| --- | --- | --- |
| Before the bracket, hand freed memory back to iOS (`ProcessMemory.flushReclaimRing`) | Flushes the reclaim ring | Returns `nil`; nothing happens |
| Read the kernel's exact interval peak (`resetKernelFootprintPeak`, `kernelFootprintPeak`) | Reported when larger than the 20 ms sample | Both return `nil`; `reportedPeak` reports the 20 ms sample |

Everything else (llama.cpp patches 002 and 003, MLX's mlx_lm-shaped prefill, counting
ONNX's mapped weight pages) runs on both builds.

What that means for a reading:

- A cell that runs first in a freshly launched app reads within about 5 MiB on either
  build (the flush also frees a few MiB parked at launch).
- On the private-API build, a cell after other cells reads within about 30 MiB of its
  fresh figure: the flush empties the reclaim ring, and what remains is the engines'
  own live memory.
- A cell that runs after other cells in the same app can read high on a public build, by
  how much depending on what ran before. Its settle step still waits for
  `phys_footprint` to stop falling. But memory parked in the reclaim ring does not fall
  on its own within those three 50 ms polls, so the loop takes it as the floor and the
  next model loads on top of it. On an iPhone 17 Pro at 512-token prompts, with patches
  002 and 003 and no flush, the later cell of each of the seven sequences tested read
  +1 to +272 MiB above its fresh figure: worst right after a Phi-3 ONNX cell (+270,
  +272), next an MLX cell after a llama.cpp cell (+49, +50). An MLX cell after ONNX was
  not tested this way.
- After an ONNX cell, a longer prompt leaves more behind. At 2048 tokens a 350M GGUF cell
  right after Phi-3 ONNX started at 1179 MiB before the flush (307 MiB at 512 tokens), so
  a public build's error there can be larger; that was not measured on a public build.
  After a llama.cpp cell the leftover did not grow with the prompt.
- Losing the kernel peak costs little: the 20 ms sample was within 1 MiB of it on
  every llama.cpp and MLX cell tested, up to 4000-token prompts.

A fleet cell (a cell from the plans that produce published results) whose runtime
asks for `private_thermal`, as the fleet's iOS plans do, asks for the memory calls too
(`private_memory` defaults to `private_thermal`). A public build refuses such a cell
(`UnrunnableClaimError.buildLacksPrivateThermal` or `.buildLacksPrivateMemory` in
`Claim.swift`), so fleet memory numbers come from the private-API build. Local jobs and
headless runs have no such check. Every result records its build:

- The runtime descriptor carries `private_thermal: true` on the private-API build and
  omits the key on a public one. `private_memory` is written only when it differs from
  `private_thermal`.
- The private-API build's `client_version` carries the `-internal` suffix.
- On a device, `headlessrun version` prints `private_memory=on|off`, and each cell's
  `AppLog.memory` line reads `kernel=off` on a public build.

The private-API build needs no warning: its flush keeps a later cell within about 30 MiB
of its fresh figure. A public build warns in the app wherever a Max Memory cell may run
after other work (the wording is in `MemoryFreshness`). It also records on each memory
cell whether the app had already run something (`JobCell.measuredAfterOtherWork`,
written as the cell starts). The mark stays on the phone: a marked result still uploads,
and nothing in it says it may read high. A public build also logs a warning after a cell
with `fresh=no` (`ProcessMemory.maxMemoryBracket`), which a headless run prints on
stderr.

## On-device validation

Measured on an iPhone 17 Pro (`LiquidAI/LFM2.5-8B-A1B` and `LiquidAI/LFM2.5-230M`;
MLX 4-bit and llama.cpp Q4_K_M), sampling every candidate counter across the run:

- **`phys_footprint` is the jetsam counter.** `phys_footprint +
  os_proc_available_memory()` was constant at **~6.98 GB** across every run. That
  is the device's per-app kill limit, so `available = limit − phys_footprint`. No
  other counter has this property.
- **The Metal allocator counter collapses without GPU offload.** With `n_gpu_layers = 0`
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
decode step runs on the prefilled context, as the `-pg <N>,1` test does on the CLI
paths.

The fresh load is mandatory: max-memory must observe the model-load allocations, so
each runtime's `maxMemoryUsage` (`LlamaExecute`, `MlxExecute`, `OnnxGenaiExecute`) loads
the model inside `ProcessMemory.maxMemoryBracket`, which settles the footprint to a clean
floor (see [Cross-cell isolation](#cross-cell-isolation-settle-to-floor)) and then
brackets the entire load + drive with the sampler. One observation is reported, matching the
single-peak rule for `max_memory_usage`.

`vl_max_memory` uses the same bracket around a vision workload: a fresh load of the
model and its projector, `vl_throughput`'s image prompt, and `parameter_decode_tokens`
decode steps, one when the cell sets none
(`LlamaExecute.vlMaxMemory`; see [vl-max-memory.md](vl-max-memory.md#ios)).
`asr_max_memory` uses it around a speech workload: a fresh load of the model and
its audio projector, the clip's audio token count, and one request with
`asr_throughput`'s prompt and `parameter_decode_tokens` samples, ending at the
last sample as the server's request does (`LlamaExecute`, the `.asrMaxMemory`
case; see
[asr-max-memory.md](asr-max-memory.md#ios)).

## Caveats

- This assumes the model is loaded without mmap (the llama path's default load mode is
  `none`, `LlamaCpp.defaultLoadMode`, unless the cell's runtime flags set `load_mode`).
  With mmap, clean file-backed GGUF pages can be evicted without cost and may not all
  count toward `phys_footprint`, which would understate the peak. The counter assumes copied-resident weights.
- iOS's per-app memory limit bounds what a cell can measure. Phi-3 mini on ONNX Runtime
  GenAI reads 2906 MiB at a 512-token prompt and 5161 MiB at 2048; at 4000 tokens iOS
  kills the app mid-cell at that limit (6656 MiB on an iPhone 17 Pro, `per-process-limit`
  in the `JetsamEvent` report), so the cell produces no reading.
- The in-process path does not emit the `llama-bench` JSON the CLI paths do, but the
  runtime's stderr (llama.cpp's `model buffer size` / `KV` / `compute buffer` lines)
  is captured and can be audited against the reported figure.

## Code References

- [`LlamaExecute.maxMemoryUsage`](../../ios/Pipette/Pipette/Runtimes/Llama/Execute/LlamaMaxMemoryUsage.swift)
- [`LlamaExecute.vlMaxMemory`](../../ios/Pipette/Pipette/Runtimes/Llama/Execute/LlamaVlMaxMemory.swift)
- [`MlxExecute.maxMemoryUsage`](../../ios/Pipette/Pipette/Runtimes/Mlx/Execute/MlxMaxMemoryUsage.swift)
- [`OnnxGenaiExecute.maxMemoryUsage`](../../ios/Pipette/Pipette/Runtimes/OnnxGenai/Execute/OnnxGenaiMaxMemoryUsage.swift)
- [`BuildFlavor.hasPrivateMemory`](../../ios/Pipette/Pipette/Config/BuildFlavor.swift)
- [`UnrunnableClaimError.buildLacksPrivateMemory`](../../ios/Pipette/Pipette/Client/Claim.swift)
- [`MemoryFreshness`](../../ios/Pipette/Pipette/Benchmarks/MemoryFreshness.swift)
- [`ProcessMemory` / `MemoryPeakSampler` / `settleToFloor`](../../ios/Pipette/Pipette/Runtimes/ProcessMemory.swift)
- [`HeadlessRunner` `diag memseq` diagnostic](../../ios/Pipette/Pipette/Headless/HeadlessRunner.swift)
