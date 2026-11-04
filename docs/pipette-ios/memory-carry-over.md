# Memory carry-over between cells on iOS

The iOS app runs every benchmark cell in one process, so memory that one cell leaves
behind can show up in the next cell's peak. This page records what causes it, what the
app does about it, the measurements, and how to repeat them. How the reading is defined,
the settle-to-floor step, and what each build's reading is worth are in
[peak-memory-ios.md](../methodology/peak-memory-ios.md).

## Where things stand

| Build | First cell after the app starts | A later cell in the same app |
| --- | --- | --- |
| [Private-API build](private-thermal-release-build.md) (`PIPETTE_PRIVATE_MEMORY`) | Exact | Within -24 to +32 MiB of its fresh value |
| Public (App Store, TestFlight, any build without the flag) | Within about 5 MiB of a private-API build | +1 to +272 MiB high at 512 tokens, worst right after an ONNX cell; possibly more at longer prompts |

On a build without the fixes, a 350M GGUF cell read 2778 MiB after a granite 3B cell,
against 417 MiB alone on the same build. `vl_max_memory` uses the same bracket, but its
carry-over has not been measured. A public build cannot flush freed memory, so it warns
instead (see
[private-API and public builds](../methodology/peak-memory-ios.md#private-api-and-public-builds)).

The fixes came in PR #1517, merged 2026-09-28. Readings from builds without it are not
comparable with later ones; a `client_version` dated before `2026.09.28` predates it.
Builds with the fixes also carry two single-cell fixes, which correct the reading of
a cell measured alone, not carry-over:

- MLX prefills in `mlx_lm`'s shape.
- ONNX counts its mapped weight pages.

Together, GGUF moved -40 to -190 MiB, MLX -90 to -285, ONNX +460.

## Test setup behind the numbers

Unless a row says otherwise: an iPhone 17 Pro on iOS 27.0 (24A437), a private-API Release
build with the app's `increased-memory-limit` entitlement (per-app limit 6656 MiB),
`max_memory_usage_512`, MiB of `phys_footprint`, and mostly two runs per figure.
Current-build runs repeat within 1 to 2 MiB.

| On-device name (`diag memseq models=`) | Source |
| --- | --- |
| `LFM2.5-350M-Q4_K_M.gguf` | `gguf-text://repo=LiquidAI/LFM2.5-350M-GGUF&path=LFM2.5-350M-Q4_K_M.gguf` |
| `granite-4.2-3b-Q4_K_M.gguf` | `gguf-text://repo=ibm-granite/granite-4.2-3b-GGUF&path=granite-4.2-3b-Q4_K_M.gguf&rev=47a3d9699d7539606c83943d717fcea7bd9f6a19` |
| `LFM2.5-8B-A1B-Q4_K_M.gguf` | `gguf-text://repo=LiquidAI/LFM2.5-8B-A1B-GGUF&path=LFM2.5-8B-A1B-Q4_K_M.gguf` |
| `cpu-int4-rtn-block-32` (Phi-3 mini) | `onnx-genai://repo=microsoft/Phi-3-mini-4k-instruct-onnx&prefix=cpu_and_mobile/cpu-int4-rtn-block-32&rev=5f5f794c1c23c9d5ee142af85df02a6cc52d6945` |
| The repository's name, such as `LFM2.5-350M-MLX-4bit` | `mlx://repo=` with six MLX 4-bit models at these revisions: `LiquidAI/LFM2.5-350M-MLX-4bit` `f6cb4e00`, `LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit` `7ccafdb0`, `mlx-community/Llama-3.2-1B-Instruct-4bit` `08231374`, `mlx-community/Qwen3-1.7B-4bit` `3b1b1768`, `LiquidAI/LFM2.5-2.6B-MLX-4bit` `04efa237`, `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` `2e92b640` |

## Causes

Three different things stay in `phys_footprint` after a cell ends, and each needed its
own answer.

### Memory an engine never gave back

On iOS, ggml took its host buffers (`ggml_aligned_malloc`) and the host side of shared
Metal buffers from `posix_memalign` and returned them with `free()`; macOS uses
`vm_allocate`. After `llama_model_free` the footprint stayed at the model's peak (2506
MiB for granite 3B), and the settle step accepted that flat line as a clean floor within
about 150 ms. Patch 002 (`ios/patches/002-ggml-metal-vm-allocate-ios.patch`) takes both
from `vm_allocate` on iOS too: after the free, granite drops to 296 to 298 MiB and the
350M to 126 to 136, and granite then 350M reads 432 to 639 instead of 2778. Upstream master
(`4b1a27fa0`, 2026-09-25) still allocates this way.

Ruled out on the way: Metal residency sets (`GGML_METAL_NO_RESIDENCY=1`,
`GGML_METAL_RESIDENCY_KEEP_ALIVE_S=1` changed nothing), and private Metal buffers
(`GGML_METAL_SHARED_BUFFERS_DISABLE=1` released the memory but added 140 MiB to every
fresh reading).

MLX hands dropped weights to its Metal buffer cache. `MLX.Memory.clearCache()` in the
settle step releases it: an LFM2.5-230M cell that entered at 5756 MB after the 8B
reported 462 MB, against 457 fresh.

### Freed memory that iOS still counts

iOS 27's allocator (xzone malloc) defers returning freed memory. Freed blocks go into the
task's reclaim ring, still counted in `phys_footprint`, until the kernel trims the ring,
a little at a time. The evidence:

- Launching with `MallocReportConfig=1 MallocDebugReport=stderr` prints the allocator's
  config: `Defer Tiny/Small/Large` 1, `Huge Cache Size` 16, `Huge Cache Max Entry Bytes`
  4294967295. So a freed 2.5 GB model can sit in the cache whole; a 5.4 GB one cannot.
- Taking memory from `vm_allocate` (patch 002) bypasses the allocator, and that alone
  removed 2.5 GB of retention.
- One engine's frees release another engine's leftover: when the ring fills, the
  allocator resizes it and the kernel first reclaims every entry.
- A probe build that freed 64 MiB still counted 86.7 MB, 75 MB of it reclaimable in the
  ring. Flushing the ring took the footprint to 19.3 MB in 0.7 ms. 256 freed 1 MiB blocks
  went from 276 to 20.3 MB in 2.8 ms.

What was parked there:

| Source | Parked after the cell | Fix |
| --- | --- | --- |
| llama.cpp's loader, one heap vector per weight tensor | 56 MiB (350M), 221 MiB (granite 3B) | Patch 003 stages every weight through one buffer of at most 16 MiB |
| llama.cpp heap | 13 to 19 MiB | Private-API-build flush |
| ONNX Runtime | about 300 MiB at a 512-token prompt, 1160 to 1180 MiB at 2048 | Private-API-build flush |

The settle step cannot see this memory: it waits for the footprint to stop falling
(three 50 ms polls within 1 MiB), and parked memory does not fall on its own in that
time. Nor is it memory pressure: the kernel drains the ring before it enforces the
footprint limit (per the published kernel source), so jetsam does not act on it. A
public build's high reading overstates what the model needs, not how close the app is
to being killed.

A private-API build empties the ring in the drain step through three private calls,
`mach_vm_reclaim_get_rings_for_task`, `mach_vm_reclaim_ring_capacity` and
`mach_vm_reclaim_ring_flush` (`ProcessMemory.flushReclaimRing`, 3 ms at most). They are
in the iOS 27.0 SDK's `usr/lib/libSystem.B.tbd` and resolved with `dlsym`, and the kernel
honors the flush only for the calling task.

The published sources lag the device. `libmalloc-812.100.31` ships iOS with deferral and
the huge cache off, while `xnu-12377.121.6` turns the ring on for iOS (1 s sampling, 1%
autotrim). Trust the config dump on the device.

### State an engine keeps for the life of the process

Some memory is live, not freed, and nothing short of a new process removes it:

- The ggml Metal device, about 14 MiB, created by `llama_backend_init`, with no teardown.
- A shared Metal driver pool, about 22 to 25 MB in the graphics ledger.
- MLX's device with its compiled pipelines, about 10 MB.

A fresh process pays this inside its first bracket and a warm one already has it, so a
later cell of the same engine reads slightly low. A different engine cannot reuse it, so
it reads high. This is the band that remains on a private-API build:

| Earlier cell, then measured cell | Later cell minus its fresh value |
| --- | --- |
| llama.cpp, then llama.cpp | -16 to -24 |
| llama.cpp, then MLX | +27 to +32 |
| llama.cpp, then ONNX | +14 to +19 |
| MLX, then llama.cpp | +6 to +9 |
| MLX, then MLX | -22 to +12 |
| ONNX, then llama.cpp | -5 to -10 |
| ONNX, then MLX | +5, +6 |

## What the app does

| Step | Builds | Code |
| --- | --- | --- |
| Drain MLX's buffer cache before each bracket | All | `ProcessMemory.drainCaches` |
| Flush the reclaim ring before each bracket | Private-API | `ProcessMemory.flushReclaimRing` |
| Wait for the footprint to stop falling (3 polls of 50 ms within 1 MiB, 4 s at most) | All | `ProcessMemory.settleToFloor` |
| llama.cpp host buffers from `vm_allocate` (patch 002) | All | `ios/patches/`, applied by `ios/build-llama.sh` |
| llama.cpp weights staged through one buffer of at most 16 MiB (patch 003) | All | same |
| Report the kernel's exact interval peak when it is larger than the 20 ms sample | Private-API | `resetKernelFootprintPeak`, `kernelFootprintPeak` |
| Log `enter= floor= peak= sampled= kernel= fresh=` for each memory cell | All | `ProcessMemory.maxMemoryBracket` |
| Record whether the app had already run a cell, and warn in the UI | Public | `MemoryFreshness`, `JobCell.measuredAfterOtherWork` |

The kernel peak is a guard: before the MLX cell moved to `mlx_lm`'s prefill shape, a
20 ms poll missed the LFM2.5-350M 4-bit prefill spike by up to 110 MiB at 4096 tokens
against a 1 ms poll. llama.cpp and ONNX peaks are plateaus, and on the current llama.cpp
and MLX cells the two figures agree.

Decided against:

- A fresh app process for every memory cell. The fleet worker would quit
  before a memory job and be relaunched: 75 to 110 s per memory job, plus a change to how
  the fleet relaunches the app. The in-process fixes already bring a private-API build
  within about 30 MiB.
- Running memory cells first in a job. Jobs keep their model and context-size order.
- Holding back results marked as measured after other work. They upload like any other.

## Measurements

### Alone in a fresh process, current build

| Model | Runtime | 512 | 2048 | 4000 |
| --- | --- | --- | --- | --- |
| LFM2.5-350M | llama.cpp | 378 | 397 | 421, 423 |
| granite-4.2-3b | llama.cpp | 2317 | 2441 | 2584 |
| LFM2.5-350M | MLX | 737 | 804 | 867 |
| LFM2.5-1.2B | MLX | 1112 | 1152 | 1175 |
| Llama-3.2-1B | MLX | 1086 | 1163 | 1256 |
| Qwen3-1.7B | MLX | 1358 | 1664 | 2072 |
| LFM2.5-2.6B | MLX | 2084 | 2150 | 2209 |
| LFM2.5-8B-A1B | MLX | 5000 | 5022 | 5150 |
| Phi-3 mini | ONNX | 2906 (2440 footprint + 466 mapped weights) | 5161, 5162 | Killed at the 6656 MiB limit |

### A later cell, current private-API build

Reading minus the model's fresh value; two values where the two runs differed. The
MLX orders at 512 tokens ran once.

| Sequence | 512 | 2048 | 4000 |
| --- | --- | --- | --- |
| Six MLX models, smallest to largest | 0 to +6 | -4 to +4 | -22 to +4 (2.6B after Qwen3 is the -22) |
| Six MLX models, largest to smallest | 0 to +11 | +1 to +11 | 0 to +12 |
| granite, then 350M GGUF | -16 | -17 | -18 |
| 350M GGUF, then granite | -19 | -20, -19 | -20 |
| 350M GGUF three times | -23, -24 | -20, -22 | -21, -23 |
| granite, then 350M MLX | +30, +31 | +30 | +30, +32 |
| 350M GGUF, then 350M MLX | +28 | +27, +28 | +28 |
| 350M MLX, then 350M GGUF | +6 | +8, +9 | +8 |
| Phi-3 ONNX, then 350M GGUF | -10, -7 | -5 | Phi-3 killed |
| 350M GGUF, then Phi-3 ONNX | +18, +19 | +14, +16 | Phi-3 killed |
| Phi-3 ONNX, then 350M MLX | +6 | +6 | Phi-3 killed |
| Phi-3 ONNX, then 1.2B MLX | +5 | +5 | Phi-3 killed |

The kernel's peak equalled the 20 ms sample on every llama.cpp and MLX cell.

### The experiment that chose the fixes

One build (llama.cpp `b11192` with patch 002 and the single-cell fixes) with each candidate behind a launch-time
switch. Later cell minus the fresh value measured under the same switch, two runs each.
"Public overflow drain" alternately `malloc`s and `free`s untouched 64 KiB blocks between
cells until the ring overflows, which makes the kernel reclaim it (6 to 13 ms).

| Sequence | None | Ring flush | Public overflow drain | Patch 003 alone | Flush + patch 003 |
| --- | --- | --- | --- | --- | --- |
| granite, then 350M GGUF | +18, +37 | -15, -16 | -12, -12 | +19, +19 | -15, -15 |
| Phi-3 ONNX, then 350M GGUF | +215, +218 | -5, -5 | -6, -4 | +270, +272 | -5, -5 |
| 350M GGUF, then 350M MLX | +107, +107 | +27 | +27, +27 | +49, +50 | +27, +27 |
| 350M GGUF, then granite | -142, +71 | -18, -18 | -16, -16 | +1, +2 | -17, -17 |
| 350M GGUF three times (third) | +3, +3 | -23, -23 | -16, -18 | +2, +2 | -21, -21 |
| 350M MLX, then 350M GGUF | +10, +10 | +9, +9 | +8, +8 | +10, +11 | +9, +9 |
| 350M GGUF, then Phi-3 ONNX | +22, +47 | +10, +13 | +22, +19 | +38, +38 | +14, +14 |

- The flush and the overflow drain both bring every order within about 30 MiB (flush
  -23 to +27, drain -18 to +27), and the two runs agree.
- Patch 003 alone is what a public build reads today (patches 002 and 003, no flush).
  It cuts fresh values (350M 436 to 378, granite 2540 to 2320) and ends granite's
  bimodal -142/+71, but leaves ONNX's parked memory.

## Repeating the investigation

### Build and device

- Use iOS 27.0 or check the allocator again first (see the config dump below): the parked
  memory is iOS 27's allocator behavior.
- Make the private-API build (`ios/build.sh --internal`, see
  [private-API builds](private-thermal-release-build.md)) for the flush and the kernel
  peak, and public to see what users get. `headlessrun version` prints
  `private_memory=on|off`; never compare runs across the two.
- Keep the `increased-memory-limit` entitlement, or the per-app limit drops and the big
  cells cannot run. `phys_footprint + os_proc_available_memory()` gives the limit.
- Pull the models with `headlessrun models pull model=<source>` (table above). On a phone
  that also serves the fleet, turn its worker off for the session, and afterwards check
  that the models it needs are still there: test pulls can make its storage quota evict
  them.

### One launch, one fresh process

Every experiment runs through `diag memseq`, which measures each model in order in one
process ([usage](usage.md)):

```bash
xcrun devicectl device process launch --device <UDID> --console --terminate-existing \
  ai.liquid.liquid-pipette headlessrun diag memseq \
  models=granite-4.2-3b-Q4_K_M.gguf,LFM2.5-350M-Q4_K_M.gguf prefill=512
```

- `--terminate-existing` makes every launch a fresh process. Switching back to the app
  keeps its process; only closing it in the App Switcher, or this flag, starts a new one.
- A one-model `memseq` in its own launch is the fresh value.
- Add launch-time environment variables with `-e '{"NAME":"value"}'`. The experiments put
  each candidate behind such a variable, read once at launch, so one binary covers every
  comparison.
- A launch takes 1 to 3 minutes. Occasionally one hangs for minutes before the app
  starts, so put a time limit on each launch (the experiments used 300 s) and run it
  again.

### Reading the logs

Per model, the console shows:

- `memseq PRE <model> carriedIn=<MB>`: the footprint the cell starts with.
- `<runtime> max_memory: enter= floor= peak= sampled= kernel= fresh=yes|no`, sizes in MB.
  `enter` minus `floor` is what the settle step removed. A fresh launch's floor is about
  23 MB; an engine's live state adds tens of MB, and a floor far above that means
  leftover memory. `kernel=` carries the kernel peak and `reset_rc=` on a private-API
  build, and reads `off` on a public one.
- `memseq REPORTED <model> [...] host=<MB>`: the reading.

Score a later cell as its reading minus the median fresh value measured with the same
build and switches, because a switch can move the fresh value too. Runs of the current
build repeat within 1 to 2 MiB, so a wider spread means something history-dependent is
back: with patch 002 but not 003, granite then 350M read 432 in one run and 639 in the
next.

### Allocator config dump

```bash
xcrun devicectl device process launch --device <UDID> --console --terminate-existing \
  -e '{"MallocReportConfig":"1","MallocDebugReport":"stderr"}' \
  ai.liquid.liquid-pipette headlessrun diag memseq models=LFM2.5-350M-Q4_K_M.gguf
```

Read `Defer Tiny/Small/Large`, `Huge Cache Size` and `Huge Cache Max Entry Bytes` from the
`XZM Config` block. Ignore `Reclaim Buffer ENABLED/DISABLED`: it reads a sysctl that only
development kernels have, so it says `DISABLED` on every retail phone.

### Ring probe and candidates

The experiment build was not kept. To rebuild it:

- **Probe:** log the footprint and the ring's reclaimable bytes, `malloc` 64 MiB, touch
  it, `free` it, wait 200 ms, log again, then flush and log a third time. The reclaimable
  count is the `reclaimable_bytes` field at offset 72 of the ring that
  `mach_vm_reclaim_get_rings_for_task` returns: a private layout that may move.
- **Overflow drain:** the ring doubles in capacity on each drain (1016 entries at first
  on macOS 27), so each drain needs about twice the frees of the last.
- Run every candidate between cells, before the kernel peak reset, never inside a
  bracket.

### Jetsam kills

A cell that exceeds the per-app limit ends with the app killed and no reading. The kill
report is on the phone:

```bash
xcrun devicectl device info files --device <UDID> --domain-type systemCrashLogs
xcrun devicectl device copy from --device <UDID> --domain-type systemCrashLogs \
  --source JetsamEvent-<date>.ips --destination .
```

Look for Pipette with reason `per-process-limit`; `rpages` times 16 KiB is its footprint.
At 4000 tokens Phi-3 died at 425984 to 426452 pages, 6656 to 6663 MiB.

### Things to keep in mind

- Never subtract the floor. The next model reuses parked pages without adding to the
  footprint, so peak minus floor undercounts: on a build without the fixes, granite
  then 350M had a floor of 332, a peak of 678 and a fresh value of 440.
- Never set MLX's `cacheLimit` to 0 during a measured cell: it changes MLX's peak.
- Do not use a 3-argument `mach_vm_deferred_reclamation_buffer_flush`: on Darwin 27 the
  routine has a fourth out parameter, and the wrong signature writes through garbage.
- The private calls are verified on iOS 27.0 only. A failed flush is logged
  (`reclaim ring flush failed: rc=`) and the cell is measured without it.
- Never ship a private-API build to the App Store or TestFlight
  ([private-API builds](private-thermal-release-build.md)).

## Open questions and untried ideas

- **ONNX Runtime through `vm_allocate`.** Backing ORT's large aligned allocations (1 MiB
  and up) with `mach_vm_allocate`, as patch 002 does for ggml, would keep ONNX's
  300 MiB out of the ring on every build, public included. Not built.
- **The public overflow drain in the app.** It worked on iOS 27.0 but relies on
  undocumented allocator internals and may silently stop working on a later iOS.
- **A fresh process inside the app.** An ExtensionKit extension process is the only
  App Store-safe route. Its memory limit, Metal access and process reuse are unknown.
- **Attributing the leftover by VM tag.** Walk regions with `vm_region_recurse_64`
  (`mach/vm_map.h`; `mach_vm.h` is not available on iOS) and sum dirty pages per tag: 1
  malloc metadata, 2 allocator segments, 3 huge segments, 88 and 100 IOSurface and
  IOAccelerator. Not run.
- **`vl_max_memory` after other cells.** Not measured.
- **MLX release order.** Scoping the model in its own block, then
  `Stream.gpu.synchronize()`, then `clearCache()`. Harmless and not applied: it changed no
  reading and would only lower what the next cell enters with (114 MB instead of 300).

## Ruled out

| Idea | Why |
| --- | --- |
| `malloc_zone_pressure_relief` | Returns 0 under xzone malloc; frees nothing |
| `MallocLargeCache=0`, `MallocSpaceEfficient=1` at launch | Ignored on iOS 27.0: the allocator config dump did not change |
| Subtracting the floor | Undercounts (see above) |
| Allocator counters in place of the footprint (MLX `peakMemory`, llama.cpp's buffer totals) | A different quantity: MLX's misses its cache spike, llama.cpp's buffers exceeded the 350M's fresh footprint |
| Tearing down and recreating the ggml Metal device | About 3 MiB gained, use-after-free risk |
| Shutting ONNX Runtime GenAI down between cells | Its globals hold nothing large |
| ONNX Runtime session options (arena off, no prepacking) | They change the measured number or leave the leftover |
| Filling memory to the per-app limit to force a trim | Fatal whenever the ring holds less than the overshoot |
| Subtracting the graphics ledger after a Metal engine | The carried graphics memory decays inside the next bracket; CPU residue remains |

## MLX on macOS is not a reference for the phone

The same six MLX models read 522 to 902 MiB more under the pipette CLI on an M3 Ultra
than on the iPhone, and the whole gap is accounted for within -8 to +11 MiB:

- A Metal GPU scratch block the driver creates on the first dispatch and never frees:
  352 MiB on an 80-core M3 Ultra, 144 on a 30-core M3 Max, an estimated 20 to 35 on the
  phone.
- MLX's command-buffer limits by GPU class: 50 ops and 50 Mi elements per buffer on a
  desktop, 20 and 40 on a phone. `MLX_MAX_OPS_PER_BUFFER=20 MLX_MAX_MB_PER_BUFFER=40`
  on the Mac reproduces the phone's MLX active peak within 2 to 23 MiB.
- Python and `mlx_lm` (a 64 MiB floor against the app's 20) and the Hugging Face
  tokenizer (42 to 197 MiB); the phone cell feeds token ids.

So do not tune iOS toward macOS MLX figures. The macOS CLI is immune to carry-over (a new
server process per cell), but `mlx_lm` in one Python process is not: +455 MiB after five
models.

## Sources

Published Apple sources read for the mechanism: `libmalloc-812.100.31`
(`src/xzone_malloc/`), `xnu-12377.121.6` (`osfmk/vm/vm_reclaim.c` for the ring,
`osfmk/kern/task.c` for draining it before the footprint limit). Xcode's
`XCTMemoryMetric` reads the same kernel interval peak and reports the absolute value;
Apple's clean-state pattern relaunches the app for every iteration, and the other
benchmarks surveyed run one model per process.
