# GMKtec GPU idle residency: MiniCPM5 E2E investigation

MiniCPM5 end-to-end (E2E) latency includes a large **cost to restore GPU memory after idle**. The standalone prefill and decode timers mostly exclude that cost. Event Tracing for Windows (ETW) shows that the driver evicts and then restores the GPU allocations of the server. A four-byte GPU keepalive every 100 ms reduced delayed E2E by **13.8–17.6% on all four GMKtec boxes**. When the keepalive stopped, the penalty returned.

This explains the tested MiniCPM5 discrepancy. The exact AMD eviction policy and the effect on other models are not known. The OLMo 3 decode dip at 2,048 and 4,096 tokens did not occur in the runs with the keepalive on. No run turned the keepalive off and on for OLMo, so the keepalive is not proven to be the fix.

The workaround ships as [`tools/gpu-keepalive`](../../tools/gpu-keepalive/README.md). For the wider diagnostic ladder on this box type (driver, thermal, SoC power limit, tuner residue, silicon bin), see [`gmktec-evo-x2.md`](gmktec-evo-x2.md); GPU idle residency is one row in that table, not a substitute for it.

## Setup and interpretation

- **Devices:** four boxes, A–D; Ryzen AI Max+ 395 / Radeon 8060S, unified memory; Windows 11 Pro build 26200; AMD driver **32.0.31041.1004**, BIOS EVO-X2 1.12.
- **Model:** `MiniCPM5-2B-Q4_K_M.gguf`, revision `8ffce18336801a527a4385b318e70822fc0f876c`.
- **Runtime:** llama.cpp **b10216 / `876a43211`**, Vulkan, 8 threads, 99 GPU layers, flash attention, no mmap, **`--ctx-checkpoints 0`**. Model/runtime hashes matched across boxes.
- **Controls:** identical server prompts, disabled prompt-cache reuse, validated token counts, delayed requests paired with immediate repeats. Collection workers were suspended during controlled comparisons; no diagnostic results were submitted.

```text
estimate_ms = standalone_prefill_ms + output_tokens * 1000 / standalone_decode_tps
excess_ms = HTTP_e2e_ms - estimate_ms
inferred_decode_tps = output_tokens * 1000 / (HTTP_e2e_ms - standalone_prefill_ms)
```

The inferred decode speed absorbs extra prefill and request overhead. It is **not measured generation speed**. Retain HTTP wall time and actual server prompt/generation timings separately.

## 1. Establish the discrepancy and reproduce it directly

Published results of a benchmark plan, plus three September 10 MiniCPM5 reruns, supplied 24 context comparisons: **all exceeded the estimate; 16 exceeded it by 15%**. The excess existed from 256 tokens and ranged **371–677 ms**. It remained **434–675 ms at 8192**, although its percentage shrank as prefill grew. This suggested an additive component, but not a universal correction constant.

Direct four-box tests at 512/4096 tokens reproduced the behavior: rapid server requests were approximately **2–3% above the estimate**, while the actual Pipette readiness procedure produced **12–24% excess**. Five repetitions per cell had E2E coefficients of variation (CVs) of 0.21–0.65% under Pipette. Standalone decode with 100 versus 256 output tokens differed by **less than 0.15%**, excluding that count difference as the main explanation.

In a single loaded server, three cycles of **35-second wait → request → immediate repeat** restored and removed the penalty without reloading or changing the prompt. At 4096, idle prefill was approximately **1715–1882 ms**, versus **1345–1367 ms** immediately afterward; generation stayed around 2.23–2.25 seconds. A plain wait matched to a sensor query's approximately 1.03-second duration also reproduced most or all of its penalty: the query itself was unnecessary.

Boxes A/C had larger recent idle penalties than B/D, while warm performance was close. This grouping did not match the outlets: boxes C and D are on a battery-backed outlet. Broader client history showed much smaller median differences. Battery power and reboot age were correlations, not tested causes.

## 2. Locate the benchmark timer boundary

Both methods waited **32–34 seconds between repetitions**, occasionally 42 seconds. A `waiting 5s` log line described one sleep, not the whole readiness interval.

| Method | Sequence before its timed work |
|---|---|
| Standalone prefill | Readiness → fresh process → benchmark warmup → timed prefill |
| Standalone decode | Readiness → fresh process → warmup/depth preparation → timed generation |
| Persistent-server E2E | Initial warmup → readiness/idle → timed prompt processing and generation |

Thus standalone decode can pay recovery during untimed depth preparation. E2E pays it inside the next request. Stock `llama-bench --delay` is before warmup, and `-r` adds no deliberate inter-repetition idle gap.

For box A at depth 4096, five complete decode invocations plus readiness took **206.95 s**, but internal decode timings summed to **4.294 s**. Each invocation lasted approximately 8–9 s including untimed load/preparation/shutdown. Five E2E requests summed to **20.620 s** within an approximately **256 s** CLI span. Decode generated 100 tokens per repetition versus E2E's 256, so these totals are not throughput comparisons.

Direct `-r 2` controls were stable: prefill 512 on box B took **130.812/131.049 ms**; decode 256 at depth 512 on box C took **1976.511/1974.246 ms**, with almost no intervening gap. Captured progress timestamps placed its first untimed depth preparation at approximately **548 ms**.

## 3. Find the idle boundary and isolate prefill

A box-A sweep paired each delay with an immediate repeat, using 512 prompt/256 output tokens, three trials per gap and alternating sweep order:

| Requested gap | Actual gap | Mean prefill | Mean HTTP E2E |
|---|---:|---:|---:|
| Immediate | 0.034–0.060 ms | 145.2 ms | 2158.0 ms |
| 700 ms | 702.9–708.5 ms | 146.1 ms | 2161.7 ms |
| 750 ms | 752.3–759.7 ms | 1053.4 ms | 3068.0 ms |
| 1 s | 1004.5–1009.4 ms | 796.2 ms | 2811.3 ms |
| 5 s | 5008.3–5012.1 ms | 597.4 ms | 2614.0 ms |

Generation stayed around **2011–2013 ms**. The observed transition was between approximately **708 and 752 ms**. Near-threshold requests were slower than later requests because they encountered removal still in progress, as the trace subsequently showed.

A **prefill-only** control used 512 prompt tokens and one sampled output, without a long autoregressive decode phase. After 7.5 s, mean prefill was **601.491 ms**, versus **146.555 ms** immediately afterward: **+454.936 ms / 4.10×**. Around 700 ms this workload was borderline, so the exact threshold depends on workload/history.

## 4. Confirm GPU eviction with ETW

A separate Windows Performance Recorder (WPR) GPU/GeneralProfile/Power trace captured two box-A cycles of the server process: **4,865,875 events, zero reported lost events/buffers**. Untraced runs supply performance figures because tracing adds overhead.

`PagingOpVirtualTransfer`/`MemoryTransfer` allocation handles were joined to `GlobalAllocation` rundown. Events came from a System worker, but the allocations belonged to `llama-server`:

| Accounted residency loss | Count | Bytes |
|---|---:|---:|
| Server-owned data allocations | 73 | 1,569,869,824 |
| GPU page tables | 44 | 3,063,808 |
| **Total** | | **1,572,933,632** |

Segment 1 residency fell **1,638,203,392 → 65,269,760 bytes**, exactly reconciling with that total. Transfers explicitly showed **1 → 0 during idle**, then **0 → 1 on request**. Segment 0 denotes system memory; on this unified-memory GPU, that does not imply PCI Express (PCIe) traffic or disk swapping. See the [Microsoft segment definitions](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-segments).

Removal began **734–736 ms** after the prior response, lasted **359–369 ms**, and restoration spanned **454–484 ms**. A one-second request arrived before removal completed; after 35 seconds it had long finished. These are event spans, not hardware-only bandwidth measurements.

Two independent accounting methods agreed: unioning allocation-relative ranges, and summing completed transfer records. Naively summing continuation records double-counted ranges. Process faults rose approximately **384,000 versus 390 warm**, despite essentially unchanged process working set/private bytes; classic kernel fault ETW recorded far fewer faults. That counter discrepancy remains unexplained. **Allocation ownership, transfers and residency reconciliation establish eviction**, not the process fault count alone.

Transitions of the GPU's F-states (functional power states) coincided with removal, supporting a power-associated mechanism. The exact AMD policy, a transition of the full device to the D3 (off) power state, and any `PeriodicTrim` callback were not established.

## 5. Test mitigation candidates

| Intervention | Evidence and conclusion |
|---|---|
| `GGML_VK_ENABLE_MEMORY_PRIORITY=1` | Four boxes, 512/4096 prompts, 1/35-second gaps, 192 measured requests. **15/16 delayed comparisons changed <1%**; the remaining improvement was about 2.36%, influenced by a slow baseline sample. Enabled runs still had **330–502 ms extra prefill** after 35 s. Did not solve it. |
| Host `mlock` | Source inspection: eligible host buffers/file mappings use `VirtualLock`; ordinary Vulkan model buffers are not classified as host. Does not automatically lock fully offloaded weights, key-value (KV) cache or scratch buffers. No mlock performance trial. |
| RustDesk/display activity | RustDesk, DWM and a virtual display did not establish active physical AMD output or continuous GPU work. No monitor/dummy-display crossover tested. |
| `/health` every 500 ms | Next E2E **2633.3 ms**, versus **2621.5 ms** idle: no benefit. |
| One-token inference every 500 ms | Next E2E **2620.8 ms**; repeated large fault bursts persisted. |
| One-token inference every 250 ms | Next E2E **2173.0 ms**, but it started **438.5 ms late**; recovery moved into background pulses. |
| Full 512-token prefill every 500 ms | Next E2E **2165.1 ms**, but pulses consumed **4.51 s per 7.5 s window** and still incurred recovery bursts. |
| One-token inference every 100 ms | Next E2E **2175.0 ms**, no model-sized fault bursts across **222 pulses**. Worked, but request/setup overhead motivated a smaller native helper. |

Priority is a [Vulkan hint](https://docs.vulkan.org/guide/latest/extensions/VK_EXT_memory_priority.html), not a residency lock. b10216 checks the environment variable's **presence**: `0` also enables it; the control leaves it unset. The 0.75-second threshold after sustained inference did not establish a safe pulse interval after tiny work. Measuring only the final request could hide recovery or delayed arrival inside pulses.

## 6. Validate a native keepalive on all four boxes

The helper measured here was a C++ prototype of [`tools/gpu-keepalive`](../../tools/gpu-keepalive/README.md), which documents the mechanism: one Direct3D 11 compute thread on the Vulkan GPU increments a four-byte buffer every **100 ms**.

Each box retained one server through **off → on → off** phases. Each phase had two priming requests, then three **7.5-second gaps → 512/256 request → immediate repeat** pairs. All **96 responses**, including priming, validated 512 evaluated prompt tokens, 256 generated tokens and zero cache reuse; **72 requests** contributed comparisons.

| Box | Off before | On | Off after | Saved vs pooled off | Reduction |
|---|---:|---:|---:|---:|---:|
| Box A | 2610.6 ms | **2166.9 ms** | 2617.1 ms | 447.0 ms | 17.10% |
| Box B | 2504.7 ms | **2157.9 ms** | 2499.0 ms | 344.0 ms | 13.75% |
| Box C | 2637.1 ms | **2164.1 ms** | 2617.1 ms | 463.0 ms | 17.62% |
| Box D | 2507.9 ms | **2151.8 ms** | 2501.3 ms | 352.8 ms | 14.09% |

Cells are means of three requests; reductions use both off-phase means. **The penalty returned on every box.** With keepalive, delayed E2E was within **1.8–10.3 ms of immediate repeats**. Prefill fell to approximately **143–149 ms**, generation stayed approximately **2004–2015 ms**, and process faults fell from approximately 384,400 to 396–399.

Each helper completed **600 verified pulses in 60 s**, without missed deadlines, using **0.49–0.73% of one CPU core**. GPU utilization and power were not measured; completion latency includes CPU scheduling/polling. No new ETW trace independently verified continuous residency under keepalive.

The remaining excess over earlier standalone prefill+decode-256 estimates was **43–56 ms / 2.04–2.63%**. Those estimates were not rerun here and used synthetic benchmark tokens rather than server text. Immediate repeats are the stronger matched reference. With three requests per timing cell, this establishes a reproducible local workaround, not long-term reliability or coverage of other models.

## Effect on collected results

All four boxes have run the keepalive permanently since about 15:15 UTC on September 11, 2026, so GMKtec collection before and after that point ran under different conditions: earlier E2E results carry the idle penalty measured here for MiniCPM5. Do not merge results across the change without accounting for it, and check jobs that straddle it individually. The E2E runner gates readiness before every measured repetition, and that wait is long enough to trigger eviction; see [end-to-end latency](../methodology/end-to-end-latency.md).

## Reproduction

Direct commands against the installed model and runtime:

```powershell
llama-bench.exe --model <gguf> --output json --progress --mmap 0 -fa on -ngl 99 -t 8 -r 2 -p 512 -n 0
llama-bench.exe --model <gguf> --output json --progress --mmap 0 -fa on -ngl 99 -t 8 -r 2 -p 0 -n 256 -d 512
llama-server.exe --model <gguf> --host 127.0.0.1 --port 58933 --ctx-checkpoints 0 --no-mmap --no-warmup -c 768 -fa on -ngl 99 -t 8
```

The comparison procedure used deterministic sampling, suppressed early EOS, and recorded actual gaps, fault counters and token counts. Stop the installed keepalive (`gpu-keepalive stop`) before reproducing an **off** condition.

The raw evidence (the ETW trace, request payload, probe and comparison scripts) is not in this repository.

A [related AMD RX 9070 XT report on HIP (AMD's GPU compute API)](https://github.com/ROCm/TheRock/issues/7221) has a different threshold of about 9.5 seconds and is a lead, not a verified fix for Radeon 8060S/Vulkan.
