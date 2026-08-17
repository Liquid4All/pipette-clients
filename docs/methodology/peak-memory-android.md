# Peak Memory: Android

This is the per-platform companion to the high-level
[peak memory methodology](peak-memory-usage.md). It documents how the Android
`pipette-llamacpp` path produces `max_host_bytes`. The Android arm64-v8a runtime
is CPU-only today, so there is no GPU counter.

## Host Counter

The runner spawns `llama-bench` directly and samples `/proc/<pid>/status` every
10 ms on a helper thread, reporting:

```
max_host_bytes = max(VmHWM, max over samples of (VmRSS + VmSwap))
```

`VmHWM` is the kernel's peak resident-set watermark, the same figure `wait4`'s
`ru_maxrss` reports. It is exact: the kernel maintains it on the fault path, so
no sampling interval can step over a transient. It is also blind to swap.

That blindness matters here. Fleet devices carry more zram than RAM (12.58 GB
against 11.39 GB on an SM-S948U1), and the peak-memory cell pins weights into
anonymous memory, so every tensor byte is swap-eligible. When the kernel
compresses those pages into zram they leave the resident set while the run still
owns and needs them, and a resident-only counter then under-reports what the run
required. Measured across SM-S948U1 handsets with a 5.68 GB model, the same
4096-token cell reported between 5.36 and 6.29 GB; available memory at run start
separated the depressed runs from the clean ones in every case observed. The low
end sits *below* the model file, which a completed no-mmap load cannot do.

`VmSwap` has no kernel watermark, so a sampled sum is the only way to count those
pages back. Neither term subsumes the other, which is why both are kept:

- On an unpressured run `VmHWM` is necessarily the larger term, since `VmRSS`
  never exceeds it and `VmSwap` is zero. Such a run therefore reports exactly what
  the previous resident-only counter did, which a repeat of a known cell confirmed
  to within 442 KB of its historical value.
- On a swapping run the sampled sum is larger and recovers the displaced pages.

Two guards reject a figure instead of submitting it:

- A sampler that captured no readings at all, i.e. a peak of zero. This matches
  the macOS `phys_footprint` guard and the Linux `VmHWM == 0` bail.
- A peak below the model file size, when the cell loads weights into anonymous
  RAM. No completed load can occupy less than its own tensor bytes, so such a
  figure means either suppression by reclaim or a run that did less than asked.
  A mapped load keeps tensors in reclaimable file pages, so the floor does not
  apply there and is not enforced.

`max_swap` and `major_faults` are recorded on a `[pipette probe]` line appended
to the `llama-bench` stderr that the result stores in `extras.json`. Everything
above that line is the tool's own output; the probe line is the runner's, and it
is the only thing in that field the tool did not print. A suppressed peak is
therefore diagnosable after the fact: near-zero major faults indicate a run that
met no pressure, while six figures indicate one that thrashed.

Counting the swapped-out pages makes the figure honest, but a run that needed zram
is still a run the device could not hold in RAM. Such a run is collected and
stored, and held back from the published results along with the rest of that model
and quantization on that device; see
[Selection policies → Swap exclusion](selection-policies.md#swap-exclusion).

`llama-bench` runs in its own process group so the deadline killer can signal the
whole group. The sampler pins the process start time from `/proc/<pid>/stat` and
discards any sample whose start time differs, so a pid recycled after the child is
reaped cannot contribute to the peak.

### Why a wrapper like `toybox time -v` cannot supply this

An earlier version of this path wrapped `llama-bench` in a vendored
`toybox time -v` and parsed its `Max RSS (KiB)` line. toybox reports that
faithfully: it prints `wait4`'s `ru_maxrss` verbatim, and it agreed with a
`/proc` `VmHWM` poll to the kilobyte across every run compared. It was removed
because the approach cannot produce the figure above, for two independent
reasons.

1. **`ru_maxrss` is resident-only, and there is no swap equivalent to read.** The
   kernel maintains a high-water mark for RSS but none for swap, so no
   read-at-exit can recover what zram holds. `VmSwap` is instantaneous, which
   means the swap term exists only if something samples it while the process is
   alive. Reading the resident counter more often would not help either: those
   pages were not resident at any single moment, so no schedule of reads against
   RSS alone could have counted them. Sampling helps only because it can read a
   second counter, `VmSwap`, that no watermark exposes.
2. **The wrapper hides the pid that needs sampling.** toybox `time` forks
   (`XVFORK`) and execs the command in the child, so the process the runner
   spawned is toybox and `llama-bench` is a grandchild. A sampler pointed at the
   spawned pid measures toybox's own footprint (about 1.6 MB), and reaching the
   real process would mean scanning `/proc` for a matching parent, which is racy.

Spawning `llama-bench` directly puts `VmHWM` and `VmSwap` in one file under one
known pid, and matches what the Linux arm already does.

### The on-device app measures something else

`pipette-android` samples `/proc/self/status` `VmRSS` in-process and subtracts a
baseline (`crates/pipette-android/src/llama.rs`, `metal_allocated_size_bytes`).
That is a current resident figure, not a high-water mark, and it does not account
for swap. An Android `max_host_bytes` from the app is therefore **not**
comparable to one from the CLI runner, and is subject to the same zram
under-reporting described above. Closing that gap is outstanding work.

## GPU Counter

`null`. The Android arm64-v8a build is CPU-only. There is no GPU runtime today.
Android GPU memory telemetry is also vendor-specific: Mali devices lack a stable
per-process counter exposed through `mali_kbase`, and the Adreno DRM path is
reserved for future work.

## Workload

`llama-bench --output json --model <gguf> --n-prompt <N> --n-gen 1 -r 1`, the
same benchmark shape as the other llama.cpp paths, plus the cell's own runtime
flags. For the peak-memory ladder those resolve to `--mmap 0`, `-fa off`,
`-ngl 0` and `-t <n>`. The load flag is the one the plausibility floor depends
on: it is what makes every tensor byte anonymous.

## Reference Measurements

`LiquidAI/LFM2-350M-GGUF` (Q4_K_M), `bench-tools-20260415-38cc8e3fd`
android-arm64-v8a, CPU backend, on a Samsung S25 Ultra (Snapdragon 8 Elite).
These predate both the current counter and the pinned runtime, so they are kept
for the reservation point they illustrate rather than as a baseline to diff
against. A 350M model stays far below the memory ceiling, so nothing should be
reclaimed and the counter change should not move them; that has not been
re-measured. The announced-buffers column is the runtime's own reported buffer
sum.

| Ctx  | `max_host_bytes` | `max_gpu_bytes` | Announced runtime buffers |
| ---: | ---: | :---: | ---: |
| 256  | 348.83 MiB | null | 338.24 MiB |
| 1024 | 367.95 MiB | null | 413.24 MiB |
| 2048 | 430.34 MiB | null | 425.24 MiB |

At `n=1024` the host peak (367.95 MiB) is below the runtime's announced sum
(413.24 MiB). This is the virtual-reservation case: the runtime sizes its compute
scratch for the worst case, but at this context length only a fraction of those
pages are faulted in, and the counter measures materialized pages, not the
reservation. By `n=2048` the workload touches the full reservation and the two
figures realign. This is why `max_host_bytes` is reported as the device-fit
signal and the announced sum is treated as a complementary, reservation-level
number, not a contradiction.

`unsloth/Qwen3.5-9B-GGUF` (Q4_K_M, 5.68 GB), llama.cpp `b10216`
android-arm64-v8a, CPU backend, 4 threads, on an SM-S948U1. This model does
approach the ceiling, so it shows both regimes. "Resident only" is what the
previous counter would have reported.

| Ctx  | `max_host_bytes` | Resident only | `max_swap` | Major faults |
| ---: | ---: | ---: | ---: | ---: |
| 256  | 6289.2 MiB | 6289.2 MiB | 0 | 41 |
| 4096 | 6275.4 MiB | 5790.3 MiB | 3017.5 MiB | 851,200 |

The 4096 cell had 3.0 GB of itself in zram. Under the resident-only counter the
two cells differ by 524 MB (7.9%); with swap counted the gap closes to 14.5 MB
(0.22%). The right answer is that they should not differ at all, because the
requirement is set by the model load and does not depend on context length for
this model. The remaining 13.8 MiB is in one direction and has a known cause: the
4096 figure rests on the sampled sum, since swap suppressed its watermark, while
the 256 figure is the exact watermark. Two estimators, and only one of them can
under-read. These figures were taken at the earlier 25 ms interval.

## Caveats

- The counter measures materialized pages, not virtual reservation. A run that
  never faults its full reserved scratch reports less than the runtime announces,
  as in the 350M table above.
- The sampled term can miss the instant of peak commitment, and can only miss it
  downward. The error is about the footprint's growth rate times the sampling
  interval: a no-mmap load faults in roughly 900 MB/s, so the shipped 10 ms
  interval costs on the order of 9 MB. Measured at 50 ms the sampler under-read
  `VmHWM` by 27 MiB. This only affects rows where swap suppressed the watermark,
  which are exactly the rows with a non-zero `max_swap`; on every other row the
  reported figure is the exact watermark.
- The app path (above) does not yet share this counter.

## Code References

- [`pipette-llamacpp max_memory_usage::android`](../../crates/pipette-llamacpp/src/execute/max_memory_usage/android.rs)
- [`pipette-llamacpp max_memory_usage::proc_footprint`](../../crates/pipette-llamacpp/src/execute/max_memory_usage/proc_footprint.rs)

## The VL runner samples resident only

`vl_max_memory` drives a resident `llama-server` rather than a one-shot
llama-bench, and polls `/proc/<pid>/status` `VmHWM` on the same interval while the
workload runs. It reports peak resident set alone, so it carries the swap
under-reporting this page describes. A VL cell needs its own floor (its mmproj
loads separately from the text weights) before it can adopt the same figure.
