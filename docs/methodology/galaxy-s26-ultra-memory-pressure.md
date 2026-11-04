# Galaxy S26 Ultra Memory Pressure

Rows from the Galaxy S26 Ultra handsets (SM-S948U1) in our fleet
sometimes carry a non-zero `observation_max_swap_bytes`, small models included.
This article records what we measured about where that swap comes from:

- **Runtime:** llama.cpp's CPU path (`android-arm64-v8a`, 4 threads, `-ngl 0`,
  the client's default `--load-mode none`).
- **Production rows:** 2026-10-16 to 2026-10-21.
- **Experiments:** 2026-10-21 and 2026-10-22, on two handsets taken out of the
  pool.

Two different effects produce the swap, and they mean different things for a
row:

- **Swap on a cell's first load.** A small model's first process after a model
  change loads into memory that the earlier models' file cache still fills, and
  the kernel swaps part of it out. The reps after it run resident.
- **Swap storms at the memory limit.** A model that needs nearly all the free
  memory sometimes moves pages in and out of zram through a whole timed
  decode, which then runs 14 to 28 times slower.

An observation spans every process of a cell
([Observations vs metrics](peak-memory-usage.md#observations-vs-metrics)), and
it records how much was swapped, not when, so the row cannot tell the two
apart. [Reading the logs](#reading-the-logs) describes the client log lines
that can.

## Key findings

**By hand, a small model swapped in its first process after a model change,
during the load.** On phone 1, the first Qwen3.5-4B process after a MiniCPM5-2B
run swapped 2,178 MiB 4 s into its load. The next five processes of the same
model swapped nothing. In a llama-bench decode cell the first process is the
warm-up, which is untimed. In a prefill cell it is rep 1, whose load is
untimed. The row's swap term includes it either way.

**That process inherits the earlier models' page cache.** `--load-mode none`
reads each model file through the page cache, and nothing drops those pages
when the tool exits. After a run of smaller models a handset can sit at about
2 GiB free, with 6 to 7 GiB cached. Unless the client drops that cache first,
as the CLI does on Android before each cell, the next model's load has to
reclaim it, and the kernel sometimes takes the new process's own just-written
pages into zram instead: 0.5 to 2.1 GiB, 3 to 4 s into the load.

**The swap does not need the worker or its launcher.** llama-bench started by
hand from the same state swaps the same way. Nothing so far ties it to the gap
between jobs, and no load's resident memory peaks above what it keeps.

**A large model at the memory limit can swap through a whole decode.**
gemma-4-12b Q4_K_M needs about 8.3 GiB, and with no model loaded a handset has
about 8 GiB free (8.8 GiB available). By hand, in one rep in six, the kernel
moved about 2.5 GiB of the weights into zram as decode started. Quantized
weights barely compress (1.05:1), so this frees almost nothing, and every token
faults weights back in. That rep moved 281 GiB in and 304 GiB out of swap,
nearly all of it during its 100-token decode, which took 478.7 s instead of 18
to 20 s. In production two of the five timed reps at depth 2048 ran 14 to 28
times slower than the fastest, and every rep at depth 4096 took more than 18
minutes. The hand storm's peak swap was barely higher than a fast rep's (2.55
against 2.50 GiB), so the row's swap term cannot tell a storm from a load that
swapped and recovered.

**Benchmark processes are exempt from the low-memory killer.** They run at
`oom_score_adj` -1000, inherited from the adb shell, so a memory shortage ends
in a reboot instead of a kill: holding 7.2 to 7.3 GiB in an idle llama-server while
loading a 2.4 or 3.6 GiB model rebooted two handsets within about two minutes.

## The phone's memory

| | SM-S948U1, Android 16, kernel 6.12 |
| --- | --- |
| `MemTotal` | 10.86 GiB |
| Swap | `zram0` only: 4 GiB, `lzo-rle`, no writeback (RAM Plus off, see [device conditions](device-conditions.md#android-samsung-galaxy-s26-ultra)) |
| Swappiness | 100 in the init scripts; the live value is unreadable from the shell |
| Free with no model loaded | about 8 GiB (8.8 GiB available) |
| `ulimit -l` in the adb shell | 64 KiB, so the `mlock` load mode cannot pin a model (and b11744 never locks the repacked weights) |
| `oom_score_adj` of a benchmark process | -1000 |
| `/proc/pressure/memory` (PSI) | readable |

Swap cannot be turned off on these handsets. Turning RAM Plus off removes
Samsung's swap to flash and its page-management changes, but the 4 GiB zram
stays, and only root, which these handsets do not allow, could remove it. This
is as far as swap can be curtailed.

The swap term also counts only anonymous pages. A file-backed page, such as an
mmap'd weight, can be evicted and later read back from its file. That costs
what a swap costs, yet it never shows in `VmSwap`: it shows as `pgmajfault`
and `workingset_refault_file` in the [log lines](#reading-the-logs).

## How a cell loads the model

A llama-bench decode cell is six processes in a row: the warm-up, which starts
without the readiness gate, then five reps, each behind it
([decode](decode-throughput.md#llamacpp)). A prefill cell has no warm-up
process. Each of its five processes warms itself at the cell's shape
([prefill](prefill-throughput.md#llamacpp)), so its first process is rep 1. Each
process loads the model from scratch. Under `--load-mode none` each copies the
whole file into anonymous memory, and on this CPU the quantized matmul weights
(Q4_0, Q4_K, Q5_K, Q6_K and Q8_0 among them) are repacked into an anonymous
`CPU_REPACK` buffer in every load mode. The worker claims its next cell as soon
as the last one is submitted, with no wait and no memory check. The row's
`observation_max_swap_bytes` is the largest of its processes'.

When a process exits, its anonymous memory is freed at once: within a second
of gemma-4-12b exiting, free memory rose from 0.09 to 8.35 GiB. The file pages
it read stay cached:

- **The same model's next process** reads them back from the cache. Llama-3.2-3B
  read up to 1.9 GiB from storage on its first load and 0 to 555 MiB on the
  next. Under `--load-mode none` the hit saves the read, not the memory: the
  process still copies the whole model.
- **A different model** gains nothing from them. On Android the CLI drops them
  before that model's cell ([Mitigations](#mitigations)). Without the drop they
  occupy RAM until reclaim evicts them, during that model's load.
- **A large model** leaves little cache behind, since its own demand evicted
  most of it while it ran. After gemma-4-12b exited, 0.6 GiB was cached and
  more than 8 GiB was free.

## Swap on a cell's first load

### Production

Across 1,837 S26 CPU cells from 2026-10-16 to 2026-10-21, all in one LFM3 text
benchmark plan (1,557 cells on llama.cpp b11744, 220 Bonsai cells on PrismML's
llama.cpp fork and 60 LFM3-1.7B cells on a llama.cpp fork with LFM3 support), 25 of the 1,463 cells under 4 GiB of host memory
swapped. They swapped 0.01 to 1.61 GiB (median 0.50 GiB), on all nine handsets,
across 15 model files and all four benchmark types (11 decode, 6 prefill, 5 peak
memory, 3 end-to-end).

What came before them:

| Predecessor job | Swapped cells | Other cells |
| --- | ---: | ---: |
| the same model file | 0 of 25 | 62 of 1,438 (4%) |
| 4 GiB or more of host memory | 1 of 25 (4%) | 276 of 1,438 (19%) |
| ended more than 120 s before this cell's first process (mostly a model download) | 2 of 25 (8%) | 121 of 1,438 (8%) |
| ended more than 120 s before this cell's claim | 0 of 25 | 33 of 1,438 (2%) |

23 of the 25 followed a smaller job on a different model. The timed values
mostly look unaffected: 18 of the 20 timing cells among them have a coefficient
of variation of 4% or less, as do 88% of the plan's 1,377 timing cells, and
their median is 1.1% against the plan's 1.2%.

### Reproduction

On phone 1 and phone 2, llama-bench b11744 was run by hand with the production
arguments from the cache-full state, about 2 GiB free and 6 to 7 GiB cached,
left by a run of smaller models or by reading four other large model files with
`cat`. Each process's `VmSwap` was sampled:

| First load of | Footprint | Swapped | Swap |
| --- | ---: | --- | --- |
| Qwen3.5-4B Q4_K_M | 3.28 GiB | 2 of 3 | 2.13 GiB, 0.66 GiB |
| gemma-4-E2B Q5_K_M | 3.56 GiB | 1 of 2 | 0.48 GiB |
| Llama-3.2-3B Q4_K_M | 2.36 GiB | 0 of 4 | |
| the same model again | | 0 of 9 | |

The case in the key findings ran on phone 1. Its first process started with
2,269 MiB free and 6,489 MiB cached. Processes 2 to 6 ran with 35 s gaps
between them, started with 3,598 to 3,800 MiB free, and swapped nothing. A
first load does not need its own file out of the cache to swap: on phone 2
Qwen3.5-4B read only 96 MiB from storage and still swapped 0.66 GiB.

Through the CLI (`pipette benchmarks run`, a client built on 2026-10-22), the same
Qwen3.5-4B decode cell swapped in none of its six processes, in four cells on
the two handsets started from the cache-full state made with `cat` (1,657 to
2,019 MiB free). The swap is probabilistic, and the rate from this state is not
established.

These were ruled out:

- **The launcher.** The CLI starts llama-bench in a session of its own, with
  stdin closed, its output piped and its library path set, and changes no
  priority, cgroup or argument. Through a CLI built on 2026-10-21, a replay of
  one production case swapped 2,024 MiB on Qwen3.5-4B from 2,283 MiB free;
  plain llama-bench swapped 2,178 MiB from 2,269 MiB free.
- **The gap between jobs**, as far as the data go. In production an idle gap of
  more than 120 s before the claim preceded none of the 25 swapped cells and 2%
  of the others, too few to show an effect. Llama-3.2-3B run back to back, with
  90 s gaps, or held as an idle server for 120 s, never swapped, but it never
  swapped on a first load by hand either.
- **A load-time peak.** Peak `VmRSS` during the load equals the steady figure
  for all four models measured.
- **One model family.** Several families swapped, in production and by hand.
- **Attribution.** The swap term is the process's own `VmSwap`, not a
  system-wide figure.

### What it means for a row

The swap happens during a load, which no cell times. In a decode cell the
swapping process is the warm-up, so the five reps the row reports ran like
processes 2 to 6 above. That rests on the hand runs; the [log
lines](#reading-the-logs) can confirm it on a production row. In a prefill cell
it is rep 1, and whether its swapped pages are back before its timed pass is not
measured; five of the six swapped prefill cells have a coefficient of variation
under 4%. Either way the swap term is non-zero. [Swap
exclusion](selection-policies.md#swap-exclusion) reads a swapping run as a
device that lacks the memory for that quantization, and these runs fit that
reading poorly: the same model on the same handset loads without swap when other
models' files do not fill the cache.

A `max_memory_usage` cell also runs its command once before the measured run
([peak memory: Android](peak-memory-android.md#measured)), and that run is the
one exposed. An end-to-end cell's llama-server loads once, before its timed
requests.

## Swap storms at the memory limit

The production case is gemma-4-12b Q4_K_M decode at depth 2048 (100 tokens) on
phone 3, 2026-10-20. The three fast reps alone give about 5 tok/s:

| Rep | Timed decode | Untimed load and fill |
| --- | ---: | ---: |
| warm-up | 517 s | 116 s |
| 1 | 23.1 s | 136 s |
| 2 | 18.4 s | 132 s |
| 3 | 252.4 s | 132 s |
| 4 | 18.1 s | 133 s |
| 5 | 506.1 s | 132 s |

Every timed rep started at 33 °C. The row published 0.611 tok/s with a
coefficient of variation of 132%, a resident peak of 8.29 GiB and 3.08 GiB of
swap. The same cell was unstable on b11349 too. At depth 4096 on phone 7 every
process was slow: the warm-up and all five reps took 1,088 to 1,283 s to decode
100 tokens, so the row published 0.084 tok/s with a coefficient of variation of
only 7.9% (8.21 GiB resident, 3.13 GiB swapped). Olmo-3-7B-Think Q4_K_M decode
at depth 4096 is also at the limit (8.48 GiB resident, 3.21 GiB swapped, CV
57%): its warm-up decode took 831 s and its timed reps 23 to 79 s.
gemma-4-12b's llama-server end-to-end cells show no storm at any length (CV 0.1
to 4.2%, 0.1% at 4096), though the three longest swapped about 3 GiB.

Rerun by hand on phone 1, one rep in six stormed. Every rep started in the same
state: 8 GiB free, 0.6 GiB cached and 2.5 GiB of swap free.

| | Fast rep | The storm |
| --- | --- | --- |
| Timed decode | 18.1 to 20.5 s | 478.7 s |
| Peak process swap | 0 to 2.50 GiB | 2.55 GiB |
| System swap in / out during the rep | at most 3.9 / 4.4 GiB | 281 / 304 GiB |
| PSI memory `full`, peak avg10 | 1.6 to 8.7% | 44.8% |
| Direct-reclaim stalls (`allocstall`) | | 722,757 |
| CPU clock | 2.5 to 2.7 GHz | 2.7 GHz |

Peak swap does not separate the two. Rep 1 peaked at 2.50 GiB with swap free at
0, yet its decode was fast, because most of the swapping happened during the
fill: 0.63 GiB was still swapped when the timed decode began, and under 1 GiB
moved each way during it. What separates a storm is the swap traffic inside the
timed window. The clocks rule out throttling. The "moderate" thermal status the
slow production reps ended at is an effect of their length: their decodes ran
4 to 8 minutes instead of 20 s.

`--load-mode mmap` does not make decode stable. On this build only `token_embd`
and the F32 tensors stay file-backed, since the repack buffer is anonymous, and
decode becomes bimodal: 18.6 to 18.8 s in two reps, 48 to 54 s in four.

## Reading the logs

The CLI logs two kinds of line around a cell. On Android, where it drops the
model store's cache ([Mitigations](#mitigations)), it adds the `before_drop`
and `after_drop` sample points and a third kind of line.

`memory pressure`, at up to five `sample_point`s:

- `before_drop` and `after_drop`, on Android only, after the downloads and on
  either side of the drop, so they show the state the cell was handed and what
  the drop freed;
- `before_cell`, after the downloads and before the cell's first process. On
  Android it follows `after_drop`;
- `before_rep` and `after_rep`, around each rep, outside the thermal and device
  readings. A `max_memory_usage` cell has no reps and logs only `before_cell`,
  after `before_drop` and `after_drop` on Android.

Each event carries the system-wide `/proc/meminfo` figures (`mem_free_bytes`,
`mem_available_bytes`, `cached_bytes`, `anon_pages_bytes`, `swap_free_bytes` and
others) and zram's `mm_stat` (`zram_orig_bytes`, `zram_compr_bytes`,
`zram_used_bytes`). Every event after the cell's first also carries what moved
since the previous one, over `window_ms`: `pswpin` and `pswpout` (pages),
`pgmajfault`, `allocstall`, `pgsteal_kswapd`, `pgsteal_direct`, the workingset
refaults, and PSI as `psi_some_ms` and `psi_full_ms`. On Android the first is
`before_drop`, so `after_drop`'s deltas cover the drop. A rep's window,
`before_rep` to `after_rep`, holds its own llama-bench load. The window from
`before_cell` to the first `before_rep` holds the warm-up, in a cell that has
one, and rep 1's readiness gate.

`dropped page cache`, on Android, between `before_drop` and `after_drop`, with
the model store's `dir`, the count of regular `files` it advised, and the
count of files and directories that `failed` to open or advise. What it freed,
and how long it took, is in `after_drop`: its `cached_bytes` and
`mem_free_bytes` against `before_drop`'s, over its `window_ms`.

`memory probe stopped`, once per probed process, the warm-up included, with its
`pid`, its `window_ms`, and, only when its `VmSwap` rose above 0,
`first_swap_ms` and `peak_swap_ms`: when that happened and when it last set a
new maximum, in ms from the probe's start.

The two effects read differently:

- **A first-load swap.** `before_drop` shows about 2 GiB of `mem_free_bytes`
  with most of RAM in `cached_bytes`. Clients whose `client_version` is dated
  2026-10-21 or earlier, and some dated 2026-10-22, kept the cache; such a
  build logs no `before_drop` line, and `before_cell` shows it. The cell's first
  `memory probe stopped` line carries a `peak_swap_ms` of a few thousand, and
  the later ones none. Through the CLI, a cell's first process logged a
  first-load swap with `peak_swap_ms=3256`: phone 1, gemma-4-E2B Q4_0,
  `max_memory_usage_256`, from 2,549 MiB free and 6,142 MiB cached in
  `before_cell`, on a build that kept the cache.
- **A storm.** One rep's `after_rep` moves `pswpin` and `pswpout` by tens of
  millions of pages (281 GiB is 74 million 4 KiB pages), and `psi_full_ms` by a
  large share of its `window_ms`.

## Mitigations

The CLI drops the model store's page cache before every cell on Android. It
applies none of the others.

- **Dropping the cache before each cell.** The shell user cannot write
  `/proc/sys/vm/drop_caches`, but
  `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on a file drops its clean pages
  without privilege. On Android the CLI does that to every regular file in the
  model store, the cell's own model included. It runs after the downloads and
  before the cell's first process, for every runtime and benchmark. The
  measurements:
  - Through the CLI on both handsets, the drop covered 242–248 files with no
    failures. It took about 1.1 s when the cache was full, and 0.17–0.28 s right
    after another drop.
  - It took free memory from 3,349–3,461 MiB to 7,684–8,034 MiB, and the cache
    from 5,322–5,469 MiB to 800–952 MiB.
  - In four earlier runs by hand, applied to every stored model file except the
    one under test, it raised free memory from 1,650–2,055 MiB to
    5,022–7,995 MiB, and it cut the cache from 6,590–7,087 MiB to
    850–3,630 MiB.
  - 40 cells with the drop and 40 without ran alternately on both handsets. The
    cells with the drop started with 7,684–8,080 MiB free, and none of their 80
    processes swapped. The cells without it started with 1,658–5,404 MiB free,
    and one process swapped: a first load from 2,549 MiB free. One swap in 80 is
    too rare to show that the drop prevents it.
- **Reading the model past the cache** (`--load-mode dio`, the client's
  `direct_io`), which opens the file with `O_DIRECT` and falls back to buffered
  reads where that fails. In four CLI cells from the cache-full state none of
  the 24 processes swapped, but the unflushed controls did not swap either, so
  this does not show that it prevents the swap.
- **Keeping the warm-up out of the row's swap term**, or reporting it on its
  own. Not implemented.
- **For storms:** flagging a rep by its swap traffic, which the `after_rep`
  lines carry for the whole rep, load and fill included; the storm by hand
  moved about 70 times what the busiest fast rep did. gemma-4-12b's end-to-end
  cells, which decode through llama-server, swapped without a storm; whether a
  decode cell on llama-server avoids storms is not measured.
  `--load-mode mmap --repack 0` may stay below the limit, but it is not
  measured, and `--repack 0` changes the kernels being measured.

## Open questions

- Why only some first loads from the cache-full state swap. In one replay
  Llama-3.2-3B swapped 1.34 GiB right after Qwen3.5-0.8B with 3.3 GiB free,
  which the mechanism above does not explain.
- What makes one gemma-4-12b rep storm and not the next. Candidates
  include llama-bench's state snapshot of about 352 MiB, allocated between the
  untimed fill and the timed decode, and the compute buffer.
- Why `--load-mode mmap` gives a 50 s mode. Its file-backed weights can be
  evicted and read back without adding to the swap term, so the file refaults
  in those reps' `after_rep` lines are the place to look.
- Whether the reboot loops seen on an NPU benchmark plan (one handset dropped off adb ten
  times, 22 to 34 s after starting gemma-4-12b, and its boot history records
  four reboots) are the same memory shortage on a process that the killer cannot take.

## Code References

- [`pipette-device memory_pressure`](../../crates/pipette-device/src/memory_pressure.rs) (the `memory pressure` event)
- [`pipette-device page_cache`](../../crates/pipette-device/src/page_cache.rs) (the drop before each cell)
- [`pipette-memprobe`'s `/proc` sampler](../../crates/pipette-memprobe/src/linux.rs) (`first_swap_ms`, `peak_swap_ms` and the `memory probe stopped` line)
- [`pipette-cli run`](../../crates/pipette-cli/src/run.rs) (where the events are logged)
- [`pipette-ops run_response`](../../crates/pipette-ops/src/run_response.rs) (the per-process merge into one observation)
- [`pipette-llamacpp runtimes`](../../crates/pipette-llamacpp/src/runtimes.rs) (`get_load_mode`, the `--load-mode none` default)
