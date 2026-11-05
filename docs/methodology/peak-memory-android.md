# Peak Memory: Android

This is the per-platform companion to the high-level
[peak memory methodology](peak-memory-usage.md). It documents how the `pipette`
CLI on an Android phone (`pipette-llamacpp`, the same code that runs on Linux)
produces `max_host_bytes`. On the `android-arm64-snapdragon` flavor it also
produces `max_gpu_bytes` and `max_npu_bytes`; see
[GPU and NPU Counters](#gpu-and-npu-counters). The Android app measures
differently ([The on-device app measures something
else](#the-on-device-app-measures-something-else)).

## Host Counter

The runner spawns `llama-bench` directly and samples `/proc/<pid>/status` every
10 ms on a helper thread, reporting:

```
max_host_bytes = max(VmHWM, max over samples of (VmRSS + VmSwap))
```

`VmHWM` is the kernel's peak resident-set watermark. It is close but not exact:
the kernel keeps it lazily, from approximate per-CPU counters, and on Linux 6.8
successive reads have dropped by up to 2 MiB between 10 ms ticks. It is also
blind to swap.

`llama-bench` drops `VmHWM` from `/proc` when it exits, so the sampler's last
read, taken after the exit and before the reap, is `waitid`'s `ru_maxrss`
instead. That figure is not `VmHWM`: it is the maximum over the process, its
image before `exec` (a copy of pipette, since the child is forked), and the
children it reaped. It counts only when it is above pipette's own `VmHWM` by
more than 4 MiB, so pipette's image is never reported as the tool's peak;
otherwise the reads while the tool ran stand.

That blindness matters here. The phones that pipette measures always carry
zram, the kernel's compressed swap in RAM (4 GiB on an
SM-S948U1 with RAM Plus off, as
[device conditions](device-conditions.md#android-samsung-galaxy-s26-ultra)
requires, and 12 GiB with RAM Plus at 8 GB), and the peak-memory cell pins
weights into anonymous memory, so every tensor byte is swap-eligible. When the
kernel compresses those pages into zram they leave the resident set while the
run still owns and needs them, and a resident-only counter then under-reports
what the run required. Measured across SM-S948U1 handsets with a 5.68 GB model,
the same 4096-token cell reported between 5.36 and 6.29 GB; available memory at
run start separated the depressed runs from the clean ones in every case
observed. The low end sits *below* the model file, which this model's
unpressured runs never do.
[Galaxy S26 Ultra memory pressure](galaxy-s26-ultra-memory-pressure.md) covers
what leaves memory short at a run's start.

`VmSwap` has no kernel watermark, so a sampled sum is the only way to count those
pages back. Neither term subsumes the other, which is why both are kept:

- On an unpressured run `VmHWM` is necessarily the larger term, since `VmRSS`
  never exceeds it and `VmSwap` is zero, so the run reports `VmHWM`.
- On a swapping run the sampled sum is larger and recovers the displaced pages.

Two guards reject a figure instead of submitting it:

- A peak of zero: no read found `VmHWM` while the tool ran.
- A sampler whose last read, at the stop, did not find the tool, running or
  exited and not yet reaped: the reads before it could have missed the end of
  the run.

The row's `client_extras` carry the four `/proc` terms behind the peak:
`peak_rss_kib`, `peak_committed_kib`, `max_swap_kib` and `major_faults`. They
stay apart from the `llama-bench` stderr, which is the tool's own output. So a
suppressed peak can be diagnosed after the run: near-zero major faults show a
run that met no pressure, and six figures show one that thrashed.

Counting the swapped-out pages gives the run's real requirement. But a run that
needed zram is still a run that the device could not hold in RAM, so it is held
back from the published results
([Selection policies → Swap exclusion](selection-policies.md#swap-exclusion)).

`llama-bench` leads a session of its own, so the deadline kill reaches whatever it
started. The sampler pins the process start time from `/proc/<pid>/stat` and
discards any sample whose start time differs, so a pid recycled after the child is
reaped cannot contribute to the peak.

### No model-file floor

None of the text, VL and ASR cells refuses a peak below the model file's
size, because the file is not a lower bound on a valid run: llama.cpp skips
tensors some files carry (the multi-token-prediction layers of several MoE
families unless asked for them; the audio encoder in Gemma 3N's projector), and
recent builds read a tensor over 4 GiB on demand even under `--load-mode none`.
The swap terms in `client_extras` are what show a run squeezed by reclaim.

### Why a wrapper like `toybox time -v` cannot supply this

Its `Max RSS` is `wait4`'s `ru_maxrss`, a resident-only watermark with no swap
equivalent to read at exit, and the wrapper makes `llama-bench` a grandchild, so
a sampler on the spawned pid measures the wrapper instead.

### The on-device app measures something else

The Android app samples `/proc/self/status` `VmRSS` in-process and subtracts a
pre-load baseline, in both engines (`MemoryAllocationPoller` in
`android/Pipette/app/src/kotlinEngine/java/ai/liquid/pipette/llama/LlamaCpp.kt`,
`spawn_metal_allocation_poller` in `crates/pipette-android/src/llama.rs`). That
is a sampled resident figure, not a high-water mark, and it does not count swap.
An Android `max_host_bytes` from the app is therefore **not** comparable to one
from the CLI runner, and is subject to the same zram under-reporting described
above.

## GPU and NPU Counters

`null` on `android-arm64-v8a`, a CPU-only build: it loads no GPU or NPU backend,
and its runs map no device memory.

`android-arm64-snapdragon` is upstream's Snapdragon package: the CPU backend plus
an Adreno (OpenCL) and a Hexagon (HTP) backend. Most of what it allocates never
reaches `/proc/<pid>/status`:

- Adreno memory comes from the KGSL driver and is mapped through
  `/dev/kgsl-3d0`. The kernel accounts it in its own pool (`GpuTotal` in
  `/proc/meminfo`); only the pages the CPU touches through those mappings count
  as resident.
- Hexagon buffers are dmabufs from the `system` heap, shared with the DSP. Their
  mappings are not resident-set pages either.

The package holds both even with nothing offloaded: the OpenCL backend and the
Hexagon session start up whatever `--n-gpu-layers` says, while llama.cpp reports
a zero-byte OpenCL and HTP compute buffer.

On this flavor the sampler also reads, on each 10 ms tick with the host's:

- `max_gpu_bytes`: the length of the process's `/dev/kgsl-3d0` mappings in
  `/proc/<pid>/maps`.
- `max_npu_bytes`: the dmabufs the process holds, each counted once by inode:
  those it has open, sized from `/proc/<pid>/fdinfo/<fd>`, and any it only maps,
  by its first mapping's length, as `dmabuf_dump` counts them. On this build only
  the Hexagon backend allocates dmabufs: a run on the Adreno GPU holds none.

Each is its own peak, and `null` when the run held none, since a reported peak
is never zero. A run whose `maps` and dmabufs could not be read fails rather than
reporting `null`. A `maps` read costs about 0.5 ms on a GPU run, twenty times a
status read, so there the sampler takes up to about a fifth of one core. The
benchmark reports bytes, not time, and the sampled runs took no longer than
unsampled ones.

All three count at any `--n-gpu-layers`: offloaded layers write their weights to
the GPU or NPU, and with Hexagon's host buffers on (`GGML_HEXAGON_HOSTBUF=1`)
llama.cpp can keep even the CPU layers' weights in the NPU's shared memory, 437
MiB of the backends' own buffers in the `-ngl 0` row below.

### Reading the three together

All three pools are the phone's one RAM, so the run needed their sum, with one
correction: the host and GPU fields overlap. During an Adreno load the CPU
writes the weights through the KGSL mappings, and those pages count as resident:
108.7 MiB of that run's `VmHWM` was `/dev/kgsl-3d0` pages (`Rss` in
`/proc/<pid>/smaps`). The fields are kept as raw peaks rather than one being
subtracted from the other ([the metric](peak-memory-usage.md#the-metric)), so
the sum overstates a GPU run by up to that much. dmabufs never count as
resident, so the NPU field does not overlap.

### Measured

`LiquidAI/LFM2.5-1.2B-Instruct-GGUF` Q4_0 (a 663.5 MiB file) and llama.cpp
`b11109` on a Galaxy S26 Ultra (SM-S948U1, SM8850, Android 16 build
S948U1UES4AZG3, kernel 6.12), as the client reports `max_memory_usage_512`
(`--load-mode none --threads 4`):

| Build and flags | `max_host_bytes` | `max_gpu_bytes` | `max_npu_bytes` |
| --- | ---: | ---: | ---: |
| `android-arm64-v8a`, `-ngl 0` | 883.5 MiB | `null` | `null` |
| `android-arm64-snapdragon`, `-ngl 0` | 903.7 MiB | 392.2 MiB | 44.9 MiB |
| `android-arm64-snapdragon`, `-ngl 99 --device GPUOpenCL` | 281.6 MiB | 1214.9 MiB | `null` |
| `android-arm64-snapdragon`, `-ngl 99 --device HTP0 -fa on -ub 1024`, `GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_OPPOLL=1` | 162.8 MiB | 15.4 MiB | 864.9 MiB |

Checked against Android's own per-process tools during the same runs, at 512 and
1024 tokens: every GPU figure sat 160 KiB under the kernel's total from
`dumpsys gpu --gpumem`, and every NPU figure equaled `dmabuf_dump <pid>`. The
host figures repeat within 3 MiB, the device ones exactly. The first Adreno run
with a cold OpenCL kernel cache (`$TMPDIR/llama.cpp/cl-cache`) read about 50 MiB
more host memory. Every llama.cpp flavor runs a warm-up first
([compile caches](peak-memory-usage.md#compile-caches)); the warm-up was not
checked on a phone.

The HTP row does run on the NPU. llama.cpp's log puts all 17 layers on `HTP0`,
and at 1024 prompt tokens that setup prefilled nearly 5000 tokens/s, against
about 400 on the package's CPU path and 1100 on the Adreno GPU. At that size it
held 1009 MiB of dmabuf (`HTP0`'s 668 MiB of weights, 15 MiB of KV cache and
275 MiB of compute buffer, plus the session's own) against a 167 MiB `VmHWM`.

### Why these sources

The client runs as the shell user (uid 2000, SELinux domain `shell`), so it can
read only what that user may. On the device above:

| Source | Shell access | Gives |
| --- | --- | --- |
| `/sys/class/kgsl/kgsl/proc/<pid>/` | denied (SELinux type `vendor_sysfs_kgsl_proc`) | KGSL's per-process counters |
| `/sys/kernel/debug/kgsl` | absent: no debugfs on a user build | KGSL's per-process counters |
| `/sys/kernel/dmabuf/buffers` | denied | per-buffer dmabuf sizes and exporters |
| `/proc/<pid>/fdinfo/<fd>` of `/dev/kgsl-3d0` | readable | no memory fields |
| `/proc/<pid>/maps` | readable for the run's own uid | the `/dev/kgsl-3d0` and `/dmabuf:` mappings |
| `/proc/<pid>/fdinfo/<fd>` of a dmabuf | readable for the run's own uid | its `size`, `ino` and `exp_name` |
| `dumpsys gpu --gpumem` | readable | the kernel's per-process GPU total, one snapshot per call |
| `dmabuf_dump <pid>` | readable | the process's dmabuf total |
| `/sys/class/kgsl/kgsl/page_alloc`, `GpuTotal` in `/proc/meminfo` | readable | system-wide totals only |

`dumpsys` asks a system service for each snapshot, too heavy to poll from the
client, so the sampler reads the two `/proc` sources and the `dumpsys` tools
serve as the check.

To repeat a check, unpack the package and run its `llama-bench` with the library
paths the client sets, then read the sources above while it runs:

```sh
cd llama-b11109 && export LD_LIBRARY_PATH=$PWD/bin:$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib
./bin/llama-bench --output json --model <gguf> --repetitions 10 --n-prompt 0 \
  --n-gen 0 -pg 512,1 --load-mode none -t 4 -ngl 0 > /dev/null 2>&1 &
sleep 8; dumpsys gpu --gpumem | grep "Proc $!"; dmabuf_dump $!
grep VmHWM /proc/$!/status
```

## Workload

[The llama-bench shape](peak-memory-usage.md#the-llama-bench-shape), as on the
other llama.cpp paths, plus the cell's own runtime flags. A cell that sets
neither `mmap` nor `load_mode` runs with `--load-mode none`, which puts every
tensor byte in anonymous memory.

## Reference Measurements

`LiquidAI/LFM2-350M-GGUF` (Q4_K_M), `bench-tools-20260415-38cc8e3fd`
android-arm64-v8a, CPU backend, on a Samsung S25 Ultra (Snapdragon 8 Elite).
Taken with an older sampler and runtime, so they show the reservation point
below, not a baseline to diff against. The announced-buffers column is the
runtime's own reported buffer sum.

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
approach the ceiling, so it shows both regimes. "`VmHWM` only" is the resident
watermark without the swap term.

| Ctx  | `max_host_bytes` | `VmHWM` only | `max_swap` | Major faults |
| ---: | ---: | ---: | ---: | ---: |
| 256  | 6289.2 MiB | 6289.2 MiB | 0 | 41 |
| 4096 | 6275.4 MiB | 5790.3 MiB | 3017.5 MiB | 851,200 |

The 4096 cell had 3017.5 MiB of itself in zram. On `VmHWM` alone the two cells
differ by 498.9 MiB (7.9%); with swap counted the gap closes to 13.8 MiB
(0.22%). For this model the requirement is set by the model load and does not
depend on context length, so the two cells need the same memory. The remaining
13.8 MiB is in one direction and has a known cause: the
4096 figure rests on the sampled sum, since swap suppressed its watermark, while
the 256 figure is the watermark. Two estimators, and only one of them can
under-read. These figures were sampled every 25 ms, not the shipped 10 ms.

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
  reported figure is the watermark.
- The app path (above) does not share this counter.
- The GPU and NPU counters are sampled every 10 ms, since no per-process
  watermark is readable, so a buffer that lives for less can be missed, and
  they end at the last tick before the tool exits.
  llama.cpp's model, KV and compute buffers live from load to exit.

## Code References

- [`pipette-llamacpp max_memory_usage`](../../crates/pipette-llamacpp/src/execute/max_memory_usage.rs) (Android and Linux share it)
- [`pipette-memprobe`'s `Peak::counting_swap`](../../crates/pipette-memprobe/src/lib.rs) (the swap-aware figure) and [`pipette-llamacpp swap_aware`](../../crates/pipette-llamacpp/src/swap_aware.rs) (its `client_extras`), for the text, VL and ASR cells
- [`pipette-memprobe`'s `/proc` sampler](../../crates/pipette-memprobe/src/linux.rs)
- [`pipette-llamacpp memory::peak`](../../crates/pipette-llamacpp/src/memory.rs) (which flavors are sampled, and for what)

