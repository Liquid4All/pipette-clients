# Peak Memory: Android

This is the per-platform companion to the high-level
[peak memory methodology](peak-memory-usage.md). It documents how the Android
`pipette-llamacpp` path produces `max_host_bytes`. The Android arm64-v8a runtime
is CPU-only today, so there is no GPU counter.

## Host Counter

The runner wraps `llama-bench` with a statically linked, vendored
`toybox time -v` binary and parses `Max RSS (KiB)` from its output, converting
KiB to bytes. `Max RSS` is the child's `wait4` resident-set high-water mark (
the same value the kernel maintains for the process), so it is read once at
child exit rather than polled. It covers the binary, libraries, heap, model
load, prefill, and the single decode step. The toybox wrapper reports the
child's rusage only, so wrapper memory is excluded; the wrapper is launched in
its own process group so the deadline killer can signal the whole group without
orphaning `llama-bench`.

The on-device Android app uses this same measurement (the process resident-set
high-water mark) rather than a separate counter, so an Android `max_host_bytes`
is comparable whether it comes from the CLI runner or the app.

## GPU Counter

`null`. The Android arm64-v8a build is CPU-only. There is no GPU runtime today.
Android GPU memory telemetry is also vendor-specific: Mali devices lack a stable
per-process counter exposed through `mali_kbase`, and the Adreno DRM path is
reserved for future work.

## Workload

`toybox time -v llama-bench --output json --n-prompt <N> --n-gen 1 -r 1`, the
same benchmark shape as the other llama.cpp paths.

## Reference Measurements

`LiquidAI/LFM2-350M-GGUF` (Q4_K_M), `bench-tools-20260415-38cc8e3fd`
android-arm64-v8a, CPU backend, on a Samsung S25 Ultra (Snapdragon 8 Elite). The
announced-buffers column is the runtime's own reported buffer sum.

| Ctx  | `max_host_bytes` (toybox `Max RSS`) | `max_gpu_bytes` | Announced runtime buffers |
| ---: | ---: | :---: | ---: |
| 256  | 348.83 MiB | null | 338.24 MiB |
| 1024 | 367.95 MiB | null | 413.24 MiB |
| 2048 | 430.34 MiB | null | 425.24 MiB |

At `n=1024` the host peak (367.95 MiB) is below the runtime's announced sum
(413.24 MiB). This is the virtual-reservation case: the runtime sizes its
compute scratch for the worst case, but at this context length only a fraction
of those pages are faulted in, and `wait4` measures resident pages, not the
reservation. By `n=2048` the workload touches the full reservation and the two
figures realign. This is why `max_host_bytes` is reported as the device-fit
signal and the announced sum is treated as a complementary, reservation-level
number, not a contradiction.

## Caveats

- The host counter is resident-set high water, not virtual reservation; a run
  that never faults the full reserved scratch reports less than the runtime
  announces, as above.
- Wrapper overhead was verified negligible against an external high-water-mark
  poll on comparable hardware.

## Code References

- [`pipette-llamacpp max_memory_usage::android`](../../crates/pipette-llamacpp/src/execute/max_memory_usage/android.rs)
