# Peak Memory: Linux

This is the per-platform companion to the high-level
[peak memory methodology](peak-memory-usage.md). On Linux the memory benchmark
is implemented for the torch-oai Docker path and for llama.cpp CPU flavors;
Linux llama.cpp GPU/accelerator flavors are not implemented and are rejected by
the dispatcher.

## torch-oai (Docker)

The torch-oai path measures the whole container, because vLLM and SGLang can use
worker processes in addition to the parent server.

### Host Counter

Host memory comes from cgroup v2 `memory.peak`, read from inside the container
after the request. The kernel maintains that peak for the lifetime of the
cgroup, so it captures model load and request-time memory even though it is read
once at the end. The path requires cgroup v2: if the kernel reports `max`
instead of a value, the run fails rather than guessing.

### GPU Counter

GPU memory is sampled from the host with
`nvidia-smi --query-compute-apps=pid,used_memory`, filtered to the container's
PIDs (cross-referenced against `docker top`), summed, and tracked as a running
maximum. The probe samples every 500 ms: a deliberately slow cadence because
each sample is a subprocess call. It starts after server readiness, so it
captures the persistent GPU allocations visible during the benchmark request; a
larger transient GPU allocation during server startup can fall outside the
sampled window. CPU containers, non-NVIDIA hosts, and runs with no matching
container PID leave `max_gpu_bytes` unset.

### Host and GPU are separate pools

cgroup `memory.peak` is host memory; the NVIDIA GPU carve-out is a separate
pool, so the two fields are disjoint.

### Workload

An OpenAI-compatible `/v1/completions` request with exact-token text,
`max_tokens = 16`, `temperature = 0.0`, and `ignore_eos = true`. The runner
starts the probe before the request, stops it after the request returns, then
validates the response usage fields. `max_tokens = 16` keeps the decode short
while still exercising allocations past prefill.

### Caveats

- `nvidia-smi` reports each process's current commitment at sample time, not a
  peak, so the GPU figure lags the true allocator high-water mark between
  samples, and the driver-plus-runtime overhead mix is not comparable across
  drivers or vendors.
- GPU attribution relies on matching host-namespaced container PIDs from
  `docker top` against the PIDs `nvidia-smi` reports.

## llama.cpp (Linux)

Implemented for CPU flavors only (`linux-arm64-cpu`, `linux-x64-cpu`,
`linux-s390x-cpu`). GPU/accelerator flavors are still rejected by the
dispatcher.

### Host Counter

The runner launches `llama-bench` directly and polls `/proc/<pid>/status`
`VmHWM` (the kernel's peak resident-set high-water mark, the same figure
`wait4`'s `ru_maxrss` reports) in a background thread every 10 ms for the
child's lifetime, tracking the running maximum as `max_host_bytes`. `VmHWM`
only grows, so any sample taken after the peak (model load plus the first
decode step) captures it; the poll is seeded with one synchronous read at spawn
so a child that exits before the thread is scheduled still yields a value.
Reading `/proc` avoids wrapping `wait4` around the child, which would race
`std`'s own reaping in `wait_with_output`.

The sampler is shared with the Android arm
([`max_memory_usage::proc_footprint`](../../crates/pipette-llamacpp/src/execute/max_memory_usage/proc_footprint.rs)),
which also collects `VmSwap`. This arm reports resident peak only, so it carries
the swap under-reporting described in
[peak memory: Android](peak-memory-android.md); adopting the swap-aware figure
here needs the same model-file floor and is left as a follow-up.

### GPU Counter

Unset. CPU flavors have no GPU allocation to attribute, so host RSS is the whole
picture. A GPU Linux flavor would need a DRM-fdinfo probe (in
`pipette-memprobe::os_counters::linux`) before it could route here without
under-reporting, which is why those flavors still bail.

### Workload

`llama-bench --n-prompt <prefill> --n-gen 1 -r 1` with `--mmap 0`, so the
weights load into anonymous memory and count fully toward RSS rather than as
file-backed mmap pages (which would under-report). `--n-gen 1` exercises one
decode step so decode-path kernels and sampling buffers land in the peak. The
child runs in its own process group under a 600 s deadline killer.

## Code References

- [`pipette-torch-oai max_memory_usage`](../../crates/pipette-torch-oai/src/execute/max_memory_usage.rs)
- [`pipette-torch-oai memprobe`](../../crates/pipette-torch-oai/src/memprobe.rs)
- [`pipette-llamacpp max_memory_usage::linux`](../../crates/pipette-llamacpp/src/execute/max_memory_usage/linux.rs) (CPU-flavor `/proc` `VmHWM` path)
