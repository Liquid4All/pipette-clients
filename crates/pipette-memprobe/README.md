# pipette-memprobe

Reads a benchmarked program's memory from the parent. Three parts:

- **`Sampler`**: one process's memory, read by pid, one implementation per OS.
  The llama.cpp, MLX, OVMS, OpenVINO, ONNX GenAI and Core AI
  engines use it.
- **`cgroup::Sampler`** (Linux only): a cgroup's memory, for a server that runs
  as several processes, such as vLLM in a container or a uv venv. torch-oai
  uses it, with `amd` and NVML for the GPU figure.
- **`metal::MetalProbeChannel`** (macOS only): a DYLD-injected shim
  ([`peakmtl/peakmtl.m`](peakmtl/peakmtl.m), built by [`build.rs`](build.rs))
  that logs the Metal allocator's peak as a diagnostic. MLX uses it. The
  reported figure stays the sampler's `phys_footprint` peak.

## Watches

A caller picks a `Watch` for what the cell reports:

- `Watch::Observe` for a timing cell. It reads every 100 ms (Windows once, at
  stop) so being watched does not move the time it reports.
- `Watch::Peak` for a memory cell's host peak.
- `Watch::PeakWithGpu`, `Watch::PeakWithNpu` and `Watch::PeakWithAccelerators`
  add the device memory the OS bills the process.

Every watch exists on every OS. Call `Watch::supported()` before anything
spawns: it refuses a watch this OS cannot read with an `UnsupportedError`
rather than read less than asked. `cgroup::Watch` has its own variants; opening
its tree and GPU readers before the launch is its check.

```rust
use pipette_memprobe::{Sampler, Watch};

let watch = Watch::Peak.supported()?;
let mut process = start_process("llama-bench", command, Duration::ZERO)?;
let sampler = Sampler::start(process.pid(), watch)?;
let (memory, peak) = process.wait_success_before_reap(timeout, || sampler.stop())?;
let max_host_bytes = peak?.host_bytes;
```

Stop the sampler while the process can still be read: a server before it is
stopped, a program that exits after its exit but before the reap
(`wait_success_before_reap`). On Linux, Android and macOS a stop that finds the
process gone refuses the peak (`PeakError::UnreadAtStop`); the Windows sampler
holds a handle, so it reads after the reap.

## Where to look

- Mechanics, counters per OS and cadences: the module docs in
  [`src/lib.rs`](src/lib.rs), [`src/linux.rs`](src/linux.rs) (Linux and
  Android), [`src/macos.rs`](src/macos.rs), [`src/windows.rs`](src/windows.rs)
  and [`src/pdh.rs`](src/pdh.rs), [`src/cgroup.rs`](src/cgroup.rs),
  [`src/drm.rs`](src/drm.rs), [`src/nvml.rs`](src/nvml.rs),
  [`src/amd.rs`](src/amd.rs) and [`src/metal.rs`](src/metal.rs).
- Method: [peak-memory-usage.md](../../docs/methodology/peak-memory-usage.md),
  [peak-memory-android.md#host-counter](../../docs/methodology/peak-memory-android.md#host-counter)
  for the `/proc` sampler,
  [peak-memory-linux.md](../../docs/methodology/peak-memory-linux.md) for the
  cgroup, DRM, NVML and AMD readers,
  [peak-memory-macos.md](../../docs/methodology/peak-memory-macos.md) for
  `phys_footprint` and the Metal shim, and
  [peak-memory-windows.md](../../docs/methodology/peak-memory-windows.md) for
  PSAPI and PDH.
