# pipette-memprobe-metal

macOS Metal peak-memory probe: DYLD-injected `peakmtl.dylib` shim
plus the Mac host counter (`phys_footprint`) that pairs with it.

This crate is the Mac-side measurement primitive used by
`pipette-llamacpp` (Mac path) and `pipette-mlx` (Apple Silicon
only). The crate is gated to `target_os = "macos"`; on non-Mac
hosts it compiles to an empty crate, so any
`use pipette_memprobe_metal::*` import must be cfg-gated by the
consumer (or live in a module that is itself Mac-only).

The wire-protocol side of this (what `max_host_bytes` /
`max_gpu_bytes` mean and what the mgmt server expects) lives in
[`pipette-mgmt/docs/methodology/peak-memory.md`](../../../pipette-mgmt/docs/methodology/peak-memory.md).

## What's not in this crate

- **Linux/Windows host counters** (`wait4 ru_maxrss` /
  `GetProcessMemoryInfo.PeakWorkingSetSize`): inlined into
  `pipette-llamacpp/src/execute/max_memory_usage/generic.rs`. They
  share no abstraction with the Mac probe and have a single consumer
  each.
- **Sidecar OS-attribution counters** (PDH, DRM fdinfo, nvidia-smi):
  live in `pipette-llamacpp/src/execute/max_memory_usage/sidecar/`.
  Diagnostic data, not the wire-schema source.
- **Android measurement**: samples `/proc/<pid>/status` for
  `max(VmHWM, VmRSS + VmSwap)`; lives in `pipette-llamacpp/src/execute/max_memory_usage/android.rs`.
- **Process-timeout helpers** (`spawn_timeout_killer`,
  `run_command_with_timeout`): moved to where they're used
  (`pipette-mlx/src/execute/python.rs`).

Each per-OS measurement strategy reads top-to-bottom in one place,
next to the benchmark dispatch. This crate exists for the one piece
that is genuinely shared and non-trivial: the Mac Metal probe with
its embedded shim, build-time SHA-256 caching, and two consumers.

## Modules

```rust
pub mod metal;        // MetalProbeChannel + MetalPeak + dyld_blocked_error
pub mod host;         // spawn_phys_footprint_poller (Mac phys_footprint)
```

### `metal::` (macOS Metal probe)

```rust
pub struct MetalProbeChannel { /* opaque */ }
impl MetalProbeChannel {
    /// Extract the bundled dylib (cached by content hash), allocate
    /// a per-run output tempfile, wire DYLD_INSERT_LIBRARIES +
    /// PIPETTE_MEMPROBE_OUT into `cmd`. Holds the tempdir until drop.
    pub fn attach(cmd: &mut Command) -> Result<Self>;

    /// After the child exits, read the shim's snapshot. Errors are
    /// operator-actionable: snapshot file missing (DYLD blocked by
    /// Hardened Runtime / Library Validation / SIP), snapshot
    /// present but missing the required `metal_peak_allocated_bytes`
    /// line (shim anomaly), or I/O error.
    pub fn read_peak(&self) -> Result<MetalPeak>;
}

#[non_exhaustive]
pub struct MetalPeak {
    pub bytes: u64,             // [MTLDevice currentAllocatedSize] peak (required)
    pub unified: Option<bool>,  // [MTLDevice hasUnifiedMemory] — diagnostic only
    pub n_devices: Option<u32>, // diagnostic only
}
```

The Metal probe is a DYLD-injected dylib (`peakmtl/peakmtl.m`,
embedded into the binary at build time) that polls
`[MTLDevice currentAllocatedSize]` every 20 ms and writes its peak
to the parent-controlled tempfile that `MetalProbeChannel::attach`
allocates. The shim writes on every peak grow plus once at `atexit`,
so abnormal exits (`_exit` skipping atexit: CPython + MLX takes
that path) still surface the peak.

### `host::` (Mac host counter)

```rust
pub fn spawn_phys_footprint_poller(pid: i32) -> PhysFootprintPoller;
pub struct PhysFootprintPoller { /* opaque */ }
impl PhysFootprintPoller { pub fn stop_and_join(self) -> u64; }
```

Polls `proc_pid_rusage(pid, RUSAGE_INFO_V4)` every 20 ms while the
child runs, taking
`max(ri_phys_footprint, ri_lifetime_max_phys_footprint)`. The
kernel's `ri_lifetime_max_phys_footprint` is a **monotonic
high-water mark** maintained inside the kernel: even if the
20 ms poll misses a sub-poll spike, the next poll observes the
elevated lifetime max. The 20 ms cadence is therefore latency for
detection, not for accuracy.

The poller's value is reported **directly** as `max_host_bytes`.
On Apple UMA, Metal allocations are billed to `phys_footprint`, so
the host bucket is a superset of `max_gpu_bytes`. Each peak is
reported independently (no cross-subtraction), because they're
two separate dimensions: the GPU allocator's high-water mark, and
the kernel's process-level high-water mark.

## Consumer pattern

A typical macOS measured-bench consumer (see
`pipette-llamacpp/src/execute/max_memory_usage/macos.rs` and
`pipette-mlx/src/execute/max_memory_usage.rs`):

```rust
use pipette_memprobe_metal::{host, metal};
use std::process::{Command, Stdio};

let mut cmd = Command::new(&exec_path);
/* … apply caller args … */

// Probe attach: dylib + tempfile + DYLD env wiring, all in one call.
let probe = metal::MetalProbeChannel::attach(&mut cmd)?;
cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

// Spawn + phys_footprint poller alongside; wait_with_output drains
// pipes and reaps. Stop the poller after wait.
let child = cmd.spawn()?;
let phys_poller = host::spawn_phys_footprint_poller(child.id() as i32);
let output = child.wait_with_output()?;
let phys_peak = phys_poller.stop_and_join();

// Read shim snapshot. Each peak is reported independently — no
// cross-subtraction. On UMA the two values overlap (Metal lives
// inside phys_footprint); each is its own dimension's peak.
// `probe` must outlive `read_peak` — when the channel drops, the
// per-run tempdir is removed and the snapshot file is gone.
let metal_peak = probe.read_peak()?;
let max_gpu_bytes  = metal_peak.bytes;
let max_host_bytes = phys_peak;
```

## Why this shape

The crate started as a kit of OS-spread "memory measurement
primitives": Mac probe, Linux/Windows host counters, OS-attribution
sidecar, generic timeout killer. In practice the Linux/Windows paths
share no abstraction with Mac (different syscall families, different
return-handle lifetimes), each non-Mac primitive had only a single
consumer, and the OS-attribution sidecar is only consumed by one
benchmark module. Concentrating the kit in one place created the
illusion of a unified interface that the actual code doesn't honor.

The current shape pulls all of that out: each per-OS measurement
strategy lives next to the benchmark that drives it, and this crate
is what's left when you keep only the parts that are genuinely
shared and non-trivial; the Mac Metal probe (DYLD shim, build.rs
compilation, SHA-256-keyed extraction) and the Mac host counter
that pairs with it.

If/when in-process Vulkan / CUDA / HIP / D3D12 probes land for
Linux/Windows, the natural shape is a sibling crate
(`pipette-memprobe-vulkan`, etc.) with the same
`*ProbeChannel::attach` API surface.

## Caveats

- **`DYLD_INSERT_LIBRARIES` is ignored** on binaries with Hardened
  Runtime + Library Validation, on `setuid` binaries, and on
  SIP-protected system binaries. The shim's constructor never runs
  in that case; `MetalProbeChannel::read_peak` returns `Err` with
  the operator-actionable diagnostic message, and the bench fails
  loudly via the consumer's `?`.
- **Polling cadence is 20 ms.** The empirical signal that this is
  enough is that the macOS Metal probe matches llama.cpp's announced
  buffer sums to within a constant +1.30 MiB of driver state across
  every prompt size we've measured (256, 2048, 8192). A missed peak
  would manifest as a variable, prompt-size-correlated delta. See
  `pipette-mgmt/docs/methodology/peak-memory.md` §3.1 "Polling
  cadence and failure-detection signal" for diagnosis when running
  on a future workload.

## Where to look for more

- Wire-protocol meaning and per-OS methodology:
  [`pipette-mgmt/docs/methodology/peak-memory.md`](../../../pipette-mgmt/docs/methodology/peak-memory.md).
- The Metal shim source: [`peakmtl/peakmtl.m`](peakmtl/peakmtl.m).
- Build-time compilation of the shim: [`build.rs`](build.rs).
- Mac consumer orchestration:
  [`pipette-llamacpp/src/execute/max_memory_usage/macos.rs`](../pipette-llamacpp/src/execute/max_memory_usage/macos.rs),
  [`pipette-mlx/src/execute/python.rs`](../pipette-mlx/src/execute/python.rs).
- Linux/Windows/Android measurement (does not use this crate):
  [`pipette-llamacpp/src/execute/max_memory_usage/`](../pipette-llamacpp/src/execute/max_memory_usage/).
