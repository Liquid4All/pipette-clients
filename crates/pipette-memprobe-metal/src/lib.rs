//! macOS Metal peak-memory probe + the Mac host counter that pairs
//! with it.
//!
//! This crate is the Mac-side measurement primitive used by
//! `pipette-llamacpp` (Mac path) and `pipette-mlx` (Apple Silicon
//! only). The crate is gated to `target_os = "macos"` — on non-Mac
//! hosts it compiles to an empty crate, so any `use
//! pipette_memprobe_metal::*` import must be cfg-gated by the
//! consumer (or live in a module that is itself Mac-only).
//!
//! See `pipette-mgmt/docs/methodology/peak-memory.md` for the
//! wire-protocol and per-OS methodology, and `README.md` for the
//! consumer pattern.
//!
//! ## Public surface
//!
//! Links are plain code below: this crate is `#![cfg(target_os = "macos")]`, so
//! off-macOS these docs are rendered with no items behind them to point at.
//!
//! - `metal` — `metal::MetalProbeChannel` (`attach` /
//!   `read_peak`) and `metal::MetalPeak`. `read_peak` returns
//!   operator-actionable errors for the DYLD-blocked and shim-anomaly
//!   cases; consumers propagate via `?`.
//! - `host` — `host::spawn_phys_footprint_poller` returning a
//!   `host::PhysFootprintPoller` (`stop_and_join`). Polls
//!   `proc_pid_rusage(RUSAGE_INFO_V4).ri_phys_footprint`. Reported
//!   directly as `max_host_bytes`; on Apple UMA this is a superset
//!   of `max_gpu_bytes` (Metal allocations are billed here too) —
//!   each peak is reported independently, no cross-subtraction.
//!
//! Linux/Windows host counters (`wait4 ru_maxrss`,
//! `GetProcessMemoryInfo.PeakWorkingSetSize`) and the OS-attribution
//! sidecar (PDH / DRM fdinfo / nvidia-smi) live in
//! `pipette-llamacpp/src/execute/max_memory_usage/` next to the
//! benchmark that consumes them — they share no abstraction with
//! the Mac probe and have a single consumer each.

#![cfg(target_os = "macos")]

pub mod host;
pub mod metal;
