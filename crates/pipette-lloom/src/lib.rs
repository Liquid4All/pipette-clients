//! lloom-backed pipette client: `lloom-serve` (`ocean1/lloom`'s LFM2 engine
//! on Metal) driven as a local sidecar.
//!
//! Public entry: `run` — the CLI owns prepare/record; this crate only executes
//! a prepared [`RunRequest`](pipette_plan_types::run::RunRequest).
//!
//! **Running** is Apple Silicon only: the binary is a Metal engine and the
//! memory cell leans on `pipette-memprobe-metal`, so `execute` is gated to
//! `target_os = "macos"` and off-platform builds skip it. **Describing** a
//! runtime or a model is not: [`models`] and [`runtimes`] are plan-types work
//! and compile everywhere, so `pipette-cli` can resolve an
//! `lloom-serve-macos://` ref while authoring or pulling on any host.
//!
//! The sidecar contract is the MLX one (`/prefill_throughput`,
//! `/decode_throughput`, `/end_to_end_latency`, `/max_memory_usage`, `/eval`
//! with `/eval/abort`, `/tokenize`, `/shutdown`, and a `{"kind":"ready"}` line
//! on stdout), spoken by a native binary instead of a Python script: no venv,
//! no script to materialize, nothing to reap between cells.

pub mod models;
pub mod runtimes;

#[cfg(target_os = "macos")]
pub mod execute;

#[cfg(target_os = "macos")]
pub use execute::run;
