//! MLX-backed pipette client.
//!
//! Public entry: `run` — CLI owns prepare/record; this crate only executes a
//! prepared [`RunRequest`](pipette_plan_types::run::RunRequest).
//!
//! **Running** is Apple Silicon only: `mlx_lm` and the Metal-side measurement
//! primitives the bench paths depend on (`pipette-memprobe-metal`'s DYLD shim,
//! `phys_footprint`) have no equivalent on other hosts, so `execute` is gated
//! to `target_os = "macos"` and off-platform builds skip it rather than fail on
//! unresolved imports. Plain code rather than links, because off macOS neither
//! item is in scope for rustdoc to resolve.
//!
//! **Describing** one is not. [`catalog`], [`models`] and [`runtimes`] are
//! plan-types work — a version, a pinned requirements body, a path projection —
//! so they compile everywhere, which is what lets `pipette-cli` resolve an
//! `mlx-macos-pipette://` ref while authoring or pulling on a Linux box.

pub mod catalog;
pub mod models;
pub mod runtimes;

#[cfg(target_os = "macos")]
pub mod execute;

#[cfg(target_os = "macos")]
pub use execute::run;
