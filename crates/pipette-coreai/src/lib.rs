//! Apple Core AI-backed pipette client.
//!
//! Public entry: `run` — CLI owns prepare/record; this crate only executes a
//! prepared [`RunRequest`](pipette_plan_types::run::RunRequest).
//!
//! **Running** is Apple Silicon + macOS 27 only: the Swift sidecar drives
//! Apple's Core AI framework (`LanguageBundle` + `EngineFactory`), which has no
//! equivalent on other hosts, and the Metal-side measurement primitives
//! (`pipette-memprobe-metal`'s DYLD shim, `phys_footprint`) have no
//! equivalent off Apple Silicon. So `execute` is gated to `target_os =
//! "macos"` and off-platform builds skip it rather than fail on unresolved
//! imports, mirroring `pipette-mlx`.
//!
//! **Describing** one is not. [`models`] and [`runtimes`] are plan-types work
//! — a path projection, a version — so they compile everywhere, which is what
//! lets `pipette-cli` resolve a `core-ai-macos://` ref while authoring or
//! pulling on a Linux box.

pub mod models;
pub mod runtimes;

#[cfg(target_os = "macos")]
pub mod execute;

#[cfg(target_os = "macos")]
pub use execute::run;
