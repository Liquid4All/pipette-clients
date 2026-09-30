//! Apple Core AI-backed pipette client.
//!
//! Public entry: `run` — CLI owns prepare/record; this crate only executes a
//! prepared `RunRequest` (`pipette_plan_types::run::RunRequest`).
//!
//! **This crate is macOS-only.** The Swift sidecar drives Apple's Core AI
//! framework (`LanguageBundle` + `EngineFactory`) and the Metal-side
//! measurement primitives have no equivalent off Apple Silicon. Describing a
//! `core-ai-macos-pipette://` ref (URI parse, plan-types identity) lives in
//! `pipette-cli` and `pipette-plan-types`, not here. Plain code rather than
//! rustdoc links for plan-types items, because off macOS the dependency is
//! gated out and rustdoc cannot resolve them (`cargo doc --workspace
//! --no-deps` on Linux/Windows).

#[cfg(target_os = "macos")]
mod models;
#[cfg(target_os = "macos")]
mod sidecar;

#[cfg(target_os = "macos")]
pub mod execute;

#[cfg(target_os = "macos")]
pub use execute::run;
#[cfg(target_os = "macos")]
pub use sidecar::{clear_sidecar_cache, require_coreai_sidecar, sidecar_obtainable};
