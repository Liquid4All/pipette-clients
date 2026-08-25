//! Apple Core AI-backed pipette client.
//!
//! Public entry: `run` — CLI owns prepare/record; this crate only executes a
//! prepared [`RunRequest`](pipette_plan_types::run::RunRequest).
//!
//! **This crate is macOS-only.** The Swift sidecar drives Apple's Core AI
//! framework (`LanguageBundle` + `EngineFactory`) and the Metal-side
//! measurement primitives have no equivalent off Apple Silicon. Describing a
//! `core-ai-macos-pipette://` ref (URI parse, plan-types identity) lives in
//! `pipette-cli` and `pipette-plan-types`, not here.

#[cfg(target_os = "macos")]
mod models;
#[cfg(target_os = "macos")]
mod sidecar;

#[cfg(target_os = "macos")]
pub mod execute;

#[cfg(target_os = "macos")]
pub use execute::run;
