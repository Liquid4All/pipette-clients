//! Apple Core AI-backed pipette client.
//!
//! Public entry: `run` — CLI owns prepare/record; this crate only executes a
//! prepared `RunRequest` (`pipette_plan_types::RunRequest`).
//!
//! **This crate is macOS-only.** The Swift sidecar drives Apple's Core AI
//! framework (`LanguageBundle` + `EngineFactory`) and the Metal-side
//! measurement primitives have no equivalent off Apple Silicon. Describing a
//! `core-ai-macos-pipette://` ref (URI parse, plan-types identity) lives in
//! `pipette-cli` and `pipette-plan-types`, not here. This page names
//! plan-types items as plain code: off macOS the build excludes the dependency,
//! so rustdoc cannot resolve links to them (`cargo doc --workspace --no-deps`
//! on Linux and Windows).

#[cfg(target_os = "macos")]
mod measurement;
#[cfg(target_os = "macos")]
mod models;
#[cfg(target_os = "macos")]
mod server;
#[cfg(target_os = "macos")]
mod sidecar;
#[cfg(target_os = "macos")]
mod throughput_http;

#[cfg(target_os = "macos")]
mod execute;

#[cfg(target_os = "macos")]
pub use execute::run;
