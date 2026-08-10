//! Workspace artifact cache: models and runtimes under `models/` / `runtimes/`.
//!
//! - [`ArtifactsContext`] — host download HTTP client + lazy tool executables
//! - [`ensure_model`] / [`ensure_runtime`] — find-or-fetch into the stores
//! - [`model`] / [`runtime`] — store layout, fetch/materialize, installers
//! - [`quota`] — disk cap accounting and the post-publish sweep
//!
//! Where an artifact lands is the stores' answer to give: callers read a path
//! off a manifest rather than rebuilding the layout, so `entry` — the
//! directory-entry constants and the atomic stage/publish engine — stays
//! internal.

mod context;
mod entry;

pub mod ensure;
pub mod model;
pub mod progress;
pub mod quota;
pub mod runtime;

pub use context::ArtifactsContext;
pub use ensure::{
    ensure_model, ensure_runtime, model_download_size, resolve_python_executable,
    runtime_download_size, runtime_tool,
};
pub use progress::{FetchProgress, ProgressSink};
