//! Workspace artifact cache: models and runtimes under `models/` / `runtimes/`.
//!
//! Callers read an artifact's path off a manifest rather than rebuilding the
//! layout, so `entry` (the directory-entry constants and the atomic
//! stage/publish engine) stays internal; only its [`sync_file`] and
//! [`sync_dir`] durability helpers are exported.

mod archive;
mod context;
mod digest;
mod entry;
mod io_kind;
mod resume;

pub mod ensure;
pub mod model;
pub mod progress;
pub mod quota;
pub mod runtime;
#[cfg(test)]
mod scripted_server;
#[cfg(test)]
mod test_support;
pub mod vault;

pub use context::ArtifactsContext;
pub use ensure::{
    ensure_model, ensure_runtime, model_download_size, runtime_download_size, runtime_tool,
};
pub use entry::{sync_dir, sync_file};
pub use progress::{FetchProgress, ProgressSink};
pub use vault::{VaultAccess, VaultFetchError};
