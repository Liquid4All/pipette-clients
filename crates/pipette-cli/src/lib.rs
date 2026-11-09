//! `pipette`, the single-binary client: the workspace stores ([`benchmarks`],
//! [`identity`], [`results`]), the management-server flows over them
//! ([`client`]), the `--runtime` / `--model` URI grammar and the clap surface in
//! [`commands`]. A run dispatches with a `match` on `Runtime` into each backend
//! crate's `run` (no handler trait). See `docs/architecture.md`.

/// This build's identity: `client_version` on every submission and what
/// `pipette --version` prints. The release's own version (the `release-metadata`
/// job in `.github/workflows/build.yml`, also the GitHub release tag, e.g.
/// `2026.08.11-640d96e8`), so a warehouse row maps to a release by equality.
/// `CARGO_PKG_VERSION` is left out: it never moves off `0.1.0`. Local builds
/// report `dev` (see `build.rs`).
pub const CLIENT_VERSION: &str = env!("PIPETTE_CLI_BUILD_VERSION");

pub mod artifact_ref;
mod attempt_watchdog;
pub(crate) mod benchmarks;
pub(crate) mod client;
pub(crate) mod client_configuration;
pub mod commands;
pub mod console;
pub(crate) mod error;
mod hf_auth;
pub(crate) mod identity;
pub mod logging;
pub mod model_uri;
mod preflight;
pub mod progress;
pub(crate) mod rep_telemetry;
pub(crate) mod results;
mod run;
pub mod runtime_uri;
mod score_validation;
mod secret_file;
pub(crate) mod storage_quota;
#[cfg(test)]
mod test_support;
pub(crate) mod workspace;
