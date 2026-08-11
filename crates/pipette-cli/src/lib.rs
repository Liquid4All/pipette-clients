//! `pipette` — the unified single-binary client.
//!
//! One binary owns the shared workspace (benchmarks / results / identity /
//! sync) and dispatches with a `match` on `Runtime` into each backend crate's
//! `run(&RunRequest, …)` entry (`pipette-llamacpp`, `pipette-mlx`,
//! `pipette-torch-oai`) — no handler trait. See `docs/pipette/design.md`.
//!
//! This crate is the client: the workspace and its stores ([`benchmarks`],
//! [`identity`], [`results`]), the management-server flows over them
//! ([`client`]), the `--runtime` / `--model` URI grammar, and the clap surface
//! in [`commands`]. What lives elsewhere: the runtime and measurement layer in
//! `pipette-ops`, the model / runtime artifact cache in `pipette-artifacts`.

/// This build's identity, reported to the management server as
/// `client_version` on every submission and printed by `pipette --version`.
///
/// Verbatim the version the release was published under — `ci/version.sh`'s
/// output, which is also the GitHub release's tag and name, e.g.
/// `2026.08.1-0-g58c2adbf16`. One string, so a warehouse row maps to a
/// downloadable release by equality and not by anyone's parsing rule.
///
/// The crate's `CARGO_PKG_VERSION` is deliberately *not* part of it. It has
/// never moved off `0.1.0` and nothing releases on it, so prefixing it made
/// every row read `0.1.0 (build …)` — a constant in front of the only part that
/// identified anything, and not a string that matches any release page.
///
/// Deliberately the *same* string the wire gets and `--version` prints: the
/// column exists so a shift in the numbers can be attributed to a harness
/// change, and that attribution starts from someone pasting what `--version`
/// printed. A second, tidier spelling for the wire would break that match for
/// nothing — the server stores the value opaquely and never parses it.
///
/// Local builds report `dev` (see `build.rs`), so a developer's submissions are
/// distinguishable from a released build's.
pub const CLIENT_VERSION: &str = env!("PIPETTE_CLI_BUILD_VERSION");

pub mod artifact_ref;
pub(crate) mod benchmarks;
pub(crate) mod client;
pub mod commands;
mod doomloop_cli;
pub(crate) mod error;
mod hf_auth;
pub(crate) mod identity;
pub mod model_uri;
pub mod progress;
pub(crate) mod results;
mod run;
pub mod runtime_uri;
mod score_validation;
pub(crate) mod storage_quota;
pub(crate) mod workspace;
