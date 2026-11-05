/// Driver version: the release this driver shipped in (e.g.
/// `2026.08.11-640d96e8`), or `dev` for a local build. The client reports the
/// same string. Which plan features parse depends on it: a transport added
/// after a release cannot be read by that release's driver.
pub const PLAN_VERSION: &str = env!("PIPETTE_CLI_BUILD_VERSION");

/// An HTTP client with no request timeout, for mgmt's uploads.
///
/// Built where it is needed rather than once per invocation: most of this
/// binary's verbs reach no server, and `run` would pay for a TLS stack it
/// never sends through.
pub(crate) fn http_client() -> anyhow::Result<pipette_http::HttpClient> {
    use anyhow::Context;

    pipette_http::HttpClient::new("pipette-ctl").context("to build the HTTP client")
}

pub mod auth;
pub mod cli;
pub(crate) mod host;
pub mod identity;
pub mod mgmt;
pub mod runner;
pub(crate) mod shell;
#[cfg(test)]
mod ssh_keygen;
pub mod state;
pub mod system_keys;
#[cfg(test)]
mod test_support;
pub(crate) mod transport;
pub mod workspace;

pub use cli::Cli;
