/// Driver version — the release this driver was published in, verbatim (e.g.
/// `2026.08.1-0-g58c2adbf16`), or `dev` for a local build. The same string the
/// client reports, so `--version` output from either is directly comparable and
/// both name the same GitHub release. Which plan features parse is a function of
/// this value: a transport added after a release cannot be read by that
/// release's driver.
pub const PLAN_VERSION: &str = env!("PIPETTE_CLI_BUILD_VERSION");

pub mod capability_rules;
pub mod cli;
pub mod generate;
pub mod runner;
pub(crate) mod shell;
pub mod ssh_keygen;
pub mod state;
pub(crate) mod transport;
pub mod workspace;

pub use cli::Cli;
