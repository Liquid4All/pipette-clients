/// Driver version, `<crate> (build <stamp>)` — the same shape the client
/// reports, so `--version` output from either is directly comparable. Which
/// plan features parse is a function of this value: a transport added after a
/// release cannot be read by that release's driver.
pub const PLAN_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (build ",
    env!("PIPETTE_CLI_BUILD_VERSION"),
    ")"
);

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
