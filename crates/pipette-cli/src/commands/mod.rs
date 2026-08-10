//! The unified root command parser and dispatch.
//!
//! Every command resolves its stores from the single
//! [`PipetteWorkspace`] and calls into
//! [`crate::client`] for anything that reaches the management server.
//! `benchmarks run` matches on `Runtime` and calls each runtime crate's
//! `run(&RunRequest, …)` entry; `runtimes` / `models` use the artifact stores
//! and per-crate catalog helpers.

pub mod auth;
pub mod benchmarks;
pub mod models;
pub mod results;
pub mod runtimes;
pub mod storage;
pub mod sync;
pub mod worker;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tabled::{settings::Style, Tabled};

use pipette_http::HttpClient;
use pipette_workspace::{resolve_work_dir, InitResult};

use crate::commands::{
    auth::AuthArgs,
    benchmarks::BenchmarkArgs,
    models::ModelsArgs,
    results::ResultsArgs,
    runtimes::{RuntimeArgs, RuntimeCommand},
    storage::StorageArgs,
    sync::{SyncArgs, SYNC_AFTER_HELP, SYNC_LONG_ABOUT},
    worker::{WorkerArgs, WORKER_AFTER_HELP, WORKER_LONG_ABOUT},
};
use crate::storage_quota::parse_quota_bytes;
use crate::workspace::{self, PipetteWorkspace};

/// Render `rows` as a psql-style table, or print `empty_message` when there are
/// none — so every `list` command handles the empty case the same way.
pub(crate) fn print_table_or<T: Tabled>(rows: &[T], empty_message: &str) {
    if rows.is_empty() {
        println!("{empty_message}");
    } else {
        println!("{}", tabled::Table::new(rows).with(Style::psql()));
    }
}

/// `User-Agent` for this client's management-server requests. One identity /
/// agent for every runtime behind the unified binary.
const USER_AGENT: &str = "pipette";
/// Same string the server is told as `client_version`; see [`crate::CLIENT_VERSION`].
const VERSION: &str = crate::CLIENT_VERSION;

const ROOT_AFTER_HELP: &str = "\
Getting started (a local benchmark needs no server and no registration):
  pipette init
  pipette benchmarks init-local
  pipette benchmarks run --benchmark local/prefill_throughput_smoke \\
    --model '<model-uri>' --runtime '<runtime-uri>'
  pipette results list

`pipette benchmarks run --help` carries worked examples and the per-cell
settings. `pipette runtimes flavors` and `pipette runtimes catalog <type>` work
before a workspace exists, so you can pick a runtime first.

Docs: docs/pipette-cli/usage.md, and models-and-runtimes.md for the `--model` /
`--runtime` / `--runtime-flags` notation.";

/// Unified CLI for running benchmarks across runtimes and submitting results.
#[derive(Parser, Debug)]
#[command(name = "pipette", version = VERSION, after_help = ROOT_AFTER_HELP)]
pub struct Cli {
    /// Working directory (default: current directory)
    #[arg(long, global = true, env = "PIPETTE_WORK_DIR")]
    pub work_dir: Option<PathBuf>,

    /// Artifact-store disk cap: bytes, or an IEC suffix (`200GiB`, `512MiB`)
    #[arg(long, global = true, env = "PIPETTE_STORAGE_QUOTA")]
    pub storage_quota: Option<String>,

    #[command(subcommand)]
    pub command: RootCommand,
}

#[derive(Subcommand, Debug)]
#[allow(clippy::large_enum_variant)] // Benchmarks(BenchmarkArgs) carries the full benchmarks-run CLI surface; boxing ripples through all call sites
pub enum RootCommand {
    /// Initialize the .pipette workspace
    Init,
    /// Manage locally cached models (per runtime)
    Models(ModelsArgs),
    /// Manage installed runtime builds
    Runtimes(RuntimeArgs),
    /// Manage client identity and server registration
    Auth(AuthArgs),
    /// List, inspect, and run benchmarks
    Benchmarks(BenchmarkArgs),
    /// List, inspect, and delete benchmark results
    Results(ResultsArgs),
    /// Sync benchmarks and results with the management server
    #[command(long_about = SYNC_LONG_ABOUT, after_help = SYNC_AFTER_HELP)]
    Sync(SyncArgs),
    /// Run the planner claim loop (claim → run → submit)
    #[command(long_about = WORKER_LONG_ABOUT, after_help = WORKER_AFTER_HELP)]
    Worker(WorkerArgs),
    /// Inspect and reclaim local artifact disk usage
    Storage(StorageArgs),
}

pub fn run() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // Teardown ^C/SIGTERM handler so an interrupted run stops its
    // server/container instead of orphaning it.
    pipette_subprocess::cleanup::install_signal_handler();
    let cli = Cli::parse();
    // clap surfaces an empty PIPETTE_WORK_DIR as Some(""); treat it as unset.
    let work_dir_arg = cli.work_dir.filter(|p| !p.as_os_str().is_empty());
    let work_dir = resolve_work_dir(work_dir_arg.as_deref())?;
    let quota_override = cli
        .storage_quota
        .as_deref()
        .map(str::trim)
        .filter(|raw| !raw.is_empty())
        .map(parse_quota_bytes)
        .transpose()?;

    // `init` runs before opening a workspace — there is nothing to open yet.
    if let RootCommand::Init = cli.command {
        match PipetteWorkspace::init(&work_dir)? {
            InitResult::Created(root) => {
                println!("Initialized pipette workspace at {}", root.display());
            }
            InitResult::AlreadyExists(root) => {
                println!(
                    "pipette workspace already initialized at {}",
                    root.display()
                );
            }
        }
        return Ok(());
    }

    // One shared HTTP client for the process — TLS + UA explicit here;
    // passed by ref into every network command (including workspace-free
    // `runtimes catalog`, which may hit GitHub for llama.cpp).
    let http = HttpClient::builder(USER_AGENT)
        .preconfigured_tls()
        .build()?;

    // `runtimes catalog <type>` and `runtimes flavors` both describe what is
    // installable without touching the store — they need no workspace, so a
    // fresh box can discover what to pull before running `init`.
    match cli.command {
        RootCommand::Runtimes(RuntimeArgs {
            command: RuntimeCommand::Catalog(args),
        }) => args.execute(&http),
        RootCommand::Runtimes(RuntimeArgs {
            command: RuntimeCommand::Flavors,
        }) => runtimes::print_llamacpp_flavors(),
        command => {
            workspace::require_workspace(&work_dir, work_dir_arg.as_deref())?;
            // Parsed here so a malformed value is reported before the workspace
            // is opened, let alone before a multi-gigabyte fetch; `open` resolves
            // it against `identity/settings.json` and carries the result.
            let ws = PipetteWorkspace::open(&work_dir, quota_override)?;
            match command {
                RootCommand::Init => anyhow::bail!("init is handled before workspace open"),
                RootCommand::Models(args) => args.execute(&ws, &http),
                RootCommand::Runtimes(args) => args.execute(&ws, &http),
                RootCommand::Auth(args) => args.execute(&ws, &http).map_err(Into::into),
                RootCommand::Benchmarks(args) => args.execute(&ws, &http),
                RootCommand::Results(args) => args.execute(&ws.results()),
                RootCommand::Sync(args) => args.execute(&ws, &http).map_err(Into::into),
                RootCommand::Worker(args) => args.execute(&ws, &http),
                RootCommand::Storage(args) => args.execute(&ws),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    /// The cap is workspace-wide, so it has to parse after any subcommand — not
    /// only on `storage`.
    #[test]
    fn storage_quota_is_a_global_arg() -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(["pipette", "models", "list", "--storage-quota", "10GiB"])?;
        assert_eq!(cli.storage_quota.as_deref(), Some("10GiB"));
        Ok(())
    }
}
