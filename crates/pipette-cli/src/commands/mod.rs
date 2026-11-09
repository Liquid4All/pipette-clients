//! The root command parser and dispatch. Every command opens its stores from
//! the one [`PipetteWorkspace`] and reaches the management server through
//! [`crate::client`].

pub mod auth;
pub mod benchmarks;
pub mod check;
pub mod models;
pub mod results;
pub mod runtimes;
pub mod storage;
pub mod sync;
pub mod worker;

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};
use tabled::{Tabled, settings::Style};
use tracing::level_filters::LevelFilter;

use pipette_http::HttpClient;
use pipette_ops::EvalShard;
use pipette_programs::{ExternalProgram, ExternalPrograms};
use pipette_workspace::{InitResult, resolve_work_dir};

use crate::commands::{
    auth::AuthArgs,
    benchmarks::{BenchmarkArgs, BenchmarkCommand},
    check::{CHECK_AFTER_HELP, CHECK_LONG_ABOUT, CheckArgs},
    models::ModelsArgs,
    results::ResultsArgs,
    runtimes::{RuntimeArgs, RuntimeCommand},
    storage::StorageArgs,
    sync::{SYNC_AFTER_HELP, SYNC_LONG_ABOUT, SyncArgs},
    worker::{WORKER_AFTER_HELP, WORKER_LONG_ABOUT, WorkerArgs},
};
use crate::console::Console;
use crate::logging::LogFormat;
use crate::storage_quota::parse_quota_bytes;
use crate::workspace::{self, PipetteWorkspace};

/// Refuse now, rather than at first spawn, for the programs `pipette worker
/// --strict` named. Nothing by default: only the operator knows what this box
/// is supposed to have.
fn require_programs_up_front(
    command: &RootCommand,
    programs: &ExternalPrograms,
) -> anyhow::Result<()> {
    let RootCommand::Worker(worker) = command else {
        return Ok(());
    };
    programs.require_working(&worker.strict_programs()).context(
        "every program named by --strict has to answer for itself before this \
         worker takes a lease",
    )
}

/// An override the invocation actually made: clap surfaces an exported-but-empty
/// env var as `Some("")`, and `PIPETTE_UV_PATH=` in a unit file means the
/// operator set nothing, not that they named the empty path.
fn named_path(path: Option<&Path>) -> Option<&Path> {
    path.filter(|p| !p.as_os_str().is_empty())
}

/// Write `rows` as a psql-style table, or `empty_message` when there are none.
pub(crate) fn write_table_or<T: Tabled>(
    out: &mut dyn Write,
    rows: &[T],
    empty_message: &str,
) -> std::io::Result<()> {
    if rows.is_empty() {
        writeln!(out, "{empty_message}")
    } else {
        writeln!(out, "{}", tabled::Table::new(rows).with(Style::psql()))
    }
}

/// `User-Agent` for the shared HTTP client [`dispatch`] builds.
const USER_AGENT: &str = "pipette";
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
before a workspace exists, so you can pick a runtime first, and
`pipette check all` says whether this host's external programs (uv, docker, the
GPU CLIs) are there and working.

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

    /// Run only group I of N (0-based) of an eval, vlm_eval or asr_eval cell,
    /// split by a hash of each sample id, into a part file beside its
    /// checkpoint. Records and reports nothing: a later run without the flag
    /// folds every group in and records the cell. `worker` takes it only
    /// behind the pool dispatcher, which holds the lease.
    #[arg(long, global = true, value_name = "I/N")]
    pub eval_shard: Option<EvalShard>,

    /// How this client operates, as a TOML file: one `[[runtimes]]` entry per
    /// runtime type with the `envs` set on every process it spawns, and a
    /// `[readiness.<gate>]` table tuning this host's readiness gate. Host
    /// facts, not cell facts — nothing in it reaches the reported
    /// `runtime_flags`. See docs/pipette-cli/usage.md.
    #[arg(long, global = true, env = "PIPETTE_CONFIG", value_name = "TOML")]
    pub config: Option<PathBuf>,

    /// Log level for every pipette crate (default info): error, warn, info,
    /// debug or trace. Dependencies stay at warn. A set RUST_LOG replaces the
    /// whole filter.
    #[arg(long, global = true, value_name = "LEVEL")]
    pub log_level: Option<LevelFilter>,

    /// Shorthand for --log-level debug; the explicit flag wins.
    #[arg(short = 'v', long, global = true)]
    pub verbose: bool,

    /// Log lines as text, or one JSON object per line (for a journal).
    #[arg(long, global = true, value_name = "FORMAT", default_value = "text")]
    pub log_format: LogFormat,

    /// The `uv` to provision venv-backed runtimes with: the binary, or a
    /// directory holding it. Default: the first `uv` on PATH.
    #[arg(long, global = true, env = "PIPETTE_UV_PATH", value_name = "PATH")]
    pub uv_path: Option<PathBuf>,

    /// The `docker` to pull and run container-backed runtimes with: the binary,
    /// or a directory holding it. Default: the first `docker` on PATH.
    #[arg(long, global = true, env = "PIPETTE_DOCKER_PATH", value_name = "PATH")]
    pub docker_path: Option<PathBuf>,

    /// The `nvidia-smi` the NVIDIA preflight runs: the binary, or a directory
    /// holding it. Default: the first on PATH, and a host with none is not an
    /// error.
    #[arg(
        long,
        global = true,
        env = "PIPETTE_NVIDIA_SMI_PATH",
        value_name = "PATH"
    )]
    pub nvidia_smi_path: Option<PathBuf>,

    /// The `rocm-smi` the AMD preflight runs: the binary, or a directory
    /// holding it. Default: the first on PATH, and a host with none is not an
    /// error.
    #[arg(
        long,
        global = true,
        env = "PIPETTE_ROCM_SMI_PATH",
        value_name = "PATH"
    )]
    pub rocm_smi_path: Option<PathBuf>,

    /// The host Python, for anything run outside a runtime's own venv: the
    /// binary, or a directory holding it. Default: `python3` then `python` on
    /// PATH. Every runtime's interpreter comes from its install instead, so a
    /// host with no Python still runs benchmarks.
    #[arg(long, global = true, env = "PIPETTE_PYTHON_PATH", value_name = "PATH")]
    pub python_path: Option<PathBuf>,

    #[command(subcommand)]
    pub command: RootCommand,
}

impl Cli {
    /// Locate every external program, once, before dispatch: a `--uv-path` typo
    /// should surface at the start of the run, not as a spawn failure an hour
    /// into a sweep. None has to be present; a missing program is reported by
    /// whatever first needs it. A struct literal rather than a positional
    /// constructor: same-typed paths in a row invite a transposition.
    fn resolve_external_programs(&self) -> ExternalPrograms {
        ExternalPrograms {
            uv: ExternalProgram::UV.resolve(named_path(self.uv_path.as_deref())),
            docker: ExternalProgram::DOCKER.resolve(named_path(self.docker_path.as_deref())),
            nvidia_smi: ExternalProgram::NVIDIA_SMI
                .resolve(named_path(self.nvidia_smi_path.as_deref())),
            rocm_smi: ExternalProgram::ROCM_SMI.resolve(named_path(self.rocm_smi_path.as_deref())),
            python: ExternalProgram::PYTHON.resolve(named_path(self.python_path.as_deref())),
        }
    }

    fn requested_log_level(&self) -> LevelFilter {
        match (self.log_level, self.verbose) {
            (Some(level), _) => level,
            (None, true) => LevelFilter::DEBUG,
            (None, false) => LevelFilter::INFO,
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum RootCommand {
    /// Initialize the .pipette workspace
    Init,
    /// Manage locally cached models
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
    /// Check that the external programs this host needs are present and working
    #[command(long_about = CHECK_LONG_ABOUT, after_help = CHECK_AFTER_HELP)]
    Check(CheckArgs),
}

/// The process entry point: the process-wide setup that can only happen once
/// (signal handler, argv, the global `tracing` subscriber, the console), then
/// [`dispatch`], which reaches for none of it.
pub fn run() -> anyhow::Result<()> {
    // Teardown ^C/SIGTERM handler so an interrupted run stops its
    // server/container instead of orphaning it.
    pipette_subprocess::install_signal_handler();
    let cli = Cli::parse();
    // After parse: the level comes off the command line. Nothing above logs.
    crate::logging::init(cli.requested_log_level(), cli.log_format);
    // Over adb or ssh stderr is the session: a cell or job it drives stops with it.
    if matches!(
        cli.command,
        RootCommand::Worker(_)
            | RootCommand::Benchmarks(BenchmarkArgs {
                command: BenchmarkCommand::Run(_)
            })
    ) {
        pipette_subprocess::stop_on_closed_output();
    }
    dispatch(cli, &mut crate::console::Stdio::default())
}

/// Resolve the host facts this invocation names, then run its command. Driven
/// only by its arguments, so a test runs a whole invocation against its own
/// workspace and console.
fn dispatch(cli: Cli, console: &mut dyn Console) -> anyhow::Result<()> {
    // Ahead of the workspace and the config, which they do not depend on, so a
    // mistyped path is reported before a multi-gigabyte download.
    let programs = cli.resolve_external_programs();
    require_programs_up_front(&cli.command, &programs)?;
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
    // Loaded before the workspace so a malformed file is reported before
    // anything heavy starts. An exported-but-empty `PIPETTE_CONFIG` is no value.
    let client_config = crate::client_configuration::load(
        cli.config
            .as_deref()
            .filter(|path| !path.as_os_str().is_empty()),
    )?;

    // `init` runs before opening a workspace: there is nothing to open yet.
    if let RootCommand::Init = cli.command {
        match PipetteWorkspace::init(&work_dir)? {
            InitResult::Created(root) => {
                writeln!(
                    console.out(),
                    "Initialized pipette workspace at {}",
                    root.display()
                )?;
            }
            InitResult::AlreadyExists(root) => {
                writeln!(
                    console.out(),
                    "pipette workspace already initialized at {}",
                    root.display()
                )?;
            }
        }
        return Ok(());
    }

    // One HTTP client shared by the network commands (including
    // workspace-free `runtimes catalog`, which may hit GitHub).
    let http = HttpClient::builder(USER_AGENT)
        .preconfigured_tls()
        .build()?;

    // `runtimes catalog <type>` and `runtimes flavors` describe what is
    // installable without touching the store, so a fresh box can discover what
    // to pull before running `init`.
    match cli.command {
        RootCommand::Runtimes(RuntimeArgs {
            command: RuntimeCommand::Catalog(args),
        }) => args.execute(&http, console),
        RootCommand::Runtimes(RuntimeArgs {
            command: RuntimeCommand::Flavors,
        }) => runtimes::write_llamacpp_flavors(console),
        // Host facts, true before `init`, so this routes above the workspace.
        RootCommand::Check(args) => args.execute(&programs, console),
        command @ (RootCommand::Init
        | RootCommand::Models(_)
        | RootCommand::Runtimes(_)
        | RootCommand::Auth(_)
        | RootCommand::Benchmarks(_)
        | RootCommand::Results(_)
        | RootCommand::Sync(_)
        | RootCommand::Worker(_)
        | RootCommand::Storage(_)) => {
            workspace::require_workspace(&work_dir, work_dir_arg.as_deref())?;
            // `open` resolves the quota override against
            // `identity/settings.json` and carries the result.
            let ws =
                PipetteWorkspace::open(&work_dir, quota_override)?.with_eval_shard(cli.eval_shard);
            match command {
                RootCommand::Init => anyhow::bail!("init is handled before workspace open"),
                RootCommand::Check(_) => {
                    anyhow::bail!("check is handled before workspace open")
                }
                RootCommand::Models(args) => args.execute(&ws, &http, &programs, console),
                RootCommand::Runtimes(args) => args.execute(&ws, &http, &programs, console),
                RootCommand::Auth(args) => args.execute(&ws, &http, console).map_err(Into::into),
                RootCommand::Benchmarks(args) => {
                    args.execute(&ws, &http, &client_config, &programs, console)
                }
                RootCommand::Results(args) => args.execute(&ws.results(), console),
                RootCommand::Sync(args) => args.execute(&ws, &http, console).map_err(Into::into),
                RootCommand::Worker(args) => args.execute(&ws, &http, &client_config, &programs),
                RootCommand::Storage(args) => args.execute(&ws, console),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use rstest::rstest;

    use super::*;
    use crate::test_support::CapturedConsole;

    /// clap checks a command's definition only when that command is parsed,
    /// so a conflicting flag on a rarely-used subcommand would otherwise
    /// surface as a panic in the field. This walks the whole tree.
    #[test]
    fn the_command_tree_is_well_formed() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn work_dir_is_a_global_arg() -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(["pipette", "results", "list", "--work-dir", "/srv/bench"])?;
        assert_eq!(cli.work_dir.as_deref(), Some(Path::new("/srv/bench")));
        Ok(())
    }

    #[test]
    fn storage_quota_is_a_global_arg() -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(["pipette", "models", "list", "--storage-quota", "10GiB"])?;
        assert_eq!(cli.storage_quota.as_deref(), Some("10GiB"));
        Ok(())
    }

    #[rstest]
    #[case::before(["pipette", "--config", "/etc/pipette/client.toml", "worker"])]
    #[case::after(["pipette", "worker", "--config", "/etc/pipette/client.toml"])]
    fn config_is_a_global_arg(#[case] argv: [&str; 4]) -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(argv)?;
        assert_eq!(
            cli.config.as_deref(),
            Some(Path::new("/etc/pipette/client.toml"))
        );
        Ok(())
    }

    #[rstest]
    #[case::before(["pipette", "--uv-path", "/opt/uv/uv", "worker"])]
    #[case::after(["pipette", "worker", "--uv-path", "/opt/uv/uv"])]
    fn the_uv_path_is_a_global_arg(#[case] argv: [&str; 4]) -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(argv)?;
        assert_eq!(cli.uv_path.as_deref(), Some(Path::new("/opt/uv/uv")));
        Ok(())
    }

    #[rstest]
    #[case::before(["pipette", "--docker-path", "/opt/bin/docker", "worker"])]
    #[case::after(["pipette", "worker", "--docker-path", "/opt/bin/docker"])]
    fn the_docker_path_is_a_global_arg(#[case] argv: [&str; 4]) -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(argv)?;
        assert_eq!(
            cli.docker_path.as_deref(),
            Some(Path::new("/opt/bin/docker"))
        );
        Ok(())
    }

    /// A named path that is not there fails at startup rather than falling back
    /// to whatever is on `PATH`, which would measure a build nobody asked for.
    #[test]
    fn a_named_program_that_is_missing_resolves_to_a_failure() -> anyhow::Result<()> {
        let cli = Cli::try_parse_from([
            "pipette",
            "--uv-path",
            "/no/such/uv",
            "--docker-path",
            "/no/such/docker",
            "results",
            "list",
        ])?;
        let programs = cli.resolve_external_programs();
        let uv = programs
            .uv
            .path()
            .err()
            .ok_or_else(|| anyhow::anyhow!("uv is missing"))?;
        let docker = programs
            .docker
            .path()
            .err()
            .ok_or_else(|| anyhow::anyhow!("docker is missing"))?;
        assert!(uv.to_string().contains("/no/such/uv"), "got {uv}");
        assert!(
            docker.to_string().contains("/no/such/docker"),
            "got {docker}"
        );
        Ok(())
    }

    #[rstest]
    #[case::empty("", None)]
    #[case::named("/opt/uv/uv", Some("/opt/uv/uv"))]
    fn an_empty_program_path_is_treated_as_unset(
        #[case] given: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(named_path(Some(Path::new(given))), expected.map(Path::new));
    }

    /// A box with no `uv` and no docker still starts a worker and lists
    /// benchmarks, results and models.
    #[rstest]
    #[case::worker(&["pipette", "worker"], false)]
    #[case::worker_strict(
        &["pipette", "worker", "--strict", "uv", "--strict", "docker"],
        true
    )]
    #[case::benchmarks_run(&["pipette", "benchmarks", "list"], false)]
    #[case::results(&["pipette", "results", "list"], false)]
    #[case::models_pull(&["pipette", "models", "list"], false)]
    fn only_strict_requires_a_program_before_the_worker_starts(
        #[case] argv: &[&str],
        #[case] refused: bool,
    ) -> anyhow::Result<()> {
        let missing = Cli::try_parse_from([
            "pipette",
            "--uv-path",
            "/no/such/uv",
            "--docker-path",
            "/no/such/docker",
            "worker",
        ])?
        .resolve_external_programs();
        let command = Cli::try_parse_from(argv)?.command;
        let outcome = require_programs_up_front(&command, &missing);
        assert_eq!(outcome.is_err(), refused, "for {argv:?}");
        if let Err(err) = outcome {
            // Named, so the operator knows which setup step they skipped.
            let msg = format!("{err:#}");
            assert!(msg.contains("/no/such/uv"), "got {msg}");
            assert!(msg.contains("/no/such/docker"), "got {msg}");
        }
        Ok(())
    }

    #[rstest]
    #[case::plain(&["pipette", "worker"], &[])]
    #[case::uv(&["pipette", "worker", "--strict", "uv"], &["uv"])]
    #[case::docker(&["pipette", "worker", "--strict", "docker"], &["docker"])]
    #[case::python(&["pipette", "worker", "--strict", "python"], &["python3/python"])]
    #[case::nvidia(&["pipette", "worker", "--strict", "nvidia-smi"], &["nvidia-smi"])]
    #[case::rocm(&["pipette", "worker", "--strict", "rocm-smi"], &["rocm-smi"])]
    #[case::repeated(
        &["pipette", "worker", "--strict", "uv", "--strict", "docker"],
        &["uv", "docker"]
    )]
    #[case::every_one(
        &[
            "pipette", "worker",
            "--strict", "uv",
            "--strict", "docker",
            "--strict", "python",
            "--strict", "nvidia-smi",
            "--strict", "rocm-smi",
        ],
        &["uv", "docker", "python3/python", "nvidia-smi", "rocm-smi"]
    )]
    fn strict_names_the_programs_it_checks(
        #[case] argv: &[&str],
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let RootCommand::Worker(worker) = Cli::try_parse_from(argv)?.command else {
            anyhow::bail!("{argv:?} should parse as the worker");
        };
        let named: Vec<String> = worker
            .strict_programs()
            .into_iter()
            .map(ExternalProgram::label)
            .collect();
        assert_eq!(named, expected);
        Ok(())
    }

    /// A host whose `uv` and `docker` resolve and whose `nvidia-smi` does not.
    /// Built rather than probed, so the case does not depend on what the
    /// machine running it has installed.
    fn a_host_without_nvidia_smi() -> ExternalPrograms {
        ExternalPrograms {
            nvidia_smi: ExternalProgram::NVIDIA_SMI.resolve(Some(Path::new("/no/such/nvidia-smi"))),
            rocm_smi: ExternalProgram::ROCM_SMI.resolve(Some(Path::new("/no/such/rocm-smi"))),
            ..ExternalPrograms::at_missing_paths()
        }
    }

    #[test]
    fn a_strict_flag_refuses_a_program_a_plain_worker_ignores() -> anyhow::Result<()> {
        let programs = a_host_without_nvidia_smi();
        let plain = Cli::try_parse_from(["pipette", "worker"])?.command;
        require_programs_up_front(&plain, &programs)?;
        let named = Cli::try_parse_from(["pipette", "worker", "--strict", "nvidia-smi"])?.command;
        let msg = format!(
            "{:#}",
            require_programs_up_front(&named, &programs)
                .err()
                .ok_or_else(|| anyhow::anyhow!("--strict should refuse a missing nvidia-smi"))?
        );
        assert!(msg.contains("--strict"), "got {msg}");
        assert!(msg.contains("/no/such/nvidia-smi"), "got {msg}");
        Ok(())
    }

    /// A Mac asks for `--strict uv` and is not refused over the GPU CLI it will
    /// never have. Asserted on the absence of `nvidia-smi` from the outcome,
    /// so the case holds wherever `uv` itself stands.
    #[test]
    fn strict_checks_nothing_it_did_not_name() -> anyhow::Result<()> {
        let programs = a_host_without_nvidia_smi();
        let command = Cli::try_parse_from(["pipette", "worker", "--strict", "uv"])?.command;
        let msg = require_programs_up_front(&command, &programs)
            .err()
            .map(|err| format!("{err:#}"))
            .unwrap_or_default();
        assert!(!msg.contains("nvidia-smi"), "got {msg}");
        Ok(())
    }

    /// A box that has not run `init` is the box asking what it is missing.
    #[test]
    fn check_is_dispatched_without_a_workspace() -> anyhow::Result<()> {
        let command = Cli::try_parse_from(["pipette", "check", "all"])?.command;
        assert!(matches!(command, RootCommand::Check(_)));
        // And is never refused up front: the command *is* the report.
        require_programs_up_front(&command, &ExternalPrograms::from_path())
    }

    /// `-v` asks for debug; `--log-level` names the level and wins over `-v`.
    #[rstest]
    #[case::default(&["pipette", "results", "list"], LevelFilter::INFO)]
    #[case::verbose(&["pipette", "-v", "results", "list"], LevelFilter::DEBUG)]
    #[case::verbose_after(&["pipette", "results", "list", "-v"], LevelFilter::DEBUG)]
    #[case::explicit(&["pipette", "--log-level", "trace", "worker"], LevelFilter::TRACE)]
    #[case::explicit_wins(&["pipette", "-v", "--log-level", "warn", "worker"], LevelFilter::WARN)]
    fn the_log_level_is_a_global_arg(
        #[case] argv: &[&str],
        #[case] expected: LevelFilter,
    ) -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(argv)?;
        assert_eq!(cli.requested_log_level(), expected);
        Ok(())
    }

    #[test]
    fn the_log_format_parses_json() -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(["pipette", "--log-format", "json", "worker"])?;
        assert_eq!(cli.log_format, LogFormat::Json);
        Ok(())
    }

    /// A fresh directory for `--work-dir`, removed on drop.
    struct WorkDir(PathBuf);

    impl WorkDir {
        fn new(label: &str) -> anyhow::Result<Self> {
            let path = std::env::temp_dir().join(format!(
                "pipette-cli-dispatch-{label}-{}-{}",
                std::process::id(),
                uuid::Uuid::new_v4()
            ));
            std::fs::create_dir_all(&path)?;
            Ok(Self(path))
        }

        fn arg(&self) -> String {
            self.0.display().to_string()
        }

        /// The full invocation against this directory, and what it wrote.
        fn dispatch(&self, argv: &[&str]) -> anyhow::Result<String> {
            let work_dir = self.arg();
            let mut full = vec!["pipette", "--work-dir", work_dir.as_str()];
            full.extend_from_slice(argv);
            let mut captured = CapturedConsole::default();
            dispatch(Cli::try_parse_from(full)?, &mut captured)?;
            Ok(captured.out_text())
        }
    }

    impl Drop for WorkDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A second `init` says the workspace was already there, and a workspace
    /// command then opens it.
    #[test]
    fn init_creates_the_workspace_once_and_then_commands_open_it() -> anyhow::Result<()> {
        let dir = WorkDir::new("init")?;
        let created = dir.dispatch(&["init"])?;
        assert!(
            created.starts_with("Initialized pipette workspace at "),
            "{created}"
        );
        assert!(created.trim_end().ends_with(".pipette"), "{created}");
        assert!(dir.0.join(".pipette").is_dir());

        let again = dir.dispatch(&["init"])?;
        assert!(
            again.starts_with("pipette workspace already initialized at "),
            "{again}"
        );
        assert_eq!(dir.dispatch(&["results", "list"])?, "No results.\n");
        Ok(())
    }

    #[test]
    fn a_workspace_command_without_a_workspace_points_at_init() -> anyhow::Result<()> {
        let dir = WorkDir::new("uninitialized")?;
        let err = dir
            .dispatch(&["results", "list"])
            .err()
            .ok_or_else(|| anyhow::anyhow!("an uninitialized directory should refuse"))?;
        let msg = format!("{err:#}");
        assert!(msg.contains("init"), "got {msg}");
        assert!(msg.contains(&dir.arg()), "got {msg}");
        Ok(())
    }

    #[test]
    fn a_missing_work_dir_is_named() -> anyhow::Result<()> {
        let dir = WorkDir::new("missing")?;
        let gone = dir.0.join("nowhere");
        let err = dispatch(
            Cli::try_parse_from([
                "pipette",
                "--work-dir",
                &gone.to_string_lossy(),
                "results",
                "list",
            ])?,
            &mut CapturedConsole::default(),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("a missing directory should refuse"))?;
        assert!(format!("{err:#}").contains("does not exist"), "got {err:#}");
        Ok(())
    }

    #[rstest]
    #[case::flavors(&["runtimes", "flavors"], "FLAVOR")]
    #[case::catalog(&["runtimes", "catalog", "uv_openvino"], "uv-openvino://version=")]
    fn discovery_commands_answer_without_a_workspace(
        #[case] argv: &[&str],
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let out = WorkDir::new("discovery")?.dispatch(argv)?;
        assert!(out.contains(needle), "got {out}");
        Ok(())
    }

    /// A blank quota is no quota at all.
    #[test]
    fn a_malformed_quota_or_config_is_refused_before_the_workspace_opens() -> anyhow::Result<()> {
        let dir = WorkDir::new("host-facts")?;
        dir.dispatch(&["init"])?;

        let err = dir
            .dispatch(&["--storage-quota", "lots", "results", "list"])
            .err()
            .ok_or_else(|| anyhow::anyhow!("a prose quota should refuse"))?;
        assert!(format!("{err:#}").contains("lots"), "got {err:#}");
        assert_eq!(
            dir.dispatch(&["--storage-quota", "   ", "results", "list"])?,
            "No results.\n"
        );
        assert_eq!(
            dir.dispatch(&["--storage-quota", "10GiB", "results", "list"])?,
            "No results.\n"
        );

        let config = dir.0.join("client.toml");
        std::fs::write(&config, "[[runtimes]]\nthis = \"is not a runtime entry\"\n")?;
        let err = dir
            .dispatch(&["--config", &config.to_string_lossy(), "results", "list"])
            .err()
            .ok_or_else(|| anyhow::anyhow!("a malformed config should refuse"))?;
        assert!(format!("{err:#}").contains("client.toml"), "got {err:#}");
        Ok(())
    }

    /// Parsed is not enough: every shard rule reads the flag off the workspace.
    #[test]
    fn an_eval_shard_after_the_subcommand_reaches_the_workspace() -> anyhow::Result<()> {
        let dir = WorkDir::new("eval-shard")?;
        dir.dispatch(&["init"])?;
        // Unseeded, so a run the flag did not reach stops at resolving it.
        let err = dir
            .dispatch(&[
                "benchmarks",
                "run",
                "--eval-shard",
                "0/2",
                "--benchmark",
                "local/eval_smoke",
                "--model",
                r#"{"type":"apple_foundation_text"}"#,
                "--runtime",
                r#"{"type":"apple_foundation"}"#,
                "--sync",
            ])
            .err()
            .ok_or_else(|| anyhow::anyhow!("a shard's --sync should refuse"))?;
        assert!(
            format!("{err:#}").contains("--sync has nothing to send"),
            "got {err:#}"
        );
        Ok(())
    }
}
