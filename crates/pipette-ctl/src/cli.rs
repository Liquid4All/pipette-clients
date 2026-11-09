use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use clap::{Args, Parser, Subcommand};

use pipette_plan_types::Plan;
use pipette_workspace::{InitResult, resolve_work_dir};

use crate::{
    auth::AuthArgs,
    mgmt::cli::RouteCommand,
    runner::{self, CellFilter, RejoinPolicy, RunOptions, Shard},
    state,
    workspace::PipetteCtlWorkspace,
};

#[derive(Parser, Debug)]
#[command(name = "pipette-ctl", version = crate::PLAN_VERSION)]
pub struct Cli {
    /// Working directory (default: current directory)
    #[arg(long, global = true, env = "PIPETTE_WORK_DIR")]
    pub work_dir: Option<PathBuf>,

    // `--adb-port` here and on `run` and `kill`, the readiness flags here and
    // on `run`, rather than `global`: that would list them on, and let them
    // through to, every management verb, none of which reaches a device.
    #[command(flatten)]
    pub adb: AdbPortArg,

    #[command(flatten)]
    pub readiness: ReadinessArgs,

    #[command(subcommand)]
    pub command: Command,
}

// `--adb-port`, taken before the subcommand or after `run` or `kill`. A `//`
// comment: clap would read a doc comment as the parent command's `about`.
#[derive(Args, Debug)]
pub struct AdbPortArg {
    /// Override the local adb server port (`adb -P <port>`) for every
    /// adb transport in the plan. Takes precedence over each
    /// `[[transports]] port = ...`. Useful when the adb server runs on
    /// a different machine reached through an SSH tunnel and you don't
    /// want to bake the tunnel port into the plan. Applies to the
    /// commands that reach a device: `run` and `kill`.
    #[arg(long)]
    pub adb_port: Option<u16>,
}

// The readiness overrides, taken before the subcommand or after `run`.
#[derive(Args, Debug)]
pub struct ReadinessArgs {
    /// Override the readiness wait ceiling (seconds) for every runner cell
    /// this run spawns. Forwarded as PIPETTE_READINESS_MAX_WAIT_SECS so it
    /// reaches remote runners over ssh/adb/slurm and iOS devices (local
    /// runners inherit the driver env regardless). Raise it on hosts that
    /// recover to a nominal thermal state slowly, e.g. fanless laptops.
    #[arg(long, env = "PIPETTE_READINESS_MAX_WAIT_SECS", value_name = "SECS")]
    pub readiness_max_wait_secs: Option<u64>,

    /// Waive every *thermal* readiness criterion for the runner cells this run
    /// spawns, keeping the load criterion. Forwarded as
    /// PIPETTE_READINESS_SKIP_THERMAL so it reaches remote runners over
    /// ssh/adb/slurm and iOS devices. For hosts or workloads where the thermal
    /// signal costs more than it buys. Changes the readiness criteria, so
    /// results collected with it are not comparable to gated ones.
    ///
    /// On macOS this waives the die-temperature check as well as the pressure
    /// enum, and the die check is the one that keeps a batch from heating
    /// underneath the gate — see `pipette-readiness/src/macos.rs`.
    // Not clap's default parser: a `bool` with `env` accepts only
    // `true`/`false`, so the `=1` this flag forwards to runners would abort the
    // driver, and the root reads the variable for every subcommand.
    #[arg(
        long,
        env = "PIPETTE_READINESS_SKIP_THERMAL",
        action = clap::ArgAction::SetTrue,
        value_parser = parse_skip_thermal,
    )]
    pub readiness_skip_thermal: bool,

    /// What the Android CLI gate does if a touch is held on the touchscreen,
    /// for the runner cells this run spawns: `off` skips touch detection,
    /// `wait` holds the gate until the touch clears or the deadline passes,
    /// and `fail` fails the cell after about 9 s. The runner receives this mode through
    /// `PIPETTE_READINESS_HELD_TOUCH`, which overrides a phone's
    /// `[readiness.android] held_touch`. Other gates ignore it.
    #[arg(
        long,
        env = "PIPETTE_READINESS_HELD_TOUCH",
        value_name = "off|wait|fail"
    )]
    pub readiness_held_touch: Option<pipette_readiness::HeldTouch>,
}

/// Parse `--readiness-skip-thermal` / `PIPETTE_READINESS_SKIP_THERMAL` through
/// the readiness crate's grammar, so the driver and the runners it spawns can
/// never disagree about what a given value means. Infallible by construction:
/// anything not recognized as a "no" is a request to skip.
fn parse_skip_thermal(raw: &str) -> Result<bool, std::convert::Infallible> {
    Ok(pipette_readiness::skip_thermal_from_str(raw))
}

impl Cli {
    pub fn execute(self) -> anyhow::Result<()> {
        let Cli {
            work_dir,
            adb,
            readiness,
            command,
        } = self;
        // clap surfaces an empty PIPETTE_WORK_DIR as Some(""); treat it as unset.
        let work_dir = work_dir.filter(|p| !p.as_os_str().is_empty());
        match command {
            Command::Init => run_init(&resolve_work_dir(work_dir.as_deref())?),
            Command::Auth(args) => args.execute(work_dir.as_deref()),
            Command::Kill(args) => {
                let adb_port = args.adb.adb_port.or(adb.adb_port);
                args.execute(adb_port)
            }
            Command::Status(args) => args.execute(&open_ws(work_dir.as_deref())?),
            Command::Commands(args) => args.execute(&open_ws(work_dir.as_deref())?),
            Command::Run(args) => {
                let adb_port = args.adb.adb_port.or(adb.adb_port);
                let readiness_max_wait_secs = args
                    .readiness
                    .readiness_max_wait_secs
                    .or(readiness.readiness_max_wait_secs);
                let readiness_skip_thermal =
                    args.readiness.readiness_skip_thermal || readiness.readiness_skip_thermal;
                let readiness_held_touch = args
                    .readiness
                    .readiness_held_touch
                    .or(readiness.readiness_held_touch);
                args.execute(
                    &open_ws(work_dir.as_deref())?,
                    adb_port,
                    readiness_max_wait_secs,
                    readiness_skip_thermal,
                    readiness_held_touch,
                )
            }
            Command::Reset(args) => args.execute(&open_ws(work_dir.as_deref())?),
            Command::Mgmt(command) => command.execute(work_dir.as_deref()),
        }
    }
}

fn open_ws(work_dir: Option<&Path>) -> anyhow::Result<PipetteCtlWorkspace> {
    PipetteCtlWorkspace::open(&resolve_work_dir(work_dir)?)
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Initialize the storage directory
    Init,
    /// Manage the user account the management verbs sign as. Filing a
    /// campaign is a user verb, so a device's `pipette auth register` does not
    /// answer for it: these verbs register this machine's key against an
    /// account or have it approved from a browser, read the account back, and
    /// add or revoke the keys it holds.
    Auth(AuthArgs),
    /// Print completion counts for a matrix plan, then draw its cells
    Status(StatusArgs),
    /// Print the exact pipette client invocations the runner would
    /// issue for each cell, without running them.
    Commands(CommandsArgs),
    /// Run missing matrix cells against the configured target(s)
    Run(RunArgs),
    /// Delete a plan's state file. All cells become `missing` on the
    /// next `run`. Does not touch other plans or remote per-sample state.
    /// With `--client`, wipes only the cells that client ran.
    Reset(ResetArgs),
    /// Kill the pipette client process on each transport in the plan.
    /// Targets the binary at `transports[*].binary_path` by basename via
    /// `taskkill` (powershell) or `pkill` (posix).
    Kill(KillArgs),
    #[command(flatten)]
    Mgmt(RouteCommand),
}

fn run_init(work_dir: &Path) -> anyhow::Result<()> {
    match PipetteCtlWorkspace::init(work_dir)? {
        InitResult::Created(root) => {
            println!("initialized pipette-ctl workspace");
            println!();
            println!("  location: {}", root.display());
            println!("  plans/    benchmark matrix execution state");
            println!();
            println!("next: pipette-ctl run --plan <file.toml>");
        }
        InitResult::AlreadyExists(root) => {
            println!("already initialized at {}", root.display());
        }
    }
    Ok(())
}

#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Path to a TOML plan
    #[arg(long)]
    pub plan: PathBuf,

    /// Which cells to draw under the counts. Defaults to `outstanding`:
    /// everything a plain `run` still has to get through.
    #[arg(long, value_enum, default_value_t = CellFilter::Outstanding)]
    pub state: CellFilter,
}

impl StatusArgs {
    pub fn execute(self, ws: &PipetteCtlWorkspace) -> anyhow::Result<()> {
        let plans_dir = ws.plans_dir();
        let plan = Plan::load(&self.plan)?;
        let info = runner::load_status(&plans_dir, &plan)?;
        println!("{}", info.render());

        let selected = runner::cells_in_state(&plans_dir, &plan, self.state)?;
        runner::split_by_state(selected)
            .into_iter()
            .for_each(|(state, cells)| {
                println!();
                println!(
                    "{} cells ({}):",
                    runner::display_cell_state(state),
                    cells.len()
                );
                println!("{}", runner::cell_ladder(&cells));
            });

        Ok(())
    }
}

#[derive(Args, Debug)]
pub struct CommandsArgs {
    /// Path to a TOML plan
    #[arg(long)]
    pub plan: PathBuf,

    /// Which cells to print commands for. Defaults to `missing` — the
    /// set a plain `run` would execute next.
    #[arg(long, value_enum, default_value_t = CellFilter::Missing)]
    pub state: CellFilter,

    /// Print only the cells in shard `INDEX/COUNT` of the ordered
    /// matrix, matching `run --shard`. Lets you preview what one split
    /// job will execute.
    #[arg(long)]
    pub shard: Option<Shard>,
}

impl CommandsArgs {
    pub fn execute(self, ws: &PipetteCtlWorkspace) -> anyhow::Result<()> {
        let plan = Plan::load(&self.plan)?;
        runner::print_commands(&ws.plans_dir(), &plan, self.state, self.shard)
    }
}

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Path to a TOML plan
    #[arg(long)]
    pub plan: PathBuf,

    /// Maximum number of cells to run in this invocation
    #[arg(long)]
    pub limit: Option<usize>,

    /// Retry cells whose latest recorded state is failed
    #[arg(long)]
    pub include_failed: bool,

    /// Re-run cells whose latest recorded state is done. Pair with
    /// `--include-failed` to re-run the whole matrix. Each re-run cell
    /// appends a fresh attempt to `state.jsonl`; nothing recorded is
    /// discarded, and the results it submits are new rows in the
    /// warehouse alongside the earlier ones.
    #[arg(long)]
    pub include_done: bool,

    /// Emit structured JSON events to stdout (one per line) instead of
    /// the human-readable summary lines. stderr logging is unchanged.
    #[arg(long)]
    pub json: bool,

    /// Run cells whose pinned transport is not in this run's
    /// `[[transports]]` list. Off by default — such cells are
    /// preserved for the original device so the on-device sample
    /// checkpoint survives. With this flag they go to whichever
    /// available worker grabs them, restarting from sample zero.
    #[arg(long)]
    pub reassign_stranded: bool,

    /// Ignore the recorded pin and route every runnable cell by
    /// `allowed_clients` only. Use when stragglers that all last ran on
    /// one device should fan back out across every allowed worker in
    /// parallel. Loses the on-device sample checkpoint — cells restart
    /// from sample zero on whichever worker grabs them.
    #[arg(long)]
    pub ignore_pins: bool,

    /// Seconds between probes of a transport that stopped answering
    /// mid-run (default: 30). With `--rejoin-max-retries`, this is how
    /// long a worker sits on a box before giving up: the default five
    /// minutes suits a phone finishing a reboot. `0` probes back to back.
    #[arg(long, value_name = "SECS")]
    pub rejoin_retry_delay_secs: Option<u64>,

    /// Probes of a transport that stopped answering before the worker
    /// gives up on it (default: 10). `0` gives up on the spot without
    /// probing. Cells still queued for that box are not failed — they go
    /// unattempted and stay Missing for a later invocation.
    #[arg(long, value_name = "COUNT")]
    pub rejoin_max_retries: Option<usize>,

    /// Run only the cells in shard `INDEX/COUNT` of the ordered matrix
    /// (round-robin by position). Use to split a plan across several
    /// independent invocations — e.g. one SLURM job per shard. The
    /// partition is stable as cells complete; each shard records its
    /// own state, so `status` shows the aggregate.
    #[arg(long)]
    pub shard: Option<Shard>,

    #[command(flatten)]
    pub adb: AdbPortArg,

    #[command(flatten)]
    pub readiness: ReadinessArgs,
}

impl RunArgs {
    pub fn execute(
        self,
        ws: &PipetteCtlWorkspace,
        adb_port: Option<u16>,
        readiness_max_wait_secs: Option<u64>,
        readiness_skip_thermal: bool,
        readiness_held_touch: Option<pipette_readiness::HeldTouch>,
    ) -> anyhow::Result<()> {
        let plan = Plan::load(&self.plan)?;
        let opts = RunOptions {
            limit: self.limit,
            include_failed: self.include_failed,
            include_done: self.include_done,
            json: self.json,
            reassign_stranded: self.reassign_stranded,
            ignore_pins: self.ignore_pins,
            adb_port,
            readiness_max_wait_secs,
            readiness_skip_thermal,
            readiness_held_touch,
            rejoin: RejoinPolicy::from_overrides(
                self.rejoin_retry_delay_secs,
                self.rejoin_max_retries,
            ),
            shard: self.shard,
        };
        runner::run_matrix(&ws.plans_dir(), &plan, opts)
    }
}

#[derive(Args, Debug)]
pub struct ResetArgs {
    /// Path to a TOML plan. The `plan_id` inside selects which state
    /// file to delete.
    #[arg(long)]
    pub plan: PathBuf,

    /// Wipe only the cells a given client ran, leaving every other
    /// device's results in place — use it to re-run one device's work.
    /// Takes a `client_id` from the plan's `[[transports]]`; a leading
    /// chunk of the `ev1_…` hash is enough. Repeatable. Remote
    /// per-sample state on the device itself is not touched.
    #[arg(long = "client", value_name = "CLIENT_ID")]
    pub clients: Vec<String>,
}

impl ResetArgs {
    pub fn execute(self, ws: &PipetteCtlWorkspace) -> anyhow::Result<()> {
        let plan = Plan::load(&self.plan)?;
        let state_file = ws.plans_dir().join(&plan.plan_id).join("state.jsonl");
        if self.clients.is_empty() {
            return remove_state_file(&state_file);
        }
        // A pattern matching no transport in the plan is a typo far more often
        // than it is intent, and the wipe would silently do nothing.
        let declared: Vec<&str> = plan
            .transports
            .iter()
            .map(|t| t.client_id().as_str())
            .collect();
        if let Some(unknown) = self
            .clients
            .iter()
            .find(|pattern| !declared.iter().any(|id| id.contains(pattern.as_str())))
        {
            anyhow::bail!(
                "no transport in {} has a client_id matching {:?}\n  declared: {}",
                plan.plan_id,
                unknown,
                declared.join(", ")
            );
        }
        let raw = match fs::read_to_string(&state_file) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!("no state at {}", state_file.display());
                return Ok(());
            }
            Err(e) => return Err(e).with_context(|| format!("reading {}", state_file.display())),
        };
        let outcome = state::filter_out_clients(&raw, &self.clients)?;
        if outcome.dropped_events == 0 {
            println!(
                "no recorded events for {}, state unchanged",
                self.clients.join(", ")
            );
            return Ok(());
        }
        // Write-then-rename: a crash mid-write must not leave a truncated
        // state file, which would read as "those cells never ran".
        let tmp = state_file.with_extension("jsonl.tmp");
        fs::write(&tmp, &outcome.kept).with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(&tmp, &state_file)
            .with_context(|| format!("replacing {}", state_file.display()))?;
        println!(
            "dropped {} event(s) from {}",
            outcome.dropped_events,
            state_file.display()
        );
        outcome
            .matched_clients
            .iter()
            .for_each(|client| println!("  {client}"));
        println!("those cells are `missing` again; `run` will re-execute them");
        Ok(())
    }
}

fn remove_state_file(state_file: &Path) -> anyhow::Result<()> {
    match fs::remove_file(state_file) {
        Ok(()) => {
            println!("removed {}", state_file.display());
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("no state at {}", state_file.display());
            Ok(())
        }
        Err(e) => Err(e).with_context(|| format!("removing {}", state_file.display())),
    }
}

#[derive(Args, Debug)]
pub struct KillArgs {
    /// Path to a TOML plan. Every `[[transports]]` entry is targeted.
    #[arg(long)]
    pub plan: PathBuf,

    #[command(flatten)]
    pub adb: AdbPortArg,
}

impl KillArgs {
    pub fn execute(self, adb_port: Option<u16>) -> anyhow::Result<()> {
        let plan = Plan::load(&self.plan)?;
        runner::kill_transports(&plan, adb_port)
    }
}
