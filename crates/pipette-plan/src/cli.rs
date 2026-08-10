use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use clap::{Args, Parser, Subcommand};
use tabled::settings::Style;

use pipette_plan_types::Plan;
use pipette_workspace::{resolve_work_dir, InitResult};

use crate::{
    generate,
    runner::{self, ListState, RunOptions, Shard},
    state,
    workspace::PipettePlanWorkspace,
};

#[derive(Parser, Debug)]
#[command(name = "pipette-plan", version = crate::PLAN_VERSION)]
pub struct Cli {
    /// Working directory (default: current directory)
    #[arg(long, global = true, env = "PIPETTE_WORK_DIR")]
    pub work_dir: Option<PathBuf>,

    /// Override the local adb server port (`adb -P <port>`) for every
    /// adb transport in the plan. Takes precedence over each
    /// `[[transports]] port = ...`. Useful when the adb server runs on
    /// a different machine reached through an SSH tunnel and you don't
    /// want to bake the tunnel port into the plan.
    #[arg(long, global = true)]
    pub adb_port: Option<u16>,

    /// Override the readiness wait ceiling (seconds) for every runner cell
    /// this run spawns. Forwarded as PIPETTE_READINESS_MAX_WAIT_SECS so it
    /// reaches remote runners over ssh/adb/slurm (local runners inherit the
    /// driver env regardless). Raise it on hosts that recover to a nominal
    /// thermal state slowly, e.g. fanless laptops.
    #[arg(
        long,
        global = true,
        env = "PIPETTE_READINESS_MAX_WAIT_SECS",
        value_name = "SECS"
    )]
    pub readiness_max_wait_secs: Option<u64>,

    /// Waive every *thermal* readiness criterion for the runner cells this run
    /// spawns, keeping the load criterion. Forwarded as
    /// PIPETTE_READINESS_SKIP_THERMAL so it reaches remote runners over
    /// ssh/adb/slurm. For hosts or workloads where the thermal signal costs more
    /// than it buys. Changes the readiness criteria, so results collected with
    /// it are not comparable to gated ones.
    ///
    /// On macOS this waives the die-temperature check as well as the pressure
    /// enum, and the die check is the one that keeps a batch from heating
    /// underneath the gate — see `pipette-readiness/src/macos.rs`.
    // The value parser is the readiness crate's own, not clap's default: a bare
    // `bool` with `env` accepts only `true`/`false`, so the `=1` this very flag
    // forwards to runners would abort the driver instead — and `global = true`
    // makes that abort every subcommand.
    #[arg(
        long,
        global = true,
        env = "PIPETTE_READINESS_SKIP_THERMAL",
        action = clap::ArgAction::SetTrue,
        value_parser = parse_skip_thermal,
    )]
    pub readiness_skip_thermal: bool,

    #[command(subcommand)]
    pub command: Command,
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
        // Each arm owns its own workspace resolution. `Init` and `Kill`
        // resolve the work dir but don't open a workspace. The remaining
        // arms need a fully-opened workspace.
        let Cli {
            work_dir,
            adb_port,
            readiness_max_wait_secs,
            readiness_skip_thermal,
            command,
        } = self;
        // clap surfaces an empty PIPETTE_WORK_DIR as Some(""); treat it as unset.
        let work_dir = work_dir.filter(|p| !p.as_os_str().is_empty());
        match command {
            Command::Init => run_init(&resolve_work_dir(work_dir.as_deref())?),
            // Expansion is a pure plan→files transform: no state to keep, so
            // no workspace to open.
            Command::Generate(args) => args.execute(),
            Command::Kill(args) => args.execute(adb_port),
            Command::Status(args) => args.execute(&open_ws(work_dir.as_deref())?, adb_port),
            Command::List(args) => args.execute(&open_ws(work_dir.as_deref())?),
            Command::Commands(args) => args.execute(&open_ws(work_dir.as_deref())?),
            Command::Run(args) => args.execute(
                &open_ws(work_dir.as_deref())?,
                adb_port,
                readiness_max_wait_secs,
                readiness_skip_thermal,
            ),
            Command::Reset(args) => args.execute(&open_ws(work_dir.as_deref())?),
        }
    }
}

fn open_ws(work_dir: Option<&Path>) -> anyhow::Result<PipettePlanWorkspace> {
    PipettePlanWorkspace::open(&resolve_work_dir(work_dir)?)
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Initialize the storage directory
    Init,
    /// Expand a scheduler-mode plan into a directory of job files for
    /// `pipette-mgmt plans ingest`. Nothing is written unless the whole
    /// plan validates.
    Generate(GenerateArgs),
    /// Print completion counts for a matrix plan
    Status(StatusArgs),
    /// List matrix cells by state
    List(ListArgs),
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
}

fn run_init(work_dir: &Path) -> anyhow::Result<()> {
    match PipettePlanWorkspace::init(work_dir)? {
        InitResult::Created(root) => {
            println!("initialized pipette-plan workspace");
            println!();
            println!("  location: {}", root.display());
            println!("  plans/    benchmark matrix execution state");
            println!();
            println!("next: pipette-plan run --plan <file.toml>");
        }
        InitResult::AlreadyExists(root) => {
            println!("already initialized at {}", root.display());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct GenerateArgs {
    /// Path to a TOML scheduler-mode plan: `[[variants]]` carrying
    /// `requires` / `clients` eligibility, and no `[[transports]]`.
    #[arg(long)]
    pub plan: PathBuf,

    /// Directory to write the job files into, created if absent. It must hold
    /// no `.json` files of its own: `plans ingest` stages every one it finds
    /// as a job of the same plan.
    #[arg(long)]
    pub out: PathBuf,
}

impl GenerateArgs {
    pub fn execute(self) -> anyhow::Result<()> {
        generate::generate(&self.plan, &self.out)
    }
}

#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Path to a TOML plan
    #[arg(long)]
    pub plan: PathBuf,
}

impl StatusArgs {
    pub fn execute(self, ws: &PipettePlanWorkspace, adb_port: Option<u16>) -> anyhow::Result<()> {
        let plans_dir = ws.plans_dir();
        let plan = Plan::load(&self.plan)?;
        let info = runner::load_status(&plans_dir, &plan, adb_port)?;
        println!("{}", tabled::Table::new(info.to_rows()).with(Style::psql()));

        if info.summary.failed > 0 {
            let failed = runner::cells_in_state(&plans_dir, &plan, ListState::Failed)?;
            let rows = runner::group_cells_by_benchmark(&failed);
            println!();
            println!("failed cells:");
            println!("{}", tabled::Table::new(&rows).with(Style::psql()));
        }
        if info.summary.missing > 0 {
            let missing = runner::cells_in_state(&plans_dir, &plan, ListState::Missing)?;
            let rows = runner::group_cells_by_benchmark(&missing);
            println!();
            println!("missing cells:");
            println!("{}", tabled::Table::new(&rows).with(Style::psql()));
        }

        Ok(())
    }
}

#[derive(Args, Debug)]
pub struct ListArgs {
    /// Path to a TOML plan
    #[arg(long)]
    pub plan: PathBuf,

    /// Filter rows by state
    #[arg(long, value_enum, default_value_t = ListState::Missing)]
    pub state: ListState,
}

impl ListArgs {
    pub fn execute(self, ws: &PipettePlanWorkspace) -> anyhow::Result<()> {
        let plan = Plan::load(&self.plan)?;
        runner::list_matrix(&ws.plans_dir(), &plan, self.state)
    }
}

#[derive(Args, Debug)]
pub struct CommandsArgs {
    /// Path to a TOML plan
    #[arg(long)]
    pub plan: PathBuf,

    /// Which cells to print commands for. Defaults to `missing` — the
    /// set a plain `run` would execute next.
    #[arg(long, value_enum, default_value_t = ListState::Missing)]
    pub state: ListState,

    /// Print only the cells in shard `INDEX/COUNT` of the ordered
    /// matrix, matching `run --shard`. Lets you preview what one split
    /// job will execute.
    #[arg(long)]
    pub shard: Option<Shard>,
}

impl CommandsArgs {
    pub fn execute(self, ws: &PipettePlanWorkspace) -> anyhow::Result<()> {
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

    /// Run only the cells in shard `INDEX/COUNT` of the ordered matrix
    /// (round-robin by position). Use to split a plan across several
    /// independent invocations — e.g. one SLURM job per shard. The
    /// partition is stable as cells complete; each shard records its
    /// own state, so `status` shows the aggregate.
    #[arg(long)]
    pub shard: Option<Shard>,
}

impl RunArgs {
    pub fn execute(
        self,
        ws: &PipettePlanWorkspace,
        adb_port: Option<u16>,
        readiness_max_wait_secs: Option<u64>,
        readiness_skip_thermal: bool,
    ) -> anyhow::Result<()> {
        let plan = Plan::load(&self.plan)?;
        let opts = RunOptions {
            limit: self.limit,
            include_failed: self.include_failed,
            json: self.json,
            reassign_stranded: self.reassign_stranded,
            ignore_pins: self.ignore_pins,
            adb_port,
            readiness_max_wait_secs,
            readiness_skip_thermal,
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
    pub fn execute(self, ws: &PipettePlanWorkspace) -> anyhow::Result<()> {
        let plan = Plan::load(&self.plan)?;
        let state_file = ws.plans_dir().join(&plan.plan_id).join("state.jsonl");
        if self.clients.is_empty() {
            return remove_state_file(&state_file);
        }
        // A pattern matching no transport in the plan is a typo far more often
        // than it is intent, and the wipe would silently do nothing.
        let declared: Vec<&str> = plan.transports.iter().map(|t| t.client_id()).collect();
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
}

impl KillArgs {
    pub fn execute(self, adb_port: Option<u16>) -> anyhow::Result<()> {
        let plan = Plan::load(&self.plan)?;
        runner::kill_transports(&plan, adb_port)
    }
}
