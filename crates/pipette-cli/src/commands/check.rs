//! `check`: whether this host's external programs are actually usable. Each
//! check runs the program rather than only locating it; [`CHECK_LONG_ABOUT`]
//! says why. No workspace: these are facts about the box.

use clap::{Args, Subcommand};
use tabled::settings::Style;
use tabled::{Table, Tabled};

use pipette_programs::{ExternalProgram, ExternalPrograms, ProgramCheck, ResolvedProgram};

use crate::console::Console;

pub const CHECK_LONG_ABOUT: &str = "\
Check the external programs this host is supposed to have.

Each subcommand locates one program and then runs it, because being on PATH is
not the same as working: a `docker` whose daemon is stopped is found and then
fails the first pull, and an `nvidia-smi` whose kernel module has drifted from
its userspace library is found and then fails every probe.

These are the same checks `pipette worker --strict <program>` gates on, so this
is how to see what a worker would refuse on without starting one. `check all`
covers every program, both GPU CLIs included, which a Mac neither has nor
needs: check the programs a Mac uses one at a time instead.

Needs no workspace, no server and no registration.";

pub const CHECK_AFTER_HELP: &str = "\
Examples:
  pipette check all           # every program in the catalog
  pipette check uv            # just uv, as `worker --strict uv` would
  pipette check docker        # is a daemon reachable from here
  pipette check nvidia-smi    # is the NVIDIA driver answering

Each program is looked for on PATH, or wherever its global flag names
(--uv-path, --docker-path, ...); `pipette --help` lists them.

The table goes to stdout and the failures to stderr, and the exit status is 0
only when every program the subcommand covers passed.";

/// Check that the external programs this host needs are present and working
#[derive(Args, Debug)]
pub struct CheckArgs {
    #[command(subcommand)]
    pub command: CheckCommand,
}

/// One variant per program, rather than a name argument: the set is closed, so
/// `--help` lists it and a typo is a parse error rather than a run that checked
/// nothing.
#[derive(Subcommand, Debug)]
pub enum CheckCommand {
    /// Every program in the catalog at once
    All,
    /// `uv`: provisions every venv-backed runtime
    Uv,
    /// `docker`: pulls and runs the container-backed runtimes, so this asks
    /// the daemon
    Docker,
    /// `python3` / `python`: anything run outside a runtime's own venv
    Python,
    /// `nvidia-smi`: the NVIDIA driver preflight
    NvidiaSmi,
    /// `rocm-smi`: the AMD preflight
    RocmSmi,
}

impl CheckArgs {
    /// `programs` is what the root command resolved, so this checks the binary
    /// a run would actually spawn, `--docker-path` included.
    pub fn execute(
        self,
        programs: &ExternalPrograms,
        console: &mut dyn Console,
    ) -> anyhow::Result<()> {
        match self.command {
            CheckCommand::All => check_all(programs, console),
            CheckCommand::Uv => check_one(ExternalProgram::UV, &programs.uv, console),
            CheckCommand::Docker => check_one(ExternalProgram::DOCKER, &programs.docker, console),
            CheckCommand::Python => check_one(ExternalProgram::PYTHON, &programs.python, console),
            CheckCommand::NvidiaSmi => {
                check_one(ExternalProgram::NVIDIA_SMI, &programs.nvidia_smi, console)
            }
            CheckCommand::RocmSmi => {
                check_one(ExternalProgram::ROCM_SMI, &programs.rocm_smi, console)
            }
        }
    }
}

/// One program: a pass says what ran and what answered, a failure is the error.
fn check_one(
    program: ExternalProgram,
    resolved: &ResolvedProgram,
    console: &mut dyn Console,
) -> anyhow::Result<()> {
    let check = program.check(resolved);
    if let Some((path, reported)) = check.answer() {
        writeln!(console.out(), "{}: ok", check.label())?;
        // The command as run, so the answer can be reproduced by hand.
        writeln!(
            console.out(),
            "  ran:      {} {}",
            path.display(),
            check.check_args()
        )?;
        writeln!(console.out(), "  reported: {reported}")?;
        return Ok(());
    }
    ProgramCheck::refuse_any_failed(&[check])
}

/// Every program, as a table plus one refusal naming everything that failed.
/// Every row prints, so one run reports every failure.
fn check_all(programs: &ExternalPrograms, console: &mut dyn Console) -> anyhow::Result<()> {
    let checks = programs.check_all();
    let rows: Vec<CheckRow> = checks.iter().map(CheckRow::new).collect();
    writeln!(console.out(), "{}", Table::new(rows).with(Style::psql()))?;
    ProgramCheck::refuse_any_failed(&checks)
}

/// One row of `check all`. The failure text is not a column: it runs to a
/// paragraph for a stopped daemon, and is printed in full below the table.
#[derive(Tabled)]
struct CheckRow {
    #[tabled(rename = "PROGRAM")]
    program: String,
    #[tabled(rename = "STATUS")]
    status: &'static str,
    #[tabled(rename = "CHECK")]
    check: String,
    #[tabled(rename = "PATH")]
    path: String,
    #[tabled(rename = "REPORTED")]
    reported: String,
}

impl CheckRow {
    fn new(check: &ProgramCheck) -> Self {
        Self {
            program: check.label(),
            status: if check.passed() { "ok" } else { "failed" },
            check: check.check_args(),
            path: check
                .path()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
            reported: check.reported().unwrap_or_default().to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use clap::Parser;
    use rstest::rstest;

    use super::*;
    use crate::commands::{Cli, RootCommand};
    use crate::test_support::CapturedConsole;

    /// Every program in the catalog has a subcommand, so there is none an
    /// operator cannot check before a worker refuses on it.
    #[test]
    fn every_program_has_a_subcommand() -> anyhow::Result<()> {
        ExternalPrograms::from_path()
            .catalog()
            .into_iter()
            .try_for_each(|(program, _)| {
                // `python3/python` is the one label that is not its subcommand.
                let verb = match program.label().as_str() {
                    "python3/python" => "python".to_owned(),
                    label => label.to_owned(),
                };
                Cli::try_parse_from(["pipette", "check", &verb]).map(drop)
            })
            .map_err(Into::into)
    }

    #[rstest]
    #[case("all")]
    #[case("uv")]
    #[case("docker")]
    #[case("python")]
    #[case("nvidia-smi")]
    #[case("rocm-smi")]
    fn check_parses_every_subcommand(#[case] verb: &str) -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(["pipette", "check", verb])?;
        assert!(matches!(cli.command, RootCommand::Check(_)));
        Ok(())
    }

    /// A program that was never located fails naming the path the operator
    /// gave, and prints nothing as ok.
    #[test]
    fn a_named_program_that_is_missing_fails_its_check() {
        let resolved = ExternalProgram::DOCKER.resolve(Some(Path::new("/no/such/d")));
        let mut captured = CapturedConsole::default();
        let err = check_one(ExternalProgram::DOCKER, &resolved, &mut captured)
            .err()
            .map(|err| format!("{err:#}"))
            .unwrap_or_default();
        assert!(err.contains("docker: "), "got {err}");
        assert!(err.contains("/no/such/d"), "got {err}");
        assert_eq!(captured.out_text(), "");
    }
}
