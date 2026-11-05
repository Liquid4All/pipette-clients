//! `pipette-ios-proxy`: a Mac standing in for tethered iPhones on the pipette
//! management server. See `README.md`; `DESIGN.md` is the contract.
//!
//! - [`cli`], [`commands`]: the verbs.
//! - [`worker`]: `run`, one claim loop per phone.
//! - [`device`]: the phone, over `xcrun devicectl`.
//! - [`app`]: the app's headless contract: launch, console, result verbs.
//! - [`server`]: the management server: claims, lease, submission.
//! - [`pool`]: the app builds, and the capabilities they advertise.
//! - [`releases`]: the app builds a release publishes, listed and pulled.
//! - [`profile`]: what the server is told about a phone.
//! - [`config`], [`secret`], [`state`], [`logging`], [`timing`], [`tool`]:
//!   configuration, keys, local state, logs, clocks, local tools.

// `allow`, not `expect`: the lint fires only under rustdoc, so clippy would call
// an `expect` unfulfilled.
#![cfg_attr(
    not(target_os = "macos"),
    allow(
        rustdoc::broken_intra_doc_links,
        reason = "the module map links to modules that exist only on macOS"
    )
)]

// Every module and dependency is macOS-scoped, so off-Mac the crate is this one
// function and `cargo clippy --workspace` stays green on the other legs. A stub
// rather than `compile_error!` for that reason; nothing ships this tool off macOS.
#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("pipette-ios-proxy: this tool drives `xcrun devicectl`, which exists only on macOS");
    std::process::ExitCode::from(1)
}

#[cfg(target_os = "macos")]
mod app;
#[cfg(target_os = "macos")]
mod cli;
#[cfg(target_os = "macos")]
mod commands;
#[cfg(target_os = "macos")]
mod config;
#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
mod logging;
#[cfg(target_os = "macos")]
mod pool;
#[cfg(target_os = "macos")]
mod profile;
#[cfg(target_os = "macos")]
mod releases;
#[cfg(target_os = "macos")]
mod secret;
#[cfg(target_os = "macos")]
mod server;
#[cfg(target_os = "macos")]
mod state;
#[cfg(target_os = "macos")]
mod timing;
#[cfg(target_os = "macos")]
mod tool;
#[cfg(target_os = "macos")]
mod worker;

#[cfg(target_os = "macos")]
use std::ffi::{OsStr, OsString};

#[cfg(target_os = "macos")]
use std::process::ExitCode;

#[cfg(target_os = "macos")]
use clap::Parser;

#[cfg(target_os = "macos")]
use crate::cli::{Cli, Command};

#[cfg(target_os = "macos")]
use crate::worker::{DeviceWait, Exclusivity};

#[cfg(target_os = "macos")]
/// Every handled failure but an operator stop exits with this, usage errors
/// included (clap would use 2). Few codes on purpose: a finer taxonomy is a
/// contract, and goes into DESIGN first.
const FAILURE_CODE: u8 = 1;

#[cfg(target_os = "macos")]
/// `run` ended because its phones stopped for faults only a person can fix
/// (DESIGN section 7), so a supervisor should not restart it into another
/// burned job. 8 is the lowest status that libc, sysexits (64-78) and the LSB
/// init-script codes (0-7) leave undefined; LSB reserves 8-99 but defines none.
const OPERATOR_CODE: u8 = 8;

#[cfg(target_os = "macos")]
/// One line on stderr (`pipette-ios-proxy: <context>: <cause>`, the full anyhow
/// chain) and code 1, or [`OPERATOR_CODE`]; never a panic or a backtrace.
fn main() -> ExitCode {
    pipette_subprocess::reset_sigpipe();
    let cli = match Cli::try_parse_from(without_options_terminator(std::env::args_os())) {
        Ok(cli) => cli,
        Err(error) => {
            let _ = error.print();
            return ExitCode::from(usage_exit_code(&error));
        }
    };
    ExitCode::from(exit_status(dispatch(&cli)))
}

#[cfg(target_os = "macos")]
/// `0` for `--help`/`--version`, [`FAILURE_CODE`] for a usage error.
fn usage_exit_code(error: &clap::Error) -> u8 {
    if error.use_stderr() { FAILURE_CODE } else { 0 }
}

#[cfg(target_os = "macos")]
/// The code for a finished verb, and the one place a handled failure is printed.
/// A stop signal that interrupted a wait is a clean exit.
fn exit_status(outcome: anyhow::Result<()>) -> u8 {
    match outcome {
        Ok(()) => 0,
        Err(error)
            if error
                .chain()
                .any(|cause| cause.is::<worker::StoppedError>()) =>
        {
            tracing::info!("{error:#}");
            0
        }
        Err(error) => {
            eprintln!("pipette-ios-proxy: {error:#}");
            if error
                .chain()
                .any(|cause| cause.is::<worker::OperatorStopError>())
            {
                OPERATOR_CODE
            } else {
                FAILURE_CODE
            }
        }
    }
}

#[cfg(target_os = "macos")]
/// Drop the first bare `--`, so `pipette-ios-proxy -- probe` runs `probe`: clap
/// would otherwise refuse the verb as a positional. Nothing here takes varargs.
fn without_options_terminator<I: IntoIterator<Item = OsString>>(argv: I) -> Vec<OsString> {
    let mut dropped = false;
    argv.into_iter()
        .filter(|arg| {
            if !dropped && arg.as_os_str() == OsStr::new("--") {
                dropped = true;
                return false;
            }
            true
        })
        .collect()
}

#[cfg(target_os = "macos")]
#[cfg(target_os = "macos")]
fn exclusivity(command: &Command) -> Exclusivity {
    match command {
        Command::Run | Command::Register | Command::Scan | Command::Uninstall => {
            Exclusivity::Exclusive
        }
        Command::Status | Command::Probe | Command::Check | Command::Builds { .. } => {
            Exclusivity::Shared
        }
    }
}

#[cfg(target_os = "macos")]
fn device_wait(command: &Command) -> DeviceWait {
    match command {
        Command::Run => DeviceWait::Wait,
        Command::Register
        | Command::Status
        | Command::Scan
        | Command::Uninstall
        | Command::Probe
        | Command::Check
        | Command::Builds { .. } => DeviceWait::Fail,
    }
}

/// Every verb but `check` and `builds` goes through `prepare`, which probes the
/// phones.
#[cfg(target_os = "macos")]
fn dispatch(cli: &Cli) -> anyhow::Result<()> {
    let command = cli.command();
    let prepare = || worker::prepare(cli, device_wait(&command), exclusivity(&command));
    match &command {
        Command::Check => commands::check(cli),
        Command::Builds { action } => commands::builds(cli, action),
        Command::Run => worker::run(prepare()?),
        Command::Register => commands::register(prepare()?),
        Command::Status => commands::status(prepare()?),
        Command::Scan => commands::scan(&mut prepare()?),
        Command::Uninstall => commands::uninstall(&mut prepare()?),
        Command::Probe => commands::probe(&prepare()?),
    }
}

#[cfg(target_os = "macos")]
#[cfg(test)]
mod tests {
    /// `device_wait` matches every variant explicitly, so a new verb cannot
    /// silently inherit `Fail`.
    #[test]
    fn only_run_waits_for_the_phone() {
        assert_eq!(device_wait(&Command::Run), DeviceWait::Wait);
        [
            Command::Register,
            Command::Status,
            Command::Scan,
            Command::Uninstall,
            Command::Probe,
        ]
        .into_iter()
        .for_each(|command| {
            assert_eq!(
                device_wait(&command),
                DeviceWait::Fail,
                "{command:?} must not wait"
            );
        });
    }

    use std::path::Path;

    use anyhow::Context;

    use super::*;
    use crate::config::Config;
    use crate::device::DevicectlFaultError;
    use crate::device::devicectl::DEVICE_NOT_FOUND_CODE;

    /// A supervisor stopping a proxy that is still waiting must not see a failure.
    #[test]
    fn a_stop_during_a_wait_exits_zero() {
        let stopped =
            anyhow::Error::from(worker::StoppedError("waiting for the phone to be attached"))
                .context("preparing the phones");
        assert_eq!(exit_status(Err(stopped)), 0);
    }

    #[test]
    fn a_config_error_and_an_unreachable_device_both_exit_one() -> anyhow::Result<()> {
        let text = format!(
            "{}\n[loop]\ncooldown_secs = 30\n",
            crate::config::tests::MINIMAL
        );
        let bad_config = Config::parse_str(&text, Path::new("pipette-ios-proxy.toml"))
            .err()
            .context("an unknown key is a config error")?;
        assert!(format!("{bad_config:#}").contains("cooldown_secs"));
        assert_eq!(exit_status(Err(bad_config)), FAILURE_CODE);

        let absent = anyhow::Error::new(DevicectlFaultError {
            domain: "com.apple.dt.CoreDeviceError".to_string(),
            code: DEVICE_NOT_FOUND_CODE,
            description: "The specified device was not found.".to_string(),
        })
        .context("probing device `UDID-1` with `devicectl device info details`");
        assert_eq!(exit_status(Err(absent)), FAILURE_CODE);

        assert_eq!(exit_status(Ok(())), 0);
        Ok(())
    }

    #[test]
    fn phones_stopped_for_an_operator_exit_with_their_own_status() {
        let stopped = anyhow::Error::from(worker::OperatorStopError { stopped: 1 })
            .context("running the phones");
        assert_eq!(exit_status(Err(stopped)), OPERATOR_CODE);
    }

    #[test]
    fn a_usage_error_exits_one_and_help_exits_zero() -> anyhow::Result<()> {
        let usage = Cli::try_parse_from(["pipette-ios-proxy", "--nope"])
            .err()
            .context("an unknown flag is a usage error")?;
        assert_eq!(usage_exit_code(&usage), FAILURE_CODE);

        let bad_verb = Cli::try_parse_from(["pipette-ios-proxy", "frobnicate"])
            .err()
            .context("an unknown verb is a usage error")?;
        assert_eq!(usage_exit_code(&bad_verb), FAILURE_CODE);

        [
            ["pipette-ios-proxy", "--help"],
            ["pipette-ios-proxy", "--version"],
        ]
        .into_iter()
        .try_for_each(|asked| {
            let answered = Cli::try_parse_from(asked)
                .err()
                .context("clap reports help and version as an `Err` carrying the text")?;
            assert_eq!(usage_exit_code(&answered), 0, "{asked:?}");
            Ok(())
        })
    }

    #[test]
    fn an_options_terminator_before_the_verb_is_dropped() -> anyhow::Result<()> {
        let argv = ["pipette-ios-proxy", "--config", "p.toml", "--", "probe"]
            .into_iter()
            .map(OsString::from);
        let cli = Cli::try_parse_from(without_options_terminator(argv))?;
        assert_eq!(cli.command(), Command::Probe);
        assert_eq!(cli.config_path(), Some(Path::new("p.toml")));

        let plain = ["pipette-ios-proxy", "status"]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>();
        assert_eq!(without_options_terminator(plain.clone()), plain);
        Ok(())
    }
}
