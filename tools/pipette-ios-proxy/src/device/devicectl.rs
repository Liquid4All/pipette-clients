//! Every conversation with the phone: `xcrun devicectl`, one method per
//! subcommand (DESIGN.md section 6). Three decisions must not be "tidied" back:
//!
//! 1. Parsed calls go through `--json-output <tmpfile>` (there is no stdout-JSON
//!    mode) and the file is read back EVEN ON FAILURE: a failed run writes
//!    `{"error":{…},"info":{"outcome":"failed"}}` with no `result`.
//! 2. `--timeout` goes on every parsed call except the launch
//!    ([`Devicectl::launch_argv`]).
//! 3. Reachability is `device info apps`, not `device info details`
//!    ([`Devicectl::reachability`]).
//!
//! Flag order is free (Swift ArgumentParser), so the `--device` / `--timeout` /
//! `--json-output` triple is appended by [`Devicectl::run_parsed`].

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::Context;
use serde_json::Value;

use super::{DeviceControl, DeviceDetails, DeviceProcess, InstalledApp, Reachability};
use crate::tool::{self, ToolError};

/// Always `xcrun`, never a resolved path: `xcrun` picks the active Xcode.
pub(crate) const PROGRAM: &str = "xcrun";

pub(crate) const INSTALL_HINT: &str =
    "install Xcode and run `xcrun devicectl --version` to check it";

/// `list devices` remembers every phone ever paired; a live `transportType` is
/// the one field that only exists while a device is attached.
const TRANSPORT_FILTER: &str = "connectionProperties.transportType != nil";

/// `connectionProperties.tunnelState` when the record is the cached one for a
/// device that is no longer attached.
const TUNNEL_UNAVAILABLE: &str = "unavailable";

/// `deviceProperties.developerModeStatus` on a phone that can launch the app.
const DEVELOPER_MODE_ENABLED: &str = "enabled";

/// What `list devices` prints when nothing matches: on STDOUT, exit 0 (devicectl
/// 518.33), so [`identifier_rows`] cannot treat every line as an identifier.
const NO_DEVICES_SENTENCE: &str = "No devices found.";

/// Grace past `call_timeout_secs` for `list devices`, which takes no `--timeout`.
const ENUMERATE_GRACE: Duration = Duration::from_secs(5);

const JSON_OUTPUT_FILE: &str = "devicectl.json";

#[derive(Debug, Clone)]
pub(crate) struct Devicectl {
    pub(crate) udid: String,
    pub(crate) call_timeout: Duration,
}

#[derive(Debug, Clone)]
struct CommandOutput {
    pub(crate) status: Option<i32>,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

impl CommandOutput {
    pub(crate) fn succeeded(&self) -> bool {
        self.status == Some(0)
    }
}

/// The `--json-output` document: `result` on success, `error` on failure,
/// `info.outcome` says which.
#[derive(Debug, Clone)]
struct DevicectlOutput {
    pub(crate) result: Option<Value>,
    pub(crate) error: Option<DevicectlFaultError>,
    pub(crate) outcome: Option<String>,
}

impl DevicectlOutput {
    fn into_result(self) -> anyhow::Result<Value> {
        match (self.result, self.error) {
            (Some(result), _) => Ok(result),
            (None, Some(fault)) => Err(fault.into()),
            (None, None) => Err(DevicectlError::EmptyDocument {
                outcome: self.outcome.unwrap_or_else(|| "absent".to_string()),
            }
            .into()),
        }
    }
}

/// A `devicectl` call that produced no usable answer. [`DevicectlFaultError`] is the
/// other case: an answer that is the device's own error.
#[derive(Debug, thiserror::Error)]
enum DevicectlError {
    #[error("`{program}` was not found: {INSTALL_HINT}")]
    NotInstalled { program: String },
    #[error(
        "no connected device: attach the iPhone, unlock it and trust this Mac (a paired device \
         that is not attached still appears in a bare `devicectl list devices`, and cannot run \
         anything)"
    )]
    NoDevice,
    #[error(
        "{} connected devices ({}): name the phone in its [[devices]] `udid`",
        .0.len(),
        .0.join(", ")
    )]
    SeveralDevices(Vec<String>),
    #[error("`{command}` failed ({status}): {diagnostic}")]
    CallFailed {
        command: String,
        status: String,
        diagnostic: String,
    },
    #[error(
        "`{command}` wrote no readable --json-output ({reason}); it exited {status}: {diagnostic}"
    )]
    NoJsonOutput {
        command: String,
        reason: String,
        status: String,
        diagnostic: String,
    },
    #[error(
        "devicectl wrote a JSON document with neither `result` nor `error` (outcome {outcome})"
    )]
    EmptyDocument { outcome: String },
    #[error("`{command}` did not finish within {secs}s and was killed: {diagnostic}")]
    TimedOut {
        command: String,
        secs: u64,
        diagnostic: String,
    },
}

/// A failed spawn: "not installed" when the program is missing, else the I/O error.
fn spawn_error(program: &str, error: std::io::Error) -> anyhow::Error {
    if error.kind() == std::io::ErrorKind::NotFound {
        DevicectlError::NotInstalled {
            program: program.to_owned(),
        }
        .into()
    } else {
        anyhow::Error::new(error).context(format!("spawning `{program}`"))
    }
}

/// `devicectl`'s own error object.
#[derive(Debug, Clone, thiserror::Error)]
#[error("devicectl {domain} {code}: {description}")]
pub(crate) struct DevicectlFaultError {
    pub(crate) domain: String,
    pub(crate) code: i64,
    /// `userInfo.NSLocalizedDescription.string` (wrapped one level deep).
    pub(crate) description: String,
}

/// The `CoreDeviceError` code for "no connected device matches".
pub(crate) const DEVICE_NOT_FOUND_CODE: i64 = 1011;

impl DevicectlFaultError {
    pub(crate) fn is_device_absent(&self) -> bool {
        self.code == DEVICE_NOT_FOUND_CODE
    }
}

impl DeviceControl for Devicectl {
    fn udid(&self) -> &str {
        &self.udid
    }

    fn reachability(&self, bundle_id: &str) -> Reachability {
        Devicectl::reachability(self, bundle_id)
    }

    fn info_apps(&self, bundle_id: Option<&str>) -> anyhow::Result<Vec<InstalledApp>> {
        Devicectl::info_apps(self, bundle_id)
    }

    fn verify_installable(
        &self,
        build: &Path,
        fleet: Option<&crate::pool::bundle::FleetSigned<'_>>,
    ) -> anyhow::Result<()> {
        crate::pool::bundle::verify_installable(build, fleet)
    }

    fn install_app(&self, path: &Path) -> anyhow::Result<()> {
        Devicectl::install_app(self, path)
    }

    fn uninstall_app(&self, bundle_id: &str) -> anyhow::Result<()> {
        Devicectl::uninstall_app(self, bundle_id)
    }

    fn pid_for_bundle(&self, bundle_id: &str) -> anyhow::Result<Option<u32>> {
        Devicectl::pid_for_bundle(self, bundle_id)
    }

    fn terminate_process(&self, pid: u32) -> anyhow::Result<()> {
        Devicectl::terminate_process(self, pid)
    }

    fn launch_console(
        &self,
        bundle_id: &str,
        app_args: &[String],
        deadline: Duration,
        first_line: Duration,
        sentinel_on: crate::app::SentinelStream,
        signals: crate::app::Signals<'_>,
        on_stdout_line: &mut dyn FnMut(&str),
    ) -> anyhow::Result<crate::app::ConsoleRun> {
        crate::app::run_console(
            self,
            bundle_id,
            app_args,
            deadline,
            first_line,
            sentinel_on,
            signals,
            on_stdout_line,
        )
    }
}

impl Devicectl {
    pub(crate) fn new(udid: String, call_timeout: Duration) -> Self {
        Self { udid, call_timeout }
    }

    /// A configured identifier is used VERBATIM (`--device` accepts many forms);
    /// with none, enumerate and require EXACTLY ONE connected device.
    pub(crate) fn resolve_target(
        configured: Option<&str>,
        call_timeout: Duration,
    ) -> anyhow::Result<String> {
        if let Some(identifier) = configured {
            let trimmed = identifier.trim();
            if !trimmed.is_empty() {
                return Ok(trimmed.to_string());
            }
        }
        let connected = Self::list_connected(call_timeout)?;
        match connected.as_slice() {
            [] => Err(DevicectlError::NoDevice.into()),
            [only] => {
                tracing::info!(udid = %only, "resolved the target device");
                Ok(only.clone())
            }
            many => Err(DevicectlError::SeveralDevices(many.to_vec()).into()),
        }
    }

    /// `list devices` takes no `--timeout`, so [`run_capture_bounded`] bounds it.
    /// A non-zero exit with no identifiers is `xcrun` failing, not an empty fleet.
    fn list_connected(call_timeout: Duration) -> anyhow::Result<Vec<String>> {
        let argv = vec![
            "devicectl".to_string(),
            "list".to_string(),
            "devices".to_string(),
            "--hide-headers".to_string(),
            "--hide-default-columns".to_string(),
            "--columns".to_string(),
            "identifier".to_string(),
            "--filter".to_string(),
            TRANSPORT_FILTER.to_string(),
        ];
        let output = run_capture_bounded(&argv, call_timeout + ENUMERATE_GRACE)?;
        let identifiers = identifier_rows(&output.stdout);
        if identifiers.is_empty() && !output.succeeded() {
            return Err(DevicectlError::CallFailed {
                command: render(&argv),
                status: describe_status(output.status),
                diagnostic: diagnostic_line(&output.stderr),
            }
            .into());
        }
        Ok(identifiers)
    }

    /// `--json-output` goes into a temp directory rather than a pre-created file,
    /// so a zero-byte artefact cannot be mistaken for "wrote nothing".
    fn run_parsed(
        &self,
        args: &[&str],
        timeout: Option<Duration>,
    ) -> anyhow::Result<DevicectlOutput> {
        let dir = tempfile::Builder::new()
            .prefix("pipette-ios-proxy-")
            .tempdir()
            .context("creating a temp directory for devicectl --json-output")?;
        let json_path = dir.path().join(JSON_OUTPUT_FILE);
        let json_arg = json_path
            .to_str()
            .context("temp directory path is not valid UTF-8")?
            .to_string();

        let mut argv = self.base_argv(args, timeout);
        argv.push("--json-output".to_string());
        argv.push(json_arg);

        let output = run_capture(&argv)?;
        // Read even on failure: the `error` object is the entire diagnostic.
        match std::fs::read_to_string(&json_path) {
            Ok(text) => parse_document(&text)
                .with_context(|| format!("reading devicectl output of `{}`", render(&argv))),
            Err(error) => Err(DevicectlError::NoJsonOutput {
                command: render(&argv),
                reason: error.to_string(),
                status: describe_status(output.status),
                diagnostic: diagnostic_line(&output.stderr),
            }
            .into()),
        }
    }

    fn base_argv(&self, args: &[&str], timeout: Option<Duration>) -> Vec<String> {
        let mut argv = vec!["devicectl".to_string()];
        argv.extend(args.iter().map(|arg| (*arg).to_string()));
        argv.push("--device".to_string());
        argv.push(self.udid.clone());
        if let Some(bound) = timeout {
            argv.push("--timeout".to_string());
            argv.push(timeout_arg(bound));
        }
        argv
    }

    fn info_details(&self) -> anyhow::Result<Value> {
        self.run_parsed(&["device", "info", "details"], Some(self.call_timeout))?
            .into_result()
    }

    pub(crate) fn identity(&self) -> anyhow::Result<DeviceDetails> {
        identity_from_result(&self.info_details()?)
    }

    /// `device info apps`, NOT `device info details`: details exits 0 with the
    /// cached record for a paired-but-absent phone; apps exits non-zero with
    /// `CoreDeviceError 1011`. The details then decide ([`reachability_of`]).
    pub(crate) fn reachability(&self, bundle_id: &str) -> Reachability {
        if let Err(error) = self.info_apps(Some(bundle_id)) {
            return Reachability::Unreachable(format!("{error:#}"));
        }
        match self.identity() {
            Ok(identity) => reachability_of(identity),
            // Apps already proved the phone answers; details is the weaker probe.
            Err(_) => Reachability::Reachable { os_version: None },
        }
    }

    /// `device info apps --bundle-id <id>`, parsed from `result.apps[]`. A
    /// `result` without an `apps` array is an error, not an empty list: "no apps"
    /// would reinstall before every job.
    pub(crate) fn info_apps(&self, bundle_id: Option<&str>) -> anyhow::Result<Vec<InstalledApp>> {
        let mut args = vec!["device", "info", "apps"];
        if let Some(id) = bundle_id {
            args.push("--bundle-id");
            args.push(id);
        }
        let result = self
            .run_parsed(&args, Some(self.call_timeout))?
            .into_result()?;
        let apps = result
            .get("apps")
            .and_then(Value::as_array)
            .context("devicectl `device info apps` returned no result.apps array")?;
        Ok(apps.iter().filter_map(installed_app).collect())
    }

    /// `device info processes`, for the terminate backstop. Two key spellings are
    /// accepted: this verb's JSON shape was never captured from a live device.
    fn info_processes(&self) -> anyhow::Result<Vec<DeviceProcess>> {
        let result = self
            .run_parsed(&["device", "info", "processes"], Some(self.call_timeout))?
            .into_result()?;
        let rows = result
            .get("runningProcesses")
            .or_else(|| result.get("processes"))
            .and_then(Value::as_array)
            .context(
                "devicectl `device info processes` returned no result.runningProcesses array",
            )?;
        Ok(rows.iter().filter_map(device_process).collect())
    }

    /// The app's pid, or `None`. `process terminate` has no bundle-id target, so:
    /// app NAME from [`Self::info_apps`], then `/<Name>.app/` in the process list.
    /// A near-miss here terminates the wrong process, hence the exact compare.
    pub(crate) fn pid_for_bundle(&self, bundle_id: &str) -> anyhow::Result<Option<u32>> {
        let apps = self.info_apps(Some(bundle_id))?;
        let Some(app) = apps.into_iter().find(|app| app.bundle_id == bundle_id) else {
            return Ok(None);
        };
        if app.name.is_empty() {
            // An empty needle would be `//` and match every process path.
            tracing::warn!(
                bundle_id = %bundle_id,
                "the installed app carries no name, so its pid cannot be looked up"
            );
            return Ok(None);
        }
        let needle = format!("/{}.app/", app.name);
        Ok(self
            .info_processes()?
            .into_iter()
            .find(|process| process.executable.contains(&needle))
            .map(|process| process.pid))
    }

    /// Installs only: the build must already be signed.
    pub(crate) fn install_app(&self, path: &Path) -> anyhow::Result<()> {
        let path_arg = path
            .to_str()
            .context("app path is not valid UTF-8, which devicectl cannot be given")?;
        self.run_parsed(
            &["device", "install", "app", path_arg],
            Some(self.call_timeout),
        )?
        .into_result()
        .with_context(|| format!("installing `{path_arg}` on {}", self.udid))?;
        tracing::info!(path = %path_arg, "installed the app build");
        Ok(())
    }

    pub(crate) fn uninstall_app(&self, bundle_id: &str) -> anyhow::Result<()> {
        self.run_parsed(
            &["device", "uninstall", "app", bundle_id],
            Some(self.call_timeout),
        )?
        .into_result()
        .with_context(|| format!("uninstalling `{bundle_id}` from {}", self.udid))?;
        tracing::info!(bundle_id = %bundle_id, "uninstalled the app");
        Ok(())
    }

    /// SIGTERM, never `--kill`: the next launch's `--terminate-existing` clears any survivor.
    pub(crate) fn terminate_process(&self, pid: u32) -> anyhow::Result<()> {
        let pid_arg = pid.to_string();
        self.run_parsed(
            &["device", "process", "terminate", "--pid", &pid_arg],
            Some(self.call_timeout),
        )?
        .into_result()
        .with_context(|| format!("terminating pid {pid} on {}", self.udid))?;
        Ok(())
    }

    /// The argv for a `--console` launch, everything after `xcrun`. Load-bearing:
    ///
    /// - `--terminate-existing`: `--console` only attaches when the app is not
    ///   already running; otherwise the launch never returns. It also makes every
    ///   run a fresh process, which a memory cell's footprint figure depends on.
    ///   Single App Mode's argument-less reopen can still land between its
    ///   terminate and launch steps; `crate::app::console::relaunch_if_silent`
    ///   catches that.
    /// - No `--timeout`: it bounds the whole command and `--console` waits for the
    ///   app to exit, so it would abort every benchmark. `crate::app` bounds
    ///   the run with `run_timeout_secs`, SIGTERM to `xcrun` (forwarded to the
    ///   app), then the terminate backstop. Never SIGKILL: it orphans the app.
    /// - `app_args` must begin with `headlessrun`, or the app opens its UI and
    ///   `--console` hangs forever.
    pub(crate) fn launch_argv(&self, bundle_id: &str, app_args: &[String]) -> Vec<String> {
        let mut argv = vec![
            "devicectl".to_string(),
            "device".to_string(),
            "process".to_string(),
            "launch".to_string(),
            "--device".to_string(),
            self.udid.clone(),
            "--terminate-existing".to_string(),
            "--console".to_string(),
            bundle_id.to_string(),
        ];
        argv.extend(app_args.iter().cloned());
        argv
    }
}

/// Spawn `xcrun <argv…>` and collect both streams. stdin is `/dev/null` so a
/// prompt cannot block the claim loop.
fn run_capture(argv: &[String]) -> anyhow::Result<CommandOutput> {
    tracing::trace!("{PROGRAM} {}", argv.join(" "));
    let output = Command::new(PROGRAM)
        .args(argv)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| spawn_error(PROGRAM, error))?;
    Ok(CommandOutput {
        status: output.status.code(),
        // Lossy: a device or app name can carry any bytes.
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

/// The identifier column of a `list devices` table. Drops [`NO_DEVICES_SENTENCE`]
/// and any other prose by the whitespace test: no identifier form contains a space.
fn identifier_rows(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| *line != NO_DEVICES_SENTENCE)
        .filter(|line| !line.chars().any(char::is_whitespace))
        .map(str::to_string)
        .collect()
}

/// [`run_capture`] with a deadline of our own, for `list devices`. SIGKILL is
/// fine here: a query has nothing on the phone to finish. Both pipes are drained
/// on their own threads.
fn run_capture_bounded(argv: &[String], deadline: Duration) -> anyhow::Result<CommandOutput> {
    tracing::trace!("{PROGRAM} {}", argv.join(" "));
    run_capture_bounded_program(PROGRAM, argv, deadline)
}

/// [`run_capture_bounded`] over an explicit program, so the deadline is testable.
fn run_capture_bounded_program(
    program: &str,
    argv: &[String],
    deadline: Duration,
) -> anyhow::Result<CommandOutput> {
    match tool::run_bounded(Command::new(program).args(argv), deadline) {
        Ok(ran) => Ok(CommandOutput {
            status: ran.status.code(),
            stdout: ran.stdout,
            stderr: ran.stderr,
        }),
        Err(ToolError::Run { source, .. }) => Err(spawn_error(program, source)),
        Err(ToolError::TimedOut { stderr, .. }) => Err(DevicectlError::TimedOut {
            command: format!("{program} {}", argv.join(" ")),
            secs: deadline.as_secs(),
            diagnostic: diagnostic_line(&stderr),
        }
        .into()),
    }
}

/// `--timeout` takes whole seconds; floors to 1, since `0` is undocumented.
fn timeout_arg(timeout: Duration) -> String {
    timeout.as_secs().max(1).to_string()
}

fn render(argv: &[String]) -> String {
    format!("{PROGRAM} {}", argv.join(" "))
}

fn describe_status(status: Option<i32>) -> String {
    match status {
        Some(code) => format!("with status {code}"),
        None => "on a signal".to_string(),
    }
}

/// The `ERROR: …` line of a failed call; the rest of stderr is noise.
fn diagnostic_line(stderr: &str) -> String {
    stderr
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("no output on stderr")
        .to_string()
}

fn parse_document(text: &str) -> anyhow::Result<DevicectlOutput> {
    let document: Value =
        serde_json::from_str(text).context("devicectl --json-output was not JSON")?;
    Ok(DevicectlOutput {
        result: document.get("result").cloned(),
        error: document.get("error").map(fault),
        outcome: string_at(&document, "/info/outcome"),
    })
}

/// `error` as a typed fault. `userInfo` values are wrapped as `{"string": …}`;
/// the unwrapped spelling is accepted too.
fn fault(error: &Value) -> DevicectlFaultError {
    let description = string_at(error, "/userInfo/NSLocalizedDescription/string")
        .or_else(|| string_at(error, "/userInfo/NSLocalizedDescription"))
        .unwrap_or_else(|| "no description".to_string());
    DevicectlFaultError {
        domain: string_at(error, "/domain").unwrap_or_else(|| "unknown domain".to_string()),
        code: error
            .get("code")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        description,
    }
}

/// Read the profile fields out of a `device info details` `result`. Only
/// `identifier` and `productType` are required.
fn identity_from_result(result: &Value) -> anyhow::Result<DeviceDetails> {
    let identifier = string_at(result, "/identifier")
        .context("devicectl `device info details` carried no result.identifier")?;
    let product_type = string_at(result, "/hardwareProperties/productType")
        .context("devicectl `device info details` carried no hardwareProperties.productType")?;
    Ok(DeviceDetails {
        identifier,
        hardware_udid: string_at(result, "/hardwareProperties/udid"),
        product_type,
        marketing_name: string_at(result, "/hardwareProperties/marketingName"),
        os_version: string_at(result, "/deviceProperties/osVersionNumber"),
        os_build: string_at(result, "/deviceProperties/osBuildUpdate"),
        // `cpuType` is an object; the string is at `.name`.
        cpu_type: string_at(result, "/hardwareProperties/cpuType/name"),
        device_name: string_at(result, "/deviceProperties/name"),
        tunnel_state: string_at(result, "/connectionProperties/tunnelState"),
        developer_mode: string_at(result, "/deviceProperties/developerModeStatus"),
    })
}

/// What the details of a phone that answered `device info apps` say. A
/// `tunnelState` of `unavailable` is a stale cache; Developer Mode off means
/// every launch would fail. A reachable phone carries the iOS version read.
fn reachability_of(identity: DeviceDetails) -> Reachability {
    if identity.tunnel_state.as_deref() == Some(TUNNEL_UNAVAILABLE) {
        return Reachability::Unreachable(format!(
            "the device record reports tunnelState `{TUNNEL_UNAVAILABLE}`, so it is cached rather \
             than live"
        ));
    }
    match identity.developer_mode.as_deref() {
        Some(DEVELOPER_MODE_ENABLED) | None => Reachability::Reachable {
            os_version: identity.os_version,
        },
        Some(status) => Reachability::Unreachable(format!(
            "developerModeStatus is `{status}`, so the app cannot launch: turn Developer Mode on \
             in the phone's Settings > Privacy & Security"
        )),
    }
}

/// One `result.apps[]` entry; dropped without a bundle identifier.
fn installed_app(entry: &Value) -> Option<InstalledApp> {
    Some(InstalledApp {
        bundle_id: string_at(entry, "/bundleIdentifier")?,
        name: string_at(entry, "/name").unwrap_or_default(),
        build_version: string_at(entry, "/bundleVersion"),
        version: string_at(entry, "/version").or_else(|| string_at(entry, "/versionNumber")),
    })
}

/// One `result.runningProcesses[]` entry; the pid is required.
fn device_process(entry: &Value) -> Option<DeviceProcess> {
    let pid = entry
        .get("processIdentifier")
        .or_else(|| entry.get("pid"))
        .and_then(Value::as_u64)?;
    Some(DeviceProcess {
        pid: u32::try_from(pid).ok()?,
        executable: string_at(entry, "/executable")
            .or_else(|| string_at(entry, "/executablePath"))
            .unwrap_or_default(),
    })
}

fn string_at(value: &Value, pointer: &str) -> Option<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    /// A real `device info details` document, trimmed.
    const DETAILS_JSON: &str = r#"{
      "info": {
        "arguments": ["devicectl","device","info","details"],
        "commandType": "devicectl.device.info.details",
        "jsonVersion": 3,
        "outcome": "success",
        "version": "518.33"
      },
      "result": {
        "connectionProperties": {
          "authenticationType": "manualPairing",
          "pairingState": "paired",
          "transportType": "wired",
          "tunnelState": "connected"
        },
        "deviceProperties": {
          "ddiServicesAvailable": true,
          "developerModeStatus": "enabled",
          "name": "Lab 17 pro 2",
          "osBuildUpdate": "23F81",
          "osVersionNumber": "26.5.1"
        },
        "hardwareProperties": {
          "cpuType": { "name": "arm64e", "subType": 2, "type": 16777228 },
          "deviceType": "iPhone",
          "ecid": 5194350912913436,
          "marketingName": "iPhone 17 Pro",
          "platform": "iOS",
          "productType": "iPhone18,1",
          "reality": "physical",
          "udid": "00008150-0012743C1100C01C"
        },
        "identifier": "1ACFAA4F-A9FD-5840-B495-072D33D2A5F5"
      }
    }"#;

    /// What a FAILED run writes: no `result`, an `error`, `info.outcome` `failed`.
    const FAILED_JSON: &str = r#"{
      "error": {
        "code": 1011,
        "domain": "com.apple.dt.CoreDeviceError",
        "userInfo": {
          "DeviceIdentifier": { "string": "ecid_5194350912913436" },
          "NSLocalizedDescription": {
            "string": "CoreDeviceService was unable to locate a device matching the requested device identifier."
          }
        }
      },
      "info": {
        "commandType": "devicectl.device.info.apps",
        "jsonVersion": 3,
        "outcome": "failed",
        "version": "518.33"
      }
    }"#;

    #[test]
    fn details_fixture_parses_including_cpu_type_as_an_object() -> anyhow::Result<()> {
        let parsed = parse_document(DETAILS_JSON).context("parsing the details fixture")?;
        assert_eq!(parsed.outcome.as_deref(), Some("success"));
        let result = parsed
            .into_result()
            .context("details fixture had no result")?;
        let identity = identity_from_result(&result).context("reading the identity fields")?;

        assert_eq!(identity.identifier, "1ACFAA4F-A9FD-5840-B495-072D33D2A5F5");
        assert_eq!(
            identity.hardware_udid.as_deref(),
            Some("00008150-0012743C1100C01C")
        );
        assert_eq!(identity.product_type, "iPhone18,1");
        assert_eq!(identity.marketing_name.as_deref(), Some("iPhone 17 Pro"));
        assert_eq!(identity.os_version.as_deref(), Some("26.5.1"));
        assert_eq!(identity.os_build.as_deref(), Some("23F81"));
        assert_eq!(identity.device_name.as_deref(), Some("Lab 17 pro 2"));
        assert_eq!(identity.tunnel_state.as_deref(), Some("connected"));
        assert_eq!(identity.developer_mode.as_deref(), Some("enabled"));
        assert_eq!(identity.cpu_type.as_deref(), Some("arm64e"));
        Ok(())
    }

    /// A live record with Developer Mode on can run a job, and carries its
    /// iOS. A cached record, or Developer Mode in any other state, cannot. A
    /// record with no status is not evidence of either.
    #[rstest]
    #[case::live("connected", Some("enabled"), None)]
    #[case::no_developer_mode_status("connected", None, None)]
    #[case::cached("unavailable", Some("enabled"), Some("tunnelState `unavailable`"))]
    #[case::developer_mode_off(
        "connected",
        Some("disabled"),
        Some("developerModeStatus is `disabled`, so the app cannot launch")
    )]
    fn the_details_decide_whether_the_phone_can_run_a_job(
        #[case] tunnel: &str,
        #[case] developer_mode: Option<&str>,
        #[case] refused: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut document: Value = serde_json::from_str(DETAILS_JSON)?;
        let result = document
            .pointer_mut("/result")
            .context("the fixture has a result")?;
        *result
            .pointer_mut("/connectionProperties/tunnelState")
            .context("the fixture has a tunnelState")? = json!(tunnel);
        let properties = result
            .pointer_mut("/deviceProperties")
            .and_then(Value::as_object_mut)
            .context("the fixture has deviceProperties")?;
        match developer_mode {
            Some(status) => properties.insert("developerModeStatus".to_owned(), json!(status)),
            None => properties.remove("developerModeStatus"),
        };

        match (reachability_of(identity_from_result(result)?), refused) {
            (Reachability::Reachable { os_version }, None) => {
                assert_eq!(os_version.as_deref(), Some("26.5.1"));
            }
            (Reachability::Unreachable(reason), Some(expected)) => {
                assert!(reason.contains(expected), "{reason}");
            }
            (reachability, refused) => {
                anyhow::bail!("{reachability:?}, expected refusal {refused:?}")
            }
        }
        Ok(())
    }

    #[test]
    fn a_failed_run_yields_the_devicectl_error_rather_than_a_parse_failure() -> anyhow::Result<()> {
        let parsed = parse_document(FAILED_JSON).context("parsing the failed-run fixture")?;
        assert_eq!(parsed.outcome.as_deref(), Some("failed"));
        assert!(parsed.result.is_none(), "a failed run carries no result");

        let fault = parsed
            .error
            .clone()
            .context("the failed-run fixture carried no error object")?;
        assert_eq!(fault.domain, "com.apple.dt.CoreDeviceError");
        assert_eq!(fault.code, DEVICE_NOT_FOUND_CODE);
        assert!(fault.is_device_absent(), "1011 is the absent-device code");
        assert!(
            fault.description.contains("unable to locate a device"),
            "the userInfo description is wrapped one level deep: {}",
            fault.description
        );

        let message = match parsed.into_result() {
            Ok(_) => anyhow::bail!("a failed-run document must not yield a result"),
            Err(error) => format!("{error:#}"),
        };
        assert!(message.contains("1011"), "{message}");
        Ok(())
    }

    #[test]
    fn launch_argv_carries_no_timeout_and_clears_a_live_instance() -> anyhow::Result<()> {
        let device = Devicectl::new("UDID-1".to_string(), Duration::from_secs(60));
        let argv = device.launch_argv(
            "ai.liquid.liquid-pipette",
            &["headlessrun".to_string(), "submit=0".to_string()],
        );
        assert_eq!(
            argv,
            vec![
                "devicectl",
                "device",
                "process",
                "launch",
                "--device",
                "UDID-1",
                "--terminate-existing",
                "--console",
                "ai.liquid.liquid-pipette",
                "headlessrun",
                "submit=0",
            ]
        );
        assert!(!argv.iter().any(|arg| arg == "--timeout"), "{argv:?}");
        assert!(!argv.iter().any(|arg| arg == "--json-output"), "{argv:?}");
        Ok(())
    }

    #[test]
    fn parsed_calls_are_bounded_and_terminate_stays_sigterm() -> anyhow::Result<()> {
        let device = Devicectl::new("UDID-1".to_string(), Duration::from_secs(45));
        let argv = device.base_argv(&["device", "info", "details"], Some(device.call_timeout));
        assert_eq!(
            argv,
            vec![
                "devicectl",
                "device",
                "info",
                "details",
                "--device",
                "UDID-1",
                "--timeout",
                "45",
            ]
        );

        let terminate = device.base_argv(
            &["device", "process", "terminate", "--pid", "4242"],
            Some(device.call_timeout),
        );
        assert!(
            !terminate.iter().any(|arg| arg == "--kill"),
            "SIGKILL drops a result mid-upload: {terminate:?}"
        );
        Ok(())
    }

    #[test]
    fn a_sub_second_timeout_floors_to_one_second() {
        assert_eq!(timeout_arg(Duration::from_millis(200)), "1");
        assert_eq!(timeout_arg(Duration::from_secs(1800)), "1800");
    }

    #[test]
    fn the_empty_fleet_sentence_is_not_a_device() {
        assert!(identifier_rows("No devices found.\n").is_empty());
        assert!(identifier_rows("").is_empty());
        assert!(identifier_rows("\n   \n").is_empty());
        assert!(identifier_rows("2 devices are not connected\n").is_empty());
    }

    #[test]
    fn identifier_rows_keep_the_identifier_column() {
        assert_eq!(
            identifier_rows(
                "1ACFAA4F-A9FD-5840-B495-072D33D2A5F5\n  00008150-0012743C1100C01C  \n\nmac.local\n"
            ),
            vec![
                "1ACFAA4F-A9FD-5840-B495-072D33D2A5F5".to_string(),
                "00008150-0012743C1100C01C".to_string(),
                "mac.local".to_string(),
            ]
        );
    }

    #[test]
    fn a_bounded_call_kills_a_child_that_never_exits() -> anyhow::Result<()> {
        let error = run_capture_bounded_program(
            "/bin/sh",
            &["-c".to_string(), "printf ready; sleep 30".to_string()],
            Duration::from_millis(300),
        )
        .err()
        .context("a child that outlives the deadline must be an error")?;
        let message = format!("{error:#}");
        assert!(message.contains("did not finish within"), "{message}");
        Ok(())
    }

    #[test]
    fn a_bounded_call_returns_a_prompt_childs_output() -> anyhow::Result<()> {
        let output = run_capture_bounded_program(
            "/bin/sh",
            &[
                "-c".to_string(),
                "printf 'UDID-1\\n'; printf 'warned\\n' >&2".to_string(),
            ],
            Duration::from_secs(10),
        )?;
        assert!(output.succeeded(), "{output:?}");
        assert_eq!(identifier_rows(&output.stdout), vec!["UDID-1".to_string()]);
        assert_eq!(output.stderr.trim(), "warned");
        Ok(())
    }
}
