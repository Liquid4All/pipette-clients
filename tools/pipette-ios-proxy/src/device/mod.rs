//! The phone, as the rest of the proxy sees it: the [`DeviceControl`] interface and
//! the data it returns. [`devicectl`] implements it over `xcrun devicectl`;
//! `fake` is the test double.

pub(crate) mod devicectl;
#[cfg(test)]
pub(crate) mod fake;
pub(crate) mod gpu_pin;

pub(crate) use self::devicectl::{Devicectl, DevicectlFaultError, INSTALL_HINT, PROGRAM};

use std::path::Path;
use std::time::Duration;

use crate::pool::bundle::FleetSigned;

#[derive(Debug, Clone)]
pub(crate) enum Reachability {
    /// `os_version`: what `device info details` read on the way, when it
    /// answered with one.
    Reachable { os_version: Option<String> },
    /// The phone cannot run a job now: absent, cached, or Developer Mode off.
    Unreachable(String),
}

/// What `device info details` says about the phone. No chip or RAM: `devicectl`
/// reports neither.
#[derive(Debug, Clone)]
pub(crate) struct DeviceDetails {
    /// The CoreDevice UUID, NOT the hardware UDID.
    pub(crate) identifier: String,
    /// The `00008150-…` hardware UDID. Shown by `probe`; nothing checks it.
    pub(crate) hardware_udid: Option<String>,
    /// e.g. `iPhone18,1`.
    pub(crate) product_type: String,
    pub(crate) marketing_name: Option<String>,
    pub(crate) os_version: Option<String>,
    pub(crate) os_build: Option<String>,
    pub(crate) cpu_type: Option<String>,
    /// The user-set device name.
    pub(crate) device_name: Option<String>,
    /// `"unavailable"` means the record is cached, not live.
    pub(crate) tunnel_state: Option<String>,
    /// Anything but `"enabled"` means the app cannot launch.
    pub(crate) developer_mode: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct InstalledApp {
    pub(crate) bundle_id: String,
    pub(crate) name: String,
    pub(crate) build_version: Option<String>,
    pub(crate) version: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct DeviceProcess {
    pub(crate) pid: u32,
    /// The executable path, matched as `/<AppName>.app/`.
    pub(crate) executable: String,
}

/// What the claim loop needs from a phone. [`Devicectl`] is the only real
/// implementation; a test fake lets tests provoke the failure branches.
/// No `identity`: the probe happens while resolving, on the concrete type.
pub(crate) trait DeviceControl: Send {
    fn udid(&self) -> &str;

    fn reachability(&self, bundle_id: &str) -> Reachability;

    fn info_apps(&self, bundle_id: Option<&str>) -> anyhow::Result<Vec<InstalledApp>>;

    /// Refuse a build `install_app` would reject, or that would replace the
    /// phone's identity (`crate::pool::bundle::verify_installable`).
    fn verify_installable(
        &self,
        build: &Path,
        fleet: Option<&FleetSigned<'_>>,
    ) -> anyhow::Result<()>;

    fn install_app(&self, path: &Path) -> anyhow::Result<()>;

    fn uninstall_app(&self, bundle_id: &str) -> anyhow::Result<()>;

    fn pid_for_bundle(&self, bundle_id: &str) -> anyhow::Result<Option<u32>>;

    fn terminate_process(&self, pid: u32) -> anyhow::Result<()>;

    /// Launch the app headless with `--console` and stream it until it exits, the
    /// deadline passes, the app prints no line by `first_line`, or a signal fires
    /// (`app::run_console`).
    #[expect(
        clippy::too_many_arguments,
        reason = "the launch, its two deadlines, its signals and its line sink"
    )]
    fn launch_console(
        &self,
        bundle_id: &str,
        app_args: &[String],
        deadline: Duration,
        first_line: Duration,
        sentinel_on: crate::app::SentinelStream,
        signals: crate::app::Signals<'_>,
        on_stdout_line: &mut dyn FnMut(&str),
    ) -> anyhow::Result<crate::app::ConsoleRun>;
}
