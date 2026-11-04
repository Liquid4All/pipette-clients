//! A phone that does what it is told and records every mutating call in order,
//! so a test asserts on the SEQUENCE the loop produced.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::{DeviceControl, InstalledApp, Reachability};
use crate::app::{ConsoleRun, SentinelStream, Signals};
use crate::pool::bundle::{BundleError, FleetSigned};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Call {
    Install(PathBuf),
    Uninstall(String),
    Launch {
        bundle_id: String,
        app_args: Vec<String>,
    },
    Terminate(u32),
}

/// An install that raises `started`, then takes `takes`: a `devicectl
/// install` that no stop or lost lease interrupts.
#[derive(Debug)]
struct SlowInstall {
    started: Arc<AtomicBool>,
    takes: Duration,
}

#[derive(Debug, Default)]
pub(crate) struct FakeDevice {
    udid: String,
    apps: RefCell<Vec<InstalledApp>>,
    /// `Some(error)` makes the next install fail and leave `apps` untouched.
    install_failure: RefCell<Option<anyhow::Error>>,
    slow_install: Option<SlowInstall>,
    /// What the fleet gates answer, one per check; passes once it is empty.
    gate_refusals: RefCell<VecDeque<BundleError>>,
    /// Every build the fleet gates were asked about, in order.
    gated: Arc<Mutex<Vec<PathBuf>>>,
    /// Raised by the fleet gates, as a stop arriving then would be.
    raise_at_gate: Option<Arc<AtomicBool>>,
    console: RefCell<VecDeque<ConsoleRun>>,
    /// `Some(reason)`: the phone has gone, as `reachability` reports it.
    gone: Option<String>,
    /// What the next probes find, one each, before `gone` applies.
    probes: RefCell<VecDeque<Reachability>>,
    /// The iOS version `reachability` reads.
    os_version: Option<String>,
    /// Shared, so a test keeps a handle after the fake is boxed into a `Phone`.
    calls: Arc<Mutex<Vec<Call>>>,
}

impl FakeDevice {
    pub(crate) fn new(udid: &str) -> Self {
        Self {
            udid: udid.to_owned(),
            ..Self::default()
        }
    }

    pub(crate) fn with_installed(self, bundle_id: &str) -> Self {
        self.apps.borrow_mut().push(InstalledApp {
            bundle_id: bundle_id.to_owned(),
            name: "Pipette".to_owned(),
            build_version: Some("1".to_owned()),
            version: Some("0.1.0".to_owned()),
        });
        self
    }

    pub(crate) fn failing_install(self, message: &str) -> Self {
        self.failing_install_with(anyhow::anyhow!("{message}"))
    }

    pub(crate) fn failing_install_with(self, error: anyhow::Error) -> Self {
        *self.install_failure.borrow_mut() = Some(error);
        self
    }

    pub(crate) fn installing_slowly(self, started: Arc<AtomicBool>, takes: Duration) -> Self {
        Self {
            slow_install: Some(SlowInstall { started, takes }),
            ..self
        }
    }

    pub(crate) fn running_ios(self, os_version: &str) -> Self {
        Self {
            os_version: Some(os_version.to_owned()),
            ..self
        }
    }

    pub(crate) fn gone(self, reason: &str) -> Self {
        Self {
            gone: Some(reason.to_owned()),
            ..self
        }
    }

    /// The next probe finds the phone gone.
    pub(crate) fn then_gone(self, reason: &str) -> Self {
        self.probes
            .borrow_mut()
            .push_back(Reachability::Unreachable(reason.to_owned()));
        self
    }

    /// The next probe finds the phone back.
    pub(crate) fn then_back(self) -> Self {
        let os_version = self.os_version.clone();
        self.probes
            .borrow_mut()
            .push_back(Reachability::Reachable { os_version });
        self
    }

    pub(crate) fn raising_at_gate(self, flag: Arc<AtomicBool>) -> Self {
        Self {
            raise_at_gate: Some(flag),
            ..self
        }
    }

    pub(crate) fn then_gate_refuses(self, refusal: BundleError) -> Self {
        self.gate_refusals.borrow_mut().push_back(refusal);
        self
    }

    pub(crate) fn gated(&self) -> GateLog {
        GateLog(Arc::clone(&self.gated))
    }

    pub(crate) fn then_console(self, run: ConsoleRun) -> Self {
        self.console.borrow_mut().push_back(run);
        self
    }

    pub(crate) fn log(&self) -> CallLog {
        CallLog(Arc::clone(&self.calls))
    }

    fn record(&self, call: Call) -> anyhow::Result<()> {
        self.calls
            .lock()
            .map_err(|_| anyhow::anyhow!("a call panicked holding the log"))?
            .push(call);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CallLog(Arc<Mutex<Vec<Call>>>);

impl CallLog {
    pub(crate) fn calls(&self) -> anyhow::Result<Vec<Call>> {
        self.0
            .lock()
            .map(|calls| calls.clone())
            .map_err(|_| anyhow::anyhow!("a call panicked holding the log"))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GateLog(Arc<Mutex<Vec<PathBuf>>>);

impl GateLog {
    pub(crate) fn builds(&self) -> anyhow::Result<Vec<PathBuf>> {
        self.0
            .lock()
            .map(|builds| builds.clone())
            .map_err(|_| anyhow::anyhow!("a gate panicked holding the log"))
    }
}

impl DeviceControl for FakeDevice {
    fn udid(&self) -> &str {
        &self.udid
    }

    fn reachability(&self, _bundle_id: &str) -> Reachability {
        if let Some(probe) = self.probes.borrow_mut().pop_front() {
            return probe;
        }
        match &self.gone {
            Some(reason) => Reachability::Unreachable(reason.clone()),
            None => Reachability::Reachable {
                os_version: self.os_version.clone(),
            },
        }
    }

    fn info_apps(&self, bundle_id: Option<&str>) -> anyhow::Result<Vec<InstalledApp>> {
        let apps = self.apps.borrow();
        Ok(apps
            .iter()
            .filter(|app| bundle_id.is_none_or(|wanted| app.bundle_id == wanted))
            .cloned()
            .collect())
    }

    fn verify_installable(
        &self,
        build: &Path,
        fleet: Option<&FleetSigned<'_>>,
    ) -> anyhow::Result<()> {
        if fleet.is_none() {
            return Ok(());
        }
        self.gated
            .lock()
            .map_err(|_| anyhow::anyhow!("a gate panicked holding the log"))?
            .push(build.to_path_buf());
        if let Some(flag) = &self.raise_at_gate {
            flag.store(true, Ordering::SeqCst);
        }
        match self.gate_refusals.borrow_mut().pop_front() {
            Some(refusal) => Err(refusal.into()),
            None => Ok(()),
        }
    }

    fn install_app(&self, path: &Path) -> anyhow::Result<()> {
        self.record(Call::Install(path.to_path_buf()))?;
        if let Some(slow) = &self.slow_install {
            slow.started.store(true, Ordering::SeqCst);
            std::thread::sleep(slow.takes);
        }
        if let Some(error) = self.install_failure.borrow_mut().take() {
            return Err(error);
        }
        // The fake cannot read a bundle id off the path; every test uses this one.
        let mut apps = self.apps.borrow_mut();
        apps.retain(|app| app.bundle_id != "ai.liquid.liquid-pipette");
        apps.push(InstalledApp {
            bundle_id: "ai.liquid.liquid-pipette".to_owned(),
            name: "Pipette".to_owned(),
            build_version: Some("1".to_owned()),
            version: Some("0.1.0".to_owned()),
        });
        Ok(())
    }

    fn uninstall_app(&self, bundle_id: &str) -> anyhow::Result<()> {
        self.record(Call::Uninstall(bundle_id.to_owned()))?;
        self.apps
            .borrow_mut()
            .retain(|app| app.bundle_id != bundle_id);
        Ok(())
    }

    fn pid_for_bundle(&self, _bundle_id: &str) -> anyhow::Result<Option<u32>> {
        Ok(None)
    }

    fn terminate_process(&self, pid: u32) -> anyhow::Result<()> {
        self.record(Call::Terminate(pid))?;
        Ok(())
    }

    fn launch_console(
        &self,
        bundle_id: &str,
        app_args: &[String],
        _deadline: Duration,
        _first_line: Duration,
        _sentinel_on: SentinelStream,
        signals: Signals<'_>,
        on_stdout_line: &mut dyn FnMut(&str),
    ) -> anyhow::Result<ConsoleRun> {
        self.record(Call::Launch {
            bundle_id: bundle_id.to_owned(),
            app_args: app_args.to_vec(),
        })?;
        let run =
            self.console.borrow_mut().pop_front().ok_or_else(|| {
                anyhow::anyhow!("the fake has no console script left for a launch")
            })?;
        // As the heartbeat does when the server takes the lease mid-run.
        if run.aborted {
            signals.abort.store(true, Ordering::SeqCst);
        }
        for line in run.stdout.lines() {
            on_stdout_line(line);
        }
        Ok(run)
    }
}
