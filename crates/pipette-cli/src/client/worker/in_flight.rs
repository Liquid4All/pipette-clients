//! Durable record of the job a worker was running when its device stopped.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

/// How often a running job refreshes `alive_at` in its marker.
pub const HEARTBEAT: Duration = Duration::from_secs(30);

/// How often the heartbeat thread checks for a stop request.
const STOP_POLL: Duration = Duration::from_millis(250);

/// The longest time from the last `alive_at` refresh to the start of the next
/// boot that still counts as a reboot during the job. It covers one missed
/// heartbeat, the reset itself, and a small clock skew.
const REBOOT_WINDOW: time::Duration = time::Duration::seconds(90);

/// One in-flight job. `boot_id` is `None` if the worker could not read the
/// kernel boot ID. `started_at` is an RFC 3339 timestamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InFlight {
    pub job_id: String,
    pub boot_id: Option<String>,
    pub started_at: String,
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub alive_at: Option<OffsetDateTime>,
}

impl InFlight {
    pub fn new(job_id: &str, boot_id: Option<String>) -> Self {
        let now = OffsetDateTime::now_utc();
        Self {
            job_id: job_id.to_owned(),
            boot_id,
            started_at: now
                .format(&Rfc3339)
                .unwrap_or_else(|_| "unknown-time".into()),
            alive_at: Some(now),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Boot {
    pub id: Option<String>,
    pub started_at: Option<OffsetDateTime>,
}

impl Boot {
    /// Each field is `None` if the worker cannot read it (always, on any host
    /// but Linux/Android), and that disables reboot detection.
    pub fn current() -> Self {
        Self {
            id: boot_id(),
            started_at: boot_started_at(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimDecision<'a> {
    Run,
    StaleJob {
        old_job_id: &'a str,
    },
    SameBoot,
    /// The marker is from an earlier boot, but it does not show the job alive
    /// when that boot ended. The worker stopped before the reboot, or a
    /// timestamp is unknown.
    EarlierBoot,
    Rebooted {
        old_boot_id: &'a str,
        new_boot_id: &'a str,
    },
}

/// The worker treats a restart as a reboot only if all of these are true:
///
/// - The server returns the same job.
/// - Both boot IDs are known, and they differ.
/// - The job refreshed `alive_at` within [`REBOOT_WINDOW`] before the current
///   boot started.
///
/// A marker without a boot ID cannot prove a reboot.
pub fn decide<'a>(marker: Option<&'a InFlight>, job_id: &str, boot: &'a Boot) -> ClaimDecision<'a> {
    let Some(marker) = marker else {
        return ClaimDecision::Run;
    };
    if marker.job_id != job_id {
        return ClaimDecision::StaleJob {
            old_job_id: &marker.job_id,
        };
    }
    match (marker.boot_id.as_deref(), boot.id.as_deref()) {
        (Some(old), Some(current)) if old != current => {
            let alive_at_reboot = marker
                .alive_at
                .and_then(|alive_at| alive_at.checked_add(REBOOT_WINDOW))
                .zip(boot.started_at)
                .is_some_and(|(latest, started_at)| latest >= started_at);
            if alive_at_reboot {
                ClaimDecision::Rebooted {
                    old_boot_id: old,
                    new_boot_id: current,
                }
            } else {
                ClaimDecision::EarlierBoot
            }
        }
        _ => ClaimDecision::SameBoot,
    }
}

pub fn boot_id() -> Option<String> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .ok()
            .map(|id| id.trim().to_owned())
            .filter(|id| !id.is_empty())
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        None
    }
}

fn boot_started_at() -> Option<OffsetDateTime> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let uptime = fs::read_to_string("/proc/uptime").ok()?;
        started_at_from_uptime(&uptime, OffsetDateTime::now_utc())
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        None
    }
}

/// The first field of `/proc/uptime` is seconds since boot. Built for tests on
/// every host, so any host covers it.
#[cfg(any(test, target_os = "linux", target_os = "android"))]
fn started_at_from_uptime(uptime: &str, now: OffsetDateTime) -> Option<OffsetDateTime> {
    let seconds = uptime.split_whitespace().next()?.parse::<f64>().ok()?;
    now.checked_sub(time::Duration::checked_seconds_f64(seconds)?)
}

#[derive(Clone)]
pub struct MarkerStore {
    path: PathBuf,
}

impl MarkerStore {
    pub fn new(dir: &Path) -> Self {
        Self {
            path: dir.join("in-flight.json"),
        }
    }

    pub fn read(&self) -> anyhow::Result<Option<InFlight>> {
        let bytes = match fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(err).with_context(|| format!("failed to read {}", self.path.display()));
            }
        };
        serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", self.path.display()))
            .map(Some)
    }

    /// Replace the marker atomically. A kernel panic leaves the old record or
    /// the new record.
    pub fn write(&self, marker: &InFlight) -> anyhow::Result<()> {
        let dir = self
            .path
            .parent()
            .context("marker has no parent directory")?;
        fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
        let temporary = dir.join(format!(".in-flight-{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| -> anyhow::Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .with_context(|| format!("failed to create {}", temporary.display()))?;
            serde_json::to_writer(&mut file, marker)?;
            file.write_all(b"\n")?;
            pipette_artifacts::sync_file(&file)?;
            fs::rename(&temporary, &self.path)
                .with_context(|| format!("failed to replace {}", self.path.display()))?;
            pipette_artifacts::sync_dir(dir)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    pub fn remove(&self) -> anyhow::Result<()> {
        match fs::remove_file(&self.path) {
            Ok(()) => {
                if let Some(dir) = self.path.parent() {
                    pipette_artifacts::sync_dir(dir)?;
                }
                Ok(())
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => {
                Err(err).with_context(|| format!("failed to remove {}", self.path.display()))
            }
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Refreshes `alive_at` every heartbeat and removes the marker on drop. An
/// orderly shutdown is never a reboot, so the marker also goes when a signal or
/// a closed session forces an exit (the shared teardown registry removes it) or
/// a signal asks for a graceful stop (the heartbeat thread sees the stop flag).
///
/// The teardown registration is bounded, so the first interrupt signal still
/// asks the worker to stop gracefully.
pub struct InFlightRun {
    store: MarkerStore,
    closing: Arc<Mutex<bool>>,
    heartbeat: Option<Heartbeat>,
    _teardown: pipette_subprocess::BoundedTeardownGuard,
}

impl InFlightRun {
    pub fn start(
        store: MarkerStore,
        marker: &InFlight,
        stop: &Arc<AtomicBool>,
    ) -> anyhow::Result<Self> {
        Self::start_with_heartbeat(store, marker, stop, HEARTBEAT)
    }

    fn start_with_heartbeat(
        store: MarkerStore,
        marker: &InFlight,
        stop: &Arc<AtomicBool>,
        interval: Duration,
    ) -> anyhow::Result<Self> {
        // Register before writing. The lock makes a forced exit that lands
        // during the write wait for its rename, then remove the marker. If it
        // lands first, the writer sees `closing` and never creates the marker.
        let closing = Arc::new(Mutex::new(false));
        let teardown_closing = Arc::clone(&closing);
        let path = store.path().to_path_buf();
        let teardown = pipette_subprocess::register_bounded_teardown(move || {
            let mut closing = teardown_closing.lock().unwrap_or_else(|p| p.into_inner());
            *closing = true;
            let result = (MarkerStore { path }).remove();
            drop(closing);
            if let Err(err) = result {
                tracing::warn!(error = %err, "could not remove in-flight marker during teardown");
            }
        });
        {
            let closing = closing.lock().unwrap_or_else(|p| p.into_inner());
            anyhow::ensure!(
                !*closing && !stop.load(Ordering::SeqCst),
                "the worker stopped before the job started"
            );
            store.write(marker)?;
        }
        let mut run = Self {
            store,
            closing,
            heartbeat: None,
            _teardown: teardown,
        };
        // On an error, `run` drops here and removes the marker it wrote.
        run.heartbeat = Some(Heartbeat::spawn(
            run.store.clone(),
            marker.clone(),
            Arc::clone(&run.closing),
            Arc::clone(stop),
            interval,
        )?);
        Ok(run)
    }
}

/// The thread that refreshes `alive_at` until the run ends, or removes the
/// marker when a stop is requested. It takes the `closing` lock for each write,
/// so it never writes after a teardown removed the marker.
struct Heartbeat {
    stopped: Arc<(Mutex<bool>, Condvar)>,
    thread: thread::JoinHandle<()>,
}

impl Heartbeat {
    fn spawn(
        store: MarkerStore,
        mut marker: InFlight,
        closing: Arc<Mutex<bool>>,
        stop: Arc<AtomicBool>,
        interval: Duration,
    ) -> anyhow::Result<Self> {
        let stopped = Arc::new((Mutex::new(false), Condvar::new()));
        let thread_stopped = Arc::clone(&stopped);
        let thread = thread::Builder::new()
            .name("in-flight-heartbeat".into())
            .spawn(move || {
                let (lock, wake) = &*thread_stopped;
                let mut stopped = lock.lock().unwrap_or_else(PoisonError::into_inner);
                let mut next_beat = Instant::now() + interval;
                loop {
                    stopped = wake
                        .wait_timeout_while(stopped, interval.min(STOP_POLL), |stopped| !*stopped)
                        .map(|(guard, _)| guard)
                        .unwrap_or_else(|p| p.into_inner().0);
                    if *stopped {
                        return;
                    }
                    let mut closing = closing.lock().unwrap_or_else(PoisonError::into_inner);
                    if *closing {
                        return;
                    }
                    if stop.load(Ordering::SeqCst) {
                        *closing = true;
                        match store.remove() {
                            Ok(()) => tracing::info!("stop requested during the job; removed the in-flight marker"),
                            Err(err) => tracing::warn!(error = %format_args!("{err:#}"), "could not remove in-flight marker after a stop request"),
                        }
                        return;
                    }
                    if Instant::now() < next_beat {
                        continue;
                    }
                    next_beat += interval;
                    marker.alive_at = Some(OffsetDateTime::now_utc());
                    if let Err(err) = store.write(&marker) {
                        tracing::warn!(error = %format_args!("{err:#}"), "could not refresh in-flight marker");
                    }
                }
            })
            .context("failed to start the in-flight heartbeat thread")?;
        Ok(Self { stopped, thread })
    }

    fn stop(self) {
        {
            let (lock, wake) = &*self.stopped;
            *lock.lock().unwrap_or_else(PoisonError::into_inner) = true;
            wake.notify_all();
        }
        if self.thread.join().is_err() {
            tracing::warn!("in-flight heartbeat thread panicked");
        }
    }
}

impl Drop for InFlightRun {
    fn drop(&mut self) {
        if let Some(heartbeat) = self.heartbeat.take() {
            heartbeat.stop();
        }
        let closing = self.closing.lock().unwrap_or_else(|p| p.into_inner());
        let result = self.store.remove();
        drop(closing);
        if let Err(err) = result {
            tracing::warn!(error = %err, "could not remove in-flight marker after job");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    const BOOT_STARTED: i64 = 1_700_000_000;

    /// `alive` and `boot_started` are offsets in seconds from `BOOT_STARTED`.
    #[rstest::rstest]
    #[case::no_marker(None, Some("new"), Some(0), ClaimDecision::Run)]
    #[case::other_job(Some(("old", Some("old"), Some(-30))), Some("new"), Some(0), ClaimDecision::StaleJob { old_job_id: "old" })]
    #[case::same_boot(Some(("job", Some("old"), Some(-30))), Some("old"), Some(0), ClaimDecision::SameBoot)]
    #[case::alive_at_reboot(Some(("job", Some("old"), Some(-30))), Some("new"), Some(0), ClaimDecision::Rebooted { old_boot_id: "old", new_boot_id: "new" })]
    #[case::edge_of_window(Some(("job", Some("old"), Some(-90))), Some("new"), Some(0), ClaimDecision::Rebooted { old_boot_id: "old", new_boot_id: "new" })]
    #[case::died_before_reboot(Some(("job", Some("old"), Some(-91))), Some("new"), Some(0), ClaimDecision::EarlierBoot)]
    #[case::alive_unknown(Some(("job", Some("old"), None)), Some("new"), Some(0), ClaimDecision::EarlierBoot)]
    #[case::boot_start_unknown(Some(("job", Some("old"), Some(-30))), Some("new"), None, ClaimDecision::EarlierBoot)]
    #[case::marker_boot_unknown(Some(("job", None, Some(-30))), Some("new"), Some(0), ClaimDecision::SameBoot)]
    #[case::current_boot_unknown(Some(("job", Some("old"), Some(-30))), None, Some(0), ClaimDecision::SameBoot)]
    fn claim_decision(
        #[case] marker: Option<(&str, Option<&str>, Option<i64>)>,
        #[case] boot_id: Option<&str>,
        #[case] boot_started: Option<i64>,
        #[case] expected: ClaimDecision<'static>,
    ) -> anyhow::Result<()> {
        let at = |offset: i64| OffsetDateTime::from_unix_timestamp(BOOT_STARTED + offset);
        let marker = marker
            .map(|(job_id, boot_id, alive)| -> anyhow::Result<InFlight> {
                Ok(InFlight {
                    job_id: job_id.into(),
                    boot_id: boot_id.map(str::to_owned),
                    started_at: "now".into(),
                    alive_at: alive.map(at).transpose()?,
                })
            })
            .transpose()?;
        let boot = Boot {
            id: boot_id.map(str::to_owned),
            started_at: boot_started.map(at).transpose()?,
        };
        assert_eq!(decide(marker.as_ref(), "job", &boot), expected);
        Ok(())
    }

    #[rstest::rstest]
    #[case::typical("12.00 40.00\n", Some(-12))]
    #[case::empty("", None)]
    #[case::garbage("up 3 days", None)]
    #[case::negative("-1.0 0.0", Some(1))]
    fn boot_start_from_uptime(
        #[case] uptime: &str,
        #[case] expected_offset: Option<i64>,
    ) -> anyhow::Result<()> {
        let now = OffsetDateTime::from_unix_timestamp(BOOT_STARTED)?;
        let started =
            started_at_from_uptime(uptime, now).map(|t| t.unix_timestamp() - BOOT_STARTED);
        assert_eq!(started, expected_offset);
        Ok(())
    }

    #[test]
    fn a_marker_without_alive_at_still_parses() -> anyhow::Result<()> {
        let marker: InFlight = serde_json::from_str(
            r#"{"job_id":"job-abc","boot_id":"boot-old","started_at":"now"}"#,
        )?;
        assert_eq!(marker.alive_at, None);
        Ok(())
    }

    #[test]
    fn marker_round_trip_and_cleanup() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let store = MarkerStore::new(dir.path());
        let marker = InFlight::new("job-abc", Some("boot-old".into()));
        {
            let _run = InFlightRun::start(store, &marker, &Arc::new(AtomicBool::new(false)))?;
            assert_eq!(MarkerStore::new(dir.path()).read()?, Some(marker));
        }
        assert_eq!(MarkerStore::new(dir.path()).read()?, None);
        Ok(())
    }

    #[test]
    fn a_running_job_refreshes_alive_at() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let store = MarkerStore::new(dir.path());
        let mut marker = InFlight::new("job-abc", Some("boot-old".into()));
        let first = OffsetDateTime::from_unix_timestamp(BOOT_STARTED)?;
        marker.alive_at = Some(first);
        {
            let _run = InFlightRun::start_with_heartbeat(
                store.clone(),
                &marker,
                &Arc::new(AtomicBool::new(false)),
                Duration::from_millis(5),
            )?;
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                let refreshed = store.read()?.and_then(|m| m.alive_at);
                if refreshed.is_some_and(|alive_at| alive_at > first) {
                    break;
                }
                anyhow::ensure!(Instant::now() < deadline, "alive_at was never refreshed");
                thread::sleep(Duration::from_millis(5));
            }
        }
        assert_eq!(store.read()?, None, "a heartbeat outlived its run");
        Ok(())
    }

    #[test]
    fn a_stop_request_removes_the_marker_while_the_job_runs() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let store = MarkerStore::new(dir.path());
        let stop = Arc::new(AtomicBool::new(false));
        let marker = InFlight::new("job-abc", Some("boot-old".into()));
        let _run = InFlightRun::start(store.clone(), &marker, &stop)?;
        assert!(store.read()?.is_some());
        stop.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + Duration::from_secs(10);
        while store.read()?.is_some() {
            anyhow::ensure!(Instant::now() < deadline, "a stop request left the marker");
            thread::sleep(Duration::from_millis(5));
        }
        Ok(())
    }

    #[test]
    fn a_job_does_not_start_after_a_stop_request() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let store = MarkerStore::new(dir.path());
        let stop = Arc::new(AtomicBool::new(true));
        let marker = InFlight::new("job-abc", None);
        assert!(InFlightRun::start(store.clone(), &marker, &stop).is_err());
        assert_eq!(store.read()?, None);
        Ok(())
    }
}
