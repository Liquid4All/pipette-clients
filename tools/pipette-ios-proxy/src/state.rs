//! The state directory (DESIGN section 3). Nothing secret lives here.
//!
//! - `builds.json`, SHARED: what each build in the pool provides; `builds.lock`
//!   is held while one process refreshes it.
//! - `<device>/lock`: held by the one process driving that phone.
//! - `<device>/installed.json`: which build this proxy last put on that phone.
//! - `<device>/device.json`: what that phone last reported about itself.
//! - `releases/<version>/`, SHARED: an app build pulled from the releases bucket;
//!   `releases.json` records each version's digest and `releases.lock` is held
//!   while one process pulls.
//! - `signed/<version>/`, SHARED: a pulled build signed for the fleet;
//!   `signed.lock` is held while one process checks, signs or copies one.
//! - `<device>/staged/`: the phone's own copy of the signed build it installs.
//!
//! Every write goes through [`write_atomically`]; `builds.json` is refused when
//! corrupt, so a torn write would leave a proxy that cannot start.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::app::Signals;
use crate::timing;

const STATE_HOME_ENV: &str = "XDG_STATE_HOME";

const STATE_DIR_NAME: &str = "pipette-ios-proxy";

const INSTALLED_FILE_NAME: &str = "installed.json";
const BUILD_INDEX_FILE_NAME: &str = "builds.json";
const DEVICE_FACTS_FILE_NAME: &str = "device.json";
const LOCK_FILE_NAME: &str = "lock";
pub(crate) const POOL_LOCK_FILE_NAME: &str = "builds.lock";
const RELEASES_DIR_NAME: &str = "releases";
const RELEASE_RECORDS_FILE_NAME: &str = "releases.json";
const RELEASES_LOCK_FILE_NAME: &str = "releases.lock";
const SIGNED_DIR_NAME: &str = "signed";
const SIGNED_LOCK_FILE_NAME: &str = "signed.lock";
const STAGED_DIR_NAME: &str = "staged";

/// Why the state directory cannot be used.
#[derive(Debug, thiserror::Error)]
enum StateError {
    #[error(
        "{path} was written by a newer pipette-ios-proxy (schema {found}; this binary reads up to \
         {SCHEMA}). Upgrade the binary, or delete the file to have it rewritten"
    )]
    SchemaTooNew { path: PathBuf, found: u32 },
    #[error(
        "another pipette-ios-proxy ({holder}) already drives {device} (it holds {lock}). One process \
         per phone: stop the other one first"
    )]
    PhoneBusy {
        device: String,
        holder: Holder,
        lock: PathBuf,
    },
    #[error(
        "neither ${STATE_HOME_ENV} nor $HOME is set, so there is nowhere to keep state: pass --state-dir"
    )]
    NoStateHome,
    #[error("gave up after waiting {secs}s for {lock}, held by {holder}")]
    LockWaitExpired {
        lock: PathBuf,
        holder: Holder,
        secs: u64,
    },
    #[error("stopped while waiting for {lock}")]
    LockWaitInterrupted { lock: PathBuf },
}

/// The state root and the shared build index under it.
#[derive(Debug, Clone)]
pub(crate) struct StatePaths {
    pub(crate) root: PathBuf,
    pub(crate) build_index: PathBuf,
}

/// Where one device's own state lives, under [`StatePaths::root`].
#[derive(Debug, Clone)]
pub(crate) struct DevicePaths {
    pub(crate) dir: PathBuf,
    pub(crate) installed: PathBuf,
    pub(crate) device_facts: PathBuf,
    pub(crate) lock: PathBuf,
    /// The copy of a signed build this phone installs, with no lock held.
    pub(crate) staged: PathBuf,
}

/// Where pulled app builds live, under [`StatePaths::root`].
#[derive(Debug, Clone)]
pub(crate) struct ReleasePaths {
    /// One directory per pulled version.
    pub(crate) dir: PathBuf,
    pub(crate) records: PathBuf,
    pub(crate) lock: PathBuf,
    /// One directory per version signed for the fleet.
    pub(crate) signed: PathBuf,
    pub(crate) signed_lock: PathBuf,
}

/// What the phone reported about itself on its last completed run, harvested from
/// the submitted `payload.json`. The only route to `device_ram_bytes`: `devicectl`
/// reports no memory size. Not the iOS version: `devicectl` reads that.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct DeviceFacts {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) chip_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) ram_bytes: Option<u64>,
}

impl DeviceFacts {
    /// Read the `device_*` fields out of a submitted payload; `None` when it
    /// carries none, so the stored answer is left alone.
    pub(crate) fn from_payload(payload: &serde_json::Value) -> Option<Self> {
        let text = |key: &str| {
            payload
                .get(key)
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        };
        let facts = Self {
            name: text("device_name"),
            chip_model: text("device_chip_model"),
            // Zero would become a `ram_bytes:0` capability.
            ram_bytes: payload
                .get("device_ram_bytes")
                .and_then(serde_json::Value::as_u64)
                .filter(|bytes| *bytes > 0),
        };
        (facts != Self::default()).then_some(facts)
    }

    pub(crate) fn load(path: &Path) -> anyhow::Result<Option<Self>> {
        load_json(path, "the device facts")
    }

    pub(crate) fn store(&self, path: &Path) -> anyhow::Result<()> {
        store_json(path, "the device facts", self)
    }
}

/// The schema of every persisted document. Bump when a change would make an older
/// binary misread a file; `serde` silently ignoring unknown fields is not enough
/// once a field is renamed.
const SCHEMA: u32 = 1;

/// Read a persisted document, `None` when absent. The schema is checked BEFORE
/// the body so a newer file is refused by version, not by a body parse error.
pub(crate) fn load_json<T: serde::de::DeserializeOwned>(
    path: &Path,
    what: &str,
) -> anyhow::Result<Option<T>> {
    #[derive(Deserialize)]
    struct Head {
        #[serde(default)]
        schema: u32,
    }

    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(
                anyhow::Error::new(error).context(format!("reading {what} at {}", path.display()))
            );
        }
    };

    // Missing means 0: a file from before versioning.
    let head: Head = serde_json::from_str(&text)
        .with_context(|| format!("reading the schema of {what} at {}", path.display()))?;
    if head.schema > SCHEMA {
        return Err(StateError::SchemaTooNew {
            path: path.to_path_buf(),
            found: head.schema,
        }
        .into());
    }

    serde_json::from_str(&text)
        .map(Some)
        .with_context(|| format!("reading {what} at {}", path.display()))
}

pub(crate) fn store_json<T: Serialize>(path: &Path, what: &str, value: &T) -> anyhow::Result<()> {
    #[derive(Serialize)]
    struct Stamped<'a, T: Serialize> {
        schema: u32,
        #[serde(flatten)]
        body: &'a T,
    }
    let text = serde_json::to_string_pretty(&Stamped {
        schema: SCHEMA,
        body: value,
    })
    .with_context(|| format!("rendering {what}"))?;
    write_atomically(path, &text).with_context(|| format!("writing {what} at {}", path.display()))
}

/// A uniquely named temp file beside the target (`rename(2)` is atomic only
/// within a filesystem), then renamed over it. Creates the parent directory.
fn write_atomically(path: &Path, contents: &str) -> anyhow::Result<()> {
    use std::io::Write;

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let mut temp = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("creating a temp file in {}", parent.display()))?;
    temp.write_all(contents.as_bytes())
        .with_context(|| format!("writing {}", temp.path().display()))?;
    temp.persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

/// Which build this proxy last installed on one phone. Advisory: a caller confirms
/// the bundle is present before skipping an install. Written empty BEFORE an
/// install so a failure part-way leaves "unknown".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct InstalledBuild {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) path: Option<PathBuf>,
    /// Version mode: the `client_version` the phone reported for it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) version: Option<String>,
}

impl InstalledBuild {
    pub(crate) fn load(path: &Path) -> anyhow::Result<Self> {
        Ok(load_json(path, "the installed build")?.unwrap_or_default())
    }

    pub(crate) fn store(&self, path: &Path) -> anyhow::Result<()> {
        store_json(path, "the installed build", self)
    }
}

impl DevicePaths {
    fn in_dir(dir: PathBuf) -> Self {
        Self {
            installed: dir.join(INSTALLED_FILE_NAME),
            device_facts: dir.join(DEVICE_FACTS_FILE_NAME),
            lock: dir.join(LOCK_FILE_NAME),
            staged: dir.join(STAGED_DIR_NAME),
            dir,
        }
    }
}

impl StatePaths {
    /// `--state-dir` overrides the WHOLE path, otherwise
    /// `$XDG_STATE_HOME/pipette-ios-proxy/` falling back to `$HOME/.local/state/…`.
    pub(crate) fn resolve(override_dir: Option<&Path>) -> anyhow::Result<Self> {
        let root = match override_dir {
            Some(dir) => dir.to_path_buf(),
            None => state_home()?.join(STATE_DIR_NAME),
        };
        let build_index = root.join(BUILD_INDEX_FILE_NAME);
        Ok(Self { root, build_index })
    }

    /// One device's own directory. Sanitised: `devicectl` accepts a device NAME,
    /// which can contain a slash.
    pub(crate) fn device(&self, udid: &str) -> DevicePaths {
        DevicePaths::in_dir(self.root.join(sanitize_path_segment(udid)))
    }

    /// The phone directories under the root, sorted: those holding the lock
    /// every verb that drives a phone takes.
    pub(crate) fn phones(&self) -> anyhow::Result<Vec<DevicePaths>> {
        let reading = || format!("reading {}", self.root.display());
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error).with_context(reading),
        };
        let mut phones = entries
            .map(|entry| entry.map(|entry| DevicePaths::in_dir(entry.path())))
            .collect::<Result<Vec<_>, _>>()
            .with_context(reading)?
            .into_iter()
            .filter(|phone| phone.lock.is_file())
            .collect::<Vec<_>>();
        phones.sort_by(|a, b| a.dir.cmp(&b.dir));
        Ok(phones)
    }

    pub(crate) fn releases(&self) -> ReleasePaths {
        ReleasePaths {
            dir: self.root.join(RELEASES_DIR_NAME),
            records: self.root.join(RELEASE_RECORDS_FILE_NAME),
            lock: self.root.join(RELEASES_LOCK_FILE_NAME),
            signed: self.root.join(SIGNED_DIR_NAME),
            signed_lock: self.root.join(SIGNED_LOCK_FILE_NAME),
        }
    }
}

/// An exclusive lock on one file (`File::try_lock`, `flock(2)` on macOS), released
/// when dropped. The kernel drops it however the process dies.
#[derive(Debug)]
pub(crate) struct StateLock {
    _file: fs::File,
}

/// Who holds a lock we could not take: the pid the holder wrote, when readable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Holder(pub(crate) Option<String>);

impl std::fmt::Display for Holder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(pid) => write!(f, "pid {pid}"),
            None => f.write_str("an unknown process"),
        }
    }
}

impl StateLock {
    /// Take the lock at `path` without waiting, creating its directory. `Err(Holder)`
    /// when another process has it.
    pub(crate) fn try_acquire(path: &Path) -> anyhow::Result<Result<Self, Holder>> {
        use std::io::Write;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;

        match file.try_lock() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => {
                let pid = fs::read_to_string(path).unwrap_or_default();
                let pid = pid.trim();
                return Ok(Err(Holder((!pid.is_empty()).then(|| pid.to_owned()))));
            }
            Err(fs::TryLockError::Error(error)) => {
                return Err(error).with_context(|| format!("locking {}", path.display()));
            }
        }

        // Best effort: the pid is only for the message the next process prints.
        let _ = file.set_len(0);
        let _ = writeln!(file, "{}", std::process::id());
        Ok(Ok(Self { _file: file }))
    }

    pub(crate) fn wait_for(
        path: &Path,
        budget: Duration,
        signals: Signals<'_>,
    ) -> anyhow::Result<Self> {
        let started = Instant::now();
        let mut announced = false;
        loop {
            let holder = match Self::try_acquire(path)? {
                Ok(lock) => return Ok(lock),
                Err(holder) => holder,
            };
            if signals.raised() {
                return Err(StateError::LockWaitInterrupted {
                    lock: path.to_path_buf(),
                }
                .into());
            }
            if started.elapsed() >= budget {
                return Err(StateError::LockWaitExpired {
                    lock: path.to_path_buf(),
                    holder,
                    secs: budget.as_secs(),
                }
                .into());
            }
            if !announced {
                announced = true;
                tracing::info!(lock = %path.display(), holder = %holder, "waiting for a lock");
            }
            std::thread::sleep(timing::LOCK_POLL.min(budget));
        }
    }

    /// The lock of the one process allowed to drive this phone: two would be two
    /// claim loops installing over each other. Non-blocking: a supervisor's
    /// restart policy handles the wait.
    pub(crate) fn for_device(paths: &DevicePaths, device: &str) -> anyhow::Result<Self> {
        match Self::try_acquire(&paths.lock)? {
            Ok(lock) => Ok(lock),
            Err(holder) => Err(StateError::PhoneBusy {
                device: device.to_owned(),
                holder,
                lock: paths.lock.clone(),
            }
            .into()),
        }
    }
}

#[cfg(test)]
impl StateLock {
    /// `take` of a lock this process just released, retried for up to a second.
    /// A child that another test thread is spawning holds a copy of every open
    /// file, and with it the `flock`, until its exec closes them.
    pub(crate) fn retake(mut take: impl FnMut() -> anyhow::Result<Self>) -> anyhow::Result<Self> {
        (0..100).fold(take(), |taken, _| {
            taken.or_else(|_| {
                std::thread::sleep(std::time::Duration::from_millis(10));
                take()
            })
        })
    }
}

/// Make an arbitrary device identifier safe as one path segment: `[A-Za-z0-9._-]`
/// kept, a leading `.` escaped so `..` cannot walk out of the tree.
fn sanitize_path_segment(raw: &str) -> String {
    let mut out: String = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if out.starts_with('.') {
        out.insert(0, '_');
    }
    if out.is_empty() {
        out.push('_');
    }
    out
}

/// `$XDG_STATE_HOME`, else `$HOME/.local/state`. Exported-but-empty counts as
/// unset: it would root the state at the cwd, `/` under launchd.
fn state_home() -> anyhow::Result<PathBuf> {
    if let Some(raw) = std::env::var_os(STATE_HOME_ENV).filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(raw));
    }
    let home = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .ok_or(StateError::NoStateHome)?;
    Ok(PathBuf::from(home).join(".local").join("state"))
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[test]
    fn a_payload_yields_the_fields_devicectl_cannot_report() -> anyhow::Result<()> {
        let payload = serde_json::json!({
            "device_name": "Lab 15 pro",
            "device_os_version": "26.5.1",
            "device_chip_model": "Apple A17 Pro",
            "device_ram_bytes": 8_589_934_592u64,
            "score": 41.2,
        });
        let facts = DeviceFacts::from_payload(&payload).context("the payload carries device_*")?;
        assert_eq!(facts.chip_model.as_deref(), Some("Apple A17 Pro"));
        assert_eq!(facts.ram_bytes, Some(8_589_934_592));
        assert_eq!(facts.name.as_deref(), Some("Lab 15 pro"));
        Ok(())
    }

    /// No fields is "nothing to learn"; blank strings and zero RAM are absent.
    #[rstest]
    #[case::no_device_fields(json!({"score": 41.2}), None)]
    #[case::blank_fields(json!({"device_chip_model": "   ", "device_name": ""}), None)]
    #[case::zero_ram(
        json!({"device_ram_bytes": 0, "device_chip_model": "Apple A17"}),
        Some(DeviceFacts { chip_model: Some("Apple A17".to_string()), ..DeviceFacts::default() })
    )]
    fn a_payload_teaches_only_what_it_carries(
        #[case] payload: serde_json::Value,
        #[case] expected: Option<DeviceFacts>,
    ) {
        assert_eq!(DeviceFacts::from_payload(&payload), expected);
    }

    #[test]
    fn the_facts_round_trip_through_their_file() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let path = dir.path().join("nested").join("device.json");
        assert!(DeviceFacts::load(&path)?.is_none());

        let facts = DeviceFacts {
            name: Some("Lab 15 pro".to_string()),
            chip_model: Some("Apple A17 Pro".to_string()),
            ram_bytes: Some(8_589_934_592),
        };
        facts.store(&path)?;
        assert_eq!(DeviceFacts::load(&path)?, Some(facts));
        Ok(())
    }

    /// A file an earlier proxy wrote, with the iOS version it kept, still loads.
    #[test]
    fn facts_with_an_ios_version_still_load() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let path = dir.path().join("device.json");
        fs::write(
            &path,
            r#"{"schema": 1, "os_version": "26.5.1", "ram_bytes": 8589934592}"#,
        )?;
        assert_eq!(
            DeviceFacts::load(&path)?,
            Some(DeviceFacts {
                ram_bytes: Some(8_589_934_592),
                ..DeviceFacts::default()
            })
        );
        Ok(())
    }

    #[test]
    fn a_write_replaces_the_file_whole_and_leaves_no_temp_file() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let path = dir.path().join("nested").join("state.json");
        write_atomically(&path, "first")?;
        write_atomically(&path, "second")?;
        assert_eq!(fs::read_to_string(&path)?, "second");

        let entries: Vec<String> = fs::read_dir(dir.path().join("nested"))?
            .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect::<Result<_, _>>()?;
        assert_eq!(entries, vec!["state.json"]);
        Ok(())
    }

    /// `flock` is per open-file-description, so a second `acquire` in this process
    /// behaves exactly as a second process would.
    #[test]
    fn a_second_driver_of_one_phone_is_refused_naming_the_holder() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let paths = StatePaths::resolve(Some(dir.path()))?.device("PHONE-A");

        let held = StateLock::for_device(&paths, "PHONE-A")?;
        let error = StateLock::for_device(&paths, "PHONE-A")
            .err()
            .context("a second holder must be refused")?;
        let message = format!("{error:#}");
        assert!(
            message.contains(&format!("pid {}", std::process::id())),
            "the message must name the holder: {message}"
        );
        assert!(message.contains("PHONE-A"), "{message}");

        drop(held);
        StateLock::retake(|| StateLock::for_device(&paths, "PHONE-A"))
            .context("the lock must be free once dropped")?;
        Ok(())
    }

    #[test]
    fn two_phones_lock_independently_and_create_their_dirs() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let paths = StatePaths::resolve(Some(&dir.path().join("root")))?;
        let _a = StateLock::for_device(&paths.device("PHONE-A"), "PHONE-A")?;
        let _b = StateLock::for_device(&paths.device("PHONE-B"), "PHONE-B")?;
        assert!(paths.device("PHONE-A").lock.is_file());
        assert!(paths.device("PHONE-B").lock.is_file());
        Ok(())
    }

    #[test]
    fn a_busy_lock_reports_its_holder_without_waiting() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let path = dir.path().join(POOL_LOCK_FILE_NAME);
        let held = StateLock::try_acquire(&path)?
            .map_err(|holder| anyhow::anyhow!("free lock reported busy by {holder}"))?;
        let Err(holder) = StateLock::try_acquire(&path)? else {
            anyhow::bail!("a held lock must not be taken twice");
        };
        assert_eq!(holder, Holder(Some(std::process::id().to_string())));
        drop(held);
        StateLock::retake(|| {
            StateLock::try_acquire(&path)?
                .map_err(|holder| anyhow::anyhow!("still held by {holder}"))
        })?;
        Ok(())
    }

    #[test]
    fn a_document_from_a_newer_binary_is_refused_by_name() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let path = dir.path().join("device.json");
        fs::write(
            &path,
            format!(
                r#"{{"schema": {}, "chip_model": "Apple A17 Pro"}}"#,
                SCHEMA + 1
            ),
        )
        .context("writing a future document")?;
        let error = DeviceFacts::load(&path)
            .err()
            .context("a newer schema must be refused")?;
        let message = format!("{error:#}");
        assert!(
            message.contains("newer") && message.contains("Upgrade"),
            "{message}"
        );
        Ok(())
    }

    #[test]
    fn a_document_without_a_schema_still_loads() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let path = dir.path().join("installed.json");
        fs::write(&path, r#"{"path": "/builds/A.app"}"#).context("writing an old document")?;
        let loaded = InstalledBuild::load(&path)?;
        assert_eq!(loaded.path, Some(PathBuf::from("/builds/A.app")));
        Ok(())
    }

    #[rstest]
    #[case::pool_mode(
        InstalledBuild { path: Some(PathBuf::from("/builds/A.app")), version: None },
        json!({ "schema": SCHEMA, "path": "/builds/A.app" })
    )]
    #[case::version_mode(
        InstalledBuild { path: None, version: Some("2026.10.02-e6a9434-internal".to_owned()) },
        json!({ "schema": SCHEMA, "version": "2026.10.02-e6a9434-internal" })
    )]
    fn a_stored_record_carries_the_current_schema(
        #[case] record: InstalledBuild,
        #[case] expected_json: serde_json::Value,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let path = dir.path().join("installed.json");
        record.store(&path)?;
        let raw: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).context("re-reading")?)
                .context("parsing")?;
        assert_eq!(raw, expected_json);
        assert_eq!(InstalledBuild::load(&path)?, record);
        Ok(())
    }

    #[test]
    fn a_state_dir_override_is_the_whole_root() -> anyhow::Result<()> {
        let paths = StatePaths::resolve(Some(Path::new("/tmp/one-off")))?;
        assert_eq!(paths.root, PathBuf::from("/tmp/one-off"));
        assert_eq!(paths.build_index, PathBuf::from("/tmp/one-off/builds.json"));
        Ok(())
    }

    #[test]
    fn the_index_is_shared_and_the_rest_is_per_device() -> anyhow::Result<()> {
        let paths = StatePaths::resolve(Some(Path::new("/tmp/fleet")))?;
        let a = paths.device("00008130-0011223344556677");
        let b = paths.device("00008140-0011223344556677");

        assert_eq!(paths.build_index, PathBuf::from("/tmp/fleet/builds.json"));
        assert_ne!(a.dir, b.dir);
        assert_ne!(a.installed, b.installed);
        assert_ne!(a.device_facts, b.device_facts);
        assert!(a.installed.starts_with(&a.dir));
        Ok(())
    }

    #[test]
    fn pulled_builds_sit_beside_the_build_index() -> anyhow::Result<()> {
        let paths = StatePaths::resolve(Some(Path::new("/tmp/fleet")))?;
        let releases = paths.releases();
        assert_eq!(releases.dir, PathBuf::from("/tmp/fleet/releases"));
        assert_eq!(releases.records, PathBuf::from("/tmp/fleet/releases.json"));
        assert_eq!(releases.lock, PathBuf::from("/tmp/fleet/releases.lock"));
        assert_eq!(releases.signed, PathBuf::from("/tmp/fleet/signed"));
        assert_eq!(
            releases.signed_lock,
            PathBuf::from("/tmp/fleet/signed.lock")
        );
        assert_eq!(
            paths.device("PHONE-A").staged,
            PathBuf::from("/tmp/fleet/PHONE-A/staged")
        );
        Ok(())
    }

    #[rstest]
    #[case::budget(false, "gave up after waiting")]
    #[case::stopped(true, "stopped while waiting")]
    fn a_bounded_lock_wait_gives_up(
        #[case] stop: bool,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        use std::sync::atomic::AtomicBool;

        let dir = tempfile::tempdir()?;
        let path = dir.path().join(SIGNED_LOCK_FILE_NAME);
        let held = StateLock::try_acquire(&path)?
            .map_err(|holder| anyhow::anyhow!("a free lock reported busy by {holder}"))?;
        let never = AtomicBool::new(false);
        let stopped = AtomicBool::new(stop);
        let signals = Signals {
            abort: &never,
            stop: &stopped,
        };
        let budget = if stop {
            Duration::from_secs(60)
        } else {
            Duration::from_millis(50)
        };

        let error = StateLock::wait_for(&path, budget, signals)
            .err()
            .context("a held lock must not be taken")?;
        assert!(format!("{error:#}").contains(needle), "{error:#}");

        drop(held);
        let free = Signals {
            abort: &never,
            stop: &never,
        };
        StateLock::retake(|| StateLock::wait_for(&path, Duration::from_millis(50), free))?;
        Ok(())
    }

    #[test]
    fn the_default_root_is_absolute_and_device_dirs_are_sanitised() -> anyhow::Result<()> {
        let paths = StatePaths::resolve(None)?;
        assert!(
            paths.root.ends_with(STATE_DIR_NAME),
            "{} should end with {STATE_DIR_NAME}",
            paths.root.display()
        );
        assert!(
            paths.root.is_absolute(),
            "a launchd job's cwd is /, so the state root cannot be relative"
        );
        assert!(
            paths
                .device("fleet/iphone 15 pro")
                .dir
                .ends_with("fleet_iphone_15_pro"),
            "a device name with a slash must not escape its directory"
        );
        Ok(())
    }

    #[test]
    fn a_device_name_cannot_walk_out_of_the_state_tree() -> anyhow::Result<()> {
        assert_eq!(
            sanitize_path_segment("00008130-0011223344556677"),
            "00008130-0011223344556677"
        );
        assert_eq!(sanitize_path_segment("fleet/iphone"), "fleet_iphone");
        assert_eq!(sanitize_path_segment(".."), "_..");
        assert_eq!(sanitize_path_segment("."), "_.");
        assert_eq!(sanitize_path_segment(""), "_");
        assert_eq!(sanitize_path_segment("iPhone (15 Pro)"), "iPhone__15_Pro_");
        Ok(())
    }
}
