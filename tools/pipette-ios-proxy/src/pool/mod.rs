//! The build pool: which app builds this Mac can install, and which runtimes each
//! provides. Nothing is declared: a build's runtimes are read off the build with
//! `headlessrun runtimes list`, in the same encoding a claim's `spec.runtime`
//! carries. That reading gives both the capability flags to advertise
//! ([`self::capabilities::flags_for`]) and the build to install for a job (typed
//! [`Runtime`] equality). Reading a build means INSTALLING it, hence the
//! persisted index.

pub(crate) mod bundle;
mod capabilities;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_plan_types::{Runtime, RuntimeBuildFlagError};

use crate::app;
use crate::config::BuildsConfig;
use crate::device::DeviceControl;
use crate::state;

const APP_EXTENSION: &str = "app";
const IPA_EXTENSION: &str = "ipa";

/// What a build looked like when it was last read: a rebuilt bundle keeps its
/// path, so length and mtime of its marker decide whether a reading still applies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct BuildStamp {
    len: u64,
    /// Whole seconds since the epoch: finer would re-read every build after an
    /// `rsync` that preserved only seconds.
    modified_secs: i64,
}

impl BuildStamp {
    /// The marker whose metadata stands for the build: `Info.plist` for an `.app`
    /// (a directory's mtime misses a replaced nested binary), the file for an `.ipa`.
    fn marker(build: &Path) -> PathBuf {
        if build.extension().and_then(|ext| ext.to_str()) == Some(APP_EXTENSION) {
            build.join("Info.plist")
        } else {
            build.to_path_buf()
        }
    }

    fn read(build: &Path) -> anyhow::Result<Self> {
        let marker = Self::marker(build);
        let metadata = std::fs::metadata(&marker)
            .with_context(|| format!("reading {} to stamp the build", marker.display()))?;
        let modified_secs = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|since| i64::try_from(since.as_secs()).unwrap_or(i64::MAX))
            .unwrap_or_default();
        Ok(Self {
            len: metadata.len(),
            modified_secs,
        })
    }

    #[cfg(test)]
    fn at(len: u64, modified: std::time::SystemTime) -> Self {
        let secs = modified
            .duration_since(UNIX_EPOCH)
            .map(|since| i64::try_from(since.as_secs()).unwrap_or(i64::MAX))
            .unwrap_or_default();
        Self {
            len,
            modified_secs: secs,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct KnownBuild {
    pub(crate) path: PathBuf,
    stamp: BuildStamp,
    /// The runtimes this build provides. Empty is distinct from never read.
    pub(crate) runtimes: Vec<Runtime>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BuildReading {
    pub(crate) runtimes: Vec<Runtime>,
}

/// Every build the proxy has read, persisted as `builds.json`. Keyed by path
/// string so the file is a plain JSON object; ordered so a rewrite does not
/// reshuffle it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct BuildIndex {
    #[serde(default)]
    entries: BTreeMap<String, KnownBuild>,
}

impl BuildIndex {
    /// Load the index; absent is empty, corrupt is an error (an empty index would
    /// be an install storm).
    fn load(path: &Path) -> anyhow::Result<Self> {
        Ok(state::load_json(path, "the build index")?.unwrap_or_default())
    }

    fn store(&self, path: &Path) -> anyhow::Result<()> {
        state::store_json(path, "the build index", self)
    }

    fn current(&self, build: &Path, stamp: &BuildStamp) -> Option<&KnownBuild> {
        self.entries
            .get(&key(build))
            .filter(|known| known.stamp == *stamp)
    }

    fn record(&mut self, build: &Path, stamp: BuildStamp, reading: BuildReading) {
        self.entries.insert(
            key(build),
            KnownBuild {
                path: build.to_path_buf(),
                stamp,
                runtimes: reading.runtimes,
            },
        );
    }

    /// Forget every build not in `present`, so a pruned pool stops advertising.
    /// Returns whether anything was forgotten, so an unchanged pool skips the write
    /// (a read-only state directory must still start).
    fn retain_present(&mut self, present: &[PathBuf]) -> bool {
        let keys: Vec<String> = present.iter().map(|path| key(path)).collect();
        let before = self.entries.len();
        self.entries
            .retain(|path, _| keys.iter().any(|k| k == path));
        self.entries.len() != before
    }

    /// The build that provides `runtime`, by typed equality (the server may send
    /// keys in any order). Ordered by path so two providers resolve stably.
    fn build_for(&self, runtime: &Runtime) -> Option<&KnownBuild> {
        self.entries
            .values()
            .find(|known| known.runtimes.contains(runtime))
    }

    /// The capability flags for everything in the pool, sorted and de-duplicated:
    /// the server voids queue standing whenever the reported set CHANGES.
    fn capabilities(&self) -> Result<Vec<String>, RuntimeBuildFlagError> {
        let mut flags = self
            .entries
            .values()
            .flat_map(|known| known.runtimes.iter())
            .map(capabilities::flags_for)
            .try_fold(Vec::new(), |mut flags, result| {
                flags.extend(result?);
                Ok::<_, RuntimeBuildFlagError>(flags)
            })?;
        flags.sort();
        flags.dedup();
        Ok(flags)
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn known(&self) -> impl Iterator<Item = &KnownBuild> {
        self.entries.values()
    }
}

fn key(build: &Path) -> String {
    build.to_string_lossy().into_owned()
}

/// Every `.app`/`.ipa` in `dir` or one directory below it (the farm keeps
/// `<commit>/Pipette.app`), sorted by path; other entries are ignored.
pub(crate) fn discover(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let is_bundle = |path: &Path| {
        matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some(APP_EXTENSION) | Some(IPA_EXTENSION)
        )
    };
    let mut builds = entries_of(dir)?
        .into_iter()
        .map(|entry| {
            if is_bundle(&entry) {
                Ok(vec![entry])
            } else if entry.is_dir() {
                Ok(entries_of(&entry)?
                    .into_iter()
                    .filter(|nested| is_bundle(nested))
                    .collect())
            } else {
                Ok(Vec::new())
            }
        })
        .collect::<anyhow::Result<Vec<Vec<PathBuf>>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    builds.sort();
    Ok(builds)
}

fn entries_of(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    std::fs::read_dir(dir)
        .with_context(|| format!("reading the build directory {}", dir.display()))?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .with_context(|| format!("reading an entry of {}", dir.display()))
        })
        .collect()
}

/// Install `build` and ask it what it provides.
pub(crate) fn read_build(
    device: &dyn DeviceControl,
    bundle_id: &str,
    build: &Path,
    timeout: Duration,
    first_line: Duration,
) -> anyhow::Result<BuildReading> {
    device
        .install_app(build)
        .with_context(|| format!("installing {} to read it", build.display()))?;

    let listing = app::list_runtimes(device, bundle_id, timeout, first_line)
        .with_context(|| format!("asking {} which runtimes it provides", build.display()))?;
    listing.unreadable.iter().for_each(|raw| {
        tracing::warn!(
            build = %build.display(),
            descriptor = %raw,
            "the build reports a runtime this proxy cannot read"
        );
    });

    Ok(BuildReading {
        runtimes: listing.runtimes,
    })
}

/// The pool: a directory of builds and everything learned about them, persisting
/// itself on every change. Reading a build (the one phone-touching step) is
/// handed in as a closure so the refresh logic is testable without a device.
#[derive(Debug)]
pub(crate) struct Pool {
    builds: BuildsConfig,
    index_path: PathBuf,
    index: BuildIndex,
}

/// No build in the pool provides a job's runtime; terminal for that job.
#[derive(Debug, thiserror::Error)]
#[error(
    "no build in {dir} provides the runtime this job names ({runtime}); the pool advertises \
     {advertised:?}"
)]
pub(crate) struct NoBuildForError {
    pub(crate) dir: PathBuf,
    pub(crate) runtime: String,
    pub(crate) advertised: Vec<String>,
}

/// Why a refreshed pool cannot be advertised: each is an operator's mistake with
/// a specific fix, and a healthy-looking proxy that never gets work is the symptom.
#[derive(Debug, thiserror::Error)]
pub(crate) enum UnadvertisablePoolError {
    #[error(
        "the build pool {dir} named by [builds] dir in {config} holds no .app or .ipa: there is \
         nothing to install, so this proxy could advertise nothing and would register a client \
         the server can never send a job to"
    )]
    Empty { dir: PathBuf, config: PathBuf },
    #[error(
        "no build in {dir} could be read, so this client would advertise nothing and the server \
         could never send it a job"
    )]
    NothingRead { dir: PathBuf },
}

impl Pool {
    /// Open the pool, loading the index and nothing else (`probe` opens it
    /// too and must not install). [`Self::refresh`] reads builds.
    pub(crate) fn open(builds: &BuildsConfig, index_path: &Path) -> anyhow::Result<Self> {
        Ok(Self {
            builds: builds.clone(),
            index_path: index_path.to_path_buf(),
            index: BuildIndex::load(index_path)?,
        })
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.builds.dir
    }

    /// `builds.lock`, beside the index: held while one process refreshes the pool.
    pub(crate) fn lock_path(&self) -> PathBuf {
        self.index_path.with_file_name(state::POOL_LOCK_FILE_NAME)
    }

    /// Re-read the index, to pick up what another process learned while this one
    /// waited for [`Self::lock_path`].
    pub(crate) fn reload(&mut self) -> anyhow::Result<()> {
        self.index = BuildIndex::load(&self.index_path)?;
        Ok(())
    }

    /// Learn anything not yet known: `read` is called once per new or changed
    /// build and INSTALLS it; the index is written after every change. Returns
    /// the build the last read left on the phone, which the caller records
    /// (installed state is per device). Advertisability is
    /// [`Self::check_advertisable`].
    pub(crate) fn refresh(
        &mut self,
        config_source: &Path,
        mut read: impl FnMut(&Path) -> anyhow::Result<BuildReading>,
    ) -> anyhow::Result<Option<PathBuf>> {
        let present = discover(&self.builds.dir)?;
        if present.is_empty() {
            return Err(UnadvertisablePoolError::Empty {
                dir: self.builds.dir.clone(),
                config: config_source.to_path_buf(),
            }
            .into());
        }
        // Before reading anything, so a deleted build stops advertising regardless.
        if self.index.retain_present(&present) {
            self.persist()?;
        }

        present.iter().try_fold(None, |last_read, build| {
            let stamp = BuildStamp::read(build)?;
            if let Some(known) = self.index.current(build, &stamp) {
                tracing::debug!(
                    build = %build.display(),
                    runtimes = %known.runtimes.len(),
                    "build already read"
                );
                return Ok(last_read);
            }
            // The closure logs the read and forgets the phone's installed state
            // itself: from the moment the install starts the old answer is wrong.
            let reading = read(build)?;
            reading.runtimes.iter().for_each(|runtime| {
                tracing::info!(
                    build = %build.display(),
                    runtime = %runtime.headless_token(),
                    "build provides a runtime"
                );
            });
            self.index.record(build, stamp, reading);
            self.persist()?;
            Ok(Some(build.clone()))
        })
    }

    /// Refuse a pool the server could never send a job to. Called before a
    /// registration, not in [`Self::refresh`], so `scan` can still list it.
    pub(crate) fn check_advertisable(&self) -> anyhow::Result<()> {
        if self.index.is_empty() {
            return Err(UnadvertisablePoolError::NothingRead {
                dir: self.builds.dir.clone(),
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn build_for(&self, runtime: &Runtime) -> Option<&KnownBuild> {
        self.index.build_for(runtime)
    }

    /// The runtime flags of every build in the pool, with one build flag per distinct
    /// runtime, sorted and unique. No `client_version:`: a claim carries no capability
    /// requirements, so a pool offering several versions could answer a pin with the
    /// wrong build.
    pub(crate) fn capabilities(&self) -> Result<Vec<String>, RuntimeBuildFlagError> {
        self.index.capabilities()
    }

    pub(crate) fn known(&self) -> impl Iterator<Item = &KnownBuild> {
        self.index.known()
    }

    /// Put a build into the pool as if it had been read.
    #[cfg(test)]
    pub(crate) fn seed(&mut self, build: &Path, runtimes: Vec<Runtime>) {
        self.index.record(
            build,
            BuildStamp {
                len: 0,
                modified_secs: 0,
            },
            BuildReading { runtimes },
        );
    }

    fn persist(&self) -> anyhow::Result<()> {
        self.index.store(&self.index_path)
    }
}

/// The build's path relative to the pool (`<commit>/Pipette.app`), so two bundles
/// with one file name still read apart.
pub(crate) fn display_name(pool_dir: &Path, build: &Path) -> String {
    build
        .strip_prefix(pool_dir)
        .unwrap_or(build)
        .display()
        .to_string()
}

#[cfg(test)]
mod tests {
    use std::time::Duration as StdDuration;

    use anyhow::Context;

    use pipette_plan_types::{
        AppleFoundation, LlamacppIosPipette, LlamacppIosPipetteFlavor, NonEmptyString, PrivateApis,
        RepositoryUrl, SourceRepository, runtime_build_flag,
    };

    use super::*;

    fn llamacpp(version: &str) -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppIosPipette(LlamacppIosPipette {
            source: SourceRepository {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new(version)?,
            },
            flavor: LlamacppIosPipetteFlavor::IosArm64,
            private_apis: PrivateApis::default(),
        }))
    }

    fn apple_foundation() -> Runtime {
        Runtime::AppleFoundation(AppleFoundation {
            private_apis: PrivateApis::default(),
        })
    }

    fn build_flag(runtime: &Runtime) -> anyhow::Result<String> {
        Ok(runtime_build_flag(runtime)?
            .context("an iOS app runtime has a build flag")?
            .into_inner())
    }

    fn llamacpp_flags(versions: &[&str]) -> anyhow::Result<Vec<String>> {
        let mut flags = vec!["runtime:llama_cpp".to_string()];
        for version in versions {
            flags.push(format!("runtime:llama_cpp:{version}"));
            flags.push(build_flag(&llamacpp(version)?)?);
        }
        flags.sort();
        Ok(flags)
    }

    fn reading(runtimes: Vec<Runtime>) -> BuildReading {
        BuildReading { runtimes }
    }

    fn stamp(len: u64) -> BuildStamp {
        BuildStamp::at(
            len,
            std::time::SystemTime::UNIX_EPOCH + StdDuration::from_secs(1_700_000_000),
        )
    }

    #[test]
    fn a_build_matches_a_runtime_whatever_key_order_it_arrives_in() -> anyhow::Result<()> {
        let mut index = BuildIndex::default();
        index.record(
            Path::new("/builds/A.app"),
            stamp(1),
            reading(vec![llamacpp("b10216")?]),
        );

        // `source` is flattened; keys deliberately out of encoder order.
        let reordered: Runtime = serde_json::from_str(
            r#"{"flavor":"ios-arm64","repository_version":"b10216",
                "repository_url":"github.com/ggml-org/llama.cpp","type":"llamacpp_ios_pipette"}"#,
        )
        .context("a descriptor with the keys in another order")?;

        let found = index
            .build_for(&reordered)
            .context("the pool provides this runtime")?;
        assert_eq!(found.path, PathBuf::from("/builds/A.app"));
        Ok(())
    }

    #[test]
    fn a_runtime_no_build_provides_matches_nothing() -> anyhow::Result<()> {
        let mut index = BuildIndex::default();
        index.record(
            Path::new("/builds/A.app"),
            stamp(1),
            reading(vec![llamacpp("b1")?]),
        );
        assert!(index.build_for(&llamacpp("b2")?).is_none());
        Ok(())
    }

    #[test]
    fn capabilities_are_the_sorted_unique_union_over_the_pool() -> anyhow::Result<()> {
        let mut index = BuildIndex::default();
        index.record(
            Path::new("/builds/B.app"),
            stamp(1),
            reading(vec![llamacpp("b10216")?, apple_foundation()]),
        );
        index.record(
            Path::new("/builds/A.app"),
            stamp(1),
            reading(vec![
                llamacpp("b9050")?,
                llamacpp("b10216")?,
                apple_foundation(),
            ]),
        );

        let flags = index.capabilities()?;
        let mut expected = llamacpp_flags(&["b10216", "b9050"])?;
        expected.push("runtime:apple_foundation".to_string());
        expected.sort();
        assert_eq!(flags, expected);

        assert_eq!(flags, index.capabilities()?);
        Ok(())
    }

    #[test]
    fn a_rebuilt_bundle_invalidates_its_reading() -> anyhow::Result<()> {
        let build = Path::new("/builds/A.app");
        let mut index = BuildIndex::default();
        index.record(build, stamp(1), reading(vec![llamacpp("b10216")?]));

        assert!(index.current(build, &stamp(1)).is_some());
        assert!(index.current(build, &stamp(2)).is_none());
        Ok(())
    }

    #[test]
    fn pruning_the_pool_drops_the_capabilities_it_carried() -> anyhow::Result<()> {
        let mut index = BuildIndex::default();
        index.record(
            Path::new("/builds/A.app"),
            stamp(1),
            reading(vec![llamacpp("b1")?]),
        );
        index.record(
            Path::new("/builds/B.app"),
            stamp(1),
            reading(vec![llamacpp("b2")?]),
        );
        assert_eq!(index.capabilities()?, llamacpp_flags(&["b1", "b2"])?);

        index.retain_present(&[PathBuf::from("/builds/A.app")]);
        assert_eq!(index.capabilities()?, llamacpp_flags(&["b1"])?);
        assert!(index.build_for(&llamacpp("b2")?).is_none());
        Ok(())
    }

    #[test]
    fn an_unread_pool_is_empty() -> anyhow::Result<()> {
        let index = BuildIndex::default();
        assert!(index.known().next().is_none());
        assert!(index.capabilities()?.is_empty());
        Ok(())
    }

    #[test]
    fn the_index_round_trips_through_its_file() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let path = dir.path().join("nested").join("builds.json");

        let mut index = BuildIndex::default();
        index.record(
            Path::new("/builds/A.app"),
            stamp(7),
            reading(vec![llamacpp("b10216")?, apple_foundation()]),
        );
        index.store(&path)?;

        let loaded = BuildIndex::load(&path)?;
        assert_eq!(loaded, index);
        assert_eq!(loaded.capabilities()?, index.capabilities()?);
        Ok(())
    }

    #[test]
    fn an_absent_index_loads_empty() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let loaded = BuildIndex::load(&dir.path().join("builds.json"))?;
        assert!(loaded.known().next().is_none());
        Ok(())
    }

    fn pool_config(dir: &Path) -> BuildsConfig {
        BuildsConfig {
            dir: dir.to_path_buf(),
        }
    }

    fn make_app(dir: &Path, name: &str, plist: &[u8]) -> anyhow::Result<PathBuf> {
        let app = dir.join(name);
        std::fs::create_dir_all(&app).context("creating the .app")?;
        std::fs::write(app.join("Info.plist"), plist).context("writing Info.plist")?;
        Ok(app)
    }

    /// Answers from `answers` by bundle name and remembers each path asked, in order.
    struct FakeReader {
        answers: BTreeMap<String, BuildReading>,
        asked: Vec<String>,
    }

    impl FakeReader {
        fn new(answers: impl IntoIterator<Item = (&'static str, BuildReading)>) -> Self {
            Self {
                answers: answers
                    .into_iter()
                    .map(|(name, reading)| (name.to_owned(), reading))
                    .collect(),
                asked: Vec::new(),
            }
        }

        fn read(&mut self, build: &Path) -> anyhow::Result<BuildReading> {
            let name = build
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            self.asked.push(name.clone());
            self.answers
                .get(&name)
                .cloned()
                .with_context(|| format!("the fake has no answer for {name}"))
        }

        fn asked(&self) -> &[String] {
            &self.asked
        }
    }

    #[test]
    fn a_refreshed_pool_is_not_read_again_by_the_next_process() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let state = tempfile::tempdir().context("a state dir")?;
        make_app(dir.path(), "A.app", b"v1")?;
        let index_path = state.path().join("builds.json");
        let config = pool_config(dir.path());

        let mut first = FakeReader::new([("A.app", reading(vec![llamacpp("b10216")?]))]);
        let mut pool = Pool::open(&config, &index_path)?;
        pool.refresh(Path::new("cfg.toml"), |b| first.read(b))?;
        assert_eq!(first.asked(), ["A.app"]);

        let mut second = FakeReader::new([]);
        let mut pool = Pool::open(&config, &index_path)?;
        pool.refresh(Path::new("cfg.toml"), |b| second.read(b))?;
        assert!(second.asked().is_empty(), "nothing should be re-read");
        assert_eq!(pool.capabilities()?, llamacpp_flags(&["b10216"])?);
        Ok(())
    }

    /// An unchanged pool is not written back: a read-only state directory must
    /// still start.
    #[test]
    fn an_unchanged_pool_refreshes_without_writing() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().context("a temp dir")?;
        let state = tempfile::tempdir().context("a state dir")?;
        make_app(dir.path(), "A.app", b"v1")?;
        let index_path = state.path().join("builds.json");
        let config = pool_config(dir.path());

        let mut first = FakeReader::new([("A.app", reading(vec![llamacpp("b1")?]))]);
        let mut pool = Pool::open(&config, &index_path)?;
        pool.refresh(Path::new("cfg.toml"), |b| first.read(b))?;
        let stored = std::fs::read_to_string(&index_path).context("the stored index")?;

        // Restored before the temp dirs are dropped so they can be removed.
        let writable = std::fs::metadata(state.path())
            .context("state dir metadata")?
            .permissions();
        std::fs::set_permissions(&index_path, std::fs::Permissions::from_mode(0o444))
            .context("making the index read-only")?;
        std::fs::set_permissions(state.path(), std::fs::Permissions::from_mode(0o555))
            .context("making the state dir read-only")?;
        let mut none = FakeReader::new([]);
        let mut pool = Pool::open(&config, &index_path)?;
        let refreshed = pool.refresh(Path::new("cfg.toml"), |b| none.read(b));
        std::fs::set_permissions(state.path(), writable).context("restoring the state dir")?;
        std::fs::set_permissions(&index_path, std::fs::Permissions::from_mode(0o644))
            .context("restoring the index")?;

        refreshed.context("an unchanged pool must not need to write")?;
        assert!(none.asked().is_empty(), "nothing should be re-read");
        assert_eq!(
            std::fs::read_to_string(&index_path).context("the index after")?,
            stored,
            "the index must be byte-identical"
        );
        Ok(())
    }

    #[test]
    fn only_a_changed_build_is_read_again() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let state = tempfile::tempdir().context("a state dir")?;
        make_app(dir.path(), "A.app", b"v1")?;
        make_app(dir.path(), "B.app", b"v1")?;
        let index_path = state.path().join("builds.json");
        let config = pool_config(dir.path());

        let mut pool = Pool::open(&config, &index_path)?;
        let mut first = FakeReader::new([
            ("A.app", reading(vec![llamacpp("b1")?])),
            ("B.app", reading(vec![llamacpp("b2")?])),
        ]);
        pool.refresh(Path::new("cfg.toml"), |b| first.read(b))?;

        // A longer plist changes the stamp even within the same mtime second.
        make_app(dir.path(), "B.app", b"v2-rebuilt")?;
        let mut second = FakeReader::new([("B.app", reading(vec![llamacpp("b3")?]))]);
        pool.refresh(Path::new("cfg.toml"), |b| second.read(b))?;

        assert_eq!(
            second.asked(),
            ["B.app"],
            "A is unchanged and must not be re-read"
        );
        let flags = pool.capabilities()?;
        assert!(flags.contains(&"runtime:llama_cpp:b3".to_string()));
        assert!(
            !flags.contains(&"runtime:llama_cpp:b2".to_string()),
            "the old reading of B must be gone: {flags:?}"
        );
        Ok(())
    }

    #[test]
    fn the_last_build_read_is_the_one_on_the_phone() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let state = tempfile::tempdir().context("a state dir")?;
        make_app(dir.path(), "A.app", b"v1")?;
        let b = make_app(dir.path(), "B.app", b"v1")?;
        let config = pool_config(dir.path());

        let mut pool = Pool::open(&config, &state.path().join("builds.json"))?;
        let mut reader = FakeReader::new([
            ("A.app", reading(vec![llamacpp("b1")?])),
            ("B.app", reading(vec![llamacpp("b2")?])),
        ]);
        let last = pool.refresh(Path::new("cfg.toml"), |p| reader.read(p))?;

        assert_eq!(last, Some(b));
        Ok(())
    }

    #[test]
    fn a_failed_read_leaves_nothing_stale_behind() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let state = tempfile::tempdir().context("a state dir")?;
        make_app(dir.path(), "A.app", b"v1")?;
        make_app(dir.path(), "B.app", b"v1")?;
        let index_path = state.path().join("builds.json");
        let config = pool_config(dir.path());

        let mut pool = Pool::open(&config, &index_path)?;
        let mut reader = FakeReader::new([("A.app", reading(vec![llamacpp("b1")?]))]);
        let error = pool
            .refresh(Path::new("cfg.toml"), |p| reader.read(p))
            .err()
            .context("B has no answer, so the refresh must fail")?;
        assert!(format!("{error:#}").contains("B.app"), "{error:#}");

        let pool = Pool::open(&config, &index_path)?;
        let names: Vec<String> = pool
            .known()
            .map(|k| display_name(pool.dir(), &k.path))
            .collect();
        assert_eq!(names, vec!["A.app"]);
        Ok(())
    }

    #[test]
    fn a_removed_build_stops_advertising_without_a_read() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let state = tempfile::tempdir().context("a state dir")?;
        make_app(dir.path(), "A.app", b"v1")?;
        let b = make_app(dir.path(), "B.app", b"v1")?;
        let index_path = state.path().join("builds.json");
        let config = pool_config(dir.path());

        let mut pool = Pool::open(&config, &index_path)?;
        let mut reader = FakeReader::new([
            ("A.app", reading(vec![llamacpp("b1")?])),
            ("B.app", reading(vec![llamacpp("b2")?])),
        ]);
        pool.refresh(Path::new("cfg.toml"), |p| reader.read(p))?;
        assert_eq!(pool.capabilities()?, llamacpp_flags(&["b1", "b2"])?);

        std::fs::remove_dir_all(&b).context("removing B")?;
        let mut none = FakeReader::new([]);
        pool.refresh(Path::new("cfg.toml"), |p| none.read(p))?;
        assert!(none.asked().is_empty());
        assert_eq!(pool.capabilities()?, llamacpp_flags(&["b1"])?);
        Ok(())
    }

    #[test]
    fn an_empty_directory_is_refused_naming_the_config() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        let state = tempfile::tempdir().context("a state dir")?;
        let mut pool = Pool::open(&pool_config(dir.path()), &state.path().join("b.json"))?;
        let error = pool
            .refresh(Path::new("/etc/pipette/ios-proxy.toml"), |_| {
                anyhow::bail!("the reader must not be called in this test")
            })
            .err()
            .context("an empty pool must be refused")?;
        let message = format!("{error:#}");
        assert!(message.contains("no .app or .ipa"), "{message}");
        assert!(
            message.contains("/etc/pipette/ios-proxy.toml"),
            "must name the config: {message}"
        );
        Ok(())
    }

    #[test]
    fn reading_a_build_installs_it_and_lists_its_runtimes() -> anyhow::Result<()> {
        use crate::app::ConsoleRun;
        use crate::device::fake::{Call, FakeDevice};

        let descriptor = r#"{"type":"llamacpp_ios_pipette","repository_url":"github.com/ggml-org/llama.cpp","repository_version":"b10216","flavor":"ios-arm64"}"#;
        let ok = |stdout: &str| ConsoleRun {
            stdout: stdout.to_owned(),
            stderr: String::new(),
            sentinel: Some(0),
            exit_code: Some(0),
            timed_out: false,
            engine_time: None,
            aborted: false,
            stopped: false,
            silent: false,
        };
        let device = FakeDevice::new("phone-a").then_console(ok(&format!(
            "[HEADLESS] runtimes count=1\n[HEADLESS] runtime type=llamacpp_ios_pipette \
                 digest=0a1b2c3d descriptor={descriptor} runtime=b10216:ios-arm64\n\
                 [HEADLESS] BENCH_DONE 0 completed\n"
        )));
        let log = device.log();

        let build = Path::new("/builds/A.app");
        let reading = read_build(
            &device,
            "ai.liquid.liquid-pipette",
            build,
            Duration::from_secs(30),
            Duration::from_secs(60),
        )?;

        assert_eq!(reading.runtimes, vec![llamacpp("b10216")?]);

        let calls = log.calls()?;
        assert_eq!(calls.len(), 2, "{calls:?}");
        assert_eq!(calls[0], Call::Install(build.to_path_buf()));
        let Call::Launch { app_args, .. } = &calls[1] else {
            anyhow::bail!("second call should be a launch: {calls:?}");
        };
        assert_eq!(app_args[1..3], ["runtimes", "list"], "{app_args:?}");
        Ok(())
    }

    #[test]
    fn discover_finds_installable_bundles_in_a_stable_order() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("a temp dir")?;
        std::fs::create_dir(dir.path().join("B.app")).context("B.app")?;
        std::fs::create_dir(dir.path().join("A.app")).context("A.app")?;
        std::fs::write(dir.path().join("C.ipa"), b"packaged").context("C.ipa")?;
        std::fs::write(dir.path().join("README.md"), b"not a build").context("README")?;
        // The farm's layout: one directory per commit, each holding `Pipette.app`.
        std::fs::create_dir_all(dir.path().join("3d91c46/Pipette.app")).context("nested")?;
        std::fs::create_dir_all(dir.path().join("2ff756e/Pipette.app")).context("nested")?;
        std::fs::create_dir_all(dir.path().join("notes/deeper/X.app")).context("too deep")?;

        let found = discover(dir.path())?;
        let names: Vec<String> = found
            .iter()
            .map(|path| display_name(dir.path(), path))
            .collect();
        assert_eq!(
            names,
            vec![
                "2ff756e/Pipette.app",
                "3d91c46/Pipette.app",
                "A.app",
                "B.app",
                "C.ipa"
            ]
        );
        Ok(())
    }
}
