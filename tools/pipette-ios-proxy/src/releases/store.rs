//! The builds `builds pull` and `run` keep in the state directory (DESIGN
//! sections 3 and 8): `releases/<version>/` holds an unpacked archive,
//! `releases.json` the digest each version was first pulled with, and
//! `releases.lock` is held while one process pulls.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256 as Sha256Hasher};

use pipette_http::HttpClient;
use pipette_plan_types::Sha256;

use super::{Build, BuildVersion, Fault, UnreadyError, s3};
use crate::app::Signals;
use crate::config::ReleasesBucket;
use crate::state::{self, Holder, ReleasePaths, StateLock};
use crate::timing;

/// What a pull or an export that was killed leaves in `releases/` or `signed/`.
/// No version can take the name: none starts with a dot.
pub(crate) const PARTIAL_PREFIX: &str = ".partial-";

const ARCHIVE_EXTENSION: &str = "xcarchive";

const WHAT: &str = "the pulled builds";

#[derive(Debug, thiserror::Error)]
enum PullError {
    #[error("{bucket} publishes no build {version}; `builds list` prints the ones it does")]
    NotPublished {
        version: BuildVersion,
        bucket: String,
    },
    #[error(
        "another pipette-ios-proxy ({holder}) is pulling a build (it holds {lock}); run this again \
         once it is done"
    )]
    Busy { holder: Holder, lock: PathBuf },
    #[error(
        "build {version} was first pulled with sha256 {recorded}, and {key} now publishes \
         {published}. A release must never change under a version, so this one is refused: \
         publish the new bytes under a new version"
    )]
    DigestChanged {
        version: BuildVersion,
        recorded: Sha256,
        published: Sha256,
        key: String,
    },
    #[error(
        "{key} downloaded with sha256 {downloaded}, and its .sha256 says {published}: nothing was \
         kept"
    )]
    DigestMismatch {
        key: String,
        downloaded: Sha256,
        published: Sha256,
    },
    #[error("`ditto -x -k` could not unpack {key}: {detail}")]
    Unpack { key: String, detail: String },
    #[error("{key} unpacked without an .{ARCHIVE_EXTENSION} in it")]
    NoArchive { key: String },
}

impl PullError {
    /// A failure every phone pulling this version meets again. Not a failed
    /// unpack: its bytes matched the sidecar, and a full disk fails it too.
    fn is_version_specific(&self) -> bool {
        match self {
            Self::NotPublished { .. }
            | Self::DigestChanged { .. }
            | Self::DigestMismatch { .. }
            | Self::NoArchive { .. } => true,
            Self::Busy { .. } | Self::Unpack { .. } => false,
        }
    }
}

/// Why [`only_child`] found no single entry; each caller decides whose fault
/// that is.
#[derive(Debug, thiserror::Error)]
pub(crate) enum OnlyChildError {
    #[error("reading {dir}")]
    Read {
        dir: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{dir} holds {found} where one {what} was expected")]
    NotOne {
        dir: PathBuf,
        what: &'static str,
        found: usize,
    },
}

#[derive(Debug, thiserror::Error)]
#[error("the download was stopped: the lease was lost or a stop arrived")]
struct DownloadStoppedError;

/// `releases.json`: the digest each version was first pulled with. A release
/// never changes under a version, so a version published with another digest is
/// refused from then on.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct Records {
    #[serde(default)]
    versions: BTreeMap<String, Record>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Record {
    sha256: Sha256,
    /// Where and when it was first read, for the operator.
    key: String,
    pulled_at: String,
}

/// What the store holds of a published build.
#[derive(Debug, Clone, PartialEq, Eq, strum::Display)]
#[strum(serialize_all = "kebab-case")]
pub(crate) enum Held {
    /// Unpacked under `releases/<version>/`, from the digest the bucket publishes.
    Pulled,
    NotPulled,
    /// The bucket publishes another digest than the one recorded.
    Changed {
        recorded: Sha256,
    },
}

impl Records {
    pub(crate) fn load(paths: &ReleasePaths) -> anyhow::Result<Self> {
        Ok(state::load_json(&paths.records, WHAT)?.unwrap_or_default())
    }

    pub(crate) fn held(&self, paths: &ReleasePaths, build: &Build) -> Held {
        match self.versions.get(build.version.as_str()) {
            Some(record) if record.sha256 != build.sha256 => Held::Changed {
                recorded: record.sha256.clone(),
            },
            Some(_) if version_dir(paths, &build.version).is_dir() => Held::Pulled,
            Some(_) | None => Held::NotPulled,
        }
    }
}

fn version_dir(paths: &ReleasePaths, version: &BuildVersion) -> PathBuf {
    paths.dir.join(version.as_str())
}

/// Make `releases/<version>/` hold the unpacked archive of `version`, and
/// return it. Downloads nothing when it already does.
pub(crate) fn pull(
    http: &HttpClient,
    releases: &ReleasesBucket,
    paths: &ReleasePaths,
    version: &BuildVersion,
) -> anyhow::Result<PathBuf> {
    // Before listing, so a second pull neither lists nor waits.
    let _lock = match StateLock::try_acquire(&paths.lock)? {
        Ok(lock) => lock,
        Err(holder) => {
            return Err(PullError::Busy {
                holder,
                lock: paths.lock.clone(),
            }
            .into());
        }
    };
    pull_locked(http, releases, paths, version, None)
}

/// [`pull`] for a claim: nothing asked of the bucket when `build` is pulled
/// already, and `releases.lock` waited for rather than refused.
pub(crate) fn ensure_pulled(
    http: &HttpClient,
    releases: &ReleasesBucket,
    paths: &ReleasePaths,
    build: &Build,
    signals: Signals<'_>,
) -> Result<PathBuf, UnreadyError> {
    // Safe without the lock: a pull moves a directory aside only when no
    // matching record vouches for it.
    if Records::load(paths)
        .map_err(UnreadyError::transient)?
        .held(paths, build)
        == Held::Pulled
    {
        return Ok(version_dir(paths, &build.version));
    }
    let _lock = StateLock::wait_for(&paths.lock, timing::LOCK_WAIT_BUDGET, signals)
        .map_err(UnreadyError::transient)?;
    pull_locked(http, releases, paths, &build.version, Some(signals)).map_err(|error| {
        let version_specific = error
            .chain()
            .filter_map(|cause| cause.downcast_ref::<PullError>())
            .any(PullError::is_version_specific);
        UnreadyError::new(
            if version_specific {
                Fault::Version
            } else {
                Fault::Transient
            },
            error,
        )
    })
}

#[derive(Debug, Clone)]
pub(crate) struct Archived {
    pub(crate) archive: PathBuf,
    pub(crate) app: PathBuf,
}

/// The one `.xcarchive` in `releases/<version>/`, and its one
/// `Products/Applications/*.app`. Anything else is this version's fault.
pub(crate) fn archived(dir: &Path) -> Result<Archived, UnreadyError> {
    let archive = only_child(dir, ARCHIVE_EXTENSION, "archive").map_err(layout_fault)?;
    let app =
        only_child(&archive.join("Products/Applications"), "app", "app").map_err(layout_fault)?;
    Ok(Archived { archive, app })
}

fn layout_fault(error: OnlyChildError) -> UnreadyError {
    let fault = match &error {
        OnlyChildError::Read { source, .. } if source.kind() != io::ErrorKind::NotFound => {
            Fault::Transient
        }
        OnlyChildError::Read { .. } | OnlyChildError::NotOne { .. } => Fault::Version,
    };
    UnreadyError::new(fault, error)
}

pub(crate) fn only_child(
    dir: &Path,
    extension: &str,
    what: &'static str,
) -> Result<PathBuf, OnlyChildError> {
    let reading = |source| OnlyChildError::Read {
        dir: dir.to_path_buf(),
        source,
    };
    let matching: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(reading)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(reading)?
        .into_iter()
        .filter(|path| {
            path.extension()
                .is_some_and(|found| found.eq_ignore_ascii_case(extension))
        })
        .collect();
    match <[PathBuf; 1]>::try_from(matching) {
        Ok([only]) => Ok(only),
        Err(matching) => Err(OnlyChildError::NotOne {
            dir: dir.to_path_buf(),
            what,
            found: matching.len(),
        }),
    }
}

/// A claim's `signals` end the download.
fn pull_locked(
    http: &HttpClient,
    releases: &ReleasesBucket,
    paths: &ReleasePaths,
    version: &BuildVersion,
    signals: Option<Signals<'_>>,
) -> anyhow::Result<PathBuf> {
    let build =
        super::find_build(http, releases, version)?.ok_or_else(|| PullError::NotPublished {
            version: version.clone(),
            bucket: releases.url.to_string(),
        })?;

    let mut records = Records::load(paths)?;
    let dir = version_dir(paths, version);
    match records.held(paths, &build) {
        Held::Changed { recorded } => {
            return Err(PullError::DigestChanged {
                version: version.clone(),
                recorded,
                published: build.sha256,
                key: build.key,
            }
            .into());
        }
        Held::Pulled => {
            tracing::info!(%version, dir = %dir.display(), "already pulled");
            return Ok(dir);
        }
        Held::NotPulled => {}
    }

    fs::create_dir_all(&paths.dir).with_context(|| format!("creating {}", paths.dir.display()))?;
    sweep_partials(&paths.dir)?;
    // Dropped on every way out, so a failure leaves nothing of the version.
    let scratch = tempfile::Builder::new()
        .prefix(PARTIAL_PREFIX)
        .tempdir_in(&paths.dir)
        .with_context(|| format!("creating a scratch directory in {}", paths.dir.display()))?;
    // No matching record vouches for what is there, so it goes with the scratch
    // directory, before any record is written.
    if let Err(error) = fs::rename(&dir, scratch.path().join("unverified"))
        && error.kind() != io::ErrorKind::NotFound
    {
        return Err(error).with_context(|| format!("moving {} aside", dir.display()));
    }

    tracing::info!(%version, key = %build.key, bytes = build.bytes, "downloading");
    let zip = scratch.path().join("archive.zip");
    let downloaded = download(http, releases, &build.key, &zip, signals)?;
    if downloaded != build.sha256 {
        return Err(PullError::DigestMismatch {
            key: build.key,
            downloaded,
            published: build.sha256,
        }
        .into());
    }
    // Recorded once the bytes are verified, whatever unpacking makes of them.
    if !records.versions.contains_key(version.as_str()) {
        records.versions.insert(
            version.to_string(),
            Record {
                sha256: build.sha256.clone(),
                key: build.key.clone(),
                pulled_at: timing::rfc3339_now(),
            },
        );
        state::store_json(&paths.records, WHAT, &records)?;
    }

    let unpacked = scratch.path().join("unpacked");
    unpack(&zip, &unpacked, &build.key)?;
    let home = archive_home(&unpacked)?.ok_or_else(|| PullError::NoArchive {
        key: build.key.clone(),
    })?;
    fs::rename(&home, &dir)
        .with_context(|| format!("moving the unpacked archive to {}", dir.display()))?;
    tracing::info!(%version, dir = %dir.display(), sha256 = %build.sha256, "pulled");
    Ok(dir)
}

/// Called under the lock, so whatever carries the prefix belongs to no one.
pub(crate) fn sweep_partials(dir: &Path) -> anyhow::Result<()> {
    let reading = || format!("reading {}", dir.display());
    fs::read_dir(dir)
        .with_context(reading)?
        .try_for_each(|entry| -> anyhow::Result<()> {
            let entry = entry.with_context(reading)?;
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with(PARTIAL_PREFIX)
            {
                fs::remove_dir_all(entry.path())
                    .with_context(|| format!("removing {}", entry.path().display()))?;
            }
            Ok(())
        })
}

/// What `builds prune` removes from `releases/` and `signed/`, sorted: every
/// version directory `keep` does not name, and every scratch directory. Any
/// other entry stays.
pub(crate) fn prunable(
    paths: &ReleasePaths,
    keep: &BTreeSet<String>,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut stale = prunable_in(&paths.dir, keep)?;
    stale.extend(prunable_in(&paths.signed, keep)?);
    stale.sort();
    Ok(stale)
}

fn prunable_in(dir: &Path, keep: &BTreeSet<String>) -> anyhow::Result<Vec<PathBuf>> {
    let reading = || format!("reading {}", dir.display());
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error).with_context(reading),
    };
    entries
        .map(|entry| -> anyhow::Result<Option<PathBuf>> {
            let entry = entry.with_context(reading)?;
            if !entry.file_type().with_context(reading)?.is_dir() {
                return Ok(None);
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let stale = name.starts_with(PARTIAL_PREFIX)
                || (name.parse::<BuildVersion>().is_ok() && !keep.contains(name.as_ref()));
            Ok(stale.then(|| entry.path()))
        })
        .filter_map(Result::transpose)
        .collect()
}

/// Returns the sha256 of what was written.
fn download(
    http: &HttpClient,
    releases: &ReleasesBucket,
    key: &str,
    to: &Path,
    signals: Option<Signals<'_>>,
) -> anyhow::Result<Sha256> {
    let mut file = HashingFile {
        file: fs::File::create(to).with_context(|| format!("creating {}", to.display()))?,
        hasher: Sha256Hasher::new(),
        signals,
    };
    // A failed write (a full disk) is reported as one; `get` reports a failed
    // read of the body as the response breaking off.
    s3::get(http, releases, key, &[], |body| -> anyhow::Result<u64> {
        io::copy(body, &mut file).with_context(|| format!("writing {}", to.display()))
    })
    .with_context(|| format!("downloading {key}"))?;
    Ok(Sha256::try_new(hex::encode(file.hasher.finalize()))?)
}

struct HashingFile<'a> {
    file: fs::File,
    hasher: Sha256Hasher,
    signals: Option<Signals<'a>>,
}

impl Write for HashingFile<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.signals.is_some_and(|signals| signals.raised()) {
            return Err(io::Error::other(DownloadStoppedError));
        }
        self.file.write_all(buf)?;
        self.hasher.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

/// `ditto`, as the release job packs with it: it keeps the bundle's symlinks
/// and extended attributes, which a later signing needs.
fn unpack(zip: &Path, into: &Path, key: &str) -> anyhow::Result<()> {
    let output = Command::new("ditto")
        .args(["-x", "-k"])
        .arg(zip)
        .arg(into)
        .stdin(Stdio::null())
        .output()
        .context("running `ditto`")?;
    if output.status.success() {
        return Ok(());
    }
    Err(PullError::Unpack {
        key: key.to_owned(),
        detail: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
    }
    .into())
}

/// The directory that holds the `.xcarchive`: `unpacked` itself, or the one
/// folder the release job wraps it in (`ditto -c -k --keepParent`).
fn archive_home(unpacked: &Path) -> anyhow::Result<Option<PathBuf>> {
    let children = |dir: &Path| -> anyhow::Result<Vec<PathBuf>> {
        let reading = || format!("reading {}", dir.display());
        let mut children = fs::read_dir(dir)
            .with_context(reading)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()
            .with_context(reading)?;
        children.sort();
        Ok(children)
    };
    let is_archive = |path: &PathBuf| {
        path.is_dir()
            && path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case(ARCHIVE_EXTENSION))
    };
    let top = children(unpacked)?;
    if top.iter().any(is_archive) {
        return Ok(Some(unpacked.to_path_buf()));
    }
    top.into_iter()
        .filter(|path| path.is_dir())
        .find_map(|folder| match children(&folder) {
            Ok(entries) if entries.iter().any(is_archive) => Some(Ok(folder)),
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::net::TcpListener;
    use std::sync::atomic::Ordering;

    use httpmock::Mock;
    use rstest::rstest;

    use super::*;
    use crate::releases::sign::testing::Flags;
    use crate::releases::testing::{FakeBucket, Published, build, listing_page, sha256_of};
    use crate::state::StatePaths;

    const VERSION: &str = "2026.10.02-5c435ba-internal";
    const KEY: &str = "2026.10.02-5c435ba/pipette-ios-2026.10.02-5c435ba-internal.xcarchive.zip";
    const OTHER_BYTES: &[u8] = b"not the bytes that were summed";

    /// What the release job uploads: a folder holding the archive and its note,
    /// packed with `ditto -c -k --keepParent`. `marker` makes the bytes differ.
    fn release_zip(marker: &str) -> anyhow::Result<Vec<u8>> {
        let stage = tempfile::tempdir()?;
        let folder = stage.path().join(format!("pipette-ios-{VERSION}"));
        let app = folder.join("Pipette.xcarchive/Products/Applications/Pipette.app");
        fs::create_dir_all(&app)?;
        fs::write(app.join("Info.plist"), marker)?;
        fs::write(folder.join("INTERNAL-BUILD.md"), "# Pipette iOS\n")?;
        zip_of(&folder, &["--keepParent"])
    }

    fn zip_of(folder: &Path, flags: &[&str]) -> anyhow::Result<Vec<u8>> {
        let zip = folder.with_extension("zip");
        let packed = Command::new("ditto")
            .args(["-c", "-k"])
            .args(flags)
            .arg(folder)
            .arg(&zip)
            .output()?;
        anyhow::ensure!(packed.status.success(), "ditto -c -k: {packed:?}");
        Ok(fs::read(&zip)?)
    }

    /// A zip as the release packs one, holding no `.xcarchive`.
    fn notes_zip() -> anyhow::Result<Vec<u8>> {
        let stage = tempfile::tempdir()?;
        let folder = stage.path().join("notes");
        fs::create_dir_all(&folder)?;
        fs::write(folder.join("README.md"), "no archive here")?;
        zip_of(&folder, &["--keepParent"])
    }

    fn pulled(bucket: &FakeBucket, paths: &ReleasePaths) -> anyhow::Result<PathBuf> {
        pull(
            &s3::http_client()?,
            &bucket.config()?,
            paths,
            &VERSION.parse()?,
        )
    }

    /// `pull`, retried while the lock is busy: a child another test thread is
    /// spawning holds a copy of the lock an earlier pull released until its
    /// exec closes it.
    fn once_free(pull: impl Fn() -> anyhow::Result<PathBuf>) -> anyhow::Result<PathBuf> {
        (0..100).fold(pull(), |outcome, _| match outcome {
            Err(error)
                if matches!(
                    error.downcast_ref::<PullError>(),
                    Some(PullError::Busy { .. })
                ) =>
            {
                std::thread::sleep(std::time::Duration::from_millis(10));
                pull()
            }
            outcome => outcome,
        })
    }

    /// A state directory, and a bucket to pull [`VERSION`] from.
    struct Fixture {
        _state: tempfile::TempDir,
        paths: ReleasePaths,
        bucket: FakeBucket,
    }

    impl Fixture {
        fn new() -> anyhow::Result<Self> {
            let state = tempfile::tempdir()?;
            let paths = StatePaths::resolve(Some(state.path()))?.releases();
            Ok(Self {
                _state: state,
                paths,
                bucket: FakeBucket::start(),
            })
        }

        /// Publishes `zip` as [`VERSION`]; returns the mock serving the archive.
        fn publish(&self, zip: &[u8]) -> Mock<'_> {
            let build = Published::build(KEY, zip, 2);
            self.bucket.serve_beside(&build);
            self.bucket.list(&build.objects(), 1000);
            self.bucket.serve(KEY, zip)
        }

        fn pull(&self) -> anyhow::Result<PathBuf> {
            pulled(&self.bucket, &self.paths)
        }

        fn ensure(&self, zip: &[u8], signals: Signals<'_>) -> Result<PathBuf, UnreadyError> {
            let build =
                build("2026.10.02-5c435ba", VERSION, zip).map_err(UnreadyError::transient)?;
            let http = s3::http_client().map_err(UnreadyError::transient)?;
            let bucket = self.bucket.config().map_err(UnreadyError::transient)?;
            ensure_pulled(&http, &bucket, &self.paths, &build, signals)
        }

        /// What `releases/` holds, by name.
        fn stored(&self) -> anyhow::Result<Vec<String>> {
            if !self.paths.dir.exists() {
                return Ok(Vec::new());
            }
            let mut names = fs::read_dir(&self.paths.dir)?
                .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
                .collect::<Result<Vec<_>, _>>()?;
            names.sort();
            Ok(names)
        }
    }

    fn info_plist(dir: &Path) -> anyhow::Result<String> {
        Ok(fs::read_to_string(dir.join(
            "Pipette.xcarchive/Products/Applications/Pipette.app/Info.plist",
        ))?)
    }

    fn refusal(outcome: anyhow::Result<PathBuf>) -> anyhow::Result<String> {
        let error = outcome.err().context("the pull must be refused")?;
        Ok(format!("{error:#}"))
    }

    #[test]
    fn a_pull_verifies_records_and_unpacks_the_archive() -> anyhow::Result<()> {
        let fixture = Fixture::new()?;
        let zip = release_zip("first")?;
        fixture.publish(&zip);

        let dir = fixture.pull()?;

        assert_eq!(dir, fixture.paths.dir.join(VERSION));
        assert_eq!(info_plist(&dir)?, "first");
        assert!(dir.join("INTERNAL-BUILD.md").is_file());
        assert_eq!(fixture.stored()?, [VERSION], "no scratch directory is left");
        let records: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&fixture.paths.records)?)?;
        assert_eq!(records["schema"], 1);
        assert_eq!(records["versions"][VERSION]["sha256"], sha256_of(&zip));
        assert_eq!(records["versions"][VERSION]["key"], KEY);
        Ok(())
    }

    #[test]
    fn a_second_pull_downloads_nothing() -> anyhow::Result<()> {
        let fixture = Fixture::new()?;
        let archive = fixture.publish(&release_zip("first")?);

        let first = fixture.pull()?;
        let recorded = fs::read_to_string(&fixture.paths.records)?;
        let second = once_free(|| fixture.pull())?;

        assert_eq!(first, second);
        archive.assert_calls(1);
        assert_eq!(fs::read_to_string(&fixture.paths.records)?, recorded);
        Ok(())
    }

    /// How the bucket serves [`VERSION`] to a pull that must be refused.
    #[derive(Debug, Clone, Copy)]
    enum BucketFault {
        /// The archive's bytes are not the ones its sidecar sums.
        OtherBytes,
        ArchiveRefused,
        SidecarRefused,
        NotListed,
        /// A second key publishes the version too.
        TwoKeys,
        /// The zip unpacks without an `.xcarchive`.
        NoArchiveInside,
        /// Bytes that match their sidecar and do not unpack.
        NotAZip,
        ListingRefused,
    }

    impl Fixture {
        fn publish_with(&self, fault: BucketFault) -> anyhow::Result<()> {
            let zip = match fault {
                BucketFault::NoArchiveInside => notes_zip()?,
                BucketFault::NotAZip => b"not a zip".to_vec(),
                BucketFault::OtherBytes
                | BucketFault::ArchiveRefused
                | BucketFault::SidecarRefused
                | BucketFault::NotListed
                | BucketFault::TwoKeys
                | BucketFault::ListingRefused => release_zip("first")?,
            };
            let build = Published::build(KEY, &zip, 2);
            let refuse = |key: String, status: u16| {
                self.bucket.server.mock(|when, then| {
                    when.path(format!("/builds/{key}"));
                    then.status(status)
                        .body("<Error><Code>InternalError</Code></Error>");
                });
            };
            match fault {
                BucketFault::OtherBytes => {
                    self.bucket.serve_beside(&build);
                    self.bucket.list(&build.objects(), 1000);
                    self.bucket.serve(KEY, OTHER_BYTES);
                }
                BucketFault::ArchiveRefused => {
                    self.bucket.serve_beside(&build);
                    self.bucket.list(&build.objects(), 1000);
                    refuse(KEY.to_owned(), 500);
                }
                BucketFault::SidecarRefused => {
                    refuse(build.sidecar.key.clone(), 503);
                    self.bucket.list(&build.objects(), 1000);
                }
                BucketFault::NotListed => {
                    self.bucket.list(&[], 1000);
                }
                BucketFault::TwoKeys => {
                    let mut objects = build.objects();
                    objects.extend(
                        Published::build(
                            &format!("branches/b/pipette-ios-{VERSION}.xcarchive.zip"),
                            &zip,
                            3,
                        )
                        .objects(),
                    );
                    self.bucket.publish(&objects, 1000);
                }
                BucketFault::NoArchiveInside | BucketFault::NotAZip => {
                    self.bucket.publish(&build.objects(), 1000);
                }
                BucketFault::ListingRefused => {
                    refuse(String::new(), 503);
                }
            }
            Ok(())
        }
    }

    /// Every refused pull names its reason and leaves no version behind; only
    /// bytes that matched their sidecar are recorded.
    #[rstest]
    #[case::digest_mismatch(
        BucketFault::OtherBytes,
        &format!("downloaded with sha256 {}, and its .sha256 says", sha256_of(OTHER_BYTES)),
        false,
        Fault::Version
    )]
    #[case::archive_refused(
        BucketFault::ArchiveRefused,
        &format!("downloading {KEY}: HTTP 500"),
        false,
        Fault::Transient
    )]
    #[case::sidecar_refused(
        BucketFault::SidecarRefused,
        &format!("reading the .sha256 and .json of {KEY}: HTTP 503"),
        false,
        Fault::Transient
    )]
    #[case::not_listed(
        BucketFault::NotListed,
        &format!("publishes no build {VERSION}"),
        false,
        Fault::Version
    )]
    #[case::two_keys(
        BucketFault::TwoKeys,
        &format!("publishes no build {VERSION}"),
        false,
        Fault::Version
    )]
    #[case::no_archive_inside(
        BucketFault::NoArchiveInside,
        "without an .xcarchive",
        true,
        Fault::Version
    )]
    #[case::not_a_zip(BucketFault::NotAZip, "could not unpack", true, Fault::Transient)]
    #[case::listing_refused(
        BucketFault::ListingRefused,
        "listing the bucket",
        false,
        Fault::Transient
    )]
    fn a_refused_pull_leaves_nothing_and_a_claim_files_it(
        #[case] fault: BucketFault,
        #[case] reason: &str,
        #[case] recorded: bool,
        #[case] class: Fault,
    ) -> anyhow::Result<()> {
        let pulled = Fixture::new()?;
        pulled.publish_with(fault)?;

        let message = refusal(pulled.pull())?;

        assert!(message.contains(reason), "{message}");
        assert_eq!(pulled.stored()?, [] as [&str; 0]);
        assert_eq!(pulled.paths.records.exists(), recorded);

        let claimed = Fixture::new()?;
        claimed.publish_with(fault)?;
        let flags = Flags::default();

        let unready = claimed
            .ensure(b"zip", flags.signals())
            .err()
            .context("the pull must fail")?;

        let message = format!("{:#}", unready.error);
        assert_eq!(unready.fault, class, "{message}");
        assert!(message.contains(reason), "{message}");
        Ok(())
    }

    #[rstest]
    #[case::one_app(&["Pipette.xcarchive/Products/Applications/Pipette.app"], Ok(()))]
    #[case::no_archive(&["notes"], Err("holds 0 where one archive"))]
    #[case::no_applications(&["Pipette.xcarchive/Products"], Err("reading"))]
    #[case::two_apps(
        &[
            "Pipette.xcarchive/Products/Applications/Pipette.app",
            "Pipette.xcarchive/Products/Applications/Other.app",
        ],
        Err("holds 2 where one app")
    )]
    fn a_pulled_layout_needs_one_archive_and_one_app(
        #[case] dirs: &[&str],
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let pulled = tempfile::tempdir()?;
        dirs.iter()
            .try_for_each(|dir| fs::create_dir_all(pulled.path().join(dir)))?;

        match (archived(pulled.path()), expected) {
            (Ok(found), Ok(())) => assert!(found.app.ends_with("Applications/Pipette.app")),
            (Err(unready), Err(needle)) => {
                assert_eq!(unready.fault, Fault::Version);
                assert!(
                    format!("{:#}", unready.error).contains(needle),
                    "{:#}",
                    unready.error
                );
            }
            (got, expected) => anyhow::bail!("got {got:?}, expected {expected:?}"),
        }
        Ok(())
    }

    #[test]
    fn a_pulled_build_needs_no_bucket() -> anyhow::Result<()> {
        let fixture = Fixture::new()?;
        let zip = release_zip("first")?;
        fixture.publish(&zip);
        let dir = fixture.pull()?;
        fixture.bucket.server.reset();
        let flags = Flags::default();

        assert_eq!(fixture.ensure(&zip, flags.signals())?, dir);
        Ok(())
    }

    #[rstest]
    #[case::waiting(true, "stopped while waiting")]
    #[case::downloading(false, "the download was stopped")]
    fn a_stop_ends_a_claims_pull(
        #[case] lock_held: bool,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new()?;
        let zip = release_zip("first")?;
        let archive = fixture.publish(&zip);
        let held = if lock_held {
            Some(
                StateLock::try_acquire(&fixture.paths.lock)?
                    .map_err(|holder| anyhow::anyhow!("a free lock reported busy by {holder}"))?,
            )
        } else {
            None
        };
        let flags = Flags::default();
        flags.stop.store(true, Ordering::SeqCst);

        let unready = fixture
            .ensure(&zip, flags.signals())
            .err()
            .context("a stopped pull must fail")?;

        drop(held);
        assert_eq!(unready.fault, Fault::Transient);
        assert!(
            format!("{:#}", unready.error).contains(needle),
            "{:#}",
            unready.error
        );
        archive.assert_calls(usize::from(!lock_held));
        assert_eq!(fixture.stored()?, [] as [&str; 0]);
        Ok(())
    }

    /// A directory no record vouches for never outlives a pull that downloads:
    /// it is replaced, or gone when the pull then fails, before or after the
    /// digest is recorded.
    #[rstest]
    #[case::unpacks(
        |fixture: &Fixture| {
            fixture.publish(&release_zip("first")?);
            Ok(())
        },
        Ok(())
    )]
    #[case::digest_mismatch(
        |fixture: &Fixture| fixture.publish_with(BucketFault::OtherBytes),
        Err("and its .sha256 says")
    )]
    #[case::does_not_unpack(
        |fixture: &Fixture| fixture.publish_with(BucketFault::NoArchiveInside),
        Err("without an .xcarchive")
    )]
    fn an_unrecorded_directory_never_outlives_a_pull(
        #[case] publish: fn(&Fixture) -> anyhow::Result<()>,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let fixture = Fixture::new()?;
        let unverified = fixture.paths.dir.join(VERSION).join("unverified");
        fs::create_dir_all(fixture.paths.dir.join(VERSION))?;
        fs::write(&unverified, "bytes no record vouches for")?;
        publish(&fixture)?;

        match (fixture.pull(), expected) {
            (Ok(dir), Ok(())) => assert_eq!(info_plist(&dir)?, "first"),
            (Err(error), Err(needle)) => {
                assert!(format!("{error:#}").contains(needle), "{error:#}");
                assert_eq!(fixture.stored()?, [] as [&str; 0]);
            }
            (got, expected) => anyhow::bail!("got {got:?}, expected {expected:?}"),
        }
        assert!(!unverified.exists());
        Ok(())
    }

    /// An archive cut short of its `Content-Length`, which httpmock cannot
    /// send: a server answers the listing, the sidecar, the manifest, then half
    /// the archive, one connection each.
    #[test]
    fn a_download_that_breaks_off_leaves_nothing() -> anyhow::Result<()> {
        let fixture = Fixture::new()?;
        let zip = release_zip("first")?;
        let build = Published::build(KEY, &zip, 2);
        let sidecar = build.sidecar.body.clone();
        let manifest = build.manifest.body.clone();
        let listing = listing_page(&build.objects(), None);
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let releases = FakeBucket::config_at(&format!("http://{}", listener.local_addr()?))?;
        let server = std::thread::spawn(move || -> io::Result<()> {
            [listing.into_bytes(), sidecar, manifest, zip]
                .into_iter()
                .zip([false, false, false, true])
                .try_for_each(|(body, breaks_off)| {
                    let (mut stream, _) = listener.accept()?;
                    let _ = stream.read(&mut [0u8; 4096])?;
                    let sent = if breaks_off {
                        body.get(..body.len() / 2).unwrap_or_default()
                    } else {
                        body.as_slice()
                    };
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )?;
                    stream.write_all(sent)
                })
        });

        let message = refusal(pull(
            &s3::http_client()?,
            &releases,
            &fixture.paths,
            &VERSION.parse()?,
        ))?;
        server
            .join()
            .map_err(|_| anyhow::anyhow!("the server thread panicked"))??;

        assert!(
            message.starts_with(&format!("downloading {KEY}: the response broke off: ")),
            "{message}"
        );
        assert_eq!(fixture.stored()?, [] as [&str; 0]);
        assert!(!fixture.paths.records.exists());
        Ok(())
    }

    /// The bucket republishes the version with other bytes and a sidecar that
    /// matches them: the first digest stands.
    #[test]
    fn a_version_whose_digest_changed_is_refused() -> anyhow::Result<()> {
        let fixture = Fixture::new()?;
        let first_zip = release_zip("first")?;
        fixture.publish(&first_zip);
        let dir = fixture.pull()?;

        let second = FakeBucket::start();
        let second_zip = release_zip("second")?;
        let build = Published::build(KEY, &second_zip, 2);
        second.serve_beside(&build);
        second.list(&build.objects(), 1000);
        let archive = second.serve(KEY, &second_zip);
        let message = refusal(once_free(|| pulled(&second, &fixture.paths)))?;

        assert!(
            message.contains(&sha256_of(&first_zip))
                && message.contains(&sha256_of(&second_zip))
                && message.contains("never change under a version"),
            "{message}"
        );
        archive.assert_calls(0);
        assert_eq!(info_plist(&dir)?, "first");

        let build = crate::releases::list_builds(&s3::http_client()?, &second.config()?)?
            .pop()
            .context("the second bucket publishes the build")?;
        assert!(matches!(
            Records::load(&fixture.paths)?.held(&fixture.paths, &build),
            Held::Changed { .. }
        ));
        // Removing the directory does not unpin the version.
        fs::remove_dir_all(&dir)?;
        assert!(
            refusal(once_free(|| pulled(&second, &fixture.paths)))?
                .contains("never change under a version")
        );
        Ok(())
    }

    #[test]
    fn a_pulled_version_whose_directory_is_gone_is_pulled_again() -> anyhow::Result<()> {
        let fixture = Fixture::new()?;
        let archive = fixture.publish(&release_zip("first")?);
        let dir = fixture.pull()?;
        let recorded = fs::read_to_string(&fixture.paths.records)?;
        fs::remove_dir_all(&dir)?;

        assert_eq!(once_free(|| fixture.pull())?, dir);

        assert_eq!(info_plist(&dir)?, "first");
        archive.assert_calls(2);
        assert_eq!(
            fs::read_to_string(&fixture.paths.records)?,
            recorded,
            "the first record stands"
        );
        Ok(())
    }

    #[test]
    fn a_pull_reads_only_its_own_sidecar() -> anyhow::Result<()> {
        let fixture = Fixture::new()?;
        let zip = release_zip("first")?;
        let own = Published::build(KEY, &zip, 2);
        let other = Published::build("branches/x/pipette-ios-x.1.xcarchive.zip", b"zip", 3);
        let bucket = &fixture.bucket;
        let own_read = bucket.serve_beside(&own);
        let other_read = bucket.serve_beside(&other);
        bucket.serve(KEY, &zip);
        let mut objects = own.objects();
        objects.extend(other.objects());
        bucket.list(&objects, 1000);

        fixture.pull()?;

        own_read.assert_calls(1);
        other_read.assert_calls(0);
        Ok(())
    }

    /// `flock` is per open-file-description, so the lock held here is what a
    /// second process would meet.
    #[test]
    fn a_pull_is_refused_while_another_process_pulls() -> anyhow::Result<()> {
        let fixture = Fixture::new()?;
        let archive = fixture.publish(&release_zip("first")?);
        let _held = StateLock::try_acquire(&fixture.paths.lock)?
            .map_err(|holder| anyhow::anyhow!("a free lock reported busy by {holder}"))?;

        let message = refusal(fixture.pull())?;

        assert!(
            message.contains(&format!("pid {}", std::process::id())),
            "{message}"
        );
        archive.assert_calls(0);
        Ok(())
    }

    #[test]
    fn a_pull_removes_what_a_killed_pull_left() -> anyhow::Result<()> {
        let fixture = Fixture::new()?;
        fixture.publish(&release_zip("first")?);
        let leftover = fixture.paths.dir.join(format!("{PARTIAL_PREFIX}dead"));
        fs::create_dir_all(&leftover)?;
        fs::write(leftover.join("archive.zip"), b"half of it")?;

        fixture.pull()?;

        assert_eq!(fixture.stored()?, [VERSION]);
        Ok(())
    }

    #[test]
    fn an_archive_packed_without_its_folder_unpacks_too() -> anyhow::Result<()> {
        let fixture = Fixture::new()?;
        let stage = tempfile::tempdir()?;
        let folder = stage.path().join("flat");
        fs::create_dir_all(folder.join("Pipette.xcarchive"))?;
        fs::write(folder.join("Pipette.xcarchive/Info.plist"), "flat")?;
        fixture.publish(&zip_of(&folder, &[])?);

        let dir = fixture.pull()?;

        assert!(dir.join("Pipette.xcarchive/Info.plist").is_file());
        Ok(())
    }
}
