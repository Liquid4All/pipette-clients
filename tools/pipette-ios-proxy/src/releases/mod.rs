//! The app builds each release publishes to the `[client_releases]` bucket
//! (DESIGN sections 8 and 9).

pub(crate) mod catalog;
pub(crate) mod s3;
pub(crate) mod sign;
pub(crate) mod store;

use std::collections::BTreeMap;
use std::str::FromStr;

use anyhow::Context;
use serde::Deserialize;
use time::OffsetDateTime;

use pipette_http::HttpClient;
use pipette_plan_types::{Runtime, Sha256, canonical_json};

use crate::config::ReleasesBucket;

const ARCHIVE_PREFIX: &str = "pipette-ios-";
const ARCHIVE_SUFFIX: &str = ".xcarchive.zip";

const SIDECAR_SUFFIX: &str = ".sha256";

/// A sidecar is one `sha256sum` line; an object past this is not one, and is
/// not read.
const SIDECAR_MAX_BYTES: u64 = 4096;

const MANIFEST_SUFFIX: &str = ".json";

/// A manifest names a handful of runtimes; an object past this is not one.
const MANIFEST_MAX_BYTES: u64 = 64 * 1024;

/// An app build's own `client_version`, such as `2026.10.02-5c435ba-internal`.
/// `[a-z0-9][a-z0-9._-]*`, which makes it one path segment in the store.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct BuildVersion(String);

#[derive(Debug, thiserror::Error)]
#[error("build version {0:?} must match [a-z0-9][a-z0-9._-]*")]
pub(crate) struct BuildVersionError(String);

impl FromStr for BuildVersion {
    type Err = BuildVersionError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let plain = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit();
        let mut chars = raw.chars();
        if chars.next().is_some_and(plain) && chars.all(|c| plain(c) || "._-".contains(c)) {
            Ok(Self(raw.to_owned()))
        } else {
            Err(BuildVersionError(raw.to_owned()))
        }
    }
}

impl BuildVersion {
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BuildVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Who can clear a failure to make a build ready (DESIGN section 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fault {
    /// Only an operator: the phone stops.
    Operator,
    /// The version itself: this phone stops advertising it.
    Version,
    /// Anything else, which may clear on its own.
    Transient,
}

#[derive(Debug, thiserror::Error)]
#[error("{error:#}")]
pub(crate) struct UnreadyError {
    pub(crate) fault: Fault,
    pub(crate) error: anyhow::Error,
}

impl UnreadyError {
    pub(crate) fn new(fault: Fault, error: impl Into<anyhow::Error>) -> Self {
        Self {
            fault,
            error: error.into(),
        }
    }

    pub(crate) fn transient(error: impl Into<anyhow::Error>) -> Self {
        Self::new(Fault::Transient, error)
    }
}

/// One published build: an archive with a valid `.sha256` and manifest
/// beside it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Build {
    pub(crate) version: BuildVersion,
    /// The name of the directory the archive sits in, `[A-Za-z0-9._-]+`: the
    /// release tag for a release build.
    pub(crate) tag: String,
    pub(crate) key: String,
    /// The first token of the sidecar.
    pub(crate) sha256: Sha256,
    pub(crate) bytes: u64,
    /// The archive's last-modified time.
    pub(crate) uploaded_at: OffsetDateTime,
    /// The manifest's runtime descriptors, as written: one this proxy cannot
    /// read is kept, so the phone's own listing still compares equal.
    pub(crate) runtimes: Vec<serde_json::Value>,
}

impl Build {
    pub(crate) fn carries(&self, runtime: &Runtime) -> bool {
        self.readable_runtimes().any(|named| &named == runtime)
    }

    pub(crate) fn readable_runtimes(&self) -> impl Iterator<Item = Runtime> + '_ {
        self.runtimes
            .iter()
            .filter_map(|raw| serde_json::from_value::<Runtime>(raw.clone()).ok())
    }

    pub(crate) fn runtime_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.runtimes.iter().map(descriptor_key).collect();
        keys.sort();
        keys
    }
}

/// One spelling per descriptor, whoever wrote it: a readable one as this
/// proxy serializes it (aliases and defaults settled), any other as written,
/// both with sorted keys.
pub(crate) fn descriptor_key(raw: &serde_json::Value) -> String {
    match serde_json::from_value::<Runtime>(raw.clone()) {
        Ok(runtime) => serde_json::to_value(&runtime)
            .map_or_else(|_| canonical_json(raw), |value| canonical_json(&value)),
        Err(_) => canonical_json(raw),
    }
}

/// `runtime` as a plan's `runtimes` entry writes it: a TOML inline table,
/// `type` first and the other keys sorted, whatever order the maps keep.
pub(crate) fn plan_toml(runtime: &Runtime) -> String {
    serde_json::to_value(runtime)
        .ok()
        .as_ref()
        .and_then(toml_value)
        .unwrap_or_else(|| runtime.to_string())
}

/// `None` for a JSON `null`, which TOML cannot write: its key is left out.
/// Keys are written bare, as a runtime's field names all can be.
fn toml_value(value: &serde_json::Value) -> Option<String> {
    use serde_json::Value;
    match value {
        Value::Null => None,
        Value::Bool(flag) => Some(flag.to_string()),
        Value::Number(number) => Some(number.to_string()),
        Value::String(text) => Some(toml::Value::String(text.clone()).to_string()),
        Value::Array(items) => {
            let items: Vec<String> = items.iter().filter_map(toml_value).collect();
            Some(format!("[{}]", items.join(", ")))
        }
        Value::Object(map) => {
            let mut fields: Vec<(&str, String)> = map
                .iter()
                .filter_map(|(key, value)| Some((key.as_str(), toml_value(value)?)))
                .collect();
            fields.sort_by_key(|(key, _)| (*key != "type", *key));
            let fields: Vec<String> = fields
                .into_iter()
                .map(|(key, value)| format!("{key} = {value}"))
                .collect();
            Some(format!("{{ {} }}", fields.join(", ")))
        }
    }
}

/// `pipette-ios-<V>.json`, which the release job writes from the sources the
/// archive was built from.
#[derive(Debug, Deserialize)]
struct Manifest {
    client_version: String,
    runtimes: Vec<serde_json::Value>,
}

fn manifest_key(archive_key: &str) -> String {
    let stem = archive_key
        .strip_suffix(ARCHIVE_SUFFIX)
        .unwrap_or(archive_key);
    format!("{stem}{MANIFEST_SUFFIX}")
}

/// A key whose file name publishes an archive with a canonical version.
struct Archive<'a> {
    object: &'a s3::Object,
    directory: Option<&'a str>,
    version: BuildVersion,
}

/// `object` as an archive, or `None`: silently for a file that is not one, with
/// a warning for one whose version is not canonical.
fn archive(object: &s3::Object) -> Option<Archive<'_>> {
    let key = object.key.as_str();
    let (directory, file_name) = match key.rsplit_once('/') {
        Some((directory, file_name)) => (Some(directory), file_name),
        None => (None, key),
    };
    let raw = file_name
        .strip_prefix(ARCHIVE_PREFIX)?
        .strip_suffix(ARCHIVE_SUFFIX)?;
    match raw.parse() {
        Ok(version) => Some(Archive {
            object,
            directory,
            version,
        }),
        Err(error) => {
            tracing::warn!(
                key,
                error = %error,
                "skipping an app build whose version is not canonical"
            );
            None
        }
    }
}

/// Whether `key`'s file name is `pipette-ios-<V>.xcarchive.zip`, whatever
/// `<V>` holds: an object that looks like a build.
fn looks_like_archive(key: &str) -> bool {
    let file_name = key.rsplit('/').next().unwrap_or(key);
    file_name.starts_with(ARCHIVE_PREFIX) && file_name.ends_with(ARCHIVE_SUFFIX)
}

/// An archive the listing names a build, before its sidecar is read.
struct Candidate {
    version: BuildVersion,
    tag: String,
    key: String,
    bytes: u64,
    uploaded_at: OffsetDateTime,
}

struct Candidates {
    found: Vec<Candidate>,
    /// Objects that look like builds but are not candidates: one still
    /// uploading among them.
    skipped: usize,
}

/// Every archive in `objects` that can be a build. Each one skipped that looks
/// like a build is logged with the reason.
fn candidates(objects: &[s3::Object]) -> Candidates {
    let bytes_by_key: BTreeMap<&str, u64> = objects
        .iter()
        .map(|object| (object.key.as_str(), object.bytes))
        .collect();
    let archives: Vec<Archive<'_>> = objects.iter().filter_map(archive).collect();
    // Every archive key counts, whatever its sidecar: a version is ambiguous
    // even while one copy is still uploading.
    let keys_per_version = archives.iter().fold(
        BTreeMap::<&BuildVersion, usize>::new(),
        |mut counts, archive| {
            *counts.entry(&archive.version).or_default() += 1;
            counts
        },
    );

    let found: Vec<Candidate> = archives
        .iter()
        .filter_map(|archive| {
            let key = archive.object.key.as_str();
            if keys_per_version
                .get(&archive.version)
                .is_some_and(|&n| n > 1)
            {
                tracing::warn!(
                    key,
                    version = %archive.version,
                    "skipping an app build whose version another key also publishes"
                );
                return None;
            }
            // URL parsing resolves these segments, so the request would leave
            // the path that was signed.
            if key.split('/').any(|segment| matches!(segment, "." | "..")) {
                tracing::warn!(
                    key,
                    "skipping an app build whose key has a `.` or `..` segment"
                );
                return None;
            }
            let Some(tag) = archive
                .directory
                .and_then(|directory| directory.rsplit('/').next())
            else {
                tracing::warn!(key, "skipping an app build that sits in no directory");
                return None;
            };
            // `builds list` prints it as a column.
            if tag.is_empty()
                || !tag
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
            {
                tracing::warn!(
                    key,
                    "skipping an app build whose directory name is not [A-Za-z0-9._-]+"
                );
                return None;
            }
            match bytes_by_key.get(format!("{key}{SIDECAR_SUFFIX}").as_str()) {
                None => {
                    tracing::warn!(key, "skipping an app build with no .sha256 sidecar");
                    return None;
                }
                Some(&bytes) if bytes > SIDECAR_MAX_BYTES => {
                    tracing::warn!(
                        key,
                        bytes,
                        "skipping an app build whose .sha256 sidecar is too large to be one"
                    );
                    return None;
                }
                Some(_) => {}
            }
            match bytes_by_key.get(manifest_key(key).as_str()) {
                None => {
                    tracing::warn!(key, "skipping an app build with no .json manifest");
                    None
                }
                Some(&bytes) if bytes > MANIFEST_MAX_BYTES => {
                    tracing::warn!(
                        key,
                        bytes,
                        "skipping an app build whose .json manifest is too large to be one"
                    );
                    None
                }
                Some(_) => Some(Candidate {
                    version: archive.version.clone(),
                    tag: tag.to_owned(),
                    key: key.to_owned(),
                    bytes: archive.object.bytes,
                    uploaded_at: archive.object.last_modified,
                }),
            }
        })
        .collect();
    let looking = objects
        .iter()
        .filter(|object| looks_like_archive(&object.key))
        .count();
    Candidates {
        skipped: looking.saturating_sub(found.len()),
        found,
    }
}

/// `candidate` as a build, or `None` (with a warning) when its sidecar does not
/// open with a digest or its manifest is not one for its version. Only a
/// failed read of either is an error.
fn verify(
    http: &HttpClient,
    releases: &ReleasesBucket,
    candidate: &Candidate,
) -> Result<Option<Build>, s3::S3Error> {
    let key = candidate.key.as_str();
    let body = s3::get_bytes(
        http,
        releases,
        &format!("{key}{SIDECAR_SUFFIX}"),
        &[],
        SIDECAR_MAX_BYTES,
    )?;
    let Some(sha256) = sidecar_digest(&body) else {
        tracing::warn!(
            key,
            "skipping an app build whose .sha256 sidecar does not open with a sha256"
        );
        return Ok(None);
    };
    let body = s3::get_bytes(http, releases, &manifest_key(key), &[], MANIFEST_MAX_BYTES)?;
    let manifest = match serde_json::from_slice::<Manifest>(&body) {
        Ok(manifest) => manifest,
        Err(error) => {
            tracing::warn!(
                key,
                error = %error,
                "skipping an app build whose .json manifest does not parse"
            );
            return Ok(None);
        }
    };
    if manifest.client_version != candidate.version.as_str() {
        tracing::warn!(
            key,
            client_version = manifest.client_version,
            "skipping an app build whose .json manifest names another version"
        );
        return Ok(None);
    }
    let unreadable = manifest
        .runtimes
        .iter()
        .filter(|raw| serde_json::from_value::<Runtime>((*raw).clone()).is_err())
        .count();
    if unreadable > 0 {
        tracing::warn!(
            key,
            unreadable,
            "the .json manifest names runtimes this proxy cannot read; no job can ask for them"
        );
    }
    Ok(Some(Build {
        version: candidate.version.clone(),
        tag: candidate.tag.clone(),
        key: candidate.key.clone(),
        sha256,
        bytes: candidate.bytes,
        uploaded_at: candidate.uploaded_at,
        runtimes: manifest.runtimes,
    }))
}

/// The bucket's candidates, newest upload first and ties by version
/// descending.
fn listed_candidates(http: &HttpClient, releases: &ReleasesBucket) -> anyhow::Result<Candidates> {
    let objects = s3::list_objects(http, releases)
        .with_context(|| format!("listing the bucket at {}", releases.url))?;
    let mut candidates = candidates(&objects);
    candidates.found.sort_by(|a, b| {
        b.uploaded_at
            .cmp(&a.uploaded_at)
            .then_with(|| b.version.cmp(&a.version))
    });
    Ok(candidates)
}

/// `candidate` as a build, or `None` when it is not one or either file cannot
/// be read.
fn verify_or_skip(
    http: &HttpClient,
    releases: &ReleasesBucket,
    candidate: &Candidate,
) -> Option<Build> {
    verify(http, releases, candidate).unwrap_or_else(|error| {
        tracing::warn!(
            key = candidate.key,
            error = %error,
            "skipping an app build whose .sha256 sidecar or .json manifest could not be read"
        );
        None
    })
}

/// Every build in the bucket, newest upload first and ties by version
/// descending. Only a failed listing fails: a sidecar or manifest that cannot
/// be read skips its build.
pub(crate) fn list_builds(
    http: &HttpClient,
    releases: &ReleasesBucket,
) -> anyhow::Result<Vec<Build>> {
    Ok(listed_candidates(http, releases)?
        .found
        .iter()
        .filter_map(|candidate| verify_or_skip(http, releases, candidate))
        .collect())
}

/// How far [`list_newest`] reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reach {
    /// The newest builds and `also`: what a phone advertises.
    Newest,
    /// Every build besides, so the listing can show that none carries a
    /// runtime.
    Whole,
}

/// What [`list_newest`] read, each part newest upload first.
#[derive(Debug)]
pub(crate) struct Listing {
    /// The newest `keep` builds, and `also`.
    pub(crate) newest: Vec<Build>,
    /// The builds past those, read only with [`Reach::Whole`].
    pub(crate) older: Vec<Build>,
    /// Whether `newest` and `older` hold every build in the bucket: none was
    /// left unread, and every object that looks like one was read as one.
    pub(crate) whole: bool,
}

/// The newest `keep` builds by upload, and `also` wherever it sits, cut before
/// any sidecar is read; with [`Reach::Whole`], the rest too. Among the newest,
/// a sidecar or manifest the bucket refuses with a 4xx skips its build, as in
/// [`list_builds`]: that refusal repeats on every read. Any other failed read
/// there fails the listing, so an outage does not change what a phone
/// advertises. Past them, any failed read only skips its build.
pub(crate) fn list_newest(
    http: &HttpClient,
    releases: &ReleasesBucket,
    keep: usize,
    also: Option<&BuildVersion>,
    reach: Reach,
) -> anyhow::Result<Listing> {
    let Candidates { found, skipped } = listed_candidates(http, releases)?;
    let (newest, older): (Vec<_>, Vec<_>) = found
        .iter()
        .enumerate()
        .partition(|(index, candidate)| *index < keep || also == Some(&candidate.version));
    let read_newest = newest
        .iter()
        .filter_map(|(_, candidate)| match verify(http, releases, candidate) {
            Err(error) if error.is_client_refusal() => {
                tracing::warn!(
                    key = candidate.key,
                    error = %error,
                    "skipping an app build whose .sha256 sidecar or .json manifest the bucket refuses"
                );
                None
            }
            outcome => outcome
                .with_context(|| format!("reading the .sha256 and .json of {}", candidate.key))
                .transpose(),
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let read_older: Vec<Build> = match reach {
        Reach::Newest => Vec::new(),
        Reach::Whole => older
            .iter()
            .filter_map(|(_, candidate)| verify_or_skip(http, releases, candidate))
            .collect(),
    };
    Ok(Listing {
        whole: skipped == 0 && read_newest.len() == newest.len() && read_older.len() == older.len(),
        newest: read_newest,
        older: read_older,
    })
}

/// The build `version` names, as [`list_builds`] would list it, reading only
/// its own sidecar and manifest. Unlike a listing, either one that cannot be
/// read fails.
pub(crate) fn find_build(
    http: &HttpClient,
    releases: &ReleasesBucket,
    version: &BuildVersion,
) -> anyhow::Result<Option<Build>> {
    let Some(candidate) = listed_candidates(http, releases)?
        .found
        .into_iter()
        .find(|candidate| &candidate.version == version)
    else {
        return Ok(None);
    };
    verify(http, releases, &candidate)
        .with_context(|| format!("reading the .sha256 and .json of {}", candidate.key))
}

/// The first whitespace-separated token of a `sha256sum` line, if it is 64
/// lowercase hex characters. Bytes, not text: only the token has to be UTF-8.
fn sidecar_digest(body: &[u8]) -> Option<Sha256> {
    let token = body
        .split(u8::is_ascii_whitespace)
        .find(|token| !token.is_empty())?;
    Sha256::try_new(std::str::from_utf8(token).ok()?).ok()
}

#[cfg(test)]
pub(crate) mod testing {
    use httpmock::Mock;
    use httpmock::prelude::*;
    use sha2::Digest;
    use time::macros::datetime;
    use url::Url;

    use pipette_plan_types::Sha256;

    use super::Build;
    use crate::config::ReleasesBucket;
    use crate::secret::Secret;

    /// The secret half of the pair AWS's documentation signs its examples with.
    pub(crate) const SECRET: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

    /// What a listing's second page is asked for with: every character class
    /// the signed query has to encode.
    const TOKEN: &str = "page/2+of=3";

    /// An object in the fake bucket, last modified at noon on `day` of October 2026.
    #[derive(Debug, Clone)]
    pub(crate) struct Published {
        pub(crate) key: String,
        pub(crate) body: Vec<u8>,
        pub(crate) day: u8,
    }

    pub(crate) struct PublishedBuild {
        pub(crate) archive: Published,
        pub(crate) sidecar: Published,
        pub(crate) manifest: Published,
    }

    /// The build flag of [`b10216`]: the plan-types vector for this descriptor.
    pub(crate) const B10216_BUILD_FLAG: &str =
        "runtime:descriptor:6cef53d3f274d3f2a4c449b830556b5c7abadbe43b88c90dac8600d4a6442dc8";

    /// The descriptor of the llama.cpp `b10216` build every fake archive
    /// carries unless a test names others.
    pub(crate) fn b10216() -> serde_json::Value {
        serde_json::json!({
            "type": "llamacpp_ios_pipette",
            "repository_url": "github.com/ggml-org/llama.cpp",
            "repository_version": "b10216",
            "flavor": "ios-arm64",
        })
    }

    /// An MLX descriptor, its packages nested and not in key order.
    pub(crate) fn mlx() -> serde_json::Value {
        let package = |repository: &str, version: &str| {
            serde_json::json!({
                "repository_url": format!("github.com/{repository}"),
                "repository_version": version,
            })
        };
        serde_json::json!({
            "type": "mlx_ios_pipette",
            "flavor": "ios-arm64",
            "packages": {
                "swift_transformers": package("huggingface/swift-transformers", "0.1.17"),
                "mlx_swift": package("ml-explore/mlx-swift", "0.21.2"),
                "mlx_swift_lm": package("ml-explore/mlx-swift-examples", "1.18.1"),
            },
        })
    }

    impl Published {
        pub(crate) fn new(key: &str, body: &[u8], day: u8) -> Self {
            Self {
                key: key.to_owned(),
                body: body.to_vec(),
                day,
            }
        }

        pub(crate) fn build(key: &str, archive: &[u8], day: u8) -> PublishedBuild {
            Self::build_carrying(key, archive, day, &[b10216()])
        }

        pub(crate) fn build_carrying(
            key: &str,
            archive: &[u8],
            day: u8,
            runtimes: &[serde_json::Value],
        ) -> PublishedBuild {
            let file_name = key.rsplit('/').next().unwrap_or(key);
            let sidecar = format!("{}  {file_name}\n", sha256_of(archive));
            let version = file_name
                .strip_prefix(super::ARCHIVE_PREFIX)
                .and_then(|rest| rest.strip_suffix(super::ARCHIVE_SUFFIX))
                .unwrap_or(file_name);
            let manifest = serde_json::json!({
                "client_version": version,
                "runtimes": runtimes,
            });
            PublishedBuild {
                archive: Self::new(key, archive, day),
                sidecar: Self::new(&format!("{key}.sha256"), sidecar.as_bytes(), day),
                manifest: Self::new(
                    &super::manifest_key(key),
                    manifest.to_string().as_bytes(),
                    day,
                ),
            }
        }
    }

    impl PublishedBuild {
        pub(crate) fn objects(self) -> Vec<Published> {
            vec![self.archive, self.sidecar, self.manifest]
        }
    }

    pub(crate) fn sha256_of(bytes: &[u8]) -> String {
        hex::encode(sha2::Sha256::digest(bytes))
    }

    /// `version` as listed from `<tag>/`, an `archive` uploaded at noon on 2
    /// October 2026, carrying no runtime.
    pub(crate) fn build(tag: &str, version: &str, archive: &[u8]) -> anyhow::Result<Build> {
        Ok(Build {
            version: version.parse()?,
            tag: tag.to_owned(),
            key: format!("{tag}/pipette-ios-{version}.xcarchive.zip"),
            sha256: Sha256::try_new(sha256_of(archive))?,
            bytes: u64::try_from(archive.len())?,
            uploaded_at: datetime!(2026-10-02 12:00 UTC),
            runtimes: Vec::new(),
        })
    }

    /// A `ListObjectsV2` page holding `objects`, truncated when `next` names
    /// the token of the page after it.
    pub(crate) fn listing_page(objects: &[Published], next: Option<&str>) -> String {
        let contents: String = objects
            .iter()
            .map(|object| {
                format!(
                    "<Contents><Key>{}</Key>\
                     <LastModified>2026-10-{:02}T12:00:00.000Z</LastModified>\
                     <ETag>&quot;9b2cf535f27731c974343645a3985328&quot;</ETag>\
                     <Size>{}</Size><StorageClass>STANDARD</StorageClass></Contents>",
                    object.key,
                    object.day,
                    object.body.len()
                )
            })
            .collect();
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">\
             <Name>builds</Name><Prefix></Prefix>{}<KeyCount>{}</KeyCount>\
             <MaxKeys>1000</MaxKeys><IsTruncated>{}</IsTruncated>{contents}\
             </ListBucketResult>",
            next.map(|token| format!("<NextContinuationToken>{token}</NextContinuationToken>"))
                .unwrap_or_default(),
            objects.len(),
            next.is_some()
        )
    }

    /// A bucket named `builds` on httpmock, reached path-style as through
    /// `[client_releases] endpoint`. It answers signed requests only.
    pub(crate) struct FakeBucket {
        pub(crate) server: MockServer,
    }

    impl FakeBucket {
        pub(crate) const ACCESS_KEY_ID: &str = "AKIAIOSFODNN7EXAMPLE";

        pub(crate) fn start() -> Self {
            Self {
                server: MockServer::start(),
            }
        }

        pub(crate) fn config(&self) -> anyhow::Result<ReleasesBucket> {
            Self::config_at(&self.server.base_url())
        }

        /// The bucket `builds` under `endpoint`, as the config resolves it.
        pub(crate) fn config_at(endpoint: &str) -> anyhow::Result<ReleasesBucket> {
            Ok(ReleasesBucket {
                url: Url::parse(endpoint)?.join("builds/")?,
                region: "us-east-1".to_owned(),
                access_key_id: Self::ACCESS_KEY_ID.to_owned(),
                secret_access_key: Secret::from(SECRET.to_owned()),
            })
        }

        pub(crate) fn serve(&self, key: &str, body: &[u8]) -> Mock<'_> {
            self.server.mock(|when, then| {
                when.method(GET)
                    .path(format!("/builds/{key}"))
                    .query_param_exists("X-Amz-Signature");
                then.status(200).body(body);
            })
        }

        /// The listing of `objects`, `page_size` keys to a page. Returns the pages.
        pub(crate) fn list(&self, objects: &[Published], page_size: usize) -> Vec<Mock<'_>> {
            let pages: Vec<&[Published]> = if objects.is_empty() {
                vec![objects]
            } else {
                objects.chunks(page_size).collect()
            };
            let last = pages.len() - 1;
            pages
                .iter()
                .enumerate()
                .map(|(index, page)| {
                    let next = (index < last).then(|| format!("{TOKEN}{index}"));
                    let body = listing_page(page, next.as_deref());
                    self.server.mock(|when, then| {
                        let when = when
                            .method(GET)
                            .path("/builds/")
                            .query_param("list-type", "2")
                            .query_param_exists("X-Amz-Signature");
                        if index == 0 {
                            when.query_param_missing("continuation-token");
                        } else {
                            when.query_param("continuation-token", format!("{TOKEN}{}", index - 1));
                        }
                        then.status(200)
                            .header("content-type", "application/xml")
                            .body(body);
                    })
                })
                .collect()
        }

        pub(crate) fn serve_beside(&self, build: &PublishedBuild) -> Mock<'_> {
            self.serve(&build.manifest.key, &build.manifest.body);
            self.serve(&build.sidecar.key, &build.sidecar.body)
        }

        pub(crate) fn publish(&self, objects: &[Published], page_size: usize) -> Vec<Mock<'_>> {
            objects.iter().for_each(|object| {
                self.serve(&object.key, &object.body);
            });
            self.list(objects, page_size)
        }

        /// One build per `(tag, version, day)`, its archive the version's name.
        /// Returns the listing's pages.
        pub(crate) fn publish_builds(
            &self,
            builds: &[(&str, &str, u8)],
            page_size: usize,
        ) -> Vec<Mock<'_>> {
            let objects: Vec<Published> = builds
                .iter()
                .flat_map(|(tag, version, day)| {
                    Published::build(
                        &format!("{tag}/pipette-ios-{version}.xcarchive.zip"),
                        version.as_bytes(),
                        *day,
                    )
                    .objects()
                })
                .collect();
            self.publish(&objects, page_size)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use httpmock::prelude::*;
    use rstest::rstest;
    use time::macros::datetime;

    use super::testing::{FakeBucket, Published, PublishedBuild, b10216, mlx, sha256_of};
    use super::*;

    const V1: &str = "2026.10.01-0000000-internal";
    const V2: &str = "2026.10.02-5c435ba-internal";

    fn archive_key(directory: &str, version: &str) -> String {
        format!("{directory}/pipette-ios-{version}.xcarchive.zip")
    }

    fn listed(bucket: &FakeBucket) -> anyhow::Result<Vec<Build>> {
        list_builds(&s3::http_client()?, &bucket.config()?)
    }

    fn versions(builds: &[Build]) -> Vec<&str> {
        builds.iter().map(|build| build.version.as_str()).collect()
    }

    #[rstest]
    #[case::release("2026.10.02-5c435ba-internal")]
    #[case::branch("feature-x.1")]
    #[case::one_digit("7")]
    #[case::underscore("a_b")]
    fn a_canonical_version_parses(#[case] raw: &str) -> anyhow::Result<()> {
        assert_eq!(raw.parse::<BuildVersion>()?.to_string(), raw);
        Ok(())
    }

    #[rstest]
    #[case::empty("")]
    #[case::uppercase("2026.10.02-ABC")]
    #[case::leading_dot(".hidden")]
    #[case::parent("..")]
    #[case::slash("a/b")]
    #[case::space("a b")]
    fn a_version_that_is_not_one_path_segment_is_refused(#[case] raw: &str) -> anyhow::Result<()> {
        let error = raw
            .parse::<BuildVersion>()
            .err()
            .context("the version must be refused")?;
        assert_eq!(
            error.to_string(),
            format!("build version {raw:?} must match [a-z0-9][a-z0-9._-]*")
        );
        Ok(())
    }

    #[test]
    fn a_release_archive_with_a_sidecar_and_a_manifest_is_a_build() -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        let key = archive_key("2026.10.02-5c435ba", V2);
        bucket.publish(&Published::build(&key, b"zip", 2).objects(), 1000);

        assert_eq!(
            listed(&bucket)?,
            [Build {
                version: V2.parse()?,
                tag: "2026.10.02-5c435ba".to_owned(),
                key,
                sha256: Sha256::try_new(sha256_of(b"zip"))?,
                bytes: 3,
                uploaded_at: datetime!(2026-10-02 12:00 UTC),
                runtimes: vec![b10216()],
            }]
        );
        Ok(())
    }

    /// The file beside an archive that a case replaces.
    #[derive(Debug, Clone, Copy)]
    enum Part {
        Sidecar,
        Manifest,
    }

    /// Every archive whose sidecar or manifest is missing (`body` `None`) or
    /// not one for it is skipped, and the good one beside it still lists.
    #[rstest]
    #[case::no_sidecar(Part::Sidecar, None)]
    #[case::empty_sidecar(Part::Sidecar, Some(Vec::new()))]
    #[case::short_digest(Part::Sidecar, Some(b"abc123  pipette-ios.xcarchive.zip\n".to_vec()))]
    #[case::uppercase_digest(
        Part::Sidecar,
        Some(b"7F5693B172A03D00C59E0FDF0B5B9D490D7198F1DC031567CDC8010C16A975C4  x\n".to_vec())
    )]
    #[case::sidecar_not_a_digest(Part::Sidecar, Some(b"<html>not found</html>".to_vec()))]
    #[case::sidecar_not_utf8(Part::Sidecar, Some(b"\xff\xfe\xfd".to_vec()))]
    #[case::leading_newline_then_short(Part::Sidecar, Some(b"\n  abc\n".to_vec()))]
    #[case::no_manifest(Part::Manifest, None)]
    #[case::manifest_not_json(Part::Manifest, Some(b"<html>not found</html>".to_vec()))]
    #[case::no_runtimes(
        Part::Manifest,
        Some(format!(r#"{{"client_version":"{V2}"}}"#).into_bytes())
    )]
    #[case::another_version(
        Part::Manifest,
        Some(format!(r#"{{"client_version":"{V1}","runtimes":[]}}"#).into_bytes())
    )]
    #[case::oversized_manifest(Part::Manifest, Some(vec![b' '; 70_000]))]
    fn an_archive_without_a_good_sidecar_or_manifest_is_skipped(
        #[case] part: Part,
        #[case] body: Option<Vec<u8>>,
    ) -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        let bad = Published::build(&archive_key("bad", V2), b"zip", 2);
        let (replaced, kept) = match part {
            Part::Sidecar => (bad.sidecar, bad.manifest),
            Part::Manifest => (bad.manifest, bad.sidecar),
        };
        let mut objects = Published::build(&archive_key("good", V1), b"zip", 1).objects();
        objects.extend(body.map(|body| Published::new(&replaced.key, &body, 2)));
        objects.extend([bad.archive, kept]);
        bucket.publish(&objects, 1000);

        assert_eq!(versions(&listed(&bucket)?), [V1]);
        Ok(())
    }

    #[test]
    fn a_build_carries_what_its_manifest_names() -> anyhow::Result<()> {
        let unknown = serde_json::json!({ "type": "tomorrows_engine_ios", "flavor": "x" });
        let bucket = FakeBucket::start();
        let key = archive_key("tag", V2);
        bucket.publish(
            &Published::build_carrying(&key, b"zip", 2, &[b10216(), unknown.clone()]).objects(),
            1000,
        );

        let builds = listed(&bucket)?;
        let build = builds.first().context("the build lists")?;

        let named: Runtime = serde_json::from_value(b10216())?;
        let mut other = b10216();
        other["repository_version"] = "b1".into();
        assert!(build.carries(&named));
        assert!(!build.carries(&serde_json::from_value(other)?));
        assert_eq!(build.runtimes, [b10216(), unknown]);
        Ok(())
    }

    /// The release job's manifest script names, in the spelling this proxy
    /// serializes, one readable runtime per llama.cpp build the archive
    /// carries and per other engine the app lists, each marked
    /// `private_thermal` when its probe found it and `private_memory` only
    /// when its probe disagrees.
    #[rstest]
    #[case::internal_build(&["--private-thermal", "--private-memory"], Some(true), None)]
    #[case::thermal_only(&["--private-thermal"], Some(true), Some(false))]
    #[case::memory_only(&["--private-memory"], None, Some(true))]
    #[case::public_build(&[], None, None)]
    fn the_release_manifest_script_writes_runtimes_this_proxy_reads(
        #[case] flags: &[&str],
        #[case] private_thermal: Option<bool>,
        #[case] private_memory: Option<bool>,
    ) -> anyhow::Result<()> {
        let frameworks = tempfile::tempdir()?;
        [
            (
                "llamacpp",
                serde_json::json!({
                    "PipetteLlamaRepositoryUrl": "github.com/ggml-org/llama.cpp",
                    "PipetteLlamaTag": "b11988",
                    "PipetteLlamaCommit": "abb76ab38",
                }),
            ),
            (
                "llamacpp-asr",
                serde_json::json!({
                    "PipetteLlamaRepositoryUrl": "github.com/example-org/llama.cpp-asr",
                    "PipetteLlamaCommit": "2898bdc6f",
                }),
            ),
        ]
        .into_iter()
        .try_for_each(|(name, info)| -> anyhow::Result<()> {
            let framework = frameworks.path().join(format!("{name}.framework"));
            std::fs::create_dir(&framework)?;
            plist::to_file_xml(framework.join("Info.plist"), &info)?;
            Ok(())
        })?;
        let output = release_manifest(frameworks.path(), flags)?;
        anyhow::ensure!(output.status.success(), "{output:?}");
        let manifest: Manifest = serde_json::from_slice(&output.stdout)?;

        assert_eq!(manifest.client_version, V2);
        let runtimes = manifest
            .runtimes
            .iter()
            .map(|raw| {
                let runtime: Runtime = serde_json::from_value(raw.clone())?;
                assert_eq!(descriptor_key(raw), canonical_json(raw));
                assert_eq!(
                    raw.get("private_thermal"),
                    private_thermal.map(serde_json::Value::from).as_ref(),
                    "{raw}"
                );
                assert_eq!(
                    raw.get("private_memory"),
                    private_memory.map(serde_json::Value::from).as_ref(),
                    "{raw}"
                );
                Ok(runtime)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let tokens: Vec<&str> = runtimes.iter().map(Runtime::headless_token).collect();
        assert_eq!(
            tokens,
            [
                "llamacpp_ios_pipette",
                "llamacpp_ios_pipette",
                "mlx_ios_pipette",
                "onnx_genai_ios_pipette",
                "apple_foundation"
            ]
        );
        let llama: Vec<String> = runtimes
            .iter()
            .filter(|runtime| matches!(runtime, Runtime::LlamacppIosPipette(_)))
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            llama,
            [
                "github.com/ggml-org/llama.cpp@b11988:ios-arm64",
                "github.com/example-org/llama.cpp-asr@2898bdc6f:ios-arm64",
            ]
        );
        Ok(())
    }

    /// Runs `ios/release-manifest.py` for `V2` over the frameworks in `frameworks`.
    fn release_manifest(
        frameworks: &Path,
        flags: &[&str],
    ) -> std::io::Result<std::process::Output> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        std::process::Command::new("python3")
            .arg(root.join("ios/release-manifest.py"))
            .arg(V2)
            .args(flags)
            .arg("--frameworks")
            .arg(frameworks)
            .arg("--root")
            .arg(&root)
            .output()
    }

    /// A framework's `Info.plist`, as `ios/build-llama.sh` writes it or cut short.
    enum InfoPlist {
        Names(serde_json::Value),
        Truncated,
    }

    fn main_build() -> InfoPlist {
        InfoPlist::Names(serde_json::json!({
            "PipetteLlamaRepositoryUrl": "github.com/ggml-org/llama.cpp",
            "PipetteLlamaTag": "b11988",
            "PipetteLlamaCommit": "abb76ab38",
        }))
    }

    /// The script writes no manifest for a set of frameworks a job could not
    /// choose from, and names what is wrong: two builds with one identity, no
    /// main build, or an `Info.plist` it cannot read.
    #[rstest]
    #[case::two_builds_one_identity(
        vec![("llamacpp", main_build()), ("llamacpp-copy", main_build())],
        "llamacpp.framework and llamacpp-copy.framework are both \
         github.com/ggml-org/llama.cpp@b11988"
    )]
    #[case::no_main_build(vec![("llamacpp-asr", main_build())], "has no llamacpp.framework")]
    #[case::truncated_info_plist(
        vec![("llamacpp", InfoPlist::Truncated)],
        "llamacpp.framework/Info.plist: "
    )]
    fn the_release_manifest_script_refuses_frameworks_a_job_cannot_choose_from(
        #[case] built: Vec<(&str, InfoPlist)>,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let frameworks = tempfile::tempdir()?;
        built
            .into_iter()
            .try_for_each(|(name, info)| -> anyhow::Result<()> {
                let framework = frameworks.path().join(format!("{name}.framework"));
                std::fs::create_dir(&framework)?;
                let path = framework.join("Info.plist");
                match info {
                    InfoPlist::Names(names) => plist::to_file_xml(path, &names)?,
                    InfoPlist::Truncated => {
                        std::fs::write(path, "<?xml version=\"1.0\"?><plist><dict>")?
                    }
                }
                Ok(())
            })?;

        let output = release_manifest(frameworks.path(), &[])?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{output:?}");
        assert!(output.stdout.is_empty(), "{output:?}");
        assert!(stderr.contains(needle), "{stderr}");
        assert!(!stderr.contains("Traceback"), "{stderr}");
        Ok(())
    }

    /// A descriptor is spelled one way whatever its key order, aliases or
    /// defaulted fields; an unreadable one keeps what it says.
    #[test]
    fn a_descriptor_key_settles_the_spelling() {
        let written = serde_json::json!({
            "flavor": "ios-arm64",
            "version": "b10216",
            "source_repo": "https://github.com/ggml-org/llama.cpp",
            "private_thermal": false,
            "type": "llamacpp_ios_pipette",
        });
        assert_eq!(descriptor_key(&written), descriptor_key(&b10216()));

        let unknown = serde_json::json!({ "type": "x", "b": 1, "a": 2 });
        assert_eq!(descriptor_key(&unknown), r#"{"a":2,"b":1,"type":"x"}"#);
    }

    /// What a plan author copies: `type` first, and TOML a plan's `runtimes`
    /// reads back as the same runtime, nested tables included.
    #[rstest]
    #[case::llamacpp(
        serde_json::json!({
            "type": "llamacpp_ios_pipette",
            "repository_url": "github.com/ggml-org/llama.cpp",
            "repository_version": "b11349",
            "flavor": "ios-arm64",
            "private_thermal": true,
        }),
        concat!(
            r#"{ type = "llamacpp_ios_pipette", flavor = "ios-arm64", private_thermal = true, "#,
            r#"repository_url = "github.com/ggml-org/llama.cpp", repository_version = "b11349" }"#,
        )
    )]
    #[case::mlx(
        mlx(),
        concat!(
            r#"{ type = "mlx_ios_pipette", flavor = "ios-arm64", packages = { "#,
            r#"mlx_swift = { repository_url = "github.com/ml-explore/mlx-swift", "#,
            r#"repository_version = "0.21.2" }, "#,
            r#"mlx_swift_lm = { repository_url = "github.com/ml-explore/mlx-swift-examples", "#,
            r#"repository_version = "1.18.1" }, "#,
            r#"swift_transformers = { "#,
            r#"repository_url = "github.com/huggingface/swift-transformers", "#,
            r#"repository_version = "0.1.17" } } }"#,
        )
    )]
    #[case::apple_foundation(
        serde_json::json!({ "type": "apple_foundation" }),
        r#"{ type = "apple_foundation" }"#
    )]
    fn a_runtime_is_written_as_plan_toml(
        #[case] raw: serde_json::Value,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        #[derive(Deserialize)]
        struct Variant {
            runtimes: Vec<Runtime>,
        }
        let runtime: Runtime = serde_json::from_value(raw)?;

        let written = plan_toml(&runtime);

        assert_eq!(written, expected);
        let variant: Variant = toml::from_str(&format!("runtimes = [{written}]"))?;
        assert_eq!(variant.runtimes, [runtime]);
        Ok(())
    }

    /// Only the first token is read, so what follows it need not be text.
    #[test]
    fn a_sidecar_names_its_first_token() -> anyhow::Result<()> {
        let digest = sha256_of(b"zip");
        let mut body = format!("\n\t{digest}  pipette-ios-").into_bytes();
        body.extend(b"\xff.xcarchive.zip\n");
        assert_eq!(sidecar_digest(&body), Some(Sha256::try_new(digest)?));
        Ok(())
    }

    #[test]
    fn an_oversized_sidecar_is_skipped_unread() -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        let key = archive_key("tag", V2);
        let mut body = format!("{}  x\n", sha256_of(b"zip")).into_bytes();
        body.resize(5000, b' ');
        let sidecar = bucket.serve(&format!("{key}.sha256"), &body);
        let manifest = Published::build(&key, b"zip", 2).manifest;
        bucket.serve(&manifest.key, &manifest.body);
        bucket.list(
            &[
                Published::new(&key, b"zip", 2),
                Published::new(&format!("{key}.sha256"), &body, 2),
                manifest,
            ],
            1000,
        );

        assert_eq!(listed(&bucket)?, []);
        sidecar.assert_calls(0);
        Ok(())
    }

    /// The newest twenty and the default, older than those; only their
    /// sidecars and manifests are read, unless the listing reads the whole
    /// bucket. Only that one holds every build.
    #[rstest]
    #[case::newest(Reach::Newest, &[][..], false)]
    #[case::whole(Reach::Whole, &["v04", "v03", "v01", "v00"][..], true)]
    fn the_newest_builds_and_the_default_are_read(
        #[case] reach: Reach,
        #[case] older: &[&str],
        #[case] whole: bool,
    ) -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        let builds: Vec<_> = (0..25u8)
            .map(|n| {
                Published::build(
                    &archive_key(&format!("t{n:02}"), &format!("v{n:02}")),
                    b"zip",
                    n + 1,
                )
            })
            .collect();
        let sidecars: Vec<_> = builds
            .iter()
            .map(|build| bucket.serve(&build.sidecar.key, &build.sidecar.body))
            .collect();
        let manifests: Vec<_> = builds
            .iter()
            .map(|build| bucket.serve(&build.manifest.key, &build.manifest.body))
            .collect();
        let objects: Vec<Published> = builds
            .into_iter()
            .flat_map(|build| build.objects())
            .collect();
        bucket.list(&objects, 1000);

        let listed = list_newest(
            &s3::http_client()?,
            &bucket.config()?,
            20,
            Some(&"v02".parse()?),
            reach,
        )?;

        let mut expected: Vec<String> = (5..25).rev().map(|n| format!("v{n:02}")).collect();
        expected.push("v02".to_owned());
        assert_eq!(versions(&listed.newest), expected);
        assert_eq!(versions(&listed.older), older);
        assert_eq!(listed.whole, whole);
        sidecars
            .iter()
            .zip(&manifests)
            .enumerate()
            .for_each(|(n, (sidecar, manifest))| {
                let read = usize::from(n >= 5 || n == 2 || reach == Reach::Whole);
                sidecar.assert_calls(read);
                manifest.assert_calls(read);
            });
        Ok(())
    }

    /// A sidecar or manifest the bucket refuses to serve skips its build, not
    /// the listing. The catalog's listing skips it too on a 4xx, which repeats,
    /// and fails on a 5xx, which may clear.
    #[rstest]
    #[case::sidecar_forbidden(".xcarchive.zip.sha256", 403, Ok(&[V1][..]))]
    #[case::manifest_forbidden(".json", 403, Ok(&[V1][..]))]
    #[case::sidecar_unavailable(".xcarchive.zip.sha256", 503, Err("HTTP 503 Service Unavailable"))]
    #[case::manifest_failing(".json", 500, Err("HTTP 500 Internal Server Error"))]
    fn an_unreadable_sidecar_or_manifest_skips_only_its_build(
        #[case] refused_suffix: &str,
        #[case] status: u16,
        #[case] expected: Result<&[&str], &str>,
    ) -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        let bad = archive_key("b", V2);
        let refused = format!("b/pipette-ios-{V2}{refused_suffix}");
        let mut objects = Published::build(&archive_key("a", V1), b"zip", 1).objects();
        objects.extend(Published::build(&bad, b"zip", 2).objects());
        objects
            .iter()
            .filter(|object| object.key != refused)
            .for_each(|object| {
                bucket.serve(&object.key, &object.body);
            });
        bucket.server.mock(|when, then| {
            when.method(GET).path(format!("/builds/{refused}"));
            then.status(status);
        });
        bucket.list(&objects, 1000);

        assert_eq!(versions(&listed(&bucket)?), [V1]);
        match (
            list_newest(
                &s3::http_client()?,
                &bucket.config()?,
                20,
                None,
                Reach::Whole,
            ),
            expected,
        ) {
            (Ok(listing), Ok(listed)) => {
                assert_eq!(versions(&listing.newest), listed);
                assert!(!listing.whole, "a refused build is not read");
            }
            (Err(error), Err(reason)) => assert_eq!(
                format!("{error:#}"),
                format!("reading the .sha256 and .json of {bad}: {reason}")
            ),
            (got, expected) => anyhow::bail!("got {got:?}, expected {expected:?}"),
        }
        Ok(())
    }

    /// The oldest of two builds, how it sits in the bucket.
    #[derive(Debug, Clone, Copy)]
    enum Oldest {
        Readable,
        /// Its archive alone, still uploading.
        Uploading,
        /// Its archive and sidecar, the manifest not yet there.
        NoManifest,
        Unparseable,
        NamingAnotherVersion,
        /// Its manifest failing with a 503.
        Failing,
    }

    /// A listing holds every build only when it read each object that looks
    /// like one as one, past the newest (`keep` 1) or among them. A read that
    /// fails past the newest skips its build, not the listing.
    #[rstest]
    #[case::readable(Oldest::Readable, 1, true)]
    #[case::uploading(Oldest::Uploading, 1, false)]
    #[case::no_manifest(Oldest::NoManifest, 1, false)]
    #[case::unparseable(Oldest::Unparseable, 1, false)]
    #[case::unparseable_among_the_newest(Oldest::Unparseable, 20, false)]
    #[case::naming_another_version(Oldest::NamingAnotherVersion, 1, false)]
    #[case::failing(Oldest::Failing, 1, false)]
    fn a_whole_listing_reads_every_build(
        #[case] oldest: Oldest,
        #[case] keep: usize,
        #[case] whole: bool,
    ) -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        let mut objects = Published::build(&archive_key("b", V2), b"zip", 2).objects();
        let PublishedBuild {
            archive,
            sidecar,
            mut manifest,
        } = Published::build(&archive_key("a", V1), b"zip", 1);
        match oldest {
            Oldest::Unparseable => manifest.body = b"{".to_vec(),
            Oldest::NamingAnotherVersion => {
                manifest.body = serde_json::json!({ "client_version": V2, "runtimes": [] })
                    .to_string()
                    .into_bytes();
            }
            Oldest::Readable | Oldest::Uploading | Oldest::NoManifest | Oldest::Failing => {}
        }
        let failing = matches!(oldest, Oldest::Failing).then(|| manifest.key.clone());
        objects.push(archive);
        match oldest {
            Oldest::Uploading => {}
            Oldest::NoManifest => objects.push(sidecar),
            Oldest::Readable
            | Oldest::Unparseable
            | Oldest::NamingAnotherVersion
            | Oldest::Failing => objects.extend([sidecar, manifest]),
        }
        objects
            .iter()
            .filter(|object| failing.as_ref() != Some(&object.key))
            .for_each(|object| {
                bucket.serve(&object.key, &object.body);
            });
        if let Some(key) = &failing {
            bucket.server.mock(|when, then| {
                when.method(GET).path(format!("/builds/{key}"));
                then.status(503);
            });
        }
        bucket.list(&objects, 1000);

        let listing = list_newest(
            &s3::http_client()?,
            &bucket.config()?,
            keep,
            None,
            Reach::Whole,
        )?;

        assert_eq!(listing.whole, whole);
        let all: Vec<Build> = listing.newest.into_iter().chain(listing.older).collect();
        let expected: &[&str] = if whole { &[V2, V1] } else { &[V2] };
        assert_eq!(versions(&all), expected);
        Ok(())
    }

    /// One version under two keys names two sets of bytes, so neither lists,
    /// whatever the second copy's sidecar.
    #[rstest]
    #[case::both_summed(Some(format!("{}  x\n", sha256_of(b"zip")).into_bytes()))]
    #[case::no_sidecar(None)]
    #[case::bad_sidecar(Some(b"not a digest".to_vec()))]
    fn a_version_published_twice_is_skipped_everywhere(
        #[case] second_sidecar: Option<Vec<u8>>,
    ) -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        let second = archive_key("branches/b", V2);
        let mut objects = Published::build(&archive_key("a", V2), b"zip", 2).objects();
        objects.push(Published::new(&second, b"zip", 3));
        if let Some(body) = second_sidecar {
            objects.push(Published::new(&format!("{second}.sha256"), &body, 3));
        }
        objects.extend(Published::build(&archive_key("c", V1), b"zip", 1).objects());
        bucket.publish(&objects, 1000);

        assert_eq!(versions(&listed(&bucket)?), [V1]);
        Ok(())
    }

    /// With a good sidecar beside each key, only an archive with a canonical
    /// version in a `[A-Za-z0-9._-]+` directory and no dot segment can be a
    /// build. Judged on the listing alone: a request for some of these keys
    /// would not reach the path they name.
    #[rstest]
    #[case::release("2026.10.02-5c435ba/pipette-ios-1.0.xcarchive.zip", true)]
    #[case::branch("branches/feature_x/pipette-ios-x.1.xcarchive.zip", true)]
    #[case::dot_segment("tag/./pipette-ios-1.0.xcarchive.zip", false)]
    #[case::parent_segment("tag/../pipette-ios-1.0.xcarchive.zip", false)]
    #[case::empty_tag("tag//pipette-ios-1.0.xcarchive.zip", false)]
    #[case::tag_with_a_space("feature x/pipette-ios-1.0.xcarchive.zip", false)]
    #[case::tag_with_a_newline("a\n9.9  injected/pipette-ios-1.0.xcarchive.zip", false)]
    #[case::cli_release(
        "2026.10.02-5c435ba/pipette-cli-x86_64-unknown-linux-gnu.tar.gz",
        false
    )]
    #[case::android_release("2026.10.02-5c435ba/pipette-android-1.apk", false)]
    #[case::other_archive_format("2026.10.02-5c435ba/pipette-ios-1.xcarchive.tar", false)]
    #[case::notes("2026.10.02-5c435ba/notes.txt", false)]
    #[case::uppercase("tag/pipette-ios-2026.10.02-ABC.xcarchive.zip", false)]
    #[case::empty_version("tag/pipette-ios-.xcarchive.zip", false)]
    #[case::at_the_root("pipette-ios-1.0.xcarchive.zip", false)]
    fn only_a_key_that_names_a_build_is_a_candidate(
        #[case] key: &str,
        #[case] is_build: bool,
    ) -> anyhow::Result<()> {
        let objects = Published::build(key, b"zip", 2)
            .objects()
            .into_iter()
            .map(|published| -> anyhow::Result<s3::Object> {
                Ok(s3::Object {
                    key: published.key,
                    bytes: u64::try_from(published.body.len())?,
                    last_modified: datetime!(2026-10-02 12:00 UTC),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let keys: Vec<String> = candidates(&objects)
            .found
            .into_iter()
            .map(|candidate| candidate.key)
            .collect();
        assert_eq!(
            keys,
            if is_build {
                vec![key.to_owned()]
            } else {
                vec![]
            }
        );
        Ok(())
    }

    #[test]
    fn a_nested_branch_build_takes_its_own_directory_as_tag() -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        let key = "branches/feature-x/pipette-ios-feature-x.1.xcarchive.zip";
        bucket.publish(&Published::build(key, b"zip", 2).objects(), 1000);

        let builds = listed(&bucket)?;
        let tags: Vec<&str> = builds.iter().map(|build| build.tag.as_str()).collect();
        assert_eq!(tags, ["feature-x"]);
        assert_eq!(versions(&builds), ["feature-x.1"]);
        assert_eq!(builds.first().map(|build| build.key.as_str()), Some(key));
        Ok(())
    }

    /// One key per page, so the order holds across pages.
    #[test]
    fn builds_list_newest_first_then_by_version() -> anyhow::Result<()> {
        let bucket = FakeBucket::start();
        let pages = bucket.publish_builds(&[("a", "1.0", 1), ("b", "3.0", 1), ("c", "2.0", 2)], 1);

        assert_eq!(versions(&listed(&bucket)?), ["2.0", "3.0", "1.0"]);
        pages.iter().for_each(|page| page.assert_calls(1));
        Ok(())
    }
}
