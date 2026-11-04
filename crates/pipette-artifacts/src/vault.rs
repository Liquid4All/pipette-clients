//! Fetching a vault resource: a private artifact the management server serves
//! only to the devices of the principal that owns it.
//!
//! The record (digest and size) is read first, then the bytes are pulled and
//! verified. The device signature is the whole credential; the plan carries no
//! token.

use std::fs;
use std::path::{Path, PathBuf};

use pipette_http::HttpClient;
use pipette_http::retry::{Disposition, RetryPolicy, retrying};
use pipette_mgmt_client::{
    ByteStream, ClientIdentity, EntityTag, ResumeFrom, ServerUrl, VaultResource, client_api,
};
use pipette_plan_types::{Runtime, Sha256, UvRuntimeSource, VaultPath, VaultResourceId};

use crate::archive::{extract_archive, sniff_archive};
use crate::io_kind::names_the_host;
use crate::progress::Reporter;
use crate::resume::{self, Expect, Partial, Restart, ResumePoint, Source};

/// How a fetch signs as the device: the server it registered with and the key
/// it registered under. Absent from a workspace with no registration.
#[derive(Clone)]
pub struct VaultAccess {
    pub server: ServerUrl,
    pub identity: ClientIdentity,
}

/// Why a vault resource could not be fetched. [`Self::is_permanent`] is the
/// worker's disposition: a refusal that every device with this owner would
/// meet identically is not worth another claim.
#[derive(Debug, thiserror::Error)]
pub enum VaultFetchError {
    /// The workspace has no registered device identity to sign the fetch as.
    #[error(
        "`{url}` is a vault resource, which is fetched as a registered device, and this \
         workspace has no registered identity; run `pipette auth register` first"
    )]
    Unregistered { url: String },
    /// `403` (the device's owner does not own it, or the device belongs to
    /// nobody) or `404` (no such resource, or its object was removed).
    #[error("the management server refused vault resource {resource_id} with {status}: {detail}")]
    Refused {
        resource_id: VaultResourceId,
        status: u16,
        detail: String,
    },
    /// The plan pins one digest and the record another. Neither is trusted
    /// over the other: the plan is what was filed, the record is what will be
    /// served, and they have to agree before a byte is pulled.
    #[error(
        "vault resource {resource_id} is pinned to sha256 {record} by its record, but the plan \
         expects {plan}"
    )]
    PinDisagrees {
        resource_id: VaultResourceId,
        plan: String,
        record: String,
    },
    /// The bytes hash to something other than the record's pin: the object was
    /// replaced behind the record, which is an operator's repair.
    #[error(
        "vault resource {resource_id} downloaded complete but hashed to {actual}, not the \
         {expected} its record pins: the object was replaced behind the record and needs \
         re-registering"
    )]
    Drift {
        resource_id: VaultResourceId,
        expected: String,
        actual: String,
    },
    /// A resumed download hashed to something other than the record's pin. A
    /// range that the server sent did not continue the file, so the next
    /// attempt downloads the object from zero.
    #[error(
        "vault resource {resource_id} resumed at byte {resumed_at} and hashed to {actual}, not \
         the {expected} that its record pins"
    )]
    ResumedDrift {
        resource_id: VaultResourceId,
        resumed_at: u64,
        expected: String,
        actual: String,
    },
    /// Every other management-server failure: transport, `5xx`, `401`, an
    /// unreadable record.
    #[error(transparent)]
    Mgmt(pipette_mgmt_client::Error),
    #[error("writing {dest} failed")]
    Io {
        dest: String,
        #[source]
        source: std::io::Error,
    },
    /// The bytes verified but did not unpack: not an archive this crate reads,
    /// one that would not extract, or a disk that would not take the tree. The
    /// producer's problem except in the last case, which
    /// [`Self::is_permanent`] tells apart.
    #[error("unpacking vault resource {resource_id} failed")]
    Archive {
        resource_id: VaultResourceId,
        #[source]
        source: anyhow::Error,
    },
}

impl VaultFetchError {
    /// Whether every retry would fail the same way: the fetch is refused on
    /// ownership, the resource is gone, the pins disagree, or the bytes drifted.
    /// A transport failure, a `5xx` or a `401` is not.
    pub fn is_permanent(&self) -> bool {
        match self {
            Self::Refused { .. }
            | Self::PinDisagrees { .. }
            | Self::Drift { .. }
            | Self::Unregistered { .. } => true,
            // The producer's bytes, unless the disk refused them: the quota
            // pre-flight cannot size an unpacked tree, so a full disk is met here.
            Self::Archive { source, .. } => !host_refused(source),
            // A `401` is about this device's signature: a skewed clock, or a
            // nonce the server saw before. Another device signs its own.
            Self::Mgmt(source) => !source.is_transient() && source.http_status() != Some(401),
            Self::Io { .. } | Self::ResumedDrift { .. } => false,
        }
    }

    /// Whether another attempt in this process can succeed. A disk that refused
    /// the bytes is this host's, and refuses the next attempt too.
    fn disposition(&self) -> Disposition {
        match self {
            Self::Mgmt(source) if source.is_transient() || source.http_status() == Some(429) => {
                Disposition::TRANSIENT
            }
            Self::Io { source, .. } if !names_the_host(source) => Disposition::TRANSIENT,
            Self::ResumedDrift { .. } => Disposition::TRANSIENT,
            Self::Unregistered { .. }
            | Self::Refused { .. }
            | Self::PinDisagrees { .. }
            | Self::Drift { .. }
            | Self::Mgmt(_)
            | Self::Io { .. }
            | Self::Archive { .. } => Disposition::Permanent,
        }
    }
}

/// Whether a failure is this host's rather than the bytes': an `io::Error` in
/// the chain whose kind names a disk or a permission giving out. `tar` and
/// `zip` both keep the kind of the write that failed.
fn host_refused(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
        .any(names_the_host)
}

/// A `403` or `404` is a refusal with a disposition of its own; the rest stays
/// what the client reported.
fn refusal(resource_id: &VaultResourceId, error: pipette_mgmt_client::Error) -> VaultFetchError {
    match error.http_status() {
        Some(status @ (403 | 404)) => VaultFetchError::Refused {
            resource_id: resource_id.clone(),
            status,
            detail: error.to_string(),
        },
        _ => VaultFetchError::Mgmt(error),
    }
}

fn require<'a>(
    access: Option<&'a VaultAccess>,
    path: &VaultPath,
) -> Result<&'a VaultAccess, VaultFetchError> {
    access.ok_or_else(|| VaultFetchError::Unregistered {
        url: path.to_string(),
    })
}

/// The record behind `path`, with the plan's pin checked against it, retried
/// under `retry`.
pub(crate) fn resource(
    access: Option<&VaultAccess>,
    http: &HttpClient,
    retry: &RetryPolicy,
    path: &VaultPath,
    pinned: Option<&Sha256>,
) -> Result<VaultResource, VaultFetchError> {
    retrying(
        retry,
        &path.to_string(),
        |_| read_record(access, http, path, pinned),
        VaultFetchError::disposition,
    )
}

/// One read of the record behind `path`, with the plan's pin checked against it.
fn read_record(
    access: Option<&VaultAccess>,
    http: &HttpClient,
    path: &VaultPath,
    pinned: Option<&Sha256>,
) -> Result<VaultResource, VaultFetchError> {
    let resource_id = path.resource_id();
    let access = require(access, path)?;
    let record =
        client_api::get_vault_resource(&access.server, http, &access.identity, resource_id)
            .map_err(|error| refusal(resource_id, error))?;
    if let Some(plan) = pinned.filter(|plan| **plan != record.sha256) {
        return Err(VaultFetchError::PinDisagrees {
            resource_id: resource_id.clone(),
            plan: plan.to_string(),
            record: record.sha256.to_string(),
        });
    }
    Ok(record)
}

/// Fetch the resource at `path` into `dest`, verified against the record's pin.
///
/// The record is read first so a refusal costs no download and the plan's own
/// `sha256`, if any, is checked against the pin before a byte moves. A partial
/// file may remain at `dest`; the store's staging discards it.
///
/// A transient failure retries the whole fetch under `retry`, the record
/// included. A broken body resumes with an `If-Range` of the pin (see
/// [`resume`]); a resumed file that fails the pin is
/// [`VaultFetchError::ResumedDrift`], and the next attempt starts from zero.
pub(crate) fn fetch_to_file(
    access: Option<&VaultAccess>,
    http: &HttpClient,
    retry: &RetryPolicy,
    path: &VaultPath,
    pinned: Option<&Sha256>,
    dest: &Path,
    reporter: &mut Reporter,
) -> Result<VaultResource, VaultFetchError> {
    let start = reporter.mark();
    let mut partial = None;
    retrying(
        retry,
        &path.to_string(),
        |_| {
            let access = require(access, path)?;
            let record = read_record(Some(access), http, path, pinned)?;
            fetch_content(access, http, &record, dest, &mut partial, start, reporter)?;
            Ok(record)
        },
        VaultFetchError::disposition,
    )
}

/// A vault object, as the server side of a download.
struct VaultContent<'a> {
    access: &'a VaultAccess,
    http: &'a HttpClient,
    record: &'a VaultResource,
    dest: &'a Path,
}

impl Source for VaultContent<'_> {
    type Body = ByteStream;
    type Error = VaultFetchError;

    fn get(&self, resume: Option<&ResumePoint>) -> Result<ByteStream, VaultFetchError> {
        let resume = resume.map(|point| ResumeFrom {
            offset: point.offset,
            if_range: point.etag.clone(),
        });
        let resource_id = &self.record.resource_id;
        client_api::get_vault_content(
            &self.access.server,
            self.http,
            &self.access.identity,
            resource_id,
            resume.as_ref(),
        )
        .map_err(|error| refusal(resource_id, error))
    }

    fn range_refused(&self, error: &VaultFetchError) -> bool {
        matches!(error, VaultFetchError::Mgmt(source) if source.http_status() == Some(416))
    }

    fn expect(&self, _body: &ByteStream) -> Expect {
        expect_of(self.record)
    }

    fn io_error(&self, source: std::io::Error) -> VaultFetchError {
        VaultFetchError::Io {
            dest: self.dest.display().to_string(),
            source,
        }
    }
}

/// What a range must match to continue a file of `record`: the pin as a
/// strong `ETag`, and the size of the record.
fn expect_of(record: &VaultResource) -> Expect {
    Expect {
        etag: EntityTag::try_new(format!("\"{}\"", record.sha256)).ok(),
        total: Some(record.bytes),
    }
}

/// One attempt to stream the bytes that `record` pins into `dest`, and verify
/// them. A `partial` for another object is dropped; a file that fails its pin
/// does not stay in `partial`.
fn fetch_content(
    access: &VaultAccess,
    http: &HttpClient,
    record: &VaultResource,
    dest: &Path,
    partial: &mut Option<Partial>,
    start: u64,
    reporter: &mut Reporter,
) -> Result<(), VaultFetchError> {
    let _span = tracing::info_span!("vault download", resource_id = %record.resource_id).entered();
    let source = VaultContent {
        access,
        http,
        record,
        dest,
    };
    if let Some(file) = partial.take_if(|file| file.expect != expect_of(record))
        && file.writer.written() > 0
    {
        tracing::info!(reason = ?Restart::RecordChanged, "restarting download from zero");
    }
    reporter.set_total_if_unknown(Some(record.bytes));
    let file = resume::attempt(&source, dest, partial, start, reporter)?;
    // A body that ends before the record's size is this host's connection, not
    // the object, and the next attempt continues it.
    let written = file.writer.written();
    if written < record.bytes {
        *partial = Some(file);
        return Err(source.io_error(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("received {written} of {} bytes", record.bytes),
        )));
    }
    let actual = file.writer.hex_digest();
    if actual != record.sha256.as_ref() {
        let (resource_id, expected) = (record.resource_id.clone(), record.sha256.to_string());
        return Err(match file.resumed_at {
            Some(resumed_at) => VaultFetchError::ResumedDrift {
                resource_id,
                resumed_at,
                expected,
                actual,
            },
            None => VaultFetchError::Drift {
                resource_id,
                expected,
                actual,
            },
        });
    }
    Ok(())
}

/// Where an archive is written before it is unpacked: inside the directory it
/// fills, so a failed fetch is discarded with the store's staging directory.
const ARCHIVE_NAME: &str = ".vault-archive";

/// Fetch the archive at `path` and unpack it into `dest_dir`, whose root is
/// the archive's root; nothing is stripped. The kind (tar, tar.gz, zip) is read
/// off the bytes, since a vault path names no file.
pub(crate) fn fetch_and_unpack(
    access: Option<&VaultAccess>,
    http: &HttpClient,
    retry: &RetryPolicy,
    path: &VaultPath,
    pinned: Option<&Sha256>,
    dest_dir: &Path,
    reporter: &mut Reporter,
) -> Result<VaultResource, VaultFetchError> {
    let archive_path = dest_dir.join(ARCHIVE_NAME);
    let record = fetch_to_file(access, http, retry, path, pinned, &archive_path, reporter)?;
    let unpack = || -> anyhow::Result<()> {
        let mut file = fs::File::open(&archive_path)?;
        let kind = sniff_archive(&mut file)?;
        extract_archive(file, kind, dest_dir)?;
        fs::remove_file(&archive_path)?;
        Ok(())
    };
    unpack().map_err(|source| VaultFetchError::Archive {
        resource_id: record.resource_id.clone(),
        source,
    })?;
    Ok(record)
}

/// Where a wheelhouse is unpacked, beside the venv it feeds.
const WHEELHOUSE_DIR: &str = "wheelhouse";

/// The directory of wheels a uv-hosted `declared` installs from, unpacked
/// under `blobs_dir`; `None` for a public source or a non-uv runtime.
pub(crate) fn ensure_wheelhouse(
    access: Option<&VaultAccess>,
    http: &HttpClient,
    retry: &RetryPolicy,
    declared: &Runtime,
    blobs_dir: &Path,
    reporter: &mut Reporter,
) -> Result<Option<PathBuf>, VaultFetchError> {
    let Some(UvRuntimeSource::VaultWheelhouse { url: path, .. }) = declared.uv_source() else {
        return Ok(None);
    };
    let dir = blobs_dir.join(WHEELHOUSE_DIR);
    fetch_and_unpack(access, http, retry, path, None, &dir, reporter)?;
    Ok(Some(dir))
}

#[cfg(test)]
pub(crate) mod test_support {
    use httpmock::Mock;
    use httpmock::prelude::*;

    use pipette_mgmt_client::{ClientIdentity, ServerUrl, generate_keypair};
    use pipette_plan_types::{ClientId, VaultPath};

    use super::VaultAccess;

    pub(crate) const RESOURCE_ID: &str = "vault_0195c3e2";
    pub(crate) const BODY: &[u8] = b"abc";
    pub(crate) const BODY_SHA: &str =
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    pub(crate) fn access(server: &MockServer) -> anyhow::Result<VaultAccess> {
        access_at(server.base_url())
    }

    /// A device registered with the server at `base_url`.
    pub(crate) fn access_at(base_url: String) -> anyhow::Result<VaultAccess> {
        let (private_key_hex, _) = generate_keypair()?;
        Ok(VaultAccess {
            server: ServerUrl::try_new(base_url)?,
            identity: ClientIdentity {
                client_id: ClientId::try_new("ev1_test")?,
                private_key_hex,
            },
        })
    }

    pub(crate) fn content_path(id: &str) -> anyhow::Result<VaultPath> {
        Ok(format!("/vault/resources/{id}/content").parse()?)
    }

    /// A ustar archive holding one regular file, the smallest thing that
    /// unpacks to a checkable tree.
    pub(crate) fn tar_with(name: &str, contents: &[u8]) -> anyhow::Result<Vec<u8>> {
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_ustar();
        header.set_size(contents.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, name, contents)?;
        Ok(builder.into_inner()?)
    }

    /// [`serve`] with the pin computed from `body`.
    pub(crate) fn serve_bytes<'s>(
        server: &'s MockServer,
        id: &str,
        body: &[u8],
    ) -> (Mock<'s>, Mock<'s>) {
        serve(server, id, &sha_of(body), body)
    }

    /// The sha256 of `body`, in lowercase hex.
    pub(crate) fn sha_of(body: &[u8]) -> String {
        use sha2::Digest;
        hex::encode(sha2::Sha256::digest(body))
    }

    /// A record pinning `sha` over `bytes` bytes, and its content.
    pub(crate) fn serve<'s>(
        server: &'s MockServer,
        id: &str,
        sha: &str,
        body: &[u8],
    ) -> (Mock<'s>, Mock<'s>) {
        let record = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/vault/resources/{id}"))
                .header_exists("X-Signature");
            then.status(200).body(format!(
                r#"{{"resource_id":"{id}","name":"ckpt","sha256":"{sha}","bytes":{}}}"#,
                body.len()
            ));
        });
        let content = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/vault/resources/{id}/content"))
                .header_exists("X-Signature");
            then.status(200).body(body);
        });
        (record, content)
    }
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;
    use rstest::rstest;

    use pipette_plan_types::Sha256;

    use super::test_support::*;
    use super::*;
    use crate::scripted_server::{Reply, ScriptedServer};

    fn http() -> anyhow::Result<HttpClient> {
        Ok(HttpClient::new("pipette-test")?)
    }

    /// One fetch of [`RESOURCE_ID`] into a scratch file, as the device `access`
    /// names (and so from the server it is registered with).
    fn fetch(
        access: Option<&VaultAccess>,
        pinned: Option<&Sha256>,
    ) -> anyhow::Result<Result<(VaultResource, Vec<u8>), VaultFetchError>> {
        let dir = tempfile::tempdir()?;
        let dest = dir.path().join("m.gguf");
        let result = fetch_to_file(
            access,
            &http()?,
            &RetryPolicy::IMMEDIATE,
            &content_path(RESOURCE_ID)?,
            pinned,
            &dest,
            &mut Reporter::silent(),
        );
        Ok(match result {
            Ok(record) => Ok((record, std::fs::read(&dest)?)),
            Err(err) => Err(err),
        })
    }

    #[test]
    fn a_fetch_reads_the_record_then_streams_and_verifies() -> anyhow::Result<()> {
        let server = MockServer::start();
        let (record, content) = serve(&server, RESOURCE_ID, BODY_SHA, BODY);

        let (got, bytes) = fetch(Some(&access(&server)?), None)??;

        record.assert();
        content.assert();
        assert_eq!(got.sha256.as_ref(), BODY_SHA);
        assert_eq!(bytes, BODY);
        Ok(())
    }

    /// A plan pin that disagrees with the record is refused before a byte is
    /// pulled; one that agrees is redundant and fine.
    #[rstest]
    #[case::agrees(BODY_SHA, true)]
    #[case::disagrees(
        "0000000000000000000000000000000000000000000000000000000000000000",
        false
    )]
    fn the_plan_pin_is_checked_against_the_record(
        #[case] pin: &str,
        #[case] ok: bool,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        let (_, content) = serve(&server, RESOURCE_ID, BODY_SHA, BODY);
        let pinned = Sha256::try_new(pin.to_owned())?;

        let result = fetch(Some(&access(&server)?), Some(&pinned))?;

        assert_eq!(result.is_ok(), ok);
        if !ok {
            let err = result.err().map(|e| e.to_string()).unwrap_or_default();
            assert!(err.contains("pinned to sha256"), "{err}");
            content.assert_calls(0);
        }
        Ok(())
    }

    /// Bytes that hash to something else are the object replaced behind the
    /// record: a permanent refusal that names the repair.
    #[test]
    fn drifted_bytes_are_refused_permanently() -> anyhow::Result<()> {
        let server = MockServer::start();
        serve(&server, RESOURCE_ID, BODY_SHA, b"not abc");

        let err = fetch(Some(&access(&server)?), None)?
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected a drift error"))?;

        assert!(matches!(err, VaultFetchError::Drift { .. }), "{err}");
        assert!(err.is_permanent());
        assert!(err.to_string().contains("re-registering"), "{err}");
        Ok(())
    }

    #[test]
    fn a_short_body_is_retriable_not_drift() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET)
                .path(format!("/vault/resources/{RESOURCE_ID}"));
            then.status(200).body(format!(
                r#"{{"resource_id":"{RESOURCE_ID}","name":"ckpt","sha256":"{BODY_SHA}","bytes":{}}}"#,
                BODY.len()
            ));
        });
        server.mock(|when, then| {
            when.method(GET)
                .path(format!("/vault/resources/{RESOURCE_ID}/content"));
            then.status(200).body(&BODY[..BODY.len() - 1]);
        });

        let err = fetch(Some(&access(&server)?), None)?
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected a short-body error"))?;

        assert!(matches!(err, VaultFetchError::Io { .. }), "{err}");
        assert!(!err.is_permanent());
        Ok(())
    }

    /// `403` and `404` on the record are permanent, asked once, and never
    /// reach the content route. A `401` is retriable on another claim, but this
    /// host signs its next attempt the same way, so it is asked once here. A
    /// `503` is transient and asked on every attempt.
    #[rstest]
    #[case(403, true, 1)]
    #[case(404, true, 1)]
    #[case(401, false, 1)]
    #[case(503, false, RetryPolicy::IMMEDIATE.attempts as usize)]
    fn a_record_refusal_maps_to_its_disposition(
        #[case] status: u16,
        #[case] permanent: bool,
        #[case] asked: usize,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        let record = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/vault/resources/{RESOURCE_ID}"));
            then.status(status).body("no");
        });
        let content = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/vault/resources/{RESOURCE_ID}/content"));
            then.status(200).body(BODY);
        });

        let err = fetch(Some(&access(&server)?), None)?
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected a refusal"))?;

        assert_eq!(err.is_permanent(), permanent, "{err}");
        assert_eq!(
            matches!(err, VaultFetchError::Refused { .. }),
            permanent,
            "{err}"
        );
        record.assert_calls(asked);
        content.assert_calls(0);
        Ok(())
    }

    /// A `503` on the record and a content body cut short are each retried,
    /// the record read again each time, until the bytes arrive and verify.
    #[test]
    fn a_transient_failure_retries_the_whole_fetch() -> anyhow::Result<()> {
        let record = format!(
            r#"{{"resource_id":"{RESOURCE_ID}","name":"ckpt","sha256":"{BODY_SHA}","bytes":3}}"#
        );
        let server = ScriptedServer::start(vec![
            Reply::status(503),
            Reply::ok(record.as_bytes()),
            Reply::cut(BODY, 1),
            Reply::ok(record.as_bytes()),
            Reply::ok(BODY),
        ])?;
        let (_, bytes) = fetch(Some(&access_at(server.url(""))?), None)??;
        assert_eq!(bytes, BODY);
        assert_eq!(server.requests().len(), 5);
        Ok(())
    }

    /// A body long enough to cut at byte 100.
    fn long_body() -> Vec<u8> {
        (0..=255).collect()
    }

    /// The `ETag` the server sends for `body`: its sha256, quoted.
    fn etag_of(body: &[u8]) -> String {
        format!("\"{}\"", sha_of(body))
    }

    /// The record reply for [`RESOURCE_ID`], pinning `body`.
    fn record_of(body: &[u8]) -> Reply {
        Reply::ok(
            format!(
                r#"{{"resource_id":"{RESOURCE_ID}","name":"ckpt","sha256":"{}","bytes":{}}}"#,
                sha_of(body),
                body.len()
            )
            .as_bytes(),
        )
    }

    /// [`long_body`] cut after 100 bytes, under its pin.
    fn cut_at_100() -> Reply {
        Reply::Cut {
            status: 200,
            headers: vec![("etag", etag_of(&long_body()))],
            body: long_body(),
            sent: 100,
        }
    }

    /// A `206` that sends `body` under `etag` and claims `content_range`.
    fn partial_reply(body: Vec<u8>, etag: String, content_range: &str) -> Reply {
        Reply::Full {
            status: 206,
            headers: vec![("etag", etag), ("content-range", content_range.to_owned())],
            body,
        }
    }

    /// The `206` that continues [`cut_at_100`].
    fn rest_from_100() -> Reply {
        let body = long_body();
        partial_reply(body[100..].to_vec(), etag_of(&body), "bytes 100-255/256")
    }

    /// The result of a scripted fetch: the file, the request headers in
    /// lowercase, and each progress count that the sink received.
    struct Fetched {
        file: Vec<u8>,
        requests: Vec<String>,
        progress: Vec<u64>,
    }

    /// Fetch [`RESOURCE_ID`] from a server that answers `script`.
    fn fetch_scripted(script: Vec<Reply>) -> anyhow::Result<Fetched> {
        use std::sync::{Arc, Mutex};

        use crate::progress::{FetchProgress, ProgressSink};

        let server = ScriptedServer::start(script)?;
        let dir = tempfile::tempdir()?;
        let dest = dir.path().join("m.gguf");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink_seen = Arc::clone(&seen);
        let sink: ProgressSink = Arc::new(move |p: FetchProgress<'_>| {
            if let Ok(mut seen) = sink_seen.lock() {
                seen.push(p.done_bytes);
            }
        });
        fetch_to_file(
            Some(&access_at(server.url(""))?),
            &http()?,
            &RetryPolicy::IMMEDIATE,
            &content_path(RESOURCE_ID)?,
            None,
            &dest,
            &mut Reporter::new(Some(sink), "m".to_owned(), None),
        )?;
        let progress = seen
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned"))?
            .clone();
        Ok(Fetched {
            file: std::fs::read(&dest)?,
            requests: server
                .requests()
                .iter()
                .map(|head| head.to_ascii_lowercase())
                .collect(),
            progress,
        })
    }

    /// [`long_body`] in reverse: another object of the same size.
    fn reversed_body() -> Vec<u8> {
        long_body().into_iter().rev().collect()
    }

    /// The index of the request that continues [`cut_at_100`] in a script
    /// from [`after_cut`].
    const RESUME_REQUEST: usize = 3;

    /// A script that serves the record of [`long_body`], cuts its body at byte
    /// 100, serves the record again, then answers with `tail`.
    fn after_cut(tail: Vec<Reply>) -> Vec<Reply> {
        [
            record_of(&long_body()),
            cut_at_100(),
            record_of(&long_body()),
        ]
        .into_iter()
        .chain(tail)
        .collect()
    }

    /// Any answer but a `206` for the same object starts from zero, and no later
    /// request asks for the range again. A record that pins another object asks
    /// for no range at all.
    #[rstest]
    #[case::resumed(after_cut(vec![rest_from_100()]), long_body(), true)]
    #[case::whole_file_sent(after_cut(vec![Reply::ok(&long_body())]), long_body(), true)]
    #[case::range_refused(
        after_cut(vec![Reply::status(416), Reply::ok(&long_body())]),
        long_body(),
        true
    )]
    #[case::another_etag(
        after_cut(vec![
            partial_reply(long_body()[100..].to_vec(), "\"other\"".to_owned(), "bytes 100-255/256"),
            Reply::ok(&long_body()),
        ]),
        long_body(),
        true
    )]
    #[case::another_offset(
        after_cut(vec![
            partial_reply(long_body(), etag_of(&long_body()), "bytes 0-255/256"),
            Reply::ok(&long_body()),
        ]),
        long_body(),
        true
    )]
    #[case::another_total(
        after_cut(vec![
            partial_reply(long_body()[100..].to_vec(), etag_of(&long_body()), "bytes 100-255/300"),
            Reply::ok(&long_body()),
        ]),
        long_body(),
        true
    )]
    #[case::weak_etag(
        after_cut(vec![
            partial_reply(
                long_body()[100..].to_vec(),
                format!("W/{}", etag_of(&long_body())),
                "bytes 100-255/256",
            ),
            Reply::ok(&long_body()),
        ]),
        long_body(),
        true
    )]
    #[case::no_content_range(
        after_cut(vec![
            Reply::Full {
                status: 206,
                headers: vec![("etag", etag_of(&long_body()))],
                body: long_body()[100..].to_vec(),
            },
            Reply::ok(&long_body()),
        ]),
        long_body(),
        true
    )]
    #[case::unknown_total(
        after_cut(vec![
            partial_reply(long_body()[100..].to_vec(), etag_of(&long_body()), "bytes 100-255/*"),
            Reply::ok(&long_body()),
        ]),
        long_body(),
        true
    )]
    // The `206` passes the resume checks but carries other bytes, so the
    // resumed file fails its pin and the next attempt starts from zero.
    #[case::resumed_then_fails_its_pin(
        after_cut(vec![
            partial_reply(vec![0; 156], etag_of(&long_body()), "bytes 100-255/256"),
            record_of(&long_body()),
            Reply::ok(&long_body()),
        ]),
        long_body(),
        true
    )]
    #[case::record_changed(
        vec![
            record_of(&long_body()),
            cut_at_100(),
            record_of(&reversed_body()),
            Reply::ok(&reversed_body()),
        ],
        reversed_body(),
        false
    )]
    fn a_broken_vault_body_resumes_only_where_the_server_continues_it(
        #[case] script: Vec<Reply>,
        #[case] expected: Vec<u8>,
        #[case] asks_for_a_range: bool,
    ) -> anyhow::Result<()> {
        let requests = script.len();
        let fetched = fetch_scripted(script)?;
        assert_eq!(fetched.file, expected);
        assert_eq!(fetched.requests.len(), requests, "{:?}", fetched.requests);
        let resumed = fetched
            .requests
            .get(RESUME_REQUEST)
            .map(String::as_str)
            .unwrap_or_default();
        let pin = etag_of(&expected).to_ascii_lowercase();
        assert_eq!(
            resumed.contains("range: bytes=100-"),
            asks_for_a_range,
            "{resumed}"
        );
        assert_eq!(
            resumed.contains(&format!("if-range: {pin}")),
            asks_for_a_range,
            "{resumed}"
        );
        // Only the request that continues the cut asks for a range.
        assert!(
            fetched
                .requests
                .iter()
                .enumerate()
                .filter(|(n, _)| *n != RESUME_REQUEST)
                .all(|(_, request)| !request.contains("range:")),
            "{:?}",
            fetched.requests
        );
        assert_eq!(
            fetched.progress.last(),
            Some(&256),
            "{:?}",
            fetched.progress
        );
        assert!(
            fetched.progress.iter().all(|done| *done <= 256),
            "{:?}",
            fetched.progress
        );
        Ok(())
    }

    #[test]
    fn a_range_answer_that_stops_short_asks_for_the_next_range() -> anyhow::Result<()> {
        let body = long_body();
        let fetched = fetch_scripted(vec![
            record_of(&body),
            cut_at_100(),
            record_of(&body),
            partial_reply(body[100..200].to_vec(), etag_of(&body), "bytes 100-199/256"),
            partial_reply(body[200..].to_vec(), etag_of(&body), "bytes 200-255/256"),
        ])?;
        assert_eq!(fetched.file, body);
        assert_eq!(fetched.requests.len(), 5, "{:?}", fetched.requests);
        let ranges: Vec<bool> = ["range: bytes=100-", "range: bytes=200-"]
            .iter()
            .zip(fetched.requests.iter().skip(3))
            .map(|(range, request)| request.contains(range))
            .collect();
        assert_eq!(ranges, vec![true, true], "{:?}", fetched.requests);
        Ok(())
    }

    #[test]
    fn a_failed_range_request_asks_for_the_range_again() -> anyhow::Result<()> {
        let body = long_body();
        let fetched = fetch_scripted(vec![
            record_of(&body),
            cut_at_100(),
            record_of(&body),
            Reply::status(503),
            record_of(&body),
            rest_from_100(),
        ])?;
        assert_eq!(fetched.file, body);
        assert_eq!(fetched.requests.len(), 6, "{:?}", fetched.requests);
        let ranged: Vec<usize> = fetched
            .requests
            .iter()
            .enumerate()
            .filter(|(_, request)| request.contains("range: bytes=100-"))
            .map(|(n, _)| n)
            .collect();
        assert_eq!(ranged, vec![3, 5], "{:?}", fetched.requests);
        Ok(())
    }

    fn write_failure(kind: std::io::ErrorKind) -> VaultFetchError {
        VaultFetchError::Io {
            dest: "m.gguf".to_owned(),
            source: kind.into(),
        }
    }

    fn mgmt_status(status: u16) -> VaultFetchError {
        VaultFetchError::Mgmt(pipette_mgmt_client::Error::HttpStatus {
            method: "GET".to_owned(),
            url: "/vault/resources/vault_0195c3e2".to_owned(),
            status,
            body: String::new(),
        })
    }

    fn resumed_drift() -> anyhow::Result<VaultFetchError> {
        Ok(VaultFetchError::ResumedDrift {
            resource_id: VaultResourceId::try_new(RESOURCE_ID)?,
            resumed_at: 100,
            expected: BODY_SHA.to_owned(),
            actual: "0".repeat(64),
        })
    }

    #[rstest]
    #[case::broken_read(
        write_failure(std::io::ErrorKind::ConnectionReset),
        Disposition::TRANSIENT
    )]
    #[case::full_disk(write_failure(std::io::ErrorKind::StorageFull), Disposition::Permanent)]
    #[case::rate_limited(mgmt_status(429), Disposition::TRANSIENT)]
    #[case::unavailable(mgmt_status(503), Disposition::TRANSIENT)]
    #[case::bad_request(mgmt_status(400), Disposition::Permanent)]
    #[case::resumed_drift(resumed_drift()?, Disposition::TRANSIENT)]
    fn a_failure_is_retried_only_when_it_can_pass(
        #[case] err: VaultFetchError,
        #[case] expected: Disposition,
    ) -> anyhow::Result<()> {
        assert_eq!(err.disposition(), expected, "{err}");
        Ok(())
    }

    /// A resumed file that fails its pin may hold bytes that broke in transit,
    /// so the job stays open to another worker. That worker starts from zero.
    #[test]
    fn a_resumed_drift_is_not_permanent() -> anyhow::Result<()> {
        let err = resumed_drift()?;
        assert!(!err.is_permanent(), "{err}");
        Ok(())
    }

    /// A record whose object is gone answers `404` on the content route alone;
    /// that refusal is as permanent as one on the record.
    #[test]
    fn a_content_refusal_after_a_good_record_is_permanent() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET)
                .path(format!("/vault/resources/{RESOURCE_ID}"));
            then.status(200).body(format!(
                r#"{{"resource_id":"{RESOURCE_ID}","name":"ckpt","sha256":"{BODY_SHA}","bytes":3}}"#
            ));
        });
        server.mock(|when, then| {
            when.method(GET)
                .path(format!("/vault/resources/{RESOURCE_ID}/content"));
            then.status(404).body("vault object not found");
        });

        let err = fetch(Some(&access(&server)?), None)?
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected a refusal"))?;
        assert!(
            matches!(&err, VaultFetchError::Refused { status: 404, .. }),
            "{err}"
        );
        assert!(err.is_permanent());
        Ok(())
    }

    #[test]
    fn an_unregistered_workspace_is_named() -> anyhow::Result<()> {
        let err = fetch(None, None)?
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected a refusal"))?;
        assert!(matches!(err, VaultFetchError::Unregistered { .. }), "{err}");
        assert!(err.to_string().contains("pipette auth register"), "{err}");
        Ok(())
    }

    /// The archive file is removed; bytes that are no archive are a permanent
    /// refusal.
    #[test]
    fn an_archive_is_unpacked_in_place() -> anyhow::Result<()> {
        let server = MockServer::start();
        serve_bytes(&server, RESOURCE_ID, &tar_with("config.json", b"{}")?);
        let dir = tempfile::tempdir()?;
        fetch_and_unpack(
            Some(&access(&server)?),
            &http()?,
            &RetryPolicy::IMMEDIATE,
            &content_path(RESOURCE_ID)?,
            None,
            dir.path(),
            &mut Reporter::silent(),
        )?;
        assert_eq!(std::fs::read(dir.path().join("config.json"))?, b"{}");
        assert!(!dir.path().join(ARCHIVE_NAME).exists());

        let server = MockServer::start();
        serve(&server, RESOURCE_ID, BODY_SHA, BODY);
        let err = fetch_and_unpack(
            Some(&access(&server)?),
            &http()?,
            &RetryPolicy::IMMEDIATE,
            &content_path(RESOURCE_ID)?,
            None,
            dir.path(),
            &mut Reporter::silent(),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected a refusal"))?;
        assert!(matches!(err, VaultFetchError::Archive { .. }), "{err}");
        assert!(err.is_permanent());
        Ok(())
    }

    #[rstest]
    #[case::disk_full(Some(std::io::ErrorKind::StorageFull), false)]
    #[case::quota(Some(std::io::ErrorKind::QuotaExceeded), false)]
    #[case::unwritable(Some(std::io::ErrorKind::PermissionDenied), false)]
    #[case::corrupt_gzip(Some(std::io::ErrorKind::InvalidData), true)]
    #[case::not_an_archive(None, true)]
    fn an_unpack_the_disk_refused_is_not_permanent(
        #[case] kind: Option<std::io::ErrorKind>,
        #[case] permanent: bool,
    ) -> anyhow::Result<()> {
        let source = match kind {
            Some(kind) => anyhow::Error::from(std::io::Error::new(kind, "failed to unpack"))
                .context("failed to unpack into /store/staging"),
            None => anyhow::anyhow!("the artifact is not a tar, tar.gz or zip archive"),
        };
        let err = VaultFetchError::Archive {
            resource_id: VaultResourceId::try_new(RESOURCE_ID)?,
            source,
        };
        assert_eq!(err.is_permanent(), permanent, "{err:#}");
        Ok(())
    }

    /// Only a wheelhouse source unpacks anything; a public one and a non-uv
    /// runtime cost no request.
    #[test]
    fn only_a_wheelhouse_source_is_unpacked() -> anyhow::Result<()> {
        use pipette_plan_types::{
            NonEmptyString, UvBuild, UvPythonVersion, UvServerVersion, UvVllm,
        };

        let server = MockServer::start();
        let (record, content) = serve_bytes(
            &server,
            RESOURCE_ID,
            &tar_with("vllm-0.10.0-py3-none-any.whl", b"PK")?,
        );
        let uv_vllm = |source| -> anyhow::Result<Runtime> {
            Ok(Runtime::UvVllm(UvVllm {
                server_version: UvServerVersion::try_new("0.10.0".to_owned())?,
                build: UvBuild::try_new("cu121".to_owned())?,
                python_version: UvPythonVersion::try_new("3.12".to_owned())?,
                source,
            }))
        };
        let contents = NonEmptyString::try_new("vllm==0.10.0\n".to_owned())?;
        let dir = tempfile::tempdir()?;
        let device = access(&server)?;

        let wheelhouse = ensure_wheelhouse(
            Some(&device),
            &http()?,
            &RetryPolicy::IMMEDIATE,
            &uv_vllm(UvRuntimeSource::VaultWheelhouse {
                contents: contents.clone(),
                url: content_path(RESOURCE_ID)?,
                principal_id: "org_1".parse()?,
                custom_label: None,
                install_flags: None,
            })?,
            dir.path(),
            &mut Reporter::silent(),
        )?;
        assert_eq!(
            wheelhouse.as_deref(),
            Some(dir.path().join(WHEELHOUSE_DIR).as_path())
        );
        assert!(
            dir.path()
                .join(WHEELHOUSE_DIR)
                .join("vllm-0.10.0-py3-none-any.whl")
                .exists()
        );
        record.assert_calls(1);
        content.assert_calls(1);

        let public = ensure_wheelhouse(
            Some(&device),
            &http()?,
            &RetryPolicy::IMMEDIATE,
            &uv_vllm(UvRuntimeSource::PipRequirementsText {
                contents,
                install_flags: None,
            })?,
            dir.path(),
            &mut Reporter::silent(),
        )?;
        assert!(public.is_none());
        record.assert_calls(1);
        Ok(())
    }

    /// The record is what the quota pre-flight sizes from, and reading it
    /// fails as the fetch would when there is no registration to ask with.
    #[test]
    fn the_size_comes_from_the_record() -> anyhow::Result<()> {
        let server = MockServer::start();
        serve(&server, RESOURCE_ID, BODY_SHA, BODY);
        let path = content_path(RESOURCE_ID)?;
        assert_eq!(
            resource(
                Some(&access(&server)?),
                &http()?,
                &RetryPolicy::IMMEDIATE,
                &path,
                None
            )?
            .bytes,
            3
        );
        assert!(matches!(
            resource(None, &http()?, &RetryPolicy::IMMEDIATE, &path, None),
            Err(VaultFetchError::Unregistered { .. })
        ));
        Ok(())
    }
}
