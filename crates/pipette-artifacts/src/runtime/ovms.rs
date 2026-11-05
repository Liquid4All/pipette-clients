//! Release-archive install for the `ovms` runtime: an OpenVINO Model Server
//! release unpacked under the store-owned `blobs/` payload dir.
//!
//! Not gated at compile time. Off Linux and Windows the cell is refused before
//! this runs, where the client builds it (`pipette-cli`'s `refuse_foreign_os`).
//! A named flavor is an archive built for one of the two, so a pull on the
//! other is refused here before anything downloads.

use std::fs;
use std::io::Write;
use std::path::Path;

use pipette_artifacts_catalog::runtime::ovms::layout::OvmsInstall;
use pipette_artifacts_catalog::runtime::ovms::{pinned_sha256, upstream_repository_url};
use pipette_http::HttpClient;
use pipette_http::retry::{Disposition, RetryPolicy, retrying};
use pipette_plan_types::{
    OvmsFlavor, OvmsSource, OvmsSourceRepository, RepositoryUrl, Runtime, Sha256,
};

use crate::archive::{extract_archive, sniff_archive};
use crate::digest::HashingWriter;
use crate::io_kind::names_the_host;
use crate::progress::{Reporter, copy_reporting};
use crate::vault::{self, VaultAccess};

/// Where the archive lands before it is unpacked: inside the directory it
/// fills, so a failed pull is discarded with the store's staging directory.
const ARCHIVE_NAME: &str = ".ovms-archive";

#[derive(Debug, thiserror::Error)]
pub enum OvmsInstallError {
    #[error("{0}")]
    NotInstallable(String),
    #[error("ovms flavor `{flavor}` is built for {built_for}, but this host is {host}")]
    ForeignFlavor {
        flavor: String,
        built_for: String,
        host: String,
    },
    #[error(
        "ovms flavor `{flavor}` names no upstream release asset at `{tag}`; install from an \
         archive URL instead"
    )]
    NoReleaseAsset { flavor: String, tag: String },
    #[error("downloading the archive from {url} failed")]
    Http {
        url: String,
        #[source]
        source: pipette_http::Error,
    },
    #[error("writing the archive from {url} failed")]
    Io {
        url: String,
        #[source]
        source: std::io::Error,
    },
    #[error("the archive from {url} hashes to {actual}, but the bundled catalog pins {expected}")]
    Sha256Mismatch {
        url: String,
        expected: String,
        actual: String,
    },
    #[error("the ovms archive unpacked no server at {0}; its top-level `ovms/` is the payload")]
    MissingExecutable(String),
}

impl OvmsInstallError {
    /// Whether another download attempt can succeed: the request got no
    /// answer or a `5xx` or `429`, or the body broke part-way. `Io` covers
    /// the body read and the disk write, so its kind decides.
    fn disposition(&self) -> Disposition {
        match self {
            Self::Http { source, .. } => source.disposition(),
            Self::Io { source, .. } if !names_the_host(source) => Disposition::TRANSIENT,
            Self::NotInstallable(_)
            | Self::ForeignFlavor { .. }
            | Self::NoReleaseAsset { .. }
            | Self::Io { .. }
            | Self::Sha256Mismatch { .. }
            | Self::MissingExecutable(_) => Disposition::Permanent,
        }
    }
}

/// Download, verify and unpack an OVMS release into `blobs_dir`.
pub(crate) fn install_ovms_archive(
    http: &HttpClient,
    retry: &RetryPolicy,
    vault: Option<&VaultAccess>,
    declared: &Runtime,
    blobs_dir: &Path,
    reporter: &mut Reporter,
) -> anyhow::Result<()> {
    let Runtime::Ovms(rt) = declared else {
        return Err(OvmsInstallError::NotInstallable(format!(
            "install_ovms_archive only installs ovms, got `{}`",
            declared.headless_token()
        ))
        .into());
    };
    refuse_foreign_flavor(&rt.flavor, std::env::consts::OS, std::env::consts::ARCH)?;
    match &rt.source {
        OvmsSource::GithubRelease(repo) => {
            let (url, pinned) = release_download(repo, &rt.flavor)?;
            download_and_unpack(http, retry, &url, pinned, blobs_dir, reporter)?;
        }
        OvmsSource::RemoteArchive { url } => {
            download_and_unpack(http, retry, url.as_ref(), None, blobs_dir, reporter)?;
        }
        // Verified against its vault record rather than the catalog.
        OvmsSource::VaultRemoteArchive { url, .. } => {
            vault::fetch_and_unpack(vault, http, retry, url, None, blobs_dir, reporter)?;
        }
        OvmsSource::RelativeDir { .. } | OvmsSource::AbsoluteDir { .. } => {
            return Err(OvmsInstallError::NotInstallable(format!(
                "cannot pull an installed ovms form ({}); pass its declared github_release, \
                 remote_archive or vault_remote_archive source instead",
                rt.source
            ))
            .into());
        }
    }
    check_payload(blobs_dir)?;
    Ok(())
}

/// The `(os, arch)` a named flavor's archive runs on, spelled as
/// [`std::env::consts`] spells them; `None` for a custom build.
fn built_for(flavor: &OvmsFlavor) -> Option<(&'static str, &'static str)> {
    match flavor {
        OvmsFlavor::WindowsX64 => Some(("windows", "x86_64")),
        OvmsFlavor::Ubuntu22X64 | OvmsFlavor::Ubuntu24X64 | OvmsFlavor::RedhatX64 => {
            Some(("linux", "x86_64"))
        }
        OvmsFlavor::Custom(_) => None,
    }
}

fn refuse_foreign_flavor(
    flavor: &OvmsFlavor,
    host_os: &str,
    host_arch: &str,
) -> Result<(), OvmsInstallError> {
    match built_for(flavor) {
        Some((os, arch)) if (os, arch) != (host_os, host_arch) => {
            Err(OvmsInstallError::ForeignFlavor {
                flavor: flavor.to_string(),
                built_for: format!("{os}-{arch}"),
                host: format!("{host_os}-{host_arch}"),
            })
        }
        _ => Ok(()),
    }
}

/// The release asset URL and the catalog's pin for it. Only upstream is pinned:
/// a fork's same-named asset is other bytes.
fn release_download(
    repo: &OvmsSourceRepository,
    flavor: &OvmsFlavor,
) -> anyhow::Result<(String, Option<&'static Sha256>)> {
    let tag = repo.repository_version.as_ref();
    let asset = flavor
        .release_asset_name(tag)
        .ok_or_else(|| OvmsInstallError::NoReleaseAsset {
            flavor: flavor.to_string(),
            tag: tag.to_owned(),
        })?;
    let url = format!(
        "https://{}/releases/download/{tag}/{asset}",
        repo.repository_url
    );
    if !is_upstream(&repo.repository_url) {
        return Ok((url, None));
    }
    let pinned = pinned_sha256(tag, flavor)?;
    if pinned.is_none() {
        tracing::warn!(
            %tag,
            %flavor,
            %url,
            "the bundled catalog does not pin this ovms release; installing it unverified"
        );
    }
    Ok((url, pinned))
}

/// GitHub matches an org/repo case-insensitively, so a respelled upstream is
/// still upstream.
fn is_upstream(repo: &RepositoryUrl) -> bool {
    repo.as_ref()
        .eq_ignore_ascii_case(upstream_repository_url().as_ref())
}

/// Also the TensorRT Edge-LLM installer's, whose remote archive has no pin.
pub(crate) fn download_and_unpack(
    http: &HttpClient,
    retry: &RetryPolicy,
    url: &str,
    pinned: Option<&Sha256>,
    blobs_dir: &Path,
    reporter: &mut Reporter,
) -> anyhow::Result<()> {
    let archive = blobs_dir.join(ARCHIVE_NAME);
    let start = reporter.mark();
    retrying(
        retry,
        url,
        |_| {
            reporter.rewind(start);
            download_archive(http, url, pinned, &archive, reporter)
        },
        OvmsInstallError::disposition,
    )?;

    let to_io = |source| OvmsInstallError::Io {
        url: url.to_owned(),
        source,
    };
    let mut reader = fs::File::open(&archive).map_err(to_io)?;
    let kind = sniff_archive(&mut reader)?;
    extract_archive(reader, kind, blobs_dir)?;
    fs::remove_file(&archive).map_err(to_io)?;
    Ok(())
}

fn download_archive(
    http: &HttpClient,
    url: &str,
    pinned: Option<&Sha256>,
    archive: &Path,
    reporter: &mut Reporter,
) -> Result<(), OvmsInstallError> {
    let mut response =
        pipette_http::send(http.client().get(url)).map_err(|source| OvmsInstallError::Http {
            url: url.to_owned(),
            source,
        })?;
    // A release archive has no listing to size it by.
    reporter.set_total_if_unknown(response.content_length());

    let to_io = |source| OvmsInstallError::Io {
        url: url.to_owned(),
        source,
    };
    let mut writer = HashingWriter::new(fs::File::create(archive).map_err(to_io)?);
    let file = url.rsplit('/').next().unwrap_or(url);
    copy_reporting(&mut response, &mut writer, file, reporter).map_err(to_io)?;
    writer.flush().map_err(to_io)?;
    let actual = writer.hex_digest();
    if let Some(expected) = pinned.filter(|expected| expected.as_ref() != actual) {
        return Err(OvmsInstallError::Sha256Mismatch {
            url: url.to_owned(),
            expected: expected.to_string(),
            actual,
        });
    }
    Ok(())
}

fn check_payload(blobs_dir: &Path) -> Result<(), OvmsInstallError> {
    let executable = OvmsInstall::under(blobs_dir).executable();
    if executable.is_file() {
        Ok(())
    } else {
        Err(OvmsInstallError::MissingExecutable(
            executable.display().to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use httpmock::MockServer;
    use rstest::rstest;
    use sha2::Digest;

    use pipette_plan_types::{NonEmptyString, Ovms, RemoteArchiveUrl};

    use super::*;
    use crate::scripted_server::{Reply, ScriptedServer};

    fn release_tar_gz(entries: &[&str]) -> anyhow::Result<Vec<u8>> {
        let mut tar = tar::Builder::new(flate2::write::GzEncoder::new(
            Vec::new(),
            flate2::Compression::fast(),
        ));
        entries.iter().try_for_each(|path| {
            let payload = b"#!/bin/sh\n";
            let mut header = tar::Header::new_gnu();
            header.set_size(payload.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            tar.append_data(&mut header, path, &payload[..])
        })?;
        Ok(tar.into_inner()?.finish()?)
    }

    /// Both archives' server paths, so the payload check passes on any host.
    const SERVER_PATHS: [&str; 2] = ["ovms/ovms.exe", "ovms/bin/ovms"];

    /// `body` at `path`; the server must outlive the returned URL.
    fn serve(path: &str, body: &[u8]) -> (MockServer, String) {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("GET").path(path);
            then.status(200).body(body);
        });
        let url = server.url(path);
        (server, url)
    }

    fn sha(bytes: &[u8]) -> anyhow::Result<Sha256> {
        Ok(Sha256::try_new(hex::encode(sha2::Sha256::digest(bytes)))?)
    }

    fn release(repository_url: RepositoryUrl, tag: &str) -> anyhow::Result<OvmsSourceRepository> {
        Ok(OvmsSourceRepository {
            repository_url,
            repository_version: NonEmptyString::try_new(tag.to_owned())?,
        })
    }

    #[rstest]
    #[case::windows_on_windows(OvmsFlavor::WindowsX64, "windows", "x86_64", true)]
    #[case::windows_on_linux(OvmsFlavor::WindowsX64, "linux", "x86_64", false)]
    #[case::ubuntu_on_linux(OvmsFlavor::Ubuntu24X64, "linux", "x86_64", true)]
    #[case::redhat_on_arm_linux(OvmsFlavor::RedhatX64, "linux", "aarch64", false)]
    #[case::ubuntu_on_macos(OvmsFlavor::Ubuntu22X64, "macos", "aarch64", false)]
    #[case::custom_anywhere(OvmsFlavor::Custom("acme".to_owned()), "macos", "aarch64", true)]
    fn a_named_flavor_installs_only_on_its_platform(
        #[case] flavor: OvmsFlavor,
        #[case] os: &str,
        #[case] arch: &str,
        #[case] admitted: bool,
    ) {
        let verdict = refuse_foreign_flavor(&flavor, os, arch);
        assert_eq!(verdict.is_ok(), admitted, "{verdict:?}");
        if let Err(err) = verdict {
            assert!(err.to_string().contains(&format!("{os}-{arch}")), "{err}");
        }
    }

    #[test]
    fn an_upstream_release_derives_its_asset_url_and_pin() -> anyhow::Result<()> {
        let (url, pinned) = release_download(
            &release(upstream_repository_url(), "v2026.4.1")?,
            &OvmsFlavor::WindowsX64,
        )?;
        assert_eq!(
            url,
            "https://github.com/openvinotoolkit/model_server/releases/download/v2026.4.1/\
             ovms_windows_2026.4.1_python_off.zip"
        );
        assert_eq!(
            pinned.map(AsRef::as_ref),
            Some("1bc2372d81bb4767829a7bf513745bd481ccbfec9550e278804ebe2051e3b678")
        );
        Ok(())
    }

    /// Upstream is matched the way GitHub matches it, case-insensitively.
    #[rstest]
    #[case::respelled_upstream("github.com/OpenVINOToolkit/Model_Server", true)]
    #[case::fork("github.com/acme/model_server", false)]
    fn only_the_upstream_repo_is_pinned(
        #[case] repo: &str,
        #[case] pinned: bool,
    ) -> anyhow::Result<()> {
        let (url, pin) = release_download(
            &release(RepositoryUrl::new(repo), "v2026.4.1")?,
            &OvmsFlavor::WindowsX64,
        )?;
        assert!(url.starts_with(&format!("https://{repo}/")), "{url}");
        assert_eq!(pin.is_some(), pinned);
        Ok(())
    }

    #[rstest]
    #[case::unversioned_asset_names("v2026.0", OvmsFlavor::WindowsX64)]
    #[case::custom_flavor("v2026.4.1", OvmsFlavor::Custom("acme".to_owned()))]
    fn a_release_with_no_asset_name_is_refused(
        #[case] tag: &str,
        #[case] flavor: OvmsFlavor,
    ) -> anyhow::Result<()> {
        let err = release_download(&release(upstream_repository_url(), tag)?, &flavor)
            .err()
            .context("no asset URL should be derived")?;
        assert!(
            err.to_string()
                .contains("install from an archive URL instead"),
            "{err}"
        );
        Ok(())
    }

    /// The staging copy is gone afterwards.
    #[test]
    fn a_download_matching_its_pin_unpacks() -> anyhow::Result<()> {
        let body = release_tar_gz(&SERVER_PATHS)?;
        let (_server, url) = serve("/ovms.tar.gz", &body);
        let tmp = tempfile::tempdir()?;
        download_and_unpack(
            &HttpClient::new("pipette-test")?,
            &RetryPolicy::IMMEDIATE,
            &url,
            Some(&sha(&body)?),
            tmp.path(),
            &mut Reporter::silent(),
        )?;
        check_payload(tmp.path())?;
        assert!(!tmp.path().join(ARCHIVE_NAME).exists());
        Ok(())
    }

    #[test]
    fn a_transient_failure_is_retried() -> anyhow::Result<()> {
        let body = release_tar_gz(&SERVER_PATHS)?;
        let server = ScriptedServer::start(vec![
            Reply::status(503),
            Reply::cut(&body, 10),
            Reply::ok(&body),
        ])?;
        let tmp = tempfile::tempdir()?;
        download_and_unpack(
            &HttpClient::new("pipette-test")?,
            &RetryPolicy::IMMEDIATE,
            &server.url("/ovms.tar.gz"),
            Some(&sha(&body)?),
            tmp.path(),
            &mut Reporter::silent(),
        )?;
        check_payload(tmp.path())?;
        assert_eq!(server.requests().len(), 3);
        Ok(())
    }

    /// `Io` is the body read or the disk write; only a read retries.
    #[rstest]
    #[case::broken_read(std::io::ErrorKind::ConnectionReset, Disposition::TRANSIENT)]
    #[case::full_disk(std::io::ErrorKind::StorageFull, Disposition::Permanent)]
    #[case::read_only(std::io::ErrorKind::ReadOnlyFilesystem, Disposition::Permanent)]
    fn an_io_failure_retries_unless_the_host_refused_it(
        #[case] kind: std::io::ErrorKind,
        #[case] expected: Disposition,
    ) {
        let err = OvmsInstallError::Io {
            url: "https://example.test/ovms.tar.gz".to_owned(),
            source: kind.into(),
        };
        assert_eq!(err.disposition(), expected);
    }

    #[test]
    fn a_download_off_its_pin_is_refused() -> anyhow::Result<()> {
        let body = release_tar_gz(&SERVER_PATHS)?;
        let (_server, url) = serve("/ovms.tar.gz", &body);
        let pinned = sha(b"other bytes")?;
        let tmp = tempfile::tempdir()?;
        let err = download_and_unpack(
            &HttpClient::new("pipette-test")?,
            &RetryPolicy::IMMEDIATE,
            &url,
            Some(&pinned),
            tmp.path(),
            &mut Reporter::silent(),
        )
        .err()
        .context("a mismatched archive must be refused")?;
        let msg = err.to_string();
        assert!(msg.contains(pinned.as_ref()), "{msg}");
        assert!(
            msg.contains(&hex::encode(sha2::Sha256::digest(&body))),
            "{msg}"
        );
        assert!(!tmp.path().join("ovms").exists(), "nothing is unpacked");
        Ok(())
    }

    fn remote_archive(url: String) -> anyhow::Result<Runtime> {
        Ok(Runtime::Ovms(Ovms {
            source: OvmsSource::RemoteArchive {
                url: RemoteArchiveUrl::try_new(url)?,
            },
            flavor: OvmsFlavor::Custom("acme-x64".to_owned()),
        }))
    }

    #[test]
    fn a_remote_archive_installs_its_ovms_payload() -> anyhow::Result<()> {
        let (_server, url) = serve("/ovms.tar.gz", &release_tar_gz(&SERVER_PATHS)?);
        let tmp = tempfile::tempdir()?;
        install_ovms_archive(
            &HttpClient::new("pipette-test")?,
            &RetryPolicy::IMMEDIATE,
            None,
            &remote_archive(url)?,
            tmp.path(),
            &mut Reporter::silent(),
        )?;
        assert!(OvmsInstall::under(tmp.path()).executable().is_file());
        Ok(())
    }

    /// An archive without the top-level `ovms/` is not an OVMS release.
    #[test]
    fn an_archive_without_the_payload_dir_is_refused() -> anyhow::Result<()> {
        let (_server, url) = serve("/flat.tar.gz", &release_tar_gz(&["bin/ovms", "ovms.exe"])?);
        let tmp = tempfile::tempdir()?;
        let err = install_ovms_archive(
            &HttpClient::new("pipette-test")?,
            &RetryPolicy::IMMEDIATE,
            None,
            &remote_archive(url)?,
            tmp.path(),
            &mut Reporter::silent(),
        )
        .err()
        .context("a flat archive must be refused")?;
        assert!(err.to_string().contains("top-level `ovms/`"), "{err}");
        Ok(())
    }

    #[test]
    fn a_vault_archive_installs_its_ovms_payload() -> anyhow::Result<()> {
        use crate::vault::test_support::{RESOURCE_ID, access, content_path, serve_bytes};

        let server = MockServer::start();
        serve_bytes(&server, RESOURCE_ID, &release_tar_gz(&SERVER_PATHS)?);
        let declared = Runtime::Ovms(Ovms {
            source: OvmsSource::VaultRemoteArchive {
                url: content_path(RESOURCE_ID)?,
                principal_id: "org_1".parse()?,
                custom_label: None,
            },
            flavor: OvmsFlavor::Custom("internal".to_owned()),
        });
        let tmp = tempfile::tempdir()?;
        install_ovms_archive(
            &HttpClient::new("pipette-test")?,
            &RetryPolicy::IMMEDIATE,
            Some(&access(&server)?),
            &declared,
            tmp.path(),
            &mut Reporter::silent(),
        )?;
        check_payload(tmp.path())?;
        Ok(())
    }

    #[rstest]
    #[case::relative(r#"{"type":"ovms","source":"relative_dir","dir":"blobs","flavor":"acme"}"#)]
    #[case::absolute(
        r#"{"type":"ovms","source":"absolute_dir","dir":"/opt/ovms","flavor":"acme"}"#
    )]
    #[case::other_runtime(
        r#"{"type":"docker_vllm","image_name":"vllm","image_tag":"v0.10.0","flavor":"nvidia_gpu"}"#
    )]
    fn a_form_that_is_not_a_pull_is_refused(#[case] json: &str) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let err = install_ovms_archive(
            &HttpClient::new("pipette-test")?,
            &RetryPolicy::IMMEDIATE,
            None,
            &serde_json::from_str(json)?,
            tmp.path(),
            &mut Reporter::silent(),
        )
        .err()
        .context("a form that is not a pull must be refused")?;
        assert!(
            matches!(
                err.downcast_ref::<OvmsInstallError>(),
                Some(OvmsInstallError::NotInstallable(_))
            ),
            "{json}: {err}"
        );
        Ok(())
    }
}
