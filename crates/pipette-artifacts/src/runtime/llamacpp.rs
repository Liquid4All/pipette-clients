//! llama.cpp install: download a prebuilt release asset or archive and unpack
//! it (no source build).

use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::Context;

use pipette_http::HttpClient;
use pipette_http::retry::{Disposition, RetryPolicy, retrying};
use pipette_plan_types::{
    LlamacppCliStockTools, LlamacppCliStockToolsRelease, LlamacppCliStockToolsSource, Runtime,
};

use crate::archive::{extract_archive, infer_archive_kind};
use crate::progress::{Reporter, copy_reporting};
use crate::vault::{self, VaultAccess};

/// Ceiling on the capacity a `content-length` may reserve up front. Well above
/// any real llama.cpp archive (tens of MB), well below a size worth aborting on.
const ARCHIVE_CAPACITY_HINT_CAP: u64 = 512 * 1024 * 1024;

const LLAMACPP_BINARIES: [&str; 2] = ["llama-bench", "llama-server"];

#[derive(Debug, thiserror::Error)]
pub enum LlamacppInstallError {
    #[error("{0}")]
    NotInstallable(String),
    /// The release names no `asset` and the flavor names none at its tag: a
    /// `Custom` flavor, or one upstream stopped building.
    #[error(
        "llama.cpp flavor `{flavor}` has no upstream release asset at `{version}`; name the \
         release file in `asset`, or install from an archive URL instead"
    )]
    NoReleaseAsset { flavor: String, version: String },
    #[error("downloading runtime archive from {url} failed")]
    Http {
        url: String,
        #[source]
        source: pipette_http::Error,
    },
    /// The body failed part-way, after the request succeeded.
    #[error("reading runtime archive from {url} failed")]
    Io {
        url: String,
        #[source]
        source: std::io::Error,
    },
}

impl LlamacppInstallError {
    /// Whether another download attempt can succeed: the request got no
    /// answer or a `5xx` or `429`, or the body broke part-way. The body is
    /// read into memory, so an `Io` is always the transfer's.
    fn disposition(&self) -> Disposition {
        match self {
            Self::Http { source, .. } => source.disposition(),
            Self::Io { .. } => Disposition::TRANSIENT,
            Self::NotInstallable(_) | Self::NoReleaseAsset { .. } => Disposition::Permanent,
        }
    }
}

/// Download and unpack a llama.cpp archive into store-owned `blobs_dir`.
///
/// A `GithubRelease` source assumes the GitHub releases URL layout; a repo
/// hosted elsewhere must use the archive `url` form.
pub(crate) fn install_llamacpp_archive(
    http: &HttpClient,
    retry: &RetryPolicy,
    vault: Option<&VaultAccess>,
    declared: &Runtime,
    blobs_dir: &Path,
    reporter: &mut Reporter,
) -> anyhow::Result<()> {
    let Runtime::LlamacppCliStockTools(rt) = declared else {
        return Err(LlamacppInstallError::NotInstallable(format!(
            "pulling `{}` runtimes is not yet implemented \
             (only llama.cpp archive/release installs are supported)",
            declared.headless_token()
        ))
        .into());
    };
    install_llamacpp(http, retry, vault, rt, blobs_dir, reporter)
}

fn install_llamacpp(
    http: &HttpClient,
    retry: &RetryPolicy,
    vault: Option<&VaultAccess>,
    rt: &LlamacppCliStockTools,
    blobs_dir: &Path,
    reporter: &mut Reporter,
) -> anyhow::Result<()> {
    let (url, kind) = match &rt.source {
        LlamacppCliStockToolsSource::GithubRelease(release) => {
            let url = release_url(rt, release)?;
            let kind = infer_archive_kind(&url);
            (url, kind)
        }
        LlamacppCliStockToolsSource::RemoteArchive { url } => {
            let download = url.as_ref().to_owned();
            let kind = infer_archive_kind(&download);
            (download, kind)
        }
        // Verified against its record and unpacked in place; the kind is read
        // off the bytes, since the path names no file.
        LlamacppCliStockToolsSource::VaultRemoteArchive { url, .. } => {
            vault::fetch_and_unpack(vault, http, retry, url, None, blobs_dir, reporter)?;
            return check_binaries(blobs_dir);
        }
        LlamacppCliStockToolsSource::RelativeDir { dir } => {
            anyhow::bail!(
                "cannot fetch a relative install form (dir={dir}); pass the \
                 declared GitHubRelease/RemoteArchive coordinate instead"
            );
        }
        LlamacppCliStockToolsSource::AbsoluteDir { dir } => {
            anyhow::bail!(
                "cannot fetch an absolute install form (dir={dir}); pass the \
                 declared GitHubRelease/RemoteArchive coordinate instead"
            );
        }
    };

    let start = reporter.mark();
    let bytes = retrying(
        retry,
        &url,
        |_| {
            reporter.rewind(start);
            read_archive(http, &url, reporter)
        },
        LlamacppInstallError::disposition,
    )?;
    extract_archive(Cursor::new(&bytes), kind, blobs_dir)?;
    check_binaries(blobs_dir)
}

fn release_url(
    rt: &LlamacppCliStockTools,
    release: &LlamacppCliStockToolsRelease,
) -> Result<String, LlamacppInstallError> {
    let version = release.repository_version.as_ref();
    let asset = rt
        .release_asset()
        .ok_or_else(|| LlamacppInstallError::NoReleaseAsset {
            flavor: rt.flavor.to_string(),
            version: version.to_owned(),
        })?;
    Ok(format!(
        "https://{}/releases/download/{version}/{asset}",
        release.repository_url
    ))
}

/// An upstream llama.cpp release at `version`, naming `asset` when given.
#[cfg(test)]
pub(crate) fn upstream_release(
    version: &str,
    asset: Option<&str>,
) -> anyhow::Result<LlamacppCliStockToolsRelease> {
    use pipette_plan_types::{FileName, NonEmptyString, RepositoryUrl};

    Ok(LlamacppCliStockToolsRelease {
        repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
        repository_version: NonEmptyString::try_new(version.to_owned())?,
        asset: asset
            .map(|asset| FileName::try_new(asset.to_owned()))
            .transpose()?,
    })
}

/// Presence only; tools are resolved later by `RuntimeManifest::resolve_tool`.
fn check_binaries(blobs_dir: &Path) -> anyhow::Result<()> {
    LLAMACPP_BINARIES
        .iter()
        .try_for_each(|base| find_binary(blobs_dir, &binary_name(base)).map(drop))
}

fn read_archive(
    http: &HttpClient,
    url: &str,
    reporter: &mut Reporter,
) -> Result<Vec<u8>, LlamacppInstallError> {
    let mut response = pipette_http::send(http.client().get(url)).map_err(|source| {
        LlamacppInstallError::Http {
            url: url.to_owned(),
            source,
        }
    })?;
    // An archive has no listing to size it beforehand; its one file's length is
    // the artifact's.
    reporter.set_total_if_unknown(response.content_length());
    // Buffered whole for `extract_archive`: an archive is tens of MB, so
    // streaming would only add a temp file. `content-length` is capped because
    // a remote claiming terabytes would abort the process on the reserve.
    let hint = response
        .content_length()
        .unwrap_or(0)
        .min(ARCHIVE_CAPACITY_HINT_CAP);
    let mut bytes = Vec::with_capacity(hint as usize);
    let file = url.rsplit('/').next().unwrap_or(url).to_owned();
    copy_reporting(&mut response, &mut bytes, &file, reporter).map_err(|source| {
        LlamacppInstallError::Io {
            url: url.to_owned(),
            source,
        }
    })?;
    Ok(bytes)
}

fn find_binary(root: &Path, expected_name: &str) -> anyhow::Result<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in
            std::fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().and_then(|name| name.to_str()) == Some(expected_name) {
                return Ok(path);
            }
        }
    }
    anyhow::bail!(
        "runtime archive is missing `{expected_name}` under {}",
        root.display()
    )
}

fn binary_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{
        DockerVllm, LlamaCppFlavor, NonEmptyString, RemoteArchiveUrl, VllmFlavor,
    };

    use super::*;
    use crate::archive::ArchiveKind;
    use crate::scripted_server::{Reply, ScriptedServer};

    #[rstest]
    #[case::lower("llama-b1234-bin-win-cpu-x64.zip", ArchiveKind::Zip)]
    #[case::upper("LLAMA.ZIP", ArchiveKind::Zip)]
    #[case::url("https://example.com/llama-b1234.zip", ArchiveKind::Zip)]
    #[case::tarball("llama-b1234-bin-ubuntu-x64.tar.gz", ArchiveKind::TarGz)]
    #[case::zip_in_stem("zip-tools.tar.gz", ArchiveKind::TarGz)]
    fn infer_archive_kind_cases(#[case] name: &str, #[case] want: ArchiveKind) {
        assert_eq!(infer_archive_kind(name), want);
    }

    #[test]
    fn non_llamacpp_runtimes_are_not_fetchable() -> anyhow::Result<()> {
        let http = HttpClient::new("pipette-test")?;
        let docker = Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm".to_owned())?,
            image_tag: NonEmptyString::try_new("v0.10.0".to_owned())?,
            flavor: VllmFlavor::NvidiaGpu,
        });
        let tmp = tempfile::tempdir()?;
        let err = install_llamacpp_archive(
            &http,
            &RetryPolicy::IMMEDIATE,
            None,
            &docker,
            tmp.path(),
            &mut Reporter::silent(),
        )
        .err()
        .context("docker runtime should not be fetchable")?;
        assert!(err.to_string().contains("not yet implemented"));
        Ok(())
    }

    /// The flavor names the file unless the release does, which is also how
    /// a flavor upstream stopped building is still reached.
    #[rstest]
    #[case::flavor_names_it(
        LlamaCppFlavor::LinuxX64Rocm,
        None,
        Some("llama-b11399-bin-ubuntu-rocm-10.0-x64.tar.gz")
    )]
    #[case::declared_wins(
        LlamaCppFlavor::LinuxX64Rocm,
        Some("llama-b11399-bin-ubuntu-vulkan-x64.tar.gz"),
        Some("llama-b11399-bin-ubuntu-vulkan-x64.tar.gz")
    )]
    #[case::past_the_flavor(
        LlamaCppFlavor::WindowsX64Cuda131,
        Some("llama-b11399-bin-win-cuda-13.4-x64.zip"),
        Some("llama-b11399-bin-win-cuda-13.4-x64.zip")
    )]
    #[case::nothing_names_it(LlamaCppFlavor::WindowsX64Cuda131, None, None)]
    fn release_url_takes_the_declared_asset(
        #[case] flavor: LlamaCppFlavor,
        #[case] asset: Option<&str>,
        #[case] file: Option<&str>,
    ) -> anyhow::Result<()> {
        let release = upstream_release("b11399", asset)?;
        let rt = LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(release.clone()),
            flavor,
        };
        match (release_url(&rt, &release), file) {
            (Ok(url), Some(file)) => assert_eq!(
                url,
                format!("https://github.com/ggml-org/llama.cpp/releases/download/b11399/{file}")
            ),
            (Err(err), None) => {
                let message = err.to_string();
                assert!(message.contains("`b11399`"), "{message}");
                assert!(message.contains("`asset`"), "{message}");
            }
            (got, want) => anyhow::bail!("got {got:?}, want {want:?}"),
        }
        Ok(())
    }

    #[test]
    fn a_custom_flavor_repository_reports_no_asset() -> anyhow::Result<()> {
        let http = HttpClient::new("pipette-test")?;
        let declared = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(upstream_release("b1", None)?),
            flavor: LlamaCppFlavor::Custom("weird".to_owned()),
        });
        let tmp = tempfile::tempdir()?;
        let err = install_llamacpp_archive(
            &http,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            tmp.path(),
            &mut Reporter::silent(),
        )
        .err()
        .context("custom flavor should have no release asset")?;
        assert!(err.to_string().contains("no upstream release asset"));
        Ok(())
    }

    #[test]
    fn extract_archive_records_binary_placements() -> anyhow::Result<()> {
        use flate2::{Compression, write::GzEncoder};

        let mut tar_gz = Vec::new();
        {
            let mut builder = tar::Builder::new(GzEncoder::new(&mut tar_gz, Compression::fast()));
            for base in LLAMACPP_BINARIES {
                let payload = b"#!/bin/sh\n";
                let mut header = tar::Header::new_gnu();
                header.set_size(payload.len() as u64);
                header.set_mode(0o755);
                header.set_cksum();
                builder.append_data(
                    &mut header,
                    format!("build/bin/{}", binary_name(base)),
                    &payload[..],
                )?;
            }
            builder.into_inner()?.finish()?;
        }

        let tmp = tempfile::tempdir()?;
        let blobs = tmp.path().join("blobs");
        std::fs::create_dir_all(&blobs)?;
        extract_archive(Cursor::new(&tar_gz), ArchiveKind::TarGz, &blobs)?;

        assert!(find_binary(&blobs, &binary_name("llama-bench")).is_ok());
        assert!(find_binary(&blobs, &binary_name("llama-server")).is_ok());
        Ok(())
    }

    fn release_tar_gz() -> anyhow::Result<Vec<u8>> {
        use flate2::{Compression, write::GzEncoder};

        let mut tar = tar::Builder::new(GzEncoder::new(Vec::new(), Compression::fast()));
        LLAMACPP_BINARIES.iter().try_for_each(|base| {
            let payload = b"#!/bin/sh\n";
            let mut header = tar::Header::new_gnu();
            header.set_size(payload.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            tar.append_data(
                &mut header,
                format!("build/bin/{}", binary_name(base)),
                &payload[..],
            )
        })?;
        Ok(tar.into_inner()?.finish()?)
    }

    fn install_from(url: &str) -> anyhow::Result<(tempfile::TempDir, anyhow::Result<()>)> {
        let declared = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::RemoteArchive {
                url: RemoteArchiveUrl::try_new(url.to_owned())?,
            },
            flavor: LlamaCppFlavor::Custom("archive".to_owned()),
        });
        let tmp = tempfile::tempdir()?;
        let result = install_llamacpp_archive(
            &HttpClient::new("pipette-test")?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            tmp.path(),
            &mut Reporter::silent(),
        );
        Ok((tmp, result))
    }

    #[rstest]
    #[case::dropped_connections(vec![Reply::Drop, Reply::Drop])]
    #[case::unavailable(vec![Reply::status(503), Reply::status(503), Reply::status(503)])]
    #[case::cut_body(vec![Reply::cut(&release_tar_gz()?, 10)])]
    fn a_transient_failure_is_retried(#[case] failures: Vec<Reply>) -> anyhow::Result<()> {
        let attempts = failures.len() + 1;
        let script = failures
            .into_iter()
            .chain([Reply::ok(&release_tar_gz()?)])
            .collect();
        let server = ScriptedServer::start(script)?;
        let (tmp, result) = install_from(&server.url("/llama.tar.gz"))?;
        result?;
        assert!(find_binary(tmp.path(), &binary_name("llama-server")).is_ok());
        assert_eq!(server.requests().len(), attempts);
        Ok(())
    }

    #[test]
    fn a_not_found_is_not_retried() -> anyhow::Result<()> {
        let server = ScriptedServer::start(vec![Reply::status(404), Reply::ok(b"")])?;
        let (_tmp, result) = install_from(&server.url("/llama.tar.gz"))?;
        let err = result.err().context("a 404 should fail the install")?;
        assert!(format!("{err:#}").contains("HTTP 404"), "{err:#}");
        assert_eq!(server.requests().len(), 1);
        Ok(())
    }

    /// The error after the last attempt still names the cause under reqwest's
    /// own message: here, the refused connect.
    #[test]
    fn a_refused_connection_keeps_its_cause() -> anyhow::Result<()> {
        let port = std::net::TcpListener::bind("127.0.0.1:0")?
            .local_addr()?
            .port();
        let (_tmp, result) = install_from(&format!("http://127.0.0.1:{port}/llama.tar.gz"))?;
        let err = result
            .err()
            .context("a closed port should fail the install")?;
        let cause = err
            .chain()
            .find_map(|cause| cause.downcast_ref::<std::io::Error>())
            .context("no io::Error in the chain")?;
        assert!(format!("{err:#}").ends_with(&cause.to_string()), "{err:#}");
        Ok(())
    }

    #[test]
    fn a_vault_archive_installs_and_is_checked_for_binaries() -> anyhow::Result<()> {
        use httpmock::MockServer;

        use crate::vault::test_support::{RESOURCE_ID, access, content_path, serve_bytes};

        let mut tar = tar::Builder::new(Vec::new());
        LLAMACPP_BINARIES.iter().try_for_each(|base| {
            let payload = b"#!/bin/sh\n";
            let mut header = tar::Header::new_ustar();
            header.set_size(payload.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            tar.append_data(
                &mut header,
                format!("bin/{}", binary_name(base)),
                &payload[..],
            )
        })?;
        let server = MockServer::start();
        serve_bytes(&server, RESOURCE_ID, &tar.into_inner()?);
        let declared = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::VaultRemoteArchive {
                url: content_path(RESOURCE_ID)?,
                principal_id: "org_1".parse()?,
                custom_label: None,
            },
            flavor: LlamaCppFlavor::Custom("internal".to_owned()),
        });
        let tmp = tempfile::tempdir()?;
        let http = HttpClient::new("pipette-test")?;
        install_llamacpp_archive(
            &http,
            &RetryPolicy::IMMEDIATE,
            Some(&access(&server)?),
            &declared,
            tmp.path(),
            &mut Reporter::silent(),
        )?;
        assert!(find_binary(tmp.path(), &binary_name("llama-server")).is_ok());
        Ok(())
    }
}
