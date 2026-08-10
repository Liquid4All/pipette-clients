//! Archive-download runtime install (llama.cpp), the runtime-side counterpart
//! to [`crate::model::fetch`].
//!
//! "Pulling" means downloading a prebuilt release asset (or an archive URL) and
//! unpacking it — there is no git build for llama.cpp. Entry point:
//! [`install_llamacpp_archive`]. Docker / UV / MLX installs live beside this
//! module and are routed by [`crate::ensure_runtime`].

use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::Context;
use flate2::read::GzDecoder;
use tar::Archive;
use zip::ZipArchive;

use pipette_http::HttpClient;
use pipette_plan_types::{LlamacppCliStockTools, LlamacppCliStockToolsSource, Runtime};

use crate::progress::{copy_reporting, Reporter};

/// Ceiling on the capacity a `content-length` may reserve up front. Well above
/// any real llama.cpp archive (tens of MB), well below a size worth aborting on.
const ARCHIVE_CAPACITY_HINT_CAP: u64 = 512 * 1024 * 1024;

/// The two binaries every llama.cpp runtime archive must contain.
const LLAMACPP_BINARIES: [&str; 2] = ["llama-bench", "llama-server"];

/// Why an archive-based runtime install failed.
#[derive(Debug, thiserror::Error)]
pub enum LlamacppInstallError {
    /// This runtime kind has no archive-download install path yet.
    #[error("{0}")]
    NotInstallable(String),
    /// A `Custom` (or otherwise unknown) llama.cpp flavor has no upstream
    /// release asset, so its download URL can't be derived.
    #[error(
        "llama.cpp flavor `{0}` has no upstream release asset; install from an archive URL instead"
    )]
    NoReleaseAsset(String),
    /// Downloading the archive failed.
    #[error("downloading runtime archive from {url} failed")]
    Http {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    /// The archive body failed part-way through — distinct from `Http`, which is
    /// the request never getting off the ground.
    #[error("reading runtime archive from {url} failed")]
    Io {
        url: String,
        #[source]
        source: std::io::Error,
    },
}

/// Download + unpack a llama.cpp archive into store-owned `blobs_dir`.
///
/// A `GitHubRelease` source resolves to the **GitHub releases** asset layout
/// (`https://<repo>/releases/download/<version>/<asset>`) — the upstream
/// `github.com/ggml-org/llama.cpp` default and forks that mirror it. A custom
/// repo that isn't GitHub-releases-hosted has no derivable asset URL; install
/// those from the archive `url` form instead.
pub(crate) fn install_llamacpp_archive(
    http: &HttpClient,
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
    install_llamacpp(http, rt, blobs_dir, reporter)
}

fn install_llamacpp(
    http: &HttpClient,
    rt: &LlamacppCliStockTools,
    blobs_dir: &Path,
    reporter: &mut Reporter,
) -> anyhow::Result<()> {
    let (url, kind) = match &rt.source {
        LlamacppCliStockToolsSource::GithubRelease(repo) => {
            let version = repo.repository_version.as_ref();
            let asset = rt
                .flavor
                .release_asset_name(version)
                .ok_or_else(|| LlamacppInstallError::NoReleaseAsset(rt.flavor.to_string()))?;
            let url = format!(
                "https://{}/releases/download/{version}/{asset}",
                repo.repository_url
            );
            let kind = infer_archive_kind(&asset);
            (url, kind)
        }
        LlamacppCliStockToolsSource::RemoteArchive { url } => {
            let download = url.download_url();
            let kind = infer_archive_kind(&download);
            (download, kind)
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

    // `blobs_dir` is the store-owned payload dir; extract straight into it.
    let bytes = read_archive(http, &url, reporter)?;
    extract_archive(&bytes, kind, blobs_dir)?;
    // Structural "both present" check — tools are resolved later via
    // RuntimeManifest::resolve_tool under install_dir.
    for base in LLAMACPP_BINARIES {
        find_binary(blobs_dir, &binary_name(base))?;
    }
    Ok(())
}

/// Download archive bytes over HTTPS (RemoteArchive is scheme-less host/path;
/// callers pass `download_url()`).
fn read_archive(http: &HttpClient, url: &str, reporter: &mut Reporter) -> anyhow::Result<Vec<u8>> {
    let mut response = http
        .client()
        .get(url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|source| LlamacppInstallError::Http {
            url: url.to_owned(),
            source,
        })?;
    // A release archive has no listing to size it beforehand, so the response is
    // where the total comes from — and it is one file, so its length is the
    // artifact's.
    reporter.set_total_if_unknown(response.content_length());
    // Buffered whole, as `extract_archive` needs it: an archive is tens of MB
    // where a model is tens of GB, and streaming it would buy nothing but a
    // temp file.
    //
    // `content-length` only *hints* the capacity, capped: it is a header from the
    // far end, and a response claiming terabytes would otherwise abort the process
    // on the reserve, before a byte had been read. The body is what actually grows
    // the vector.
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

/// The archive kind from a filename/URL: `.zip` → zip, else tar.gz.
fn infer_archive_kind(name: &str) -> ArchiveKind {
    if name.ends_with(".zip") {
        ArchiveKind::Zip
    } else {
        ArchiveKind::TarGz
    }
}

#[derive(Clone, Copy)]
enum ArchiveKind {
    Zip,
    TarGz,
}

/// Unpack `archive_bytes` into `destination`, restoring unix exec bits for zip
/// entries so `llama-server` stays runnable. Ported from `pipette-llamacpp`.
///
/// Path-traversal safety: the zip path normalizes each entry via
/// `mangled_name()`; the tar path relies on `tar::Archive::unpack`, which
/// refuses to write outside `destination` (rejecting `..` components).
fn extract_archive(
    archive_bytes: &[u8],
    kind: ArchiveKind,
    destination: &Path,
) -> anyhow::Result<()> {
    match kind {
        ArchiveKind::Zip => {
            let mut archive = ZipArchive::new(Cursor::new(archive_bytes))
                .context("failed to read zip runtime archive")?;
            for index in 0..archive.len() {
                let mut entry = archive
                    .by_index(index)
                    .with_context(|| format!("failed to read zip entry #{index}"))?;
                let out_path = destination.join(entry.mangled_name());
                if entry.is_dir() {
                    std::fs::create_dir_all(&out_path)
                        .with_context(|| format!("failed to create {}", out_path.display()))?;
                    continue;
                }
                if let Some(parent) = out_path.parent() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("failed to create {}", parent.display()))?;
                }
                let mut out_file = std::fs::File::create(&out_path)
                    .with_context(|| format!("failed to create {}", out_path.display()))?;
                std::io::copy(&mut entry, &mut out_file)
                    .with_context(|| format!("failed to extract {}", out_path.display()))?;
                #[cfg(unix)]
                if let Some(mode) = entry.unix_mode() {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::set_permissions(&out_path, std::fs::Permissions::from_mode(mode))
                        .with_context(|| {
                            format!("failed to set permissions on {}", out_path.display())
                        })?;
                }
            }
        }
        ArchiveKind::TarGz => {
            let mut archive = Archive::new(GzDecoder::new(Cursor::new(archive_bytes)));
            archive
                .unpack(destination)
                .with_context(|| format!("failed to unpack into {}", destination.display()))?;
        }
    }
    Ok(())
}

/// Recursively locate `expected_name` under `root`; errors if absent.
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
    use pipette_plan_types::{
        DockerVllm, LlamaCppFlavor, NonEmptyString, RepositoryUrl, SourceRepository, VllmFlavor,
    };

    use super::*;

    #[test]
    fn infer_archive_kind_from_name() {
        assert!(matches!(infer_archive_kind("x.zip"), ArchiveKind::Zip));
        assert!(matches!(infer_archive_kind("x.tar.gz"), ArchiveKind::TarGz));
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
        let err = install_llamacpp_archive(&http, &docker, tmp.path(), &mut Reporter::silent())
            .err()
            .context("docker runtime should not be fetchable")?;
        assert!(err.to_string().contains("not yet implemented"));
        Ok(())
    }

    #[test]
    fn a_custom_flavor_repository_reports_no_asset() -> anyhow::Result<()> {
        let http = HttpClient::new("pipette-test")?;
        let declared = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new("b1".to_owned())?,
            }),
            flavor: LlamaCppFlavor::Custom("weird".to_owned()),
        });
        let tmp = tempfile::tempdir()?;
        // No network: the error is raised while deriving the asset URL.
        let err = install_llamacpp_archive(&http, &declared, tmp.path(), &mut Reporter::silent())
            .err()
            .context("custom flavor should have no release asset")?;
        assert!(err.to_string().contains("no upstream release asset"));
        Ok(())
    }

    #[test]
    fn remote_archive_url_is_scheme_less_https_download() -> anyhow::Result<()> {
        let url = pipette_plan_types::RemoteArchiveUrl::try_new(
            "https://github.com/org/repo/releases/download/v1/llama.tar.gz".to_owned(),
        )?;
        assert_eq!(
            url.as_ref(),
            "github.com/org/repo/releases/download/v1/llama.tar.gz"
        );
        assert_eq!(
            url.download_url(),
            "https://github.com/org/repo/releases/download/v1/llama.tar.gz"
        );
        assert!(pipette_plan_types::RemoteArchiveUrl::try_new(
            "file:///tmp/llama.tar.gz".to_owned()
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn extract_archive_records_binary_placements() -> anyhow::Result<()> {
        use flate2::{write::GzEncoder, Compression};

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
        extract_archive(&tar_gz, ArchiveKind::TarGz, &blobs)?;

        assert!(find_binary(&blobs, &binary_name("llama-bench")).is_ok());
        assert!(find_binary(&blobs, &binary_name("llama-server")).is_ok());
        Ok(())
    }
}
