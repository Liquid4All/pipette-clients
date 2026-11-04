//! Archive install for the `trt_edgellm` runtime: a TensorRT Edge-LLM server
//! archive unpacked under the store-owned `blobs/` payload dir.
//!
//! Upstream publishes no server archive, so there is no catalog pin: a remote
//! archive installs unverified (name it by its content hash), a vault archive
//! is verified against its record. The engines it builds are not part of the
//! install; they go to the runtime's compile cache.

use std::path::Path;

use pipette_artifacts_catalog::runtime::trt_edgellm::TrtEdgellmInstall;
use pipette_http::HttpClient;
use pipette_http::retry::RetryPolicy;
use pipette_plan_types::{Runtime, TrtEdgellmFlavor, TrtEdgellmSource};

use crate::progress::Reporter;
use crate::runtime::ovms::download_and_unpack;
use crate::vault::{self, VaultAccess};

#[derive(Debug, thiserror::Error)]
pub enum TrtEdgellmInstallError {
    #[error("{0}")]
    NotInstallable(String),
    #[error("trt_edgellm flavor `{flavor}` is built for {built_for}, but this host is {host}")]
    ForeignFlavor {
        flavor: String,
        built_for: String,
        host: String,
    },
    #[error(
        "the trt_edgellm archive unpacked no {0}; its top-level `trt-edgellm-server/` is the \
         payload"
    )]
    MissingFile(String),
}

/// Download, verify and unpack a server archive into `blobs_dir`.
pub(crate) fn install_trt_edgellm_archive(
    http: &HttpClient,
    retry: &RetryPolicy,
    vault: Option<&VaultAccess>,
    declared: &Runtime,
    blobs_dir: &Path,
    reporter: &mut Reporter,
) -> anyhow::Result<()> {
    let Runtime::TrtEdgellm(rt) = declared else {
        return Err(TrtEdgellmInstallError::NotInstallable(format!(
            "install_trt_edgellm_archive only installs trt_edgellm, got `{}`",
            declared.headless_token()
        ))
        .into());
    };
    refuse_foreign_flavor(&rt.flavor, std::env::consts::OS, std::env::consts::ARCH)?;
    match &rt.source {
        TrtEdgellmSource::RemoteArchive { url } => {
            download_and_unpack(http, retry, url.as_ref(), None, blobs_dir, reporter)?;
        }
        TrtEdgellmSource::VaultRemoteArchive { url, .. } => {
            vault::fetch_and_unpack(vault, http, retry, url, None, blobs_dir, reporter)?;
        }
        TrtEdgellmSource::RelativeDir { .. } | TrtEdgellmSource::AbsoluteDir { .. } => {
            return Err(TrtEdgellmInstallError::NotInstallable(format!(
                "cannot pull an installed trt_edgellm form ({}); pass its declared \
                 remote_archive or vault_remote_archive source instead",
                rt.source
            ))
            .into());
        }
    }
    check_payload(blobs_dir)?;
    Ok(())
}

/// A named flavor is a JetPack build: Linux on arm64. A custom one installs
/// anywhere, and the server's own checks decide.
fn refuse_foreign_flavor(
    flavor: &TrtEdgellmFlavor,
    host_os: &str,
    host_arch: &str,
) -> Result<(), TrtEdgellmInstallError> {
    let (os, arch) = match flavor {
        TrtEdgellmFlavor::JetsonOrinCuda13 => ("linux", "aarch64"),
        TrtEdgellmFlavor::Custom(_) => return Ok(()),
    };
    if (os, arch) == (host_os, host_arch) {
        Ok(())
    } else {
        Err(TrtEdgellmInstallError::ForeignFlavor {
            flavor: flavor.to_string(),
            built_for: format!("{os}-{arch}"),
            host: format!("{host_os}-{host_arch}"),
        })
    }
}

fn check_payload(blobs_dir: &Path) -> Result<(), TrtEdgellmInstallError> {
    let install = TrtEdgellmInstall::under(blobs_dir);
    match install.required().into_iter().find(|path| !path.is_file()) {
        Some(missing) => Err(TrtEdgellmInstallError::MissingFile(
            missing.display().to_string(),
        )),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use httpmock::MockServer;
    use rstest::rstest;

    use pipette_plan_types::{RemoteArchiveUrl, TrtEdgellm};

    use super::*;

    const PAYLOAD: [&str; 2] = [
        "trt-edgellm-server/bin/trt-edgellm-server",
        "trt-edgellm-server/lib/libNvInfer_edgellm_plugin.so",
    ];

    fn tar_gz(entries: &[&str]) -> anyhow::Result<Vec<u8>> {
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

    fn install(entries: &[&str]) -> anyhow::Result<(tempfile::TempDir, anyhow::Result<()>)> {
        let server = MockServer::start();
        let body = tar_gz(entries)?;
        server.mock(|when, then| {
            when.method("GET").path("/trt.tar.gz");
            then.status(200).body(&body);
        });
        // A custom flavor, so the test runs on any host.
        let declared = Runtime::TrtEdgellm(TrtEdgellm {
            source: TrtEdgellmSource::RemoteArchive {
                url: RemoteArchiveUrl::try_new(server.url("/trt.tar.gz"))?,
            },
            flavor: TrtEdgellmFlavor::Custom("test".to_owned()),
        });
        let tmp = tempfile::tempdir()?;
        let result = install_trt_edgellm_archive(
            &HttpClient::new("pipette-test")?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            tmp.path(),
            &mut Reporter::silent(),
        );
        Ok((tmp, result))
    }

    #[test]
    fn a_remote_archive_installs_its_payload() -> anyhow::Result<()> {
        let (tmp, result) = install(&PAYLOAD)?;
        result?;
        assert!(TrtEdgellmInstall::under(tmp.path()).server().is_file());
        Ok(())
    }

    /// Each required file is checked, the plugin included.
    #[rstest]
    #[case::no_plugin(&PAYLOAD[..1], "libNvInfer_edgellm_plugin.so")]
    #[case::flat(&["bin/trt-edgellm-server"], "trt-edgellm-server/bin/trt-edgellm-server")]
    fn an_archive_missing_a_file_is_refused(
        #[case] entries: &[&str],
        #[case] missing: &str,
    ) -> anyhow::Result<()> {
        let (_tmp, result) = install(entries)?;
        let err = result
            .err()
            .context("an incomplete archive must be refused")?;
        // The message quotes a native path, `\` separated on Windows.
        assert!(
            err.to_string().replace('\\', "/").contains(missing),
            "{err}"
        );
        Ok(())
    }

    #[rstest]
    #[case::orin(TrtEdgellmFlavor::JetsonOrinCuda13, "linux", "aarch64", true)]
    #[case::x64_linux(TrtEdgellmFlavor::JetsonOrinCuda13, "linux", "x86_64", false)]
    #[case::mac(TrtEdgellmFlavor::JetsonOrinCuda13, "macos", "aarch64", false)]
    #[case::custom(TrtEdgellmFlavor::Custom("x".to_owned()), "macos", "aarch64", true)]
    fn a_named_flavor_installs_only_on_jetson_hosts(
        #[case] flavor: TrtEdgellmFlavor,
        #[case] os: &str,
        #[case] arch: &str,
        #[case] admitted: bool,
    ) {
        assert_eq!(refuse_foreign_flavor(&flavor, os, arch).is_ok(), admitted);
    }

    #[test]
    fn an_installed_form_is_not_a_pull() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let declared: Runtime = serde_json::from_str(
            r#"{"type":"trt_edgellm","source":"absolute_dir","dir":"/opt/trt","flavor":"x"}"#,
        )?;
        let err = install_trt_edgellm_archive(
            &HttpClient::new("pipette-test")?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            tmp.path(),
            &mut Reporter::silent(),
        )
        .err()
        .context("an installed form must be refused")?;
        assert!(
            matches!(
                err.downcast_ref::<TrtEdgellmInstallError>(),
                Some(TrtEdgellmInstallError::NotInstallable(_))
            ),
            "{err}"
        );
        Ok(())
    }
}
