//! The server archive under a bound runtime's install, resolved from the
//! plan's [`Runtime`].
//!
//! Install is `pipette_artifacts::ensure_runtime`.

use std::ops::Deref;
use std::path::Path;

use pipette_artifacts_catalog::runtime::trt_edgellm::TrtEdgellmInstall;
use pipette_plan_types::{Runtime, TrtEdgellmSource};

/// The archive a bound `trt_edgellm` runtime unpacked, with every file it
/// launches present.
#[derive(Debug)]
pub struct ResolvedTrtEdgellmInstall(TrtEdgellmInstall);

impl TryFrom<&Runtime> for ResolvedTrtEdgellmInstall {
    type Error = anyhow::Error;

    fn try_from(runtime: &Runtime) -> anyhow::Result<Self> {
        let Runtime::TrtEdgellm(rt) = runtime else {
            anyhow::bail!("expected trt_edgellm, got `{}`", runtime.headless_token());
        };
        let dir = match &rt.source {
            TrtEdgellmSource::AbsoluteDir { dir } => Path::new(dir.as_ref()),
            other @ (TrtEdgellmSource::RemoteArchive { .. }
            | TrtEdgellmSource::VaultRemoteArchive { .. }
            | TrtEdgellmSource::RelativeDir { .. }) => {
                anyhow::bail!("expected a trt_edgellm AbsoluteDir (after bind_under); got {other}")
            }
        };
        let install = TrtEdgellmInstall::under(dir);
        if let Some(missing) = install.required().iter().find(|path| !path.is_file()) {
            anyhow::bail!(
                "trt_edgellm file missing at {}; the install may be corrupt, re-pull the runtime",
                missing.display()
            );
        }
        Ok(Self(install))
    }
}

impl Deref for ResolvedTrtEdgellmInstall {
    type Target = TrtEdgellmInstall;

    fn deref(&self) -> &TrtEdgellmInstall {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bound_at(dir: &Path) -> anyhow::Result<Runtime> {
        Ok(serde_json::from_value(serde_json::json!({
            "type": "trt_edgellm", "source": "absolute_dir", "dir": dir,
            "flavor": "jetson-orin-cuda-13"
        }))?)
    }

    #[test]
    fn a_complete_install_is_found() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let install = TrtEdgellmInstall::under(tmp.path());
        install.required().iter().try_for_each(|path| {
            std::fs::create_dir_all(path.parent().unwrap_or_else(|| tmp.path()))?;
            std::fs::write(path, b"")
        })?;
        let found = ResolvedTrtEdgellmInstall::try_from(&bound_at(tmp.path())?)?;
        assert_eq!(found.server(), install.server());
        Ok(())
    }

    #[rstest::rstest]
    #[case::empty_install(None, "trt_edgellm file missing")]
    #[case::other_runtime(Some(r#"{"type":"apple_foundation"}"#), "expected trt_edgellm")]
    #[case::unbound(
        Some(r#"{"type":"trt_edgellm","source":"remote_archive","url":"https://example.test/t.tar.gz","flavor":"jetson-orin-cuda-13"}"#),
        "after bind_under"
    )]
    fn a_runtime_that_cannot_launch_is_refused(
        #[case] json: Option<&str>,
        #[case] reason: &str,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let runtime = match json {
            Some(json) => serde_json::from_str(json)?,
            None => bound_at(tmp.path())?,
        };
        let Err(err) = ResolvedTrtEdgellmInstall::try_from(&runtime) else {
            anyhow::bail!("expected a rejection");
        };
        assert!(err.to_string().contains(reason), "{err}");
        Ok(())
    }
}
