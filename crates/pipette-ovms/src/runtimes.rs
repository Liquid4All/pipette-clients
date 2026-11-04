//! The OVMS release under a bound runtime's install, resolved from the plan's
//! [`Runtime`].
//!
//! Install is `pipette_artifacts::ensure_runtime`.

use std::ops::Deref;
use std::path::Path;

use pipette_artifacts_catalog::runtime::ovms::layout::OvmsInstall;
use pipette_plan_types::{OvmsSource, Runtime};

/// The release a bound `ovms` runtime unpacked, with its server present.
#[derive(Debug)]
pub struct ResolvedOvmsInstall(OvmsInstall);

impl TryFrom<&Runtime> for ResolvedOvmsInstall {
    type Error = anyhow::Error;

    fn try_from(runtime: &Runtime) -> anyhow::Result<Self> {
        let Runtime::Ovms(rt) = runtime else {
            anyhow::bail!("expected ovms, got `{}`", runtime.headless_token());
        };
        let dir = match &rt.source {
            OvmsSource::AbsoluteDir { dir } => Path::new(dir.as_ref()),
            other @ (OvmsSource::GithubRelease(_)
            | OvmsSource::RemoteArchive { .. }
            | OvmsSource::VaultRemoteArchive { .. }
            | OvmsSource::RelativeDir { .. }) => {
                anyhow::bail!("expected an ovms AbsoluteDir (after bind_under); got {other}")
            }
        };
        let install = OvmsInstall::under(dir);
        let server = install.executable();
        anyhow::ensure!(
            server.is_file(),
            "ovms server missing at {}; the install may be corrupt, re-pull the runtime",
            server.display()
        );
        Ok(Self(install))
    }
}

impl Deref for ResolvedOvmsInstall {
    type Target = OvmsInstall;

    fn deref(&self) -> &OvmsInstall {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{bound_at, unbound};

    fn runtime(runtime: serde_json::Value) -> anyhow::Result<Runtime> {
        Ok(serde_json::from_value(runtime)?)
    }

    #[test]
    fn a_bound_install_with_its_server_is_found() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let server = OvmsInstall::under(tmp.path()).executable();
        if let Some(parent) = server.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&server, b"")?;
        let install = ResolvedOvmsInstall::try_from(&runtime(bound_at(tmp.path()))?)?;
        assert_eq!(install.executable(), server);
        Ok(())
    }

    #[test]
    fn a_missing_server_is_refused_by_path() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let Err(err) = ResolvedOvmsInstall::try_from(&runtime(bound_at(tmp.path()))?) else {
            anyhow::bail!("expected a missing-server rejection");
        };
        assert!(err.to_string().contains("ovms server missing"), "{err}");
        Ok(())
    }

    #[rstest::rstest]
    #[case::unbound(unbound(), "after bind_under")]
    #[case::other_runtime(serde_json::json!({"type": "apple_foundation"}), "expected ovms")]
    fn a_runtime_that_is_not_a_bound_ovms_is_refused(
        #[case] bound: serde_json::Value,
        #[case] reason: &str,
    ) -> anyhow::Result<()> {
        let Err(err) = ResolvedOvmsInstall::try_from(&runtime(bound)?) else {
            anyhow::bail!("expected a rejection");
        };
        assert!(err.to_string().contains(reason), "{err}");
        Ok(())
    }
}
