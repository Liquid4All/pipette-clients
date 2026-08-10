//! Installer for the `uv_openvino` runtime: an `openvino-genai` venv under the
//! store-owned `blobs/` payload dir.
//!
//! Unlike [`crate::runtime::uv`] (Linux) and [`crate::runtime::mlx`] (macOS)
//! this one is not platform-gated. OpenVINO publishes x86_64 wheels for Linux
//! and Windows, and Windows is where Intel NPU hardware lives, so gating to
//! Linux would exclude the runtime's main target. `pipette-venv` handles the
//! `Scripts\`-vs-`bin/` layout difference.
//!
//! Nothing here reads the runtime's `device`: one wheel serves CPU, GPU and
//! NPU, so all three share an install.

use std::path::Path;

use anyhow::Context;

use pipette_plan_types::Runtime;
use pipette_venv::{install_venv, VenvInstall};

pub fn install_openvino_runtime(
    uv: &Path,
    declared: &Runtime,
    blobs_dir: &Path,
) -> anyhow::Result<()> {
    let (python_version, source) = match declared {
        Runtime::UvOpenvino(rt) => (rt.python_version.as_ref(), &rt.source),
        other => anyhow::bail!(
            "install_openvino_runtime only installs uv_openvino, got `{}`",
            other.headless_token()
        ),
    };

    install_venv(
        uv,
        blobs_dir,
        VenvInstall {
            python_version,
            source,
            // The catalog row is already a resolved pin set, so a lockfile
            // would restate it. Matches mlx, which pins the same way.
            compile_lockfile: false,
        },
    )
    .with_context(|| format!("installing `{}`", declared.headless_token()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pipette_plan_types::{
        NonEmptyString, UvOpenvino, UvPythonVersion, UvRuntimeSource, UvServerVersion,
    };

    use super::*;

    fn openvino_runtime() -> anyhow::Result<Runtime> {
        Ok(Runtime::UvOpenvino(UvOpenvino {
            server_version: UvServerVersion::try_new("2026.2.1".to_owned())?,
            python_version: UvPythonVersion::try_new("3.11".to_owned())?,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("openvino-genai==2026.2.1.0\n".to_owned())?,
                install_flags: None,
            },
        }))
    }

    #[test]
    fn rejects_a_runtime_it_does_not_install() -> anyhow::Result<()> {
        let uv = PathBuf::from("/nonexistent/uv");
        let Err(err) = install_openvino_runtime(
            &uv,
            &Runtime::AppleFoundation(Default::default()),
            Path::new("/tmp"),
        ) else {
            anyhow::bail!("expected a wrong-runtime rejection");
        };
        assert!(
            format!("{err:#}").contains("only installs uv_openvino"),
            "got {err:#}"
        );
        Ok(())
    }

    /// No platform gate: the failure a caller hits on any host is the missing
    /// `uv`, not a "requires Linux"-style refusal.
    #[test]
    fn is_not_platform_gated() -> anyhow::Result<()> {
        let uv = PathBuf::from("/nonexistent/uv");
        let tmp = tempfile::tempdir()?;
        let Err(err) = install_openvino_runtime(&uv, &openvino_runtime()?, tmp.path()) else {
            anyhow::bail!("expected the missing uv binary to fail the install");
        };
        let msg = format!("{err:#}");
        assert!(!msg.contains("requires Linux"), "got {msg}");
        assert!(!msg.contains("requires macOS"), "got {msg}");
        Ok(())
    }
}
