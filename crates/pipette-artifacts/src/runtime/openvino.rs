//! Installer for the `uv_openvino` runtime: an `openvino-genai` venv under the
//! store-owned `blobs/` payload dir.
//!
//! Not gated at compile time, unlike [`crate::runtime::uv`]: OpenVINO ships
//! Linux and Windows wheels, and Windows is where Intel NPU hardware lives.
//! Other OSes are refused earlier by `pipette-cli`'s `refuse_foreign_os`. One
//! wheel serves CPU, GPU and NPU, so all three share an install.

use std::path::Path;

use anyhow::Context;

use pipette_plan_types::Runtime;
use pipette_venv::{VenvInstall, install_venv};

/// `wheelhouse` is the unpacked vault wheelhouse for that source arm.
pub fn install_openvino_runtime(
    uv: &Path,
    declared: &Runtime,
    blobs_dir: &Path,
    wheelhouse: Option<&Path>,
) -> anyhow::Result<()> {
    let (python_version, source) = match declared {
        Runtime::UvOpenvino(rt) => (rt.python_version.as_ref(), &rt.source),
        other @ (Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::DockerVllm(_)
        | Runtime::DockerSglang(_)
        | Runtime::UvVllm(_)
        | Runtime::UvSglang(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_)
        | Runtime::AppleCoreAiMacosPipette(_)
        | Runtime::AppleFoundation(_)) => anyhow::bail!(
            "install_openvino_runtime only installs uv_openvino, got `{}`",
            other.headless_token()
        ),
    };

    install_venv(
        uv,
        blobs_dir,
        VenvInstall {
            wheelhouse,
            python_version,
            source,
            // The catalog row is already a resolved pin set, so a lockfile
            // would restate it.
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
            server_version: UvServerVersion::try_new("2026.4.1".to_owned())?,
            python_version: UvPythonVersion::try_new("3.11".to_owned())?,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("openvino-genai==2026.4.1.0\n".to_owned())?,
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
            None,
        ) else {
            anyhow::bail!("expected a wrong-runtime rejection");
        };
        assert!(
            format!("{err:#}").contains("only installs uv_openvino"),
            "got {err:#}"
        );
        Ok(())
    }

    #[test]
    fn is_not_platform_gated() -> anyhow::Result<()> {
        let uv = PathBuf::from("/nonexistent/uv");
        let tmp = tempfile::tempdir()?;
        let Err(err) = install_openvino_runtime(&uv, &openvino_runtime()?, tmp.path(), None) else {
            anyhow::bail!("expected the missing uv binary to fail the install");
        };
        let msg = format!("{err:#}");
        assert!(!msg.contains("requires Linux"), "got {msg}");
        assert!(!msg.contains("requires macOS"), "got {msg}");
        Ok(())
    }
}
