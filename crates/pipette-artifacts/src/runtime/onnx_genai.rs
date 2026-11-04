//! Installer for the `uv_onnx_genai` runtime: an `onnxruntime-genai` venv under
//! the store-owned `blobs/` payload dir.
//!
//! Not platform-gated, unlike [`crate::runtime::uv`] and
//! [`crate::runtime::mlx`]: the CPU execution provider runs on macOS, Linux and
//! Windows. The provider is already pinned in the requirements body; it is read
//! here only for the error context.

use std::path::Path;

use anyhow::Context;

use pipette_plan_types::Runtime;
use pipette_venv::{VenvInstall, install_venv};

/// `wheelhouse` is the unpacked vault wheelhouse for that source arm.
pub fn install_onnx_genai_runtime(
    uv: &Path,
    declared: &Runtime,
    blobs_dir: &Path,
    wheelhouse: Option<&Path>,
) -> anyhow::Result<()> {
    let (execution_provider, python_version, source) = match declared {
        Runtime::UvOnnxGenai(rt) => (
            &rt.execution_provider,
            rt.python_version.as_ref(),
            &rt.source,
        ),
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
        | Runtime::UvOpenvino(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_)
        | Runtime::AppleCoreAiMacosPipette(_)
        | Runtime::AppleFoundation(_)) => anyhow::bail!(
            "install_onnx_genai_runtime only installs uv_onnx_genai, got `{}`",
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
            // The catalog row is already a full pin set, so a compiled
            // lockfile would only restate it.
            compile_lockfile: false,
        },
    )
    .with_context(|| {
        format!(
            "installing `{}` for the {execution_provider} execution provider",
            declared.headless_token()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pipette_plan_types::{
        NonEmptyString, OnnxExecutionProvider, UvOnnxGenai, UvPythonVersion, UvRuntimeSource,
        UvServerVersion,
    };

    use super::*;

    fn onnx_genai_runtime() -> anyhow::Result<Runtime> {
        Ok(Runtime::UvOnnxGenai(UvOnnxGenai {
            server_version: UvServerVersion::try_new("0.16.0".to_owned())?,
            execution_provider: OnnxExecutionProvider::Cpu,
            python_version: UvPythonVersion::try_new("3.11".to_owned())?,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("onnxruntime-genai==0.16.0\n".to_owned())?,
                install_flags: None,
            },
        }))
    }

    #[test]
    fn rejects_a_runtime_it_does_not_install() -> anyhow::Result<()> {
        let uv = PathBuf::from("/nonexistent/uv");
        let Err(err) = install_onnx_genai_runtime(
            &uv,
            &Runtime::AppleFoundation(Default::default()),
            Path::new("/tmp"),
            None,
        ) else {
            anyhow::bail!("expected a wrong-runtime rejection");
        };
        assert!(
            format!("{err:#}").contains("only installs uv_onnx_genai"),
            "got {err:#}"
        );
        Ok(())
    }

    #[test]
    fn is_not_platform_gated() -> anyhow::Result<()> {
        let uv = PathBuf::from("/nonexistent/uv");
        let tmp = tempfile::tempdir()?;
        let Err(err) = install_onnx_genai_runtime(&uv, &onnx_genai_runtime()?, tmp.path(), None)
        else {
            anyhow::bail!("expected the missing uv binary to fail the install");
        };
        let msg = format!("{err:#}");
        assert!(!msg.contains("requires Linux"), "got {msg}");
        assert!(!msg.contains("requires macOS"), "got {msg}");
        assert!(!msg.contains("requires Windows"), "got {msg}");
        Ok(())
    }
}
