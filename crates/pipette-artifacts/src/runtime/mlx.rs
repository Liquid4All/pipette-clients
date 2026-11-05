//! MLX (macOS) runtime install: a `uv` venv under the store-owned `blobs/`
//! payload dir, the same layout as [`crate::runtime::uv`]. Apple Silicon only;
//! other platforms fail at install time.

use std::path::Path;

#[cfg(target_os = "macos")]
use anyhow::Context;

use pipette_plan_types::Runtime;

/// The plan's MLX variant pins no Python version, unlike the uv runtimes.
#[cfg(target_os = "macos")]
const DEFAULT_PYTHON_VERSION: &str = "3.12";

/// Install a desktop MLX runtime into store-owned `blobs_dir` via `uv`.
/// `wheelhouse` is the unpacked vault wheelhouse for that source arm.
pub fn install_mlx_runtime(
    uv: &Path,
    declared: &Runtime,
    blobs_dir: &Path,
    wheelhouse: Option<&Path>,
) -> anyhow::Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (uv, declared, blobs_dir, wheelhouse);
        anyhow::bail!(
            "installing `{}` runtimes requires macOS (Apple Silicon MLX)",
            declared.headless_token()
        );
    }
    #[cfg(target_os = "macos")]
    {
        install_mlx_into_blobs(uv, declared, blobs_dir, wheelhouse)
    }
}

#[cfg(target_os = "macos")]
fn install_mlx_into_blobs(
    uv: &Path,
    declared: &Runtime,
    blobs_dir: &Path,
    wheelhouse: Option<&Path>,
) -> anyhow::Result<()> {
    use pipette_venv::{VenvInstall, install_venv};

    let source = match declared {
        Runtime::MlxMacosPipette(rt) => &rt.source,
        other @ (Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::DockerVllm(_)
        | Runtime::DockerSglang(_)
        | Runtime::UvVllm(_)
        | Runtime::UvSglang(_)
        | Runtime::UvOpenvino(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_)
        | Runtime::AppleCoreAiMacosPipette(_)
        | Runtime::AppleFoundation(_)) => anyhow::bail!(
            "install_mlx_runtime only installs mlx_macos_pipette, got `{}`",
            other.headless_token()
        ),
    };

    install_venv(
        uv,
        blobs_dir,
        VenvInstall {
            wheelhouse,
            python_version: DEFAULT_PYTHON_VERSION,
            source,
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
        MlxMacosPipette, MlxMacosPipetteFlavor, NonEmptyString, RelativePath, UvRuntimeSource,
    };

    use super::*;

    fn mlx_catalog() -> anyhow::Result<Runtime> {
        Ok(Runtime::MlxMacosPipette(MlxMacosPipette {
            version: NonEmptyString::try_new("0.31.3".to_owned())?,
            flavor: MlxMacosPipetteFlavor::MacosArm64,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("mlx-lm==0.31.3\n".to_owned())?,
                install_flags: None,
            },
        }))
    }

    #[test]
    fn rejects_non_mlx_declared() -> anyhow::Result<()> {
        let uv = PathBuf::from("/nonexistent/uv");
        let docker = Runtime::DockerVllm(pipette_plan_types::DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_owned())?,
            image_tag: NonEmptyString::try_new("v0.1".to_owned())?,
            flavor: pipette_plan_types::VllmFlavor::Cpu,
        });
        let Err(err) = install_mlx_runtime(&uv, &docker, Path::new("/tmp"), None) else {
            anyhow::bail!("docker should not be MLX-fetchable");
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("mlx_macos_pipette") || msg.contains("macOS") || msg.contains("Docker"),
            "{msg}"
        );
        Ok(())
    }

    #[test]
    fn rejects_preinstalled() -> anyhow::Result<()> {
        let uv = PathBuf::from("/nonexistent/uv");
        let rt = Runtime::MlxMacosPipette(MlxMacosPipette {
            version: NonEmptyString::try_new("0.31.3".to_owned())?,
            flavor: MlxMacosPipetteFlavor::MacosArm64,
            source: UvRuntimeSource::RelativePreinstalled {
                dir: RelativePath::try_new("venv".to_owned())?,
            },
        });
        let Err(err) = install_mlx_runtime(&uv, &rt, Path::new("/tmp"), None) else {
            anyhow::bail!("preinstalled is not fetchable");
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("preinstalled") || msg.contains("macOS"),
            "{msg}"
        );
        Ok(())
    }

    #[test]
    fn catalog_declared_is_accepted_shape() -> anyhow::Result<()> {
        let _ = mlx_catalog()?;
        Ok(())
    }
}
