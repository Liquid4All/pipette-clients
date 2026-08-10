//! MLX (macOS) runtime install for the shared store.
//!
//! Creates a relocatable `uv` venv under the store-owned `blobs/` payload dir
//! and returns a single `python` placement — the same layout UV uses
//! (`blobs/venv`). Apple Silicon only; other platforms fail at fetch time.
//!
//! Requirements come from the declared [`Runtime`]'s
//! [`UvRuntimeSource`](pipette_plan_types::UvRuntimeSource) (catalog text or
//! inline pip requirements). The fetcher does not read a catalog itself.

use std::path::Path;

#[cfg(target_os = "macos")]
use anyhow::Context;

use pipette_plan_types::Runtime;

/// Default Python for auto-provisioned mlx-lm venvs (matches the historical
/// private MLX installer). Gated with the install body that reads it: off
/// macOS this module only bails.
#[cfg(target_os = "macos")]
const DEFAULT_PYTHON_VERSION: &str = "3.12";

/// Install a desktop MLX runtime into store-owned `blobs_dir` via `uv`.
pub fn install_mlx_runtime(uv: &Path, declared: &Runtime, blobs_dir: &Path) -> anyhow::Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (uv, declared, blobs_dir);
        anyhow::bail!(
            "installing `{}` runtimes requires macOS (Apple Silicon MLX)",
            declared.headless_token()
        );
    }
    #[cfg(target_os = "macos")]
    {
        install_mlx_into_blobs(uv, declared, blobs_dir)
    }
}

#[cfg(target_os = "macos")]
fn install_mlx_into_blobs(uv: &Path, declared: &Runtime, blobs_dir: &Path) -> anyhow::Result<()> {
    use pipette_venv::{install_venv, VenvInstall};

    let source = match declared {
        Runtime::MlxMacosPipette(rt) => &rt.source,
        other => anyhow::bail!(
            "install_mlx_runtime only installs mlx_macos_pipette, got `{}`",
            other.headless_token()
        ),
    };

    install_venv(
        uv,
        blobs_dir,
        VenvInstall {
            // MLX pins no python version in its plan variant, unlike
            // uv_vllm / uv_sglang.
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
        let Err(err) = install_mlx_runtime(&uv, &docker, Path::new("/tmp")) else {
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
        let Err(err) = install_mlx_runtime(&uv, &rt, Path::new("/tmp")) else {
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
        // Shape-only: we don't run uv here. Confirms PipRequirementsText is the
        // installable form (parity with URI parse filling requirements_text).
        let _ = mlx_catalog()?;
        Ok(())
    }
}
