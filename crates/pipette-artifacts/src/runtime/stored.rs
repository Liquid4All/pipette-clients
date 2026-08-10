//! [`to_stored`] — map a declared runtime to its effective (installed) form.
//!
//! Parallel to model [`crate::model::to_stored`]: plans/URIs use fetch
//! coordinates; the manifest `stored` field is the entry-relative effective
//! runtime (RelativeDir / RelativePreinstalled / docker identity).

use pipette_plan_types::{
    LlamacppCliStockTools, LlamacppCliStockToolsSource, MlxMacosPipette, RelativePath, Runtime,
    UvOpenvino, UvRuntimeSource, UvSglang, UvVllm,
};

use crate::entry::BLOBS_DIR_NAME;

/// Why [`to_stored`] could not build an effective runtime.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RuntimeStoredError {
    #[error("runtime `{0}` has no store-local form (not installable on this host)")]
    NotStorable(String),
    #[error("`{0}` is not a valid entry-relative path")]
    InvalidPath(String),
}

/// Entry-relative payload root used for file-tree installs (`blobs`).
fn blobs_dir() -> Result<RelativePath, RuntimeStoredError> {
    RelativePath::try_new(BLOBS_DIR_NAME.to_owned())
        .map_err(|_| RuntimeStoredError::InvalidPath(BLOBS_DIR_NAME.to_owned()))
}

/// Entry-relative UV/MLX venv root (`blobs/venv`).
fn blobs_venv_dir() -> Result<RelativePath, RuntimeStoredError> {
    let p = format!("{BLOBS_DIR_NAME}/venv");
    RelativePath::try_new(p.clone()).map_err(|_| RuntimeStoredError::InvalidPath(p))
}

/// Effective form of `declared` under the store entry (models-style `stored`).
///
/// | declared | stored |
/// |----------|--------|
/// | llama release/archive | same kind, `RelativeDir { dir: blobs }` |
/// | llama already RelativeDir | re-home to `blobs` |
/// | docker | clone (image is the handle; no tree) |
/// | uv/mlx installable | same kind, `RelativePreinstalled { dir: blobs/venv }` |
/// | uv/mlx already RelativePreinstalled | re-home to `blobs/venv` |
/// | apk/ios/AFM | error |
pub fn to_stored(declared: &Runtime) -> Result<Runtime, RuntimeStoredError> {
    match declared {
        Runtime::LlamacppCliStockTools(rt) => {
            Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
                source: LlamacppCliStockToolsSource::RelativeDir { dir: blobs_dir()? },
                flavor: rt.flavor.clone(),
            }))
        }
        Runtime::DockerVllm(rt) => Ok(Runtime::DockerVllm(rt.clone())),
        Runtime::DockerSglang(rt) => Ok(Runtime::DockerSglang(rt.clone())),
        Runtime::UvVllm(rt) => Ok(Runtime::UvVllm(UvVllm {
            server_version: rt.server_version.clone(),
            build: rt.build.clone(),
            python_version: rt.python_version.clone(),
            source: UvRuntimeSource::RelativePreinstalled {
                dir: blobs_venv_dir()?,
            },
        })),
        Runtime::UvSglang(rt) => Ok(Runtime::UvSglang(UvSglang {
            server_version: rt.server_version.clone(),
            build: rt.build.clone(),
            python_version: rt.python_version.clone(),
            source: UvRuntimeSource::RelativePreinstalled {
                dir: blobs_venv_dir()?,
            },
        })),
        Runtime::UvOpenvino(rt) => Ok(Runtime::UvOpenvino(UvOpenvino {
            server_version: rt.server_version.clone(),
            python_version: rt.python_version.clone(),
            source: UvRuntimeSource::RelativePreinstalled {
                dir: blobs_venv_dir()?,
            },
        })),
        Runtime::MlxMacosPipette(rt) => Ok(Runtime::MlxMacosPipette(MlxMacosPipette {
            version: rt.version.clone(),
            flavor: rt.flavor,
            source: UvRuntimeSource::RelativePreinstalled {
                dir: blobs_venv_dir()?,
            },
        })),
        Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::AppleFoundation(_) => Err(RuntimeStoredError::NotStorable(declared.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{
        DockerVllm, LlamaCppFlavor, LlamacppCliStockToolsSource, MlxMacosPipette,
        MlxMacosPipetteFlavor, NonEmptyString, RepositoryUrl, SourceRepository, UvBuild,
        UvPythonVersion, UvRuntimeSource, UvServerVersion, VllmFlavor,
    };

    use super::*;

    #[test]
    fn llama_release_stores_as_local_dir_blobs() -> anyhow::Result<()> {
        let declared = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new("b9305".to_owned())?,
            }),
            flavor: LlamaCppFlavor::MacosArm64,
        });
        let stored = to_stored(&declared)?;
        let Runtime::LlamacppCliStockTools(rt) = stored else {
            anyhow::bail!("expected llama");
        };
        assert!(matches!(
            rt.source,
            LlamacppCliStockToolsSource::RelativeDir { ref dir } if dir.as_ref() == "blobs"
        ));
        assert_eq!(rt.flavor, LlamaCppFlavor::MacosArm64);
        Ok(())
    }

    #[test]
    fn docker_stored_is_declared_clone() -> anyhow::Result<()> {
        let declared = Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_owned())?,
            image_tag: NonEmptyString::try_new("v0.10.0".to_owned())?,
            flavor: VllmFlavor::NvidiaGpu,
        });
        assert_eq!(to_stored(&declared)?, declared);
        Ok(())
    }

    #[test]
    fn uv_stores_as_preinstalled_blobs_venv() -> anyhow::Result<()> {
        let declared = Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.22.0".to_owned())?,
            build: UvBuild::try_new("cu129".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("vllm==0.22.0\n".to_owned())?,
                install_flags: None,
            },
        });
        let Runtime::UvVllm(rt) = to_stored(&declared)? else {
            anyhow::bail!("expected uv-vllm");
        };
        assert!(matches!(
            rt.source,
            UvRuntimeSource::RelativePreinstalled { ref dir } if dir.as_ref() == "blobs/venv"
        ));
        Ok(())
    }

    #[test]
    fn mlx_stores_as_preinstalled_blobs_venv() -> anyhow::Result<()> {
        let declared = Runtime::MlxMacosPipette(MlxMacosPipette {
            version: NonEmptyString::try_new("0.31.3".to_owned())?,
            flavor: MlxMacosPipetteFlavor::MacosArm64,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("mlx-lm==0.31.3\n".to_owned())?,
                install_flags: None,
            },
        });
        let Runtime::MlxMacosPipette(rt) = to_stored(&declared)? else {
            anyhow::bail!("expected mlx");
        };
        assert!(matches!(
            rt.source,
            UvRuntimeSource::RelativePreinstalled { ref dir } if dir.as_ref() == "blobs/venv"
        ));
        Ok(())
    }

    #[test]
    fn on_device_is_not_storable() {
        assert!(matches!(
            to_stored(&Runtime::AppleFoundation(Default::default())),
            Err(RuntimeStoredError::NotStorable(_))
        ));
    }
}
