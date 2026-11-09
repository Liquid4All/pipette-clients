//! [`to_stored`]: a declared runtime's entry-relative installed form, the
//! manifest's `stored` field. Parallel to [`crate::model::to_stored`].

use pipette_plan_types::{
    LlamacppCliStockTools, LlamacppCliStockToolsSource, MlxMacosPipette, Ovms, OvmsSource,
    RelativePath, Runtime, TrtEdgellm, TrtEdgellmSource, UvOnnxGenai, UvOpenvino, UvRuntimeSource,
    UvSglang, UvVllm,
};
use pipette_venv::stored_venv_dir;

use crate::entry::BLOBS_DIR_NAME;

/// Why [`to_stored`] could not build an effective runtime.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RuntimeStoredError {
    #[error("runtime `{0}` has no store-local form (not installable on this host)")]
    NotStorable(String),
    #[error("`{0}` is not a valid entry-relative path")]
    InvalidPath(String),
}

fn blobs_dir() -> Result<RelativePath, RuntimeStoredError> {
    RelativePath::try_new(BLOBS_DIR_NAME.to_owned())
        .map_err(|_| RuntimeStoredError::InvalidPath(BLOBS_DIR_NAME.to_owned()))
}

fn blobs_venv_dir() -> Result<RelativePath, RuntimeStoredError> {
    let p = stored_venv_dir(BLOBS_DIR_NAME);
    RelativePath::try_new(p.clone()).map_err(|_| RuntimeStoredError::InvalidPath(p))
}

/// Docker is cloned unchanged: the image is the handle and there is no tree.
pub fn to_stored(declared: &Runtime) -> Result<Runtime, RuntimeStoredError> {
    match declared {
        Runtime::LlamacppCliStockTools(rt) => {
            Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
                source: LlamacppCliStockToolsSource::RelativeDir { dir: blobs_dir()? },
                flavor: rt.flavor.clone(),
            }))
        }
        Runtime::Ovms(rt) => Ok(Runtime::Ovms(Ovms {
            source: OvmsSource::RelativeDir { dir: blobs_dir()? },
            flavor: rt.flavor.clone(),
        })),
        Runtime::TrtEdgellm(rt) => Ok(Runtime::TrtEdgellm(TrtEdgellm {
            source: TrtEdgellmSource::RelativeDir { dir: blobs_dir()? },
            flavor: rt.flavor.clone(),
        })),
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
        Runtime::UvOnnxGenai(rt) => Ok(Runtime::UvOnnxGenai(UvOnnxGenai {
            server_version: rt.server_version.clone(),
            execution_provider: rt.execution_provider.clone(),
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
        Runtime::AppleCoreAiMacosPipette(rt) => Ok(Runtime::AppleCoreAiMacosPipette(
            pipette_plan_types::AppleCoreAiMacosPipette {
                packages: rt.packages.clone(),
                source: pipette_plan_types::AppleCoreAiMacosPipetteSource::RelativeDir {
                    dir: blobs_dir()?,
                },
            },
        )),
        Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::AppleFoundation(_) => Err(RuntimeStoredError::NotStorable(declared.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{
        DockerVllm, LlamaCppFlavor, LlamacppCliStockToolsRelease, LlamacppCliStockToolsSource,
        MlxMacosPipette, MlxMacosPipetteFlavor, NonEmptyString, RepositoryUrl, UvBuild,
        UvPythonVersion, UvRuntimeSource, UvServerVersion, VllmFlavor,
    };

    use super::*;

    #[test]
    fn llama_release_stores_as_local_dir_blobs() -> anyhow::Result<()> {
        let declared = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(LlamacppCliStockToolsRelease {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new("b9305".to_owned())?,
                asset: None,
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
        // A literal: installed manifests record this path.
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
