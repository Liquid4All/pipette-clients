//! Store install for the uv-backed vLLM / SGLang runtimes (Linux only); the
//! venv work is [`pipette_venv`]'s.

use std::path::Path;

#[cfg(target_os = "linux")]
use anyhow::Context;

use pipette_plan_types::Runtime;
#[cfg(target_os = "linux")]
use pipette_venv::{VenvInstall, install_venv};

/// Install a UV vLLM/SGLang runtime into store-owned `blobs_dir` via `uv`.
/// `wheelhouse` is the unpacked vault wheelhouse for that source arm.
pub fn install_uv_runtime(
    uv: &Path,
    declared: &Runtime,
    blobs_dir: &Path,
    wheelhouse: Option<&Path>,
) -> anyhow::Result<()> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (uv, declared, blobs_dir, wheelhouse);
        anyhow::bail!(
            "installing `{}` runtimes requires Linux (uv vLLM/SGLang)",
            declared.headless_token()
        );
    }
    #[cfg(target_os = "linux")]
    {
        install_uv_into_blobs(uv, declared, blobs_dir, wheelhouse)
    }
}

#[cfg(target_os = "linux")]
fn install_uv_into_blobs(
    uv: &Path,
    declared: &Runtime,
    blobs_dir: &Path,
    wheelhouse: Option<&Path>,
) -> anyhow::Result<()> {
    let (python_version, source) = match declared {
        Runtime::UvVllm(rt) => (rt.python_version.as_ref(), &rt.source),
        Runtime::UvSglang(rt) => (rt.python_version.as_ref(), &rt.source),
        other @ (Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::DockerVllm(_)
        | Runtime::DockerSglang(_)
        | Runtime::UvOpenvino(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_)
        | Runtime::AppleCoreAiMacosPipette(_)
        | Runtime::AppleFoundation(_)) => anyhow::bail!(
            "install_uv_runtime only installs uv_vllm / uv_sglang, got `{}`",
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
            // vLLM/SGLang dependency sets are large and resolve differently over
            // time; record what this install actually resolved to.
            compile_lockfile: true,
        },
    )
    .with_context(|| format!("installing `{}`", declared.headless_token()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pipette_plan_types::{DockerVllm, NonEmptyString, VllmFlavor};

    use super::*;

    fn docker_runtime() -> anyhow::Result<Runtime> {
        Ok(Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_owned())?,
            image_tag: NonEmptyString::try_new("v0.1".to_owned())?,
            flavor: VllmFlavor::Cpu,
        }))
    }

    // Off Linux the gate is reached before the variant and source checks.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn install_uv_runtime_reports_the_linux_requirement() -> anyhow::Result<()> {
        let uv = PathBuf::from("/nonexistent/uv");
        let Err(err) = install_uv_runtime(&uv, &docker_runtime()?, Path::new("/tmp"), None) else {
            anyhow::bail!("expected a platform rejection");
        };
        assert!(format!("{err:#}").contains("requires Linux"), "got {err:#}");
        Ok(())
    }

    #[cfg(target_os = "linux")]
    mod linux {
        use rstest::rstest;

        use pipette_plan_types::{
            RelativePath, UvBuild, UvPythonVersion, UvRuntimeSource, UvServerVersion, UvVllm,
        };

        use super::*;

        fn preinstalled_uv_runtime() -> anyhow::Result<Runtime> {
            Ok(Runtime::UvVllm(UvVllm {
                server_version: UvServerVersion::try_new("0.21.0".to_owned())?,
                build: UvBuild::try_new("cpu".to_owned())?,
                python_version: UvPythonVersion::try_new("3.12".to_owned())?,
                source: UvRuntimeSource::RelativePreinstalled {
                    dir: RelativePath::try_new("venv".to_owned())?,
                },
            }))
        }

        #[rstest]
        #[case::wrong_variant(docker_runtime(), "only installs uv_vllm / uv_sglang")]
        #[case::unfetchable_source(preinstalled_uv_runtime(), "cannot fetch a preinstalled")]
        fn install_uv_runtime_rejects(
            #[case] declared: anyhow::Result<Runtime>,
            #[case] expected: &str,
        ) -> anyhow::Result<()> {
            let uv = PathBuf::from("/nonexistent/uv");
            let Err(err) = install_uv_runtime(&uv, &declared?, Path::new("/tmp"), None) else {
                anyhow::bail!("expected a rejection before any uv invocation");
            };
            let msg = format!("{err:#}");
            assert!(msg.contains(expected), "expected {expected:?}, got {msg}");
            Ok(())
        }
    }
}
