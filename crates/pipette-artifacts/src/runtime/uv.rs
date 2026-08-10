//! Store install for the uv-backed vLLM / SGLang runtimes.
//!
//! Dispatches on the declared plan [`Runtime`] variant and hands the venv
//! work to [`pipette_venv`]; the store layout and `ensure_runtime` wiring
//! stay here, alongside the mlx installer that mirrors it.
//!
//! A GPU build ends with an in-venv torch probe, so an install that resolved
//! wheels the host driver can't drive fails before the entry is published.

use std::path::Path;

// Only the Linux install body uses these; the other arm bails immediately.
#[cfg(target_os = "linux")]
use anyhow::Context;

use pipette_plan_types::Runtime;
#[cfg(target_os = "linux")]
use pipette_plan_types::{flavor_from_uv_build, UvBuild, VllmFlavor};
#[cfg(target_os = "linux")]
use pipette_venv::{assert_torch_available, install_venv, VenvInstall};

/// Install a UV vLLM/SGLang runtime into store-owned `blobs_dir` via `uv`.
pub fn install_uv_runtime(uv: &Path, declared: &Runtime, blobs_dir: &Path) -> anyhow::Result<()> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (uv, declared, blobs_dir);
        anyhow::bail!(
            "installing `{}` runtimes requires Linux (uv vLLM/SGLang)",
            declared.headless_token()
        );
    }
    #[cfg(target_os = "linux")]
    {
        install_uv_into_blobs(uv, declared, blobs_dir)
    }
}

#[cfg(target_os = "linux")]
fn install_uv_into_blobs(uv: &Path, declared: &Runtime, blobs_dir: &Path) -> anyhow::Result<()> {
    let (python_version, source, build) = match declared {
        Runtime::UvVllm(rt) => (rt.python_version.as_ref(), &rt.source, &rt.build),
        Runtime::UvSglang(rt) => (rt.python_version.as_ref(), &rt.source, &rt.build),
        other => anyhow::bail!(
            "install_uv_runtime only installs uv_vllm / uv_sglang, got `{}`",
            other.headless_token()
        ),
    };

    let python = install_venv(
        uv,
        blobs_dir,
        VenvInstall {
            python_version,
            source,
            // vLLM/SGLang dependency sets are large and resolve differently over
            // time; record what this install actually resolved to.
            compile_lockfile: true,
        },
    )
    .with_context(|| format!("installing `{}`", declared.headless_token()))?;

    // `python` is still in the staging tree — the venv is `--relocatable`, so it
    // runs there, and probing before the rename is the point: an error here
    // aborts the publish instead of recording an unusable runtime.
    if let Some(backend) = torch_backend_label(build) {
        assert_torch_available(&python, backend).with_context(|| {
            format!("probing torch in the venv for `{declared}`; nothing was recorded")
        })?;
    }
    Ok(())
}

/// Torch GPU backend a uv build targets, or `None` for a CPU build — CPU torch
/// has no GPU backend, so those installs skip the probe.
#[cfg(target_os = "linux")]
fn torch_backend_label(build: &UvBuild) -> Option<&'static str> {
    match flavor_from_uv_build(build) {
        VllmFlavor::NvidiaGpu => Some("CUDA"),
        VllmFlavor::AmdGpu => Some("ROCm"),
        VllmFlavor::Cpu => None,
    }
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

    // Off Linux the platform gate is the whole behavior — the variant and
    // source checks below are never reached, so assert the gate itself rather
    // than let a looser assertion pass for the wrong reason.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn install_uv_runtime_reports_the_linux_requirement() -> anyhow::Result<()> {
        let uv = PathBuf::from("/nonexistent/uv");
        let Err(err) = install_uv_runtime(&uv, &docker_runtime()?, Path::new("/tmp")) else {
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
            let Err(err) = install_uv_runtime(&uv, &declared?, Path::new("/tmp")) else {
                anyhow::bail!("expected a rejection before any uv invocation");
            };
            let msg = format!("{err:#}");
            assert!(msg.contains(expected), "expected {expected:?}, got {msg}");
            Ok(())
        }

        // `None` is what skips the probe, so a CPU build mislabelled as a GPU
        // one would send every `+cpu` install into a check it cannot pass.
        #[rstest]
        #[case::cuda("cu121", Some("CUDA"))]
        #[case::rocm("rocm624", Some("ROCm"))]
        #[case::cpu("cpu", None)]
        fn torch_backend_label_cases(
            #[case] build: &str,
            #[case] expected: Option<&str>,
        ) -> anyhow::Result<()> {
            let build = UvBuild::try_new(build.to_owned())?;
            assert_eq!(torch_backend_label(&build), expected);
            Ok(())
        }
    }
}
