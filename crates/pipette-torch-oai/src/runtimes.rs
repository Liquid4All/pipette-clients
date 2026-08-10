//! Bound-runtime projection for a prepared `RunRequest`, the runtime-side
//! counterpart to [`crate::models`].
//!
//! Install is `pipette_artifacts::ensure_runtime`. No install path is derived
//! here: the bound runtime already carries the venv, so torch-oai never
//! rebuilds the store's layout or re-reads a manifest to find one.

use std::path::PathBuf;

use pipette_plan_types::RuntimeType;

/// Bound UV venv root after prepare: `AbsolutePreinstalled` install dir
/// (the venv itself — same layout mlx uses for `require_mlx_python`).
pub fn require_uv_venv(req: &pipette_plan_types::run::RunRequest) -> anyhow::Result<PathBuf> {
    pipette_venv::require_bound_venv(
        &req.runtime.bound,
        &[RuntimeType::UvVllm, RuntimeType::UvSglang],
    )
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use pipette_plan_types as plan_types;

    use pipette_plan_types::{
        AbsolutePath, NonEmptyString, UvBuild, UvPythonVersion, UvRuntimeSource, UvServerVersion,
        UvVllm,
    };

    use super::*;

    #[test]
    fn require_uv_venv_rejects_a_non_uv_runtime() -> anyhow::Result<()> {
        let runtime = plan_types::Runtime::DockerVllm(plan_types::DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_string())?,
            image_tag: NonEmptyString::try_new("v0.20.2".to_string())?,
            flavor: pipette_plan_types::VllmFlavor::NvidiaGpu,
        });
        let req = stub_req(runtime);
        let err = require_uv_venv(&req)
            .err()
            .context("expected a rejection error")?;
        assert!(
            err.to_string().contains("expected uv_vllm / uv_sglang"),
            "got {err:#}"
        );
        Ok(())
    }

    #[test]
    fn require_uv_venv_rejects_a_venv_without_python() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let runtime = plan_types::Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.21.0".to_string())?,
            build: UvBuild::try_new("cpu".to_string())?,
            python_version: UvPythonVersion::try_new("3.12".to_string())?,
            source: UvRuntimeSource::AbsolutePreinstalled {
                dir: AbsolutePath::try_new(tmp.path().to_string_lossy().into_owned())?,
            },
        });
        let req = stub_req(runtime);
        let err = require_uv_venv(&req)
            .err()
            .context("expected a rejection error")?;
        // `{:#}` — the detail comes from the shared ops helper, one level below
        // this function's own context.
        let msg = format!("{err:#}");
        assert!(msg.contains("python missing"), "got {msg}");
        Ok(())
    }

    /// A `RunRequest` carrying `runtime` as both declared and bound. Only the
    /// bound runtime matters to [`require_uv_venv`].
    fn stub_req(runtime: plan_types::Runtime) -> pipette_plan_types::run::RunRequest {
        use pipette_plan_types::benchmark::{BenchmarkDefinition, PrefillThroughput};
        use pipette_plan_types::run::DeclaredBound;

        let model = plan_types::Model::AppleFoundationText;
        pipette_plan_types::run::RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
                benchmark_id: "prefill".into(),
                parameter_prefill_tokens: 512,
            }),
        }
    }
}
