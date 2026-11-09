//! TensorRT Edge-LLM execute: kind dispatch from a prepared [`RunRequest`].
//!
//! Only the VL cells run: the server is a VLA and VLM server, and the plan
//! vocabulary pairs this runtime with `gguf_vision` alone.

mod vl_max_memory;
mod vl_throughput;

use std::path::Path;

use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::readiness::ReadinessGate;
use pipette_plan_types::{
    BenchmarkDefinition, RunRequest, RunResponse, TrtEdgellmExtraRuntimeConfiguration,
    TrtEdgellmGgufVision,
};

use crate::models::ResolvedGgufVisionModel;
use crate::runtimes::ResolvedTrtEdgellmInstall;
use crate::server::Launch;

/// `compile_cache` is where the server keeps its converted checkpoints and
/// engines, keyed by everything an engine depends on; `client_config` is
/// applied whole to the server and its build.
pub fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    compile_cache: &Path,
    client_config: Option<&TrtEdgellmExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let launch = || -> anyhow::Result<Launch> {
        let flags: TrtEdgellmGgufVision = req.runtime_flags_ref()?.try_into()?;
        Ok(Launch::new(
            &ResolvedTrtEdgellmInstall::try_from(&req.runtime.bound)?,
            &ResolvedGgufVisionModel::try_from(&req.model.bound)?,
            &flags,
            compile_cache,
            client_config,
        ))
    };
    match &req.benchmark {
        BenchmarkDefinition::VlThroughput(benchmark) => vl_throughput::run(
            benchmark,
            &req.benchmark_flags_ref()?,
            &launch()?,
            readiness_gate,
            observer,
        ),
        BenchmarkDefinition::VlMaxMemory(benchmark) => {
            vl_max_memory::run(benchmark, &req.benchmark_flags_ref()?, &launch()?, observer)
        }
        other @ (BenchmarkDefinition::PrefillThroughput(_)
        | BenchmarkDefinition::DecodeThroughput(_)
        | BenchmarkDefinition::EndToEndLatency(_)
        | BenchmarkDefinition::MaxMemoryUsage(_)
        | BenchmarkDefinition::Eval(_)
        | BenchmarkDefinition::VlmEval(_)
        | BenchmarkDefinition::AsrEval(_)
        | BenchmarkDefinition::VlSpecDecode(_)
        | BenchmarkDefinition::AsrThroughput(_)
        | BenchmarkDefinition::AsrMaxMemory(_)) => anyhow::bail!(
            "TensorRT Edge-LLM runs vl_throughput and vl_max_memory only, not {}",
            other.benchmark_type()
        ),
    }
}

#[cfg(test)]
mod tests {
    use pipette_artifacts_catalog::runtime::trt_edgellm::TrtEdgellmInstall;
    use pipette_ops::measurement::VlThroughputMeasurementError;
    use pipette_plan_types::{
        BenchmarkId, DeclaredBound, MaxMemoryUsageBenchmark, Model, NonNegativeCount,
        PositiveCount, VlThroughputBenchmark,
    };

    use super::*;

    /// A one-token decode has no decode time, so it is refused before the
    /// engine build. The fixture's server is an empty file, so a build that
    /// ran would fail with another error.
    #[test]
    fn a_one_token_decode_is_refused_before_the_build() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        TrtEdgellmInstall::under(tmp.path())
            .required()
            .iter()
            .try_for_each(|path| {
                std::fs::create_dir_all(path.parent().unwrap_or_else(|| tmp.path()))?;
                std::fs::write(path, b"")
            })?;
        ["model.gguf", "mmproj.gguf"]
            .iter()
            .try_for_each(|name| std::fs::write(tmp.path().join(name), b"gguf"))?;
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(serde_json::from_value(serde_json::json!({
                "type": "trt_edgellm", "source": "absolute_dir", "dir": tmp.path(),
                "flavor": "jetson-orin-cuda-13"
            }))?),
            model: DeclaredBound::already_bound(serde_json::from_value(serde_json::json!({
                "type": "gguf_vision", "source": "absolute_files",
                "model": tmp.path().join("model.gguf"), "mmproj": tmp.path().join("mmproj.gguf"),
            }))?),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::VlThroughput(VlThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new("vl_throughput_256x256_t87_d1_f2")?,
                parameter_image_width: PositiveCount::try_new(256)?,
                parameter_image_height: PositiveCount::try_new(256)?,
                parameter_text_tokens: NonNegativeCount::try_new(87)?,
                parameter_decode_tokens: PositiveCount::try_new(1)?,
                parameter_num_images: NonNegativeCount::try_new(2)?,
            }),
        };
        let observer = MeasurementObserver::QUIET;
        let Err(err) = run(&req, &|| Ok(()), observer, &tmp.path().join("cache"), None) else {
            anyhow::bail!("a one-token decode cell was measured");
        };
        assert!(
            matches!(
                err.downcast_ref::<VlThroughputMeasurementError>(),
                Some(VlThroughputMeasurementError::TooFewTokens { .. })
            ),
            "{err:#}"
        );
        Ok(())
    }

    #[test]
    fn a_text_cell_is_refused_by_name() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(serde_json::from_value(serde_json::json!({
                "type": "trt_edgellm", "source": "absolute_dir", "dir": tmp.path(),
                "flavor": "jetson-orin-cuda-13"
            }))?),
            model: DeclaredBound::already_bound(Model::AppleFoundationText),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsageBenchmark {
                benchmark_id: BenchmarkId::try_new("max_memory_usage_128".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(128)?,
            }),
        };
        let observer = MeasurementObserver::QUIET;
        let Err(err) = run(&req, &|| Ok(()), observer, tmp.path(), None) else {
            anyhow::bail!("a text cell ran");
        };
        let message = format!("{err:#}");
        assert!(
            message.contains("vl_throughput and vl_max_memory only"),
            "{message}"
        );
        assert!(message.contains("max_memory_usage"), "{message}");
        Ok(())
    }
}
