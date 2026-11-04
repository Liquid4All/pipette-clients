//! Runs an end-to-end cell on a docker vLLM runtime through `run`, with the
//! `docker`, the image and the model directory named on the command line.

use std::path::PathBuf;

use anyhow::Context;

use pipette_ops::EvalCompletionsStore;
use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    AbsolutePath, BatchSize, BenchmarkDefinition, BenchmarkId, Dataset, DeclaredBound, DockerVllm,
    EndToEndLatencyBenchmark, Model, ModelSource, NonEmptyString, NonNegativeCount, PositiveCount,
    RunRequest, Runtime, Torch, VllmFlavor,
};
use pipette_programs::{ExternalPrograms, ResolvedProgram};

struct NoCorpus(PathBuf);

impl RunContext for NoCorpus {
    fn fetch_dataset(&self, _: &Dataset) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("this cell reads no corpus")
    }

    fn datasets_dir(&self) -> PathBuf {
        self.0.join("datasets")
    }

    fn eval_completions(&self) -> EvalCompletionsStore {
        EvalCompletionsStore::new(self.0.join("evals"))
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(docker), Some(image), Some(model)) = (args.next(), args.next(), args.next()) else {
        anyhow::bail!("usage: start_docker_server <docker> <image> <model dir>");
    };
    let (name, tag) = image.split_once(':').context("the image names no tag")?;
    let req = RunRequest {
        runtime: DeclaredBound::already_bound(Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new(name.to_owned())?,
            image_tag: NonEmptyString::try_new(tag.to_owned())?,
            flavor: VllmFlavor::Cpu,
        })),
        model: DeclaredBound::already_bound(Model::Torch(Torch {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(model)?,
            },
        })),
        runtime_flags: None,
        model_flags: None,
        benchmark_flags: None,
        benchmark: BenchmarkDefinition::EndToEndLatency(EndToEndLatencyBenchmark {
            benchmark_id: BenchmarkId::try_new("end_to_end_latency_512_256".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
            parameter_decode_tokens: PositiveCount::try_new(256)?,
            parameter_batch_size: BatchSize::one(),
        }),
    };
    pipette_torch_oai::run(
        &req,
        &NoCorpus(std::env::temp_dir()),
        &|| Ok(()),
        MeasurementObserver {
            on_before_rep_start: &|| Ok(()),
            on_after_rep_end: &|| Ok(()),
        },
        None,
        &ExternalPrograms {
            docker: ResolvedProgram::at(docker),
            ..ExternalPrograms::from_path()
        },
    )?;
    Ok(())
}
