//! Live run of every cell this engine measures, through [`pipette_onnx_genai::run`]
//! against a real venv and model: the server starts, sizes its prompts through
//! `/tokenize` and streams exactly the requested tokens, or the cell fails.
//!
//! **`#[ignore]`.** Everything comes in by environment:
//!
//! - `PIPETTE_LIVE_ONNX_GENAI_VENV`: the venv root holding `onnxruntime-genai`
//! - `PIPETTE_LIVE_ONNX_GENAI_MODEL`: a GenAI model directory
//! - `PIPETTE_LIVE_ONNX_GENAI_EP`: the venv's provider (default `cpu`)
//!
//! ```text
//! cargo test -p pipette-onnx-genai --test live_server_cells -- --ignored --nocapture
//! ```

use anyhow::Context;

use pipette_ops::measurement::MeasurementObserver;
use pipette_plan_types::{
    AbsolutePath, BatchSize, BenchmarkDefinition, BenchmarkId, DeclaredBound,
    DecodeThroughputBenchmark, EndToEndLatencyBenchmark, MaxMemoryUsageBenchmark, Model,
    ModelSource, NonNegativeCount, OnnxGenai, PositiveCount, PrefillThroughputBenchmark,
    RunRequest, Runtime,
};

/// Below any server with its runtime imported. uv's Windows launcher holds
/// about 5 MiB, so a run that watched it, not the interpreter, fails.
const SERVER_FLOOR_BYTES: u64 = 64 << 20;

fn env(name: &str) -> anyhow::Result<String> {
    std::env::var(name).with_context(|| format!("set {name}; see the file's doc comment"))
}

#[test]
#[ignore = "needs an onnxruntime-genai venv and model; see the doc comment"]
fn every_cell_runs_through_the_server() -> anyhow::Result<()> {
    let runtime: Runtime = serde_json::from_value(serde_json::json!({
        "type": "uv_onnx_genai", "server_version": "0.16.0", "python_version": "3.11",
        "execution_provider": std::env::var("PIPETTE_LIVE_ONNX_GENAI_EP")
            .unwrap_or_else(|_| "cpu".to_owned()),
        "source": {"type": "absolute_preinstalled", "dir": env("PIPETTE_LIVE_ONNX_GENAI_VENV")?}
    }))?;
    let model = Model::OnnxGenai(OnnxGenai {
        source: ModelSource::AbsoluteDir {
            dir: AbsolutePath::try_new(env("PIPETTE_LIVE_ONNX_GENAI_MODEL")?)?,
        },
    });
    let one = BatchSize::one();
    [
        BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("prefill_throughput_128".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(128)?,
            parameter_batch_size: one,
        }),
        BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("decode_throughput_128_32".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(128)?,
            parameter_decode_tokens: PositiveCount::try_new(32)?,
            parameter_batch_size: one,
        }),
        BenchmarkDefinition::EndToEndLatency(EndToEndLatencyBenchmark {
            benchmark_id: BenchmarkId::try_new("end_to_end_latency_128_32".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(128)?,
            parameter_decode_tokens: PositiveCount::try_new(32)?,
            parameter_batch_size: one,
        }),
        BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsageBenchmark {
            benchmark_id: BenchmarkId::try_new("max_memory_usage_128".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(128)?,
        }),
    ]
    .into_iter()
    .try_for_each(|benchmark| -> anyhow::Result<()> {
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(runtime.clone()),
            model: DeclaredBound::already_bound(model.clone()),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        };
        let observer = MeasurementObserver::QUIET;
        let response = pipette_onnx_genai::run(&req, &|| Ok(()), observer, None)
            .with_context(|| req.benchmark.benchmark_id().to_owned())?;
        println!(
            "{}: {:?} (memory {:?})",
            req.benchmark.benchmark_id(),
            response.result_data,
            response.memory,
        );
        // Every cell watches its server, the timing cells too.
        anyhow::ensure!(
            response.memory.max_host_bytes > Some(SERVER_FLOOR_BYTES),
            "{}",
            req.benchmark.benchmark_id()
        );
        Ok(())
    })
}
