//! Decode throughput over `mlx_lm.server`, through [`measurement::decode()`].
//!
//! The server reports no per-phase rates, so the reading is the interval from
//! the first token to the last of a streamed `/v1/completions`, on the client's
//! clock. EOS is biased away so the full `max_tokens` comes out.

use std::num::NonZeroUsize;

use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::measurement::{
    self, DecodeMeasurementPerPromptReading, DecodeMeasurementReading, MeasurementObserver,
    MeasurementWarmup,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    DecodeThroughputBenchmark, MlxMacosPipetteExtraRuntimeConfiguration, RunResponse,
};
use pipette_subprocess::start_process;

use crate::models::ResolvedMlxModel;
use crate::runtimes::ResolvedMlxPython;
use crate::{completions, server, tokenizer};

pub(super) fn run(
    benchmark: &DecodeThroughputBenchmark,
    venv_python: &ResolvedMlxPython,
    model_dir: &ResolvedMlxModel,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    // Before the model loads: the entry refuses a single token only once the
    // warm-up has run.
    anyhow::ensure!(
        benchmark.parameter_decode_tokens.into_inner() >= 2,
        "parameter_decode_tokens must be at least 2: the first token is the prefill's, so a \
         decode rate starts at the second; got {}",
        benchmark.parameter_decode_tokens
    );
    let (prompt, eos_ids) = tokenizer::prompt(
        venv_python,
        model_dir,
        benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
        server::TEXT_TRUSTS_REMOTE_CODE,
    )?;
    let request = completions::completion_request(
        &prompt,
        benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
        &eos_ids,
    );
    let http = HttpClient::loopback("pipette", completions::TIMING_TIMEOUT)?;

    readiness_gate()?;
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let completions_url = completions::url(&url);
    let mut mlx_server = start_process(
        server::TEXT_SERVER_MODULE,
        server::text_command(venv_python, model_dir, port, client_config)?,
        server::STOP_GRACE,
    )?;
    let sampler = Sampler::start(mlx_server.pid(), SupportedWatch::OBSERVE)?;
    mlx_server.wait_ready(server::READY_TIMEOUT, server::ready(&url)?)?;
    let result = measurement::decode(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Once,
        (),
        |()| {
            let generation = pipette_completions::stream(&http, &completions_url, &request)?;
            let reading = DecodeMeasurementPerPromptReading::from_generation(&generation)?;
            Ok(DecodeMeasurementReading::PerPrompt(vec![reading]))
        },
    );
    let (memory, _) = sampler.stop();
    let result_data = result.map_err(|err| err.context(mlx_server.stop_with_tail()))?;
    Ok(RunResponseBuilder::new(result_data)
        .process(&mut mlx_server)
        .memory(memory)
        .build())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use pipette_plan_types::{BatchSize, BenchmarkId, NonNegativeCount, PositiveCount};

    use super::*;
    use crate::models::test_support::bound_model;
    use crate::runtimes::test_support::bound_runtime;

    /// Refused before the gate and the tokenizer, too.
    #[test]
    fn a_single_decode_token_is_refused_before_the_model_loads() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let benchmark = DecodeThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("decode_throughput_512_1")?,
            parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
            parameter_decode_tokens: PositiveCount::try_new(1)?,
            parameter_batch_size: BatchSize::one(),
        };
        let gated = Cell::new(false);
        let Err(err) = run(
            &benchmark,
            &ResolvedMlxPython::try_from(&bound_runtime(tmp.path())?)?,
            &ResolvedMlxModel::try_from(&bound_model(tmp.path(), false)?)?,
            &|| {
                gated.set(true);
                Ok(())
            },
            MeasurementObserver::QUIET,
            None,
        ) else {
            anyhow::bail!("a one-token decode cell was measured");
        };
        assert!(format!("{err:#}").contains("at least 2"), "{err:#}");
        assert!(!gated.get(), "refused only after the gate");
        Ok(())
    }
}
