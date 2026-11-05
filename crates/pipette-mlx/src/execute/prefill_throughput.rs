//! Prefill throughput over `mlx_lm.server`, through [`measurement::prefill()`].
//!
//! The server reports no per-phase rates, so the reading is the time to the
//! first token of a streamed one-token `/v1/completions`, on the client's clock.

use std::num::NonZeroUsize;

use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::measurement::{
    self, MeasurementObserver, MeasurementWarmup, PrefillMeasurementPerPromptReading,
    PrefillMeasurementReading,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    MlxMacosPipetteExtraRuntimeConfiguration, PrefillThroughputBenchmark, RunResponse,
};
use pipette_subprocess::start_process;

use crate::models::ResolvedMlxModel;
use crate::runtimes::ResolvedMlxPython;
use crate::{completions, server, tokenizer};

pub(super) fn run(
    benchmark: &PrefillThroughputBenchmark,
    venv_python: &ResolvedMlxPython,
    model_dir: &ResolvedMlxModel,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let (prompt, eos_ids) = tokenizer::prompt(
        venv_python,
        model_dir,
        benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
        server::TEXT_TRUSTS_REMOTE_CODE,
    )?;
    let request = completions::completion_request(&prompt, 1, &eos_ids);
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
    let result = measurement::prefill(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Once,
        (),
        |()| {
            let generation = pipette_completions::stream(&http, &completions_url, &request)?;
            let reading = PrefillMeasurementPerPromptReading::from_generation(&generation)?;
            Ok(PrefillMeasurementReading::PerPrompt(vec![reading]))
        },
    );
    let (memory, _) = sampler.stop();
    let result_data = result.map_err(|err| err.context(mlx_server.stop_with_tail()))?;
    Ok(RunResponseBuilder::new(result_data)
        .process(&mut mlx_server)
        .memory(memory)
        .build())
}
