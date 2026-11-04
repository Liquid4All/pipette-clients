//! Neither vLLM nor SGLang reports a per-request decode time, so the reading
//! is the interval from the first token to the last of a streamed
//! `/v1/completions`, on the client's clock. A batched entry keeps
//! `parameter_batch_size` of these in flight, each timed from its own send.

use std::num::NonZeroUsize;

use pipette_completions::TimingRequest;
use pipette_http::HttpClient;
use pipette_ops::measurement::{
    self, DecodeMeasurementPerPromptReading, DecodeMeasurementReading, MeasurementObserver,
    MeasurementWarmup,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{DecodeThroughputBenchmark, PositiveSeconds};

use crate::{completions, response, server::ServerState};

pub(super) fn run(
    benchmark: &DecodeThroughputBenchmark,
    http_timeout: Option<PositiveSeconds>,
    model: &str,
    state: &ServerState,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
) -> anyhow::Result<RunResponseBuilder> {
    // Before the prompt is sized: the entry refuses a single token only once
    // the warm-up has run.
    anyhow::ensure!(
        benchmark.parameter_decode_tokens.into_inner() >= 2,
        "parameter_decode_tokens must be at least 2: the first token is the prefill's, so a \
         decode rate starts at the second; got {}",
        benchmark.parameter_decode_tokens
    );
    let batch_size = benchmark.parameter_batch_size;
    readiness_gate()?;
    let timeout = measurement::http_timeout_for(http_timeout, batch_size.as_u32());
    let base_url = state.base_url();
    let http = HttpClient::loopback("pipette", timeout)?;
    let prompt_text = completions::build_prompt_text(
        &http,
        &base_url,
        model,
        benchmark
            .parameter_prefill_tokens
            .into_inner()
            .unsigned_abs(),
    )?;
    let url = completions::url(&base_url);
    let request = TimingRequest::text(
        Some(model),
        &prompt_text,
        benchmark
            .parameter_decode_tokens
            .into_inner()
            .unsigned_abs(),
    );
    let in_flight = batch_size.in_flight()?;
    let result_data = measurement::decode(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Once,
        (),
        |()| {
            let readings = measurement::in_flight(in_flight, |_| {
                let generation = pipette_completions::stream(&http, &url, &request)?;
                DecodeMeasurementPerPromptReading::from_generation(&generation)
            })?;
            Ok(DecodeMeasurementReading::PerPrompt(readings))
        },
    )?;
    Ok(response::timing_response(
        result_data,
        state,
        format!(
            "model={model} max_tokens={} prefill_target_tokens={} stream=true",
            benchmark.parameter_decode_tokens, benchmark.parameter_prefill_tokens
        ),
        batch_size,
    ))
}
