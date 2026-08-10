//! Prefill and decode throughput.
//!
//! Both reduce a rate the driver reports rather than the wall clock, because
//! the wall clock of a one-shot process includes the pipeline compile — up to
//! ~18s on NPU against a sub-second workload. Timing the process would measure
//! the compiler.

use pipette_ops::measurement;
use pipette_ops::prompt_seed::PROMPT_SEED_TEXT;
use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::{RunRequest, RunResponse};

use super::driver::{DriverRequest, Mode};
use super::{invoke, take_output, Cell, LastOutput};

pub(super) fn run_prefill(
    req: &RunRequest,
    compile_cache: &std::path::Path,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_prefill_throughput()
        .map_err(anyhow::Error::from)?;
    let prefill_tokens = benchmark.parameter_prefill_tokens;
    let cell = Cell::bind(req, compile_cache)?;
    let model_dir = cell.model_dir_str()?.to_owned();
    let properties = cell.properties();
    let last = LastOutput::default();

    let measured = measurement::run(
        "openvino/prefill_throughput",
        readiness_gate,
        observer,
        // Nothing to reset between reps: each is a new process.
        |_| Ok(()),
        |_| {
            invoke(
                &cell,
                &DriverRequest {
                    model_dir: &model_dir,
                    device: crate::runtimes::device_property(&cell.device),
                    mode: Mode::Prefill,
                    prefill_tokens,
                    decode_tokens: 1,
                    prompt: None,
                    warmup: Some(cell.warmup(prefill_tokens, 1)),
                    properties: properties.clone(),
                    prompt_seed: PROMPT_SEED_TEXT,
                },
                &last,
            )
        },
        |_, rep| {
            measurement::expect_tokens("input_tokens", rep.value.input_tokens, prefill_tokens)?;
            // Prefill is time-to-first-token: the one generated token is the
            // first, so TTFT is the prefill time with no decode in it.
            measurement::positive_finite("ttft_ms", rep.value.ttft_ms)
        },
    )?;
    let stats = measured.stats();
    let (stdout, stderr) = take_output(last);
    Ok(cell.respond(
        BenchmarkResultData::PrefillThroughput {
            prefill_time_ms: stats.mean_ms,
            prefill_time_ms_stddev: Some(stats.stddev_ms),
        },
        stdout,
        stderr,
    ))
}

pub(super) fn run_decode(
    req: &RunRequest,
    compile_cache: &std::path::Path,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_decode_throughput()
        .map_err(anyhow::Error::from)?;
    let prefill_tokens = benchmark.parameter_prefill_tokens;
    let decode_tokens = benchmark.parameter_decode_tokens;
    let cell = Cell::bind(req, compile_cache)?;
    let model_dir = cell.model_dir_str()?.to_owned();
    let properties = cell.properties();
    let last = LastOutput::default();

    let measured = measurement::run(
        "openvino/decode_throughput",
        readiness_gate,
        observer,
        |_| Ok(()),
        |_| {
            invoke(
                &cell,
                &DriverRequest {
                    model_dir: &model_dir,
                    device: crate::runtimes::device_property(&cell.device),
                    mode: Mode::Decode,
                    prefill_tokens,
                    decode_tokens,
                    prompt: None,
                    warmup: Some(cell.warmup(prefill_tokens, decode_tokens)),
                    properties: properties.clone(),
                    prompt_seed: PROMPT_SEED_TEXT,
                },
                &last,
            )
        },
        |_, rep| {
            measurement::expect_tokens("input_tokens", rep.value.input_tokens, prefill_tokens)?;
            measurement::expect_tokens(
                "generated_tokens",
                rep.value.generated_tokens,
                decode_tokens,
            )?;
            // TPOT excludes the first token, so it is decode alone — the
            // prefill sits in TTFT, which this cell does not report.
            let tpot = rep
                .value
                .tpot_ms
                .ok_or_else(|| anyhow::anyhow!("driver reported no tpot_ms for a decode cell"))?;
            // The measured rate over the whole requested count, deliberately:
            // this cell reports what generating `decode_tokens` costs at the
            // decode rate. End-to-end latency is where the first token is
            // accounted separately, because there the prefill is in the number.
            Ok(measurement::positive_finite("tpot_ms", tpot)? * f64::from(decode_tokens))
        },
    )?;
    let stats = measured.stats();
    let (stdout, stderr) = take_output(last);
    Ok(cell.respond(
        BenchmarkResultData::DecodeThroughput {
            decode_time_ms: stats.mean_ms,
            decode_time_ms_stddev: Some(stats.stddev_ms),
        },
        stdout,
        stderr,
    ))
}
