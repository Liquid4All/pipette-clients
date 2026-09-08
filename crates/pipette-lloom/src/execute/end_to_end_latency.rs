use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_ops::measurement;
use pipette_ops::prompt_seed;
use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use super::{server, throughput_http};
use crate::models::require_lloom_model_dir;
use crate::runtimes::require_lloom_serve;

const ENDPOINT: &str = "/end_to_end_latency";
const TOKENIZE_ENDPOINT: &str = "/tokenize";

/// `/tokenize` counts with the special tokens the request path adds, so a
/// prompt built to `P` here reports `prompt_tokens == P` on the run.
#[derive(Debug, Serialize)]
struct TokenizeRequest {
    prompt: String,
    add_special_tokens: bool,
}

#[derive(Debug, Deserialize)]
struct TokenizeResponse {
    count: usize,
}

#[derive(Debug, Serialize)]
struct EndToEndLatencyRequest {
    prompt: String,
    decode_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct EndToEndLatencyResponse {
    total_ms: f64,
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug)]
struct LatencySample {
    elapsed_ms: f64,
    server_total_ms: f64,
    prompt_tokens: u32,
    completion_tokens: u32,
}

pub(super) fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_end_to_end_latency()
        .map_err(anyhow::Error::from)?;
    let prefill_tokens = benchmark.parameter_prefill_tokens;
    let decode_tokens = benchmark.parameter_decode_tokens;
    let binary = require_lloom_serve(req)?;
    let model_dir = require_lloom_model_dir(req)?;

    readiness_gate()?;
    let server = server::start_server(&binary, &model_dir, None)?;

    let count_tokens_via_server = |text: &str| -> anyhow::Result<usize> {
        let response: TokenizeResponse = throughput_http::post_json(
            &server.base_url,
            TOKENIZE_ENDPOINT,
            &TokenizeRequest {
                prompt: text.to_string(),
                add_special_tokens: true,
            },
        )?;
        Ok(response.count)
    };
    let prompt = prompt_seed::build_prompt_text(prefill_tokens, count_tokens_via_server)?;

    log::info!("end_to_end_latency: warm-up run ({prefill_tokens}p/{decode_tokens}g)");
    validate_response(
        &run_latency_request(&server.base_url, &prompt, decode_tokens)?,
        prefill_tokens,
        decode_tokens,
    )
    .context("invalid /end_to_end_latency warmup")?;

    let measured = measurement::run(
        "end_to_end_latency",
        readiness_gate,
        observer,
        |_| Ok(()),
        |_| run_latency_request(&server.base_url, &prompt, decode_tokens),
        |idx, rep| {
            validate_response(&rep.value, prefill_tokens, decode_tokens)
                .with_context(|| format!("invalid {ENDPOINT} trial {idx}"))?;
            Ok(rep.elapsed_ms())
        },
    )?;
    let stats = measured.stats();
    let samples = measured
        .into_iter()
        .map(|rep| {
            let elapsed_ms = rep.elapsed_ms();
            sample_from(rep.value, elapsed_ms)
        })
        .collect::<Vec<_>>();
    let stdout = response_stdout(&samples);
    Ok(RunResponse {
        executable: Some(server.executable.clone()),
        command: server.command_preview.clone(),
        ..RunResponse::new(
            BenchmarkResultData::EndToEndLatency {
                total_time_ms: stats.mean_ms,
                total_time_ms_stddev: Some(stats.stddev_ms),
            },
            stdout,
            server.stderr(),
        )
    })
}

fn run_latency_request(
    base_url: &str,
    prompt: &str,
    decode_tokens: u32,
) -> anyhow::Result<EndToEndLatencyResponse> {
    throughput_http::post_json(
        base_url,
        ENDPOINT,
        &EndToEndLatencyRequest {
            prompt: prompt.to_string(),
            decode_tokens,
        },
    )
}

fn sample_from(response: EndToEndLatencyResponse, elapsed_ms: f64) -> LatencySample {
    LatencySample {
        elapsed_ms,
        server_total_ms: response.total_ms,
        prompt_tokens: response.prompt_tokens,
        completion_tokens: response.completion_tokens,
    }
}

fn validate_response(
    response: &EndToEndLatencyResponse,
    expected_prompt_tokens: u32,
    expected_completion_tokens: u32,
) -> anyhow::Result<()> {
    if !response.total_ms.is_finite() || response.total_ms <= 0.0 {
        anyhow::bail!("invalid total_ms: {}", response.total_ms);
    }
    validate_token_count(
        "prompt_tokens",
        response.prompt_tokens,
        expected_prompt_tokens,
    )?;
    validate_token_count(
        "completion_tokens",
        response.completion_tokens,
        expected_completion_tokens,
    )
}

fn validate_token_count(metric: &str, actual: u32, expected: u32) -> anyhow::Result<()> {
    if actual != expected {
        anyhow::bail!("{ENDPOINT} returned {metric} {actual}, expected {expected}");
    }
    Ok(())
}

fn response_stdout(samples: &[LatencySample]) -> String {
    let mut stdout = String::new();
    for (idx, sample) in samples.iter().enumerate() {
        stdout.push_str(&format!(
            "rep {}/{}: {:.3} ms (server {:.3} ms, prompt_tokens={}, completion_tokens={})\n",
            idx + 1,
            samples.len(),
            sample.elapsed_ms,
            sample.server_total_ms,
            sample.prompt_tokens,
            sample.completion_tokens
        ));
    }
    stdout
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_response() -> EndToEndLatencyResponse {
        EndToEndLatencyResponse {
            total_ms: 10.0,
            prompt_tokens: 100,
            completion_tokens: 256,
        }
    }

    #[test]
    fn rejects_token_mismatches_and_bad_times() {
        assert!(validate_response(&valid_response(), 100, 256).is_ok());
        let mut r = valid_response();
        r.prompt_tokens = 99;
        assert!(validate_response(&r, 100, 256).is_err());
        let mut r = valid_response();
        r.completion_tokens = 255;
        assert!(validate_response(&r, 100, 256).is_err());
        let mut r = valid_response();
        r.total_ms = 0.0;
        assert!(validate_response(&r, 100, 256).is_err());
    }

    #[test]
    fn formats_token_counts_in_stdout() {
        let s = response_stdout(&[sample_from(valid_response(), 12.0)]);
        assert!(s.contains(
            "rep 1/1: 12.000 ms (server 10.000 ms, prompt_tokens=100, completion_tokens=256)"
        ));
    }
}
