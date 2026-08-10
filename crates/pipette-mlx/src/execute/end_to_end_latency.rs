use anyhow::Context;
use serde::{Deserialize, Serialize};

use pipette_ops::measurement;
use pipette_ops::prompt_seed;
use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use super::{server, throughput_http};
use crate::models::require_mlx_model_dir;
use crate::runtimes::require_mlx_python;

const ENDPOINT: &str = "/end_to_end_latency";
const TOKENIZE_ENDPOINT: &str = "/tokenize";

#[derive(Debug, Serialize)]
struct TokenizeRequest {
    prompt: String,
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
    /// What the server timed around generation alone. The recorded metric is
    /// the client's wall clock — the same one llama.cpp and vLLM report, so
    /// the runtimes stay comparable — and the difference between the two is
    /// this server's HTTP and JSON cost. Recorded rather than dropped so that
    /// overhead is inspectable in the result instead of invisible inside it.
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
    let venv_python = require_mlx_python(req)?;
    let model_dir = require_mlx_model_dir(req)?;

    readiness_gate()?;
    let server = server::start_server(&venv_python, &model_dir, None)?;
    let count_tokens_via_server = |text: &str| -> anyhow::Result<usize> {
        let response: TokenizeResponse = throughput_http::post_json(
            &server.base_url,
            TOKENIZE_ENDPOINT,
            &TokenizeRequest {
                prompt: text.to_string(),
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
        // No untimed per-rep setup: the server holds no state a rep resets.
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
    validate_ms("total_ms", response.total_ms)?;
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

fn validate_ms(metric: &str, value: f64) -> anyhow::Result<()> {
    if !value.is_finite() || value <= 0.0 {
        anyhow::bail!("invalid {metric}: {value}");
    }
    Ok(())
}

fn validate_token_count(metric: &str, actual: u32, expected: u32) -> anyhow::Result<()> {
    if actual != expected {
        anyhow::bail!("{ENDPOINT} returned {metric} {actual}, expected {expected}");
    }
    Ok(())
}

fn response_stdout(samples: &[LatencySample]) -> String {
    let mut stdout = String::new();
    samples.iter().enumerate().for_each(|(idx, sample)| {
        append_sample_line(
            &mut stdout,
            &format!("rep {}/{}", idx + 1, samples.len()),
            sample,
        );
    });
    stdout
}

fn append_sample_line(stdout: &mut String, label: &str, sample: &LatencySample) {
    stdout.push_str(&format!(
        "{label}: {:.3} ms (server {:.3} ms, prompt_tokens={}, completion_tokens={})\n",
        sample.elapsed_ms, sample.server_total_ms, sample.prompt_tokens, sample.completion_tokens
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    type ResponseMutator = fn(&mut EndToEndLatencyResponse);

    fn valid_response() -> EndToEndLatencyResponse {
        EndToEndLatencyResponse {
            total_ms: 10.0,
            prompt_tokens: 100,
            completion_tokens: 256,
        }
    }

    fn valid_sample(elapsed_ms: f64) -> LatencySample {
        LatencySample {
            elapsed_ms,
            server_total_ms: elapsed_ms - 1.0,
            prompt_tokens: 100,
            completion_tokens: 256,
        }
    }

    #[test]
    fn converts_response_to_sample() {
        let sample = sample_from(valid_response(), 12.25);

        assert_eq!(sample.elapsed_ms, 12.25);
        assert_eq!(sample.prompt_tokens, 100);
        assert_eq!(sample.completion_tokens, 256);
    }

    #[test]
    fn rejects_response_token_mismatches() {
        let cases: &[(&str, ResponseMutator)] = &[
            ("prompt_tokens", |response| {
                response.prompt_tokens = 99;
            }),
            ("completion_tokens", |response| {
                response.completion_tokens = 255;
            }),
        ];

        cases.iter().for_each(|(name, mutate)| {
            let mut response = valid_response();
            mutate(&mut response);
            assert!(
                validate_response(&response, 100, 256).is_err(),
                "case should fail: {name}"
            );
        });
    }

    #[test]
    fn formats_token_counts_in_stdout() {
        let samples = vec![
            valid_sample(10.0),
            valid_sample(12.0),
            valid_sample(11.0),
            valid_sample(13.0),
            valid_sample(14.0),
        ];
        let stdout = response_stdout(&samples);

        assert!(stdout.contains(
            "rep 1/5: 10.000 ms (server 9.000 ms, prompt_tokens=100, completion_tokens=256)"
        ));
        assert!(stdout.contains(
            "rep 5/5: 14.000 ms (server 13.000 ms, prompt_tokens=100, completion_tokens=256)"
        ));
        assert!(!stdout.contains("warmup"));
    }
}
