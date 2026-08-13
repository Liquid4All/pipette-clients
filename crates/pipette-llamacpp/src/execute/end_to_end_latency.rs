use std::time::Duration;

use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;

use pipette_http::HttpClient;
use pipette_ops::measurement::{self, REPS};
use pipette_ops::prompt_seed;
use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::reserved_flags::llamacpp_cli_stock_tools as reserved;
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use crate::common::send_json;
use crate::models::require_gguf_text;
use crate::runtime_flags::{self, MmapPolicy};
use crate::runtimes::require_llama_server;
use crate::server;

/// Default HTTP timeout when `benchmark_flags.http_timeout` is unset.
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 120;

/// End-to-end latency cell: bound `llama-server` + GGUF text, typed body.
pub fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_end_to_end_latency()
        .map_err(anyhow::Error::from)?;
    let llama_server = require_llama_server(req)?;
    let model_path = require_gguf_text(req)?;
    let flags = runtime_flags::for_server(
        req,
        benchmark
            .parameter_prefill_tokens
            .saturating_add(benchmark.parameter_decode_tokens),
        MmapPolicy::PinInRam,
    )?;
    let extra_flags = server::args_for(&flags).build(reserved::SERVER, "end_to_end_latency")?;

    readiness_gate()?;
    let prompt_tokens = benchmark.parameter_prefill_tokens;
    let decode_tokens = benchmark.parameter_decode_tokens;
    let http_timeout = http_timeout_from_req(req);
    let mut server = server::start(&llama_server, &model_path, None, &extra_flags)?;
    server.observe_memory();

    let result = (|| -> anyhow::Result<RunResponse> {
        if let Err(e) = server::wait_until_ready(&server.base_url, &mut server.child, http_timeout)
        {
            let stderr = server::shutdown_and_collect_stderr(&mut server);
            if stderr.is_empty() {
                anyhow::bail!("{e}");
            } else {
                anyhow::bail!("{e}\nserver stderr:\n{stderr}");
            }
        }

        let client = HttpClient::blocking_with_timeout("pipette", http_timeout)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let eog_token_ids = server::discover_eog_token_ids(&server);
        let prompt = prompt_seed::build_prompt_text(prompt_tokens, |text| {
            count_tokens(&client, &server.base_url, text)
        })?;
        let stdout = String::new();

        // Warm-up runs the cell's own shape, not the light shared one: llama.cpp
        // selects and compiles kernels per tensor shape, so a 32-token warm-up
        // leaves the 2048-token prefill's pipelines to be built inside rep 1 —
        // measured as a lone first rep 30% above the rest. Reusing the measured
        // prompt is safe because requests set `cache_prompt: false`.
        log::info!("end_to_end_latency: warm-up run ({prompt_tokens}p/{decode_tokens}g)");
        let warmup = send_completion_request(
            &client,
            &server.base_url,
            &prompt,
            decode_tokens,
            &eog_token_ids,
        )?;
        validate_completion(prompt_tokens, decode_tokens, &warmup)?;

        let measured = measurement::run(
            "end_to_end_latency",
            readiness_gate,
            observer,
            // No untimed per-rep setup: the server holds no state a rep resets.
            |_| Ok(()),
            |_| {
                send_completion_request(
                    &client,
                    &server.base_url,
                    &prompt,
                    decode_tokens,
                    &eog_token_ids,
                )
            },
            |_, rep| {
                validate_completion(prompt_tokens, decode_tokens, &rep.value)?;
                Ok(rep.elapsed_ms())
            },
        )?;
        let stats = measured.stats();
        let stdout = measured
            .into_iter()
            .enumerate()
            .fold(stdout, |mut stdout, (i, rep)| {
                append_sample_line(
                    &mut stdout,
                    &format!("rep {}/{}", i + 1, REPS),
                    rep.elapsed_ms(),
                    &rep.value,
                );
                stdout
            });

        Ok(RunResponse {
            executable: Some(llama_server.display().to_string()),
            command: server.command_preview.clone(),
            runtime_flags: Some(flags.clone()),
            memory: server.memory_observation(),
            ..RunResponse::new(
                BenchmarkResultData::EndToEndLatency {
                    total_time_ms: stats.mean_ms,
                    total_time_ms_stddev: Some(stats.stddev_ms),
                },
                stdout,
                String::new(),
            )
        })
    })();

    let _ = server.child.kill();
    let _ = server.child.wait();
    result
}

fn http_timeout_from_req(req: &RunRequest) -> Duration {
    Duration::from_secs(
        req.benchmark_flags
            .as_ref()
            .and_then(|f| f.http_timeout())
            .unwrap_or(DEFAULT_HTTP_TIMEOUT_SECS),
    )
}

fn count_tokens(client: &Client, base_url: &str, text: &str) -> anyhow::Result<usize> {
    // `add_special: true` matches `/completion`'s behavior on tokenizers
    // that auto-prepend a BOS (Gemma, LFM2). On Granite and Qwen2 where no
    // BOS is added, the flag is a no-op. Keeps our pre-flight count exactly
    // in sync with the server-side count `/completion` will report back.
    let response: TokenizeResponse = send_json(
        client
            .post(format!("{base_url}/tokenize"))
            .json(&json!({ "content": text, "add_special": true })),
        "/tokenize",
    )?;
    Ok(response.tokens.len())
}

fn send_completion_request(
    client: &Client,
    base_url: &str,
    prompt_text: &str,
    decode_tokens: u32,
    eog_token_ids: &[u32],
) -> anyhow::Result<CompletionResponse> {
    // Build logit_bias entries that suppress EOG tokens.
    // `false` in llama.cpp logit_bias format means `-INFINITY`.
    // This works around a llama.cpp server bug where `ignore_eos: true`
    // does not actually apply EOG logit biases.
    let logit_bias: Vec<(u32, bool)> = eog_token_ids.iter().map(|&id| (id, false)).collect();

    // Sending `prompt` as a **string** (not a token-ID array) means the
    // server's tokenize step runs inside the timed window, so the recorded
    // total_time_ms is a true end-to-end latency (tokenize + prefill +
    // decode). `validate_completion` cross-checks the server's reported
    // `prompt_n` against our target N.
    send_json(
        client.post(format!("{base_url}/completion")).json(&json!({
            "prompt": prompt_text,
            "temperature": 0.0,
            "n_predict": decode_tokens,
            "ignore_eos": true,
            "logit_bias": logit_bias,
            "cache_prompt": false,
        })),
        "/completion",
    )
}

/// A rep's line in the recorded stdout. The measurement harness has already
/// logged the sample itself, so this is the copy that travels with the result,
/// carrying the token counts the reading describes.
fn append_sample_line(
    stdout: &mut String,
    label: &str,
    elapsed_ms: f64,
    completion: &CompletionResponse,
) {
    let line = format!(
        "{label}: {:.3} ms (prompt_tokens={}, completion_tokens={})",
        elapsed_ms, completion.timings.prompt_n, completion.timings.predicted_n
    );
    stdout.push_str(&line);
    stdout.push('\n');
}

fn validate_completion(
    prompt_tokens: u32,
    decode_tokens: u32,
    completion: &CompletionResponse,
) -> anyhow::Result<()> {
    if completion.timings.prompt_n != prompt_tokens {
        anyhow::bail!(
            "llama-server prompt timing mismatch: expected {} prompt tokens, got {}",
            prompt_tokens,
            completion.timings.prompt_n
        );
    }
    if completion.timings.predicted_n != decode_tokens {
        anyhow::bail!(
            "llama-server generation timing mismatch: expected {} generated tokens, got {}",
            decode_tokens,
            completion.timings.predicted_n
        );
    }
    let stopped_at_limit = completion.stopped_limit
        || (completion.stop && completion.stop_type.as_deref() == Some("limit"));
    if !stopped_at_limit {
        anyhow::bail!("llama-server did not stop at the requested generation limit");
    }
    Ok(())
}

use crate::server::TokenizeResponse;

#[derive(Debug, Deserialize)]
struct CompletionResponse {
    #[serde(default)]
    stopped_limit: bool,
    #[serde(default)]
    stop: bool,
    #[serde(default)]
    stop_type: Option<String>,
    timings: CompletionTimings,
}

#[derive(Debug, Deserialize)]
struct CompletionTimings {
    prompt_n: u32,
    predicted_n: u32,
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    #[test]
    fn validate_completion_accepts_matching_tokens() -> anyhow::Result<()> {
        let completion = CompletionResponse {
            stopped_limit: true,
            stop: true,
            stop_type: Some("limit".to_string()),
            timings: CompletionTimings {
                prompt_n: 100,
                predicted_n: 256,
            },
        };

        validate_completion(100, 256, &completion)?;
        Ok(())
    }

    #[test]
    fn validate_completion_rejects_prompt_mismatch() -> anyhow::Result<()> {
        let completion = CompletionResponse {
            stopped_limit: true,
            stop: true,
            stop_type: Some("limit".to_string()),
            timings: CompletionTimings {
                prompt_n: 99,
                predicted_n: 256,
            },
        };

        let error = validate_completion(100, 256, &completion)
            .err()
            .context("expected prompt-mismatch rejection")?;

        assert!(error
            .to_string()
            .contains("expected 100 prompt tokens, got 99"));
        Ok(())
    }

    #[test]
    fn formats_token_counts_in_stdout() {
        let completion = CompletionResponse {
            stopped_limit: true,
            stop: true,
            stop_type: Some("limit".to_string()),
            timings: CompletionTimings {
                prompt_n: 512,
                predicted_n: 256,
            },
        };
        let mut stdout = String::new();

        append_sample_line(&mut stdout, "rep 1/5", 1234.5, &completion);

        assert_eq!(
            stdout,
            "rep 1/5: 1234.500 ms (prompt_tokens=512, completion_tokens=256)\n"
        );
    }

    #[test]
    fn validate_completion_accepts_stop_type_limit_shape() -> anyhow::Result<()> {
        let completion = CompletionResponse {
            stopped_limit: false,
            stop: true,
            stop_type: Some("limit".to_string()),
            timings: CompletionTimings {
                prompt_n: 8,
                predicted_n: 8,
            },
        };

        validate_completion(8, 8, &completion)?;
        Ok(())
    }
}
