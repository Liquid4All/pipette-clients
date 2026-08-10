use std::time::Duration;

use pipette_ops::measurement::{self, REPS};
use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use super::{build_prompt_text, http_timeout, validate_completion_usage};
use crate::{
    openai::{self, CompletionPrompt, CompletionRequest},
    server::ServerState,
};

/// End-to-end latency cell: OpenAI `/v1/completions` against a launched server.
pub(super) fn run(
    req: &RunRequest,
    model: &str,
    state: &ServerState,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let body = req
        .benchmark
        .as_end_to_end_latency()
        .map_err(anyhow::Error::from)?;
    let prefill_tokens = body.parameter_prefill_tokens;
    let decode_tokens = body.parameter_decode_tokens;

    readiness_gate()?;
    let timeout = http_timeout(req);
    // Send a string prompt so the server's tokenize step is part of the
    // measured request, matching the llama.cpp and MLX latency paths.
    let prompt_text = build_prompt_text(&state.base_url(), model, prefill_tokens, timeout)?;

    log::info!("end_to_end_latency: warm-up run");
    let (warmup_prompt_tokens, warmup_completion_tokens) = response_token_counts(
        &run_latency_request(
            &state.base_url(),
            model,
            &prompt_text,
            decode_tokens,
            timeout,
        )?,
        prefill_tokens,
        decode_tokens,
    )?;
    log::info!(
        "warmup: prompt_tokens={}, completion_tokens={}",
        warmup_prompt_tokens,
        warmup_completion_tokens,
    );

    let measured = measurement::run(
        "end_to_end_latency",
        readiness_gate,
        observer,
        // No untimed per-rep setup: the server holds no state a rep resets.
        |_| Ok(()),
        |_| {
            run_latency_request(
                &state.base_url(),
                model,
                &prompt_text,
                decode_tokens,
                timeout,
            )
        },
        |_, rep| {
            response_token_counts(&rep.value, prefill_tokens, decode_tokens)?;
            Ok(rep.elapsed_ms())
        },
    )?;
    let stats = measured.stats();
    // The per-rep check passed, so each response's counts *are* the expected
    // ones — the line reports what was asked for and confirmed, not a re-read.
    // The samples themselves are logged by the harness as it takes them; these
    // lines are the copy that travels with the result.
    let stdout =
        measured
            .into_iter()
            .enumerate()
            .fold(String::new(), |mut stdout, (rep, measured)| {
                let line = format_rep_line(
                    rep,
                    REPS,
                    measured.elapsed_ms(),
                    prefill_tokens,
                    decode_tokens,
                );
                stdout.push_str(&line);
                stdout.push('\n');
                stdout
            });

    Ok(RunResponse {
        executable: Some(state.executable().display().to_string()),
        command: command_preview(state, model, prefill_tokens, decode_tokens),
        ..RunResponse::new(
            BenchmarkResultData::EndToEndLatency {
                total_time_ms: stats.mean_ms,
                total_time_ms_stddev: Some(stats.stddev_ms),
            },
            stdout,
            String::new(),
        )
    })
}

fn run_latency_request(
    base_url: &str,
    model: &str,
    prompt_text: &str,
    decode_tokens: u32,
    timeout: Duration,
) -> anyhow::Result<openai::CompletionResponse> {
    let request = CompletionRequest {
        model: model.to_string(),
        prompt: CompletionPrompt::Text(prompt_text.to_string()),
        max_tokens: Some(decode_tokens),
        temperature: Some(0.0),
        // Force the server to decode the full `decode_tokens` count even
        // if the model wants to emit EOS early.
        ignore_eos: Some(true),
    };
    openai::complete(base_url, &request, timeout)
}

fn response_token_counts(
    response: &openai::CompletionResponse,
    prefill_tokens: u32,
    decode_tokens: u32,
) -> anyhow::Result<(u32, u32)> {
    let usage = validate_completion_usage(response.usage.as_ref(), prefill_tokens, decode_tokens)?;
    Ok((usage.prompt_tokens, usage.completion_tokens))
}

fn format_rep_line(
    rep: usize,
    total: usize,
    elapsed_ms: f64,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> String {
    format!(
        "rep {}/{}: {:.3} ms (prompt_tokens={}, completion_tokens={})",
        rep + 1,
        total,
        elapsed_ms,
        prompt_tokens,
        completion_tokens,
    )
}

fn command_preview(state: &ServerState, model: &str, prefill: u32, decode: u32) -> Vec<String> {
    vec![
        format!("POST {}/v1/completions", state.base_url()),
        format!("model={model} max_tokens={decode} prefill_target_tokens={prefill}"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rep_line_format_matches_runner_contract() {
        let line = format_rep_line(0, 5, 12.5, 100, 200);

        assert_eq!(
            line,
            "rep 1/5: 12.500 ms (prompt_tokens=100, completion_tokens=200)"
        );
        assert!(!line.contains("warmup"));
    }
}
