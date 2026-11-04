//! The corpus benchmark on SGLang: `vl_spec_decode`, one VL model over a real
//! image + text corpus, with a DSpark drafter on or off.
//!
//! [`crate::launch`] assembles the server's launch with or without the
//! drafter; this module adds the SGLang half of a turn: the request envelope,
//! and where the counters are read from. SGLang reports no per-request
//! prefill or generation time, so the row's `prompt_ms` and `predicted_ms`
//! are the client's TTFT and first-to-last-token spans, comparable within
//! SGLang rows only. See `docs/methodology/vl-spec-decode.md`.

use anyhow::Context;
use serde_json::json;

use pipette_http::HttpClient;
use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_ops::vl_spec_decode::{Speculation, TimingsSource};
use pipette_plan_types::{BenchmarkFlags, SamplingTemperature, VlSpecDecodeBenchmark};
use pipette_vl_spec_decode_corpus::{VlSpecDecodeCorpus, turn};

use crate::server::ServerState;

/// `speculation` is the cell's `speculative_enabled`, resolved against the
/// model.
#[expect(
    clippy::too_many_arguments,
    reason = "the benchmark's parts, the server, the corpus, and the seams its entry takes"
)]
pub(super) fn run(
    benchmark: &VlSpecDecodeBenchmark,
    speculation: Speculation,
    benchmark_flags: Option<&BenchmarkFlags>,
    model: &str,
    state: &ServerState,
    corpus: &VlSpecDecodeCorpus,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
) -> anyhow::Result<RunResponseBuilder> {
    let params = turn::Params::new(
        benchmark,
        benchmark_flags.and_then(BenchmarkFlags::http_timeout),
        benchmark_flags.and_then(BenchmarkFlags::bench_timeout),
        speculation,
    )?;

    let client = HttpClient::loopback("pipette", params.http_timeout)
        .context("failed to build the corpus HTTP client")?
        .client()
        .clone();
    let envelope = envelope(model, params.temperature)?;
    let base_url = state.base_url();
    let response = if params.batch_size.get() == 1 {
        turn::measure(
            benchmark,
            &params,
            corpus,
            readiness_gate,
            observer,
            &|messages, max_tokens| envelope.stream(&client, &base_url, messages, max_tokens),
        )?
    } else {
        // The entry sends a batch as one call; SGLang's chat endpoint takes one
        // conversation per request.
        turn::run(
            &params,
            corpus,
            readiness_gate,
            observer,
            &|messages, max_tokens| envelope.send(&client, &base_url, messages, max_tokens),
        )?
    };
    Ok(response
        .executable(state.executable.display().to_string())
        .command(vec![
            format!("POST {base_url}{}", turn::ENDPOINT),
            format!(
                "model={model} max_tokens={} batch_size={}",
                params.max_tokens, params.batch_size
            ),
        ]))
}

/// SGLang's part of the body: the trailing `usage` chunk for the token counts
/// and its per-request draft counters, read on the client's clock. No prompt
/// cache to disable per request: the server runs with `--disable-radix-cache`,
/// and a turn served from cache is refused on read, which needs
/// `--enable-cache-report` from [`crate::cell_flags::cache_report_argv`].
fn envelope(model: &str, temperature: SamplingTemperature) -> anyhow::Result<turn::Envelope> {
    turn::Envelope::new(
        json!({
            "model": model,
            "stream_options": { "include_usage": true },
            "return_spec_tokens_details": true,
        }),
        TimingsSource::ClientClock,
        temperature,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The envelope asks for everything the reader needs and nothing that
    /// changes the generation: greedy, streamed, usage on the tail, the draft
    /// counters by name.
    #[test]
    fn the_request_asks_for_the_counters_it_reads() -> anyhow::Result<()> {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let envelope = envelope("m", SamplingTemperature::greedy())?;
        assert_eq!(
            envelope.body(messages.clone(), 7),
            json!({
                "model": "m",
                "messages": messages,
                "max_tokens": 7,
                "temperature": 0.0,
                "stream": true,
                "stream_options": { "include_usage": true },
                "return_spec_tokens_details": true,
            })
        );
        assert_eq!(envelope.source(), TimingsSource::ClientClock);
        Ok(())
    }
}
