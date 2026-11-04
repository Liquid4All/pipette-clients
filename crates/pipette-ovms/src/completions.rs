//! The OVMS surface a cell drives: `/v3/tokenize` to size a prompt and
//! `/v3/completions` to time it, through [`pipette_completions`].
//!
//! Prompts are text: OVMS refuses token ids (`prompt is not a string`). The
//! stream's empty opening chunk is the first token and its text trails by a
//! few tokens, so the first token is stamped on the opening chunk and the last
//! on the finish chunk.

use anyhow::Context;
use reqwest::Method;
use serde::Deserialize;

use pipette_completions::TimingRequest;
use pipette_http::HttpClient;
use pipette_ops::generation::Generation;
use pipette_ops::measurement;
use pipette_plan_types::{BatchSize, PositiveSeconds};

use crate::server::MODEL_NAME;

/// The bound on each wait of a timing request, per in-flight slot, when the
/// cell pins no `http_timeout`: for the headers, then for each chunk, not for
/// the whole exchange. The longest is the prefill before the first chunk.
const TIMING_TIMEOUT_SECS: u64 = 3600;

/// One client per timing cell, so the reps reuse its pooled connection.
pub(crate) fn timing_client(
    http_timeout: Option<PositiveSeconds>,
    batch_size: BatchSize,
) -> anyhow::Result<HttpClient> {
    Ok(HttpClient::loopback(
        "pipette",
        measurement::http_timeout_with_default(
            http_timeout,
            batch_size.as_u32(),
            TIMING_TIMEOUT_SECS,
        ),
    )?)
}

/// A prompt the server tokenizes to exactly `target_tokens`, sized with its own
/// `/v3/tokenize`, which counts the BOS the completion will add.
pub(crate) fn prompt_text(
    http: &HttpClient,
    base_url: &str,
    target_tokens: u32,
) -> anyhow::Result<String> {
    pipette_ops::prompt_seed::build_prompt_text(target_tokens, |text| {
        count_tokens(http, base_url, text)
    })
    .context("sizing the prompt with /v3/tokenize")
}

fn count_tokens(http: &HttpClient, base_url: &str, text: &str) -> anyhow::Result<usize> {
    #[derive(Deserialize)]
    struct Tokens {
        tokens: Vec<u32>,
    }
    let url = format!("{base_url}/v3/tokenize");
    let body = serde_json::json!({ "model": MODEL_NAME, "text": text });
    let tokens: Tokens = http
        .json_request(Method::POST, &url, None, Some(body))
        .with_context(|| format!("POST {url}"))?;
    Ok(tokens.tokens.len())
}

pub(crate) fn url(base_url: &str) -> String {
    format!("{base_url}/v3/completions")
}

/// Fail if the opening chunk left before the prefill finished, which would
/// read TTFT near zero. In a one-token stream it lands with the token, so it
/// cannot lead the finish chunk by more than it trails the send.
pub(crate) fn check_opening_chunk(
    http: &HttpClient,
    base_url: &str,
    prompt: &str,
) -> anyhow::Result<()> {
    opening_chunk_lands_with_its_token(&pipette_completions::stream(
        http,
        &url(base_url),
        &TimingRequest::text(Some(MODEL_NAME), prompt, 1),
    )?)
}

fn opening_chunk_lands_with_its_token(generation: &Generation) -> anyhow::Result<()> {
    let lead_ms = generation.last_token_ms - generation.first_token_ms;
    anyhow::ensure!(
        lead_ms <= generation.first_token_ms,
        "the /v3/completions opening chunk arrived at {:.1} ms, {lead_ms:.1} ms before \
         the finish of a one-token request; this OVMS sends it before the prefill, so \
         it cannot stand for the first token",
        generation.first_token_ms
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use httpmock::MockServer;

    use pipette_ops::generation::Counts;

    use super::*;

    /// OVMS 2026.4.0 on an Intel host, `max_tokens: 6`, verbatim bar the
    /// chunked-transfer framing: an empty opening chunk, two text segments for
    /// six tokens, the finish chunk, then the usage trailer.
    const CAPTURED_STREAM: &str = "data: {\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":null}],\"created\":1790274602,\"model\":\"m\",\"object\":\"text_completion.chunk\"}\n\n\
data: {\"choices\":[{\"index\":0,\"logprobs\":null,\"text\":\"!\",\"finish_reason\":null}],\"created\":1790274602,\"model\":\"m\",\"object\":\"text_completion.chunk\",\"usage\":null}\n\n\
data: {\"choices\":[{\"index\":0,\"logprobs\":null,\"text\":\" \u{1f30d}\",\"finish_reason\":null}],\"created\":1790274602,\"model\":\"m\",\"object\":\"text_completion.chunk\",\"usage\":null}\n\n\
data: {\"choices\":[{\"index\":0,\"logprobs\":null,\"text\":\"\",\"finish_reason\":\"length\"}],\"created\":1790274602,\"model\":\"m\",\"object\":\"text_completion.chunk\",\"usage\":null}\n\n\
data: {\"choices\":[],\"created\":1790274602,\"model\":\"m\",\"object\":\"text_completion.chunk\",\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":6,\"total_tokens\":9}}\n\n\
data: [DONE]\n\n";

    /// The empty opening chunk is the first token and the finish chunk the
    /// last and the end; the usage trailer is none.
    #[test]
    fn the_captured_stream_stamps_the_opening_chunk_then_finish() -> anyhow::Result<()> {
        let generation = pipette_completions::test_support::read_ticking(CAPTURED_STREAM)?;
        assert_eq!(
            (
                generation.first_token_ms,
                generation.last_token_ms,
                generation.end_ms
            ),
            (10.0, 40.0, 40.0)
        );
        assert_eq!(
            generation.counts_at_limit()?,
            Counts {
                prompt_tokens: 3,
                completion_tokens: 6
            }
        );
        Ok(())
    }

    /// Probed on the Arc GPU: the opening chunk of a one-token stream lands with
    /// its token. One flushed at the send would lead by the whole prefill.
    #[rstest::rstest]
    #[case::with_the_token(64.3, 64.5, true)]
    #[case::at_the_send(0.8, 64.5, false)]
    fn the_opening_chunk_must_land_with_its_token(
        #[case] first_token_ms: f64,
        #[case] last_token_ms: f64,
        #[case] accepted: bool,
    ) {
        let generation = Generation {
            first_token_ms,
            last_token_ms,
            ..Generation::default()
        };
        assert_eq!(
            opening_chunk_lands_with_its_token(&generation).is_ok(),
            accepted,
            "{generation:?}"
        );
    }

    #[test]
    fn the_prompt_is_sized_by_the_servers_tokenizer() -> anyhow::Result<()> {
        let server = MockServer::start();
        let tokenize = server.mock(|when, then| {
            when.method("POST")
                .path("/v3/tokenize")
                .json_body_includes(r#"{"model":"pipette"}"#);
            then.status(200)
                .json_body(serde_json::json!({ "tokens": [1, 2, 3, 4] }));
        });
        let text = prompt_text(
            &HttpClient::loopback("pipette", Duration::from_secs(5))?,
            &server.base_url(),
            4,
        )?;
        assert!(!text.is_empty());
        assert!(tokenize.calls() > 0);
        Ok(())
    }

    #[test]
    fn requests_are_greedy_and_run_to_max_tokens() -> anyhow::Result<()> {
        assert_eq!(
            serde_json::to_value(TimingRequest::text(Some(MODEL_NAME), "hi", 4))?,
            serde_json::json!({
                "model": "pipette", "prompt": "hi", "max_tokens": 4,
                "temperature": 0.0, "ignore_eos": true
            })
        );
        Ok(())
    }
}
