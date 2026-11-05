//! The vLLM / SGLang calls the cells make: `/tokenize` to size a prompt, and
//! `/v1/completions`, read through [`pipette_completions`]. Both servers send
//! a chunk per decode step, empty while the detokenizer holds back an
//! incomplete UTF-8 sequence, so every chunk with a choice is a token
//! (vLLM 0.21 `entrypoints/openai/completion/serving.py:364-372`, SGLang
//! 0.5.12 `entrypoints/openai/serving_completions.py:310-361`).
//!
//! Chat completions stay hand-built in `execute/eval.rs`: the body differs per
//! call site (`guided_choice`, `top_logprobs`, streamed free text).

use anyhow::Context;
use reqwest::Method;
use serde::{Deserialize, Serialize};

use pipette_http::HttpClient;

#[derive(Debug, Serialize)]
pub struct TokenizeRequest<'a> {
    pub model: &'a str,
    pub prompt: &'a str,
    /// Keep explicit at call sites so prompt construction uses the same
    /// special-token policy as the timed inference request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub add_special_tokens: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
}

/// `POST /tokenize` (no `/v1/` prefix), served by both vLLM and SGLang.
pub fn tokenize(
    http: &HttpClient,
    base_url: &str,
    request: &TokenizeRequest<'_>,
) -> anyhow::Result<TokenizeResponse> {
    let url = format!("{base_url}/tokenize");
    let body = serde_json::to_value(request).context("failed to serialize request body")?;
    http.json_request(Method::POST, &url, None, Some(body))
        .with_context(|| format!("POST {url}"))
}

pub fn url(base_url: &str) -> String {
    format!("{base_url}/v1/completions")
}

/// Build a prompt string that the server will tokenize to exactly
/// `target_tokens`. Sending text keeps the server's tokenize step inside the
/// timed request, and lets `usage` confirm the length every rep.
pub(crate) fn build_prompt_text(
    http: &HttpClient,
    base_url: &str,
    model: &str,
    target_tokens: u32,
) -> anyhow::Result<String> {
    pipette_ops::prompt_seed::build_prompt_text(target_tokens, |text| {
        let response = tokenize(
            http,
            base_url,
            &TokenizeRequest {
                model,
                prompt: text,
                // Match /v1/completions string-prompt accounting. Some
                // tokenizers add BOS here, and the e2e validator checks the
                // server-reported prompt_tokens from the timed request.
                add_special_tokens: Some(true),
            },
        )
        .context("failed to /tokenize while building prompt")?;
        Ok(response.tokens.len())
    })
}

#[cfg(test)]
mod tests {

    use pipette_ops::generation::Counts;

    /// A stream as vLLM 0.21 and SGLang 0.5.12 send it, one chunk per decode
    /// step with the finish reason on the last token's: the first token is
    /// half of a two-byte character, whose text is held back, so its chunk is
    /// empty. Then a choice-less usage chunk and `[DONE]`.
    const STREAM: &str = "\
data: {\"id\":\"cmpl-1\",\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":null}],\"usage\":null}\n\
\n\
data: {\"id\":\"cmpl-1\",\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"\u{e9}\",\"finish_reason\":null}],\"usage\":null}\n\
\n\
data: {\"id\":\"cmpl-1\",\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\" cat\",\"finish_reason\":\"length\"}],\"usage\":null}\n\
\n\
data: {\"id\":\"cmpl-1\",\"object\":\"text_completion\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\
\n\
data: [DONE]\n";

    /// The empty first chunk is the first token, not the first text, and the
    /// finish chunk the last and the end; the usage chunk is none.
    #[test]
    fn every_step_is_a_token_text_or_not() -> anyhow::Result<()> {
        let generation = pipette_completions::test_support::read_ticking(STREAM)?;
        assert_eq!(
            (
                generation.first_token_ms,
                generation.last_token_ms,
                generation.end_ms
            ),
            (10.0, 30.0, 30.0)
        );
        assert_eq!(
            generation.counts_at_limit()?,
            Counts {
                prompt_tokens: 7,
                completion_tokens: 3
            }
        );
        Ok(())
    }
}
