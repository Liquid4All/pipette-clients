//! OpenAI-compat JSON helpers for vLLM / SGLang.
//!
//! Transport is [`pipette_http::HttpClient`]; this module owns request/response
//! shapes and thin RPC wrappers. Chat completions stay hand-built in
//! `execute/eval.rs` (per-call `guided_choice` / `top_logprobs` / streaming).

use std::time::Duration;

use anyhow::Context;
use reqwest::Method;
use serde::{Deserialize, Serialize};

use pipette_http::HttpClient;

const USER_AGENT: &str = "pipette";

// Chat completions are issued directly from `execute/eval.rs` with hand-built
// `serde_json::Value` bodies (the request shape differs per call site —
// guided_choice vs top_logprobs vs streaming free-text — and includes
// engine-specific extras like `chat_template_kwargs`). A typed
// `ChatRequest`/`chat()` helper lived here previously but was unused; if a
// future caller wants a typed surface, give it an explicit `chat_template_kwargs`
// field so the contract stays in one place.

/// Token-count usage block reported by both `/v1/chat/completions` and
/// `/v1/completions`. Currently only the latter consumes it; kept here so a
/// future chat helper can reuse the same shape.
#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// --- Tokenize ---------------------------------------------------------------

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
    #[serde(default)]
    pub count: u32,
}

/// vLLM and SGLang both expose `POST /tokenize` (no `/v1/` prefix) that
/// returns the token id sequence for a prompt. Used to build exact-length
/// prompts for latency / memory benchmarks.
pub fn tokenize(
    base_url: &str,
    request: &TokenizeRequest<'_>,
    timeout: Duration,
) -> anyhow::Result<TokenizeResponse> {
    json_post(base_url, "/tokenize", request, timeout)
}

// --- Completions (raw, non-chat) -------------------------------------------

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Text(String),
    Tokens(Vec<u32>),
}

#[derive(Debug, Serialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: CompletionPrompt,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// vLLM-specific: prevent the server from stopping early on the
    /// model's EOS token. Without this, latency benchmarks with EOS-prone
    /// models complete in 1-2 decode steps instead of `max_tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    #[serde(default)]
    pub choices: Vec<CompletionChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionChoice {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// POST `/v1/completions` (OpenAI-compat raw completions endpoint). Unlike
/// `/v1/chat/completions` this accepts a token-id array as `prompt`, so we
/// can hand the server the exact prefill length we want.
pub fn complete(
    base_url: &str,
    request: &CompletionRequest,
    timeout: Duration,
) -> anyhow::Result<CompletionResponse> {
    json_post(base_url, "/v1/completions", request, timeout)
}

fn json_post<B: Serialize, T: for<'de> Deserialize<'de>>(
    base_url: &str,
    path: &str,
    body: &B,
    timeout: Duration,
) -> anyhow::Result<T> {
    let http = HttpClient::with_request_timeout(USER_AGENT, timeout)
        .context("failed to build HTTP client")?;
    let url = format!("{base_url}{path}");
    let body = serde_json::to_value(body).context("failed to serialize request body")?;
    http.json_request(Method::POST, &url, None, Some(body))
        .with_context(|| format!("POST {url}"))
}
