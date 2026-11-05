//! llama-server's HTTP dialect as pipette's VL cells speak it: `/health`,
//! `/tokenize` and the timed, unstreamed `/v1/chat/completions` with its
//! server-clock `timings`. `pipette-llamacpp` and `pipette-trt-edgellm` both
//! drive servers that answer it, and runtime crates may not depend on each
//! other.

use std::time::{Duration, Instant};

use anyhow::Context;
use reqwest::Method;
use reqwest::header::ACCEPT;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use pipette_http::HttpClient;
use pipette_ops::generation::{
    self, Arrived, Counts, Finish, Generation, Observation, ServerClock,
};
use pipette_plan_types::RuntimeOutput;
use pipette_subprocess::NotReadyError;

use crate::{Messages, TimingRequest};

/// Where a server started on `port` listens.
pub fn url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

/// The readiness check for `Process::wait_ready`: `GET /health` on `url`
/// answers 2xx. `poll` bounds one request, so a socket that accepts and then
/// stalls costs one poll rather than the whole wait.
pub fn health(
    url: &str,
    poll: Duration,
) -> anyhow::Result<impl FnMut(&RuntimeOutput) -> Result<(), NotReadyError>> {
    let client = HttpClient::loopback("pipette", poll)?.client().clone();
    let health = format!("{url}/health");
    Ok(move |_: &RuntimeOutput| {
        pipette_http::send(client.get(&health))
            .context("GET /health")
            .map_err(NotReadyError::yet)?;
        Ok(())
    })
}

/// `err` with the stopped server's report as detail and its own root cause as
/// the headline: as plain context the report would head the error, and a
/// readiness wait that timed out would read "llama-server was still running".
/// The chain keeps every cause and its type.
pub fn with_server_tail(
    err: anyhow::Error,
    stopped: impl std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
) -> anyhow::Error {
    let cause = err.root_cause().to_string();
    err.context(stopped).context(cause)
}

/// The server's `GET /props`, whole.
pub fn props(http: &HttpClient, url: &str) -> anyhow::Result<Map<String, Value>> {
    http.json_request(Method::GET, &format!("{url}/props"), None, None)
        .context("GET /props")
}

/// How many tokens a raw `/completion` prompt is on the running server.
/// `add_special: true` matches `/completion` on tokenizers that prepend a BOS
/// (Gemma, LFM2).
pub fn tokenize(http: &HttpClient, url: &str, text: &str) -> anyhow::Result<usize> {
    tokenize_with(http, url, text, true)
}

/// Filler of exactly `tokens` tokens for a chat message's text part, never
/// starting or ending in whitespace: the chat templates of Gemma 4, Qwen3.5 and
/// MiniCPM-V trim a message's content, so a filler cut after a space would
/// reach their prompts a token short.
pub fn message_filler(http: &HttpClient, url: &str, tokens: u32) -> anyhow::Result<String> {
    trimmed_filler(tokens, |text| tokenize_message_text(http, url, text))
}

/// [`message_filler`] over any counter.
fn trimmed_filler(
    tokens: u32,
    mut count: impl FnMut(&str) -> anyhow::Result<usize>,
) -> anyhow::Result<String> {
    let text = pipette_ops::prompt_seed::build_prompt_text(tokens, |text| count(text.trim()))?;
    Ok(text.trim().to_owned())
}

/// How many tokens `text` is as a chat message's content: without special
/// tokens, since the chat template adds the BOS and the role markers. Counted
/// with them, a filler built to `n` tokens comes out `n - 1` on a tokenizer
/// that prepends a BOS.
pub fn tokenize_message_text(http: &HttpClient, url: &str, text: &str) -> anyhow::Result<usize> {
    tokenize_with(http, url, text, false)
}

fn tokenize_with(
    http: &HttpClient,
    url: &str,
    text: &str,
    add_special: bool,
) -> anyhow::Result<usize> {
    #[derive(serde::Deserialize)]
    struct TokenizeResponse {
        tokens: Vec<u32>,
    }
    let response: TokenizeResponse = http
        .json_request(
            Method::POST,
            &format!("{url}/tokenize"),
            None,
            Some(serde_json::json!({ "content": text, "add_special": add_special })),
        )
        .context("POST /tokenize")?;
    Ok(response.tokens.len())
}

/// The chat endpoint, because `/completion` does not run the image through the
/// vision tower. The reply's `timings` are the server's own clock; the send and
/// the reply's arrival are stamped on this client's, for the whole request. An
/// error keeps the server's body: a ctx-overflow or image-decode 400 carries the
/// message a bare status would drop.
pub fn chat(
    http: &HttpClient,
    url: &str,
    text: &str,
    image_b64: &str,
    decode_tokens: u32,
    num_images: u32,
) -> anyhow::Result<Generation> {
    chat_reply(http, url, text, image_b64, decode_tokens, num_images).map(|reply| reply.generation)
}

/// [`chat`]'s reply whole, with the image tokens a server that counts them
/// reports.
pub fn chat_reply(
    http: &HttpClient,
    url: &str,
    text: &str,
    image_b64: &str,
    decode_tokens: u32,
    num_images: u32,
) -> anyhow::Result<ChatReply> {
    post_chat_reply(
        http,
        url,
        &build_chat_body(text, image_b64, decode_tokens, num_images)?,
    )
}

/// One `/v1/chat/completions` reply as read.
#[derive(Debug)]
pub struct ChatReply {
    pub generation: Generation,
    /// `timings.image_n`, the image share of `prompt_n`, from a server that
    /// counts it (TensorRT Edge-LLM's); llama-server sends none.
    pub image_tokens: Option<u32>,
}

/// The body, base64 included, is encoded before the clock starts and the reply
/// stamped before it is parsed, as the text cells' requests are.
pub fn post_chat(http: &HttpClient, url: &str, body: &Value) -> anyhow::Result<Generation> {
    post_chat_reply(http, url, body).map(|reply| reply.generation)
}

fn post_chat_reply(http: &HttpClient, url: &str, body: &Value) -> anyhow::Result<ChatReply> {
    let endpoint = format!("{url}/v1/chat/completions");
    let request = http
        .client()
        .post(&endpoint)
        .header(ACCEPT, "application/json")
        .json(body);
    // Before the clock starts, so a stalled request is visible at `-v`.
    tracing::debug!(%endpoint, "sending request");
    let sent = Instant::now();
    let response = pipette_http::send(request).context("POST /v1/chat/completions")?;
    let status = response.status().as_u16();
    let bytes = response
        .bytes()
        .context("POST /v1/chat/completions: failed to read the reply")?;
    let arrived = Instant::now();
    // After the stamp, so a log write at `-v` is not timed.
    tracing::debug!(%endpoint, status, elapsed = ?(arrived - sent), "response");
    let reply: VlCompletionResponse = serde_json::from_slice(&bytes)
        .context("POST /v1/chat/completions: the reply is not JSON")?;
    let image_tokens = reply.timings.and_then(|timings| timings.image_n);
    let generation = generation::generation(sent, &reply.observations(arrived))
        .context("POST /v1/chat/completions")?;
    Ok(ChatReply {
        generation,
        image_tokens,
    })
}

/// Build the `/v1/chat/completions` request body for a VL run: `num_images`
/// image parts (+ optional text) as OpenAI-style multimodal content. Pure so
/// the content shape is unit-testable without a live server.
/// `num_images == 0` sends the text alone.
fn build_chat_body(
    text_prompt: &str,
    image_b64: &str,
    decode_tokens: u32,
    num_images: u32,
) -> anyhow::Result<Value> {
    let image_url = format!("data:image/png;base64,{image_b64}");
    let mut content: Vec<Value> = Vec::with_capacity(num_images as usize + 1);
    if !text_prompt.is_empty() {
        content.push(json!({ "type": "text", "text": text_prompt }));
    }
    for _ in 0..num_images {
        content.push(json!({ "type": "image_url", "image_url": { "url": image_url } }));
    }
    // No parts → empty string so the chat template still renders.
    let message_content = if content.is_empty() {
        json!("")
    } else {
        json!(content)
    };
    let request = TimingRequest::new(
        None,
        Messages::user(message_content),
        decode_tokens,
        ServerKeys {
            ignore_eos: true,
            cache_prompt: false,
        },
    );
    serde_json::to_value(request).context("failed to encode the chat request")
}

/// llama-server's own keys on a timing request.
#[derive(Debug, Serialize)]
pub struct ServerKeys {
    /// The server masks every end-of-generation token under it (llama.cpp
    /// b8718+, #21203), so the reply runs to `max_tokens`.
    pub ignore_eos: bool,
    pub cache_prompt: bool,
}

/// Read leniently: each cell checks only the fields it uses.
#[derive(Debug, Deserialize)]
struct VlCompletionResponse {
    #[serde(default)]
    choices: Vec<VlChoice>,
    #[serde(default)]
    timings: Option<VlCompletionTimings>,
}

#[derive(Debug, Deserialize)]
struct VlChoice {
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct VlCompletionTimings {
    prompt_n: u32,
    prompt_ms: f64,
    predicted_n: u32,
    predicted_ms: f64,
    #[serde(default)]
    image_n: Option<u32>,
}

impl VlCompletionResponse {
    fn observations(self, at: Instant) -> Vec<Observation> {
        let finish = self
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.finish_reason)
            .map(|reason| match reason.as_str() {
                "length" => Finish::Limit,
                "stop" => Finish::Stop,
                _ => Finish::Other(reason),
            });
        let timings = self.timings;
        [
            Some(Arrived::Token),
            finish.map(Arrived::Finish),
            timings.map(|timings| {
                Arrived::Counts(Counts {
                    prompt_tokens: timings.prompt_n,
                    completion_tokens: timings.predicted_n,
                })
            }),
            timings.map(|timings| {
                Arrived::ServerClock(ServerClock {
                    prefill_ms: timings.prompt_ms,
                    decode_ms: timings.predicted_ms,
                })
            }),
            Some(Arrived::End),
        ]
        .into_iter()
        .flatten()
        .map(|what| Observation {
            prompt: 0,
            at,
            what,
        })
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// The fake count makes trailing whitespace a token of its own, as a
    /// template that trims would drop.
    #[rstest]
    #[case(1)]
    #[case(7)]
    #[case(64)]
    #[case(1024)]
    fn the_message_filler_has_no_whitespace_to_trim(#[case] tokens: u32) -> anyhow::Result<()> {
        let count = |text: &str| -> anyhow::Result<usize> {
            let trailing = text.len() - text.trim_end().len();
            Ok(text.split_whitespace().count() + trailing)
        };
        let text = trimmed_filler(tokens, count)?;
        assert_eq!(text, text.trim(), "{text:?}");
        assert_eq!(count(&text)?, tokens as usize);
        Ok(())
    }

    // Count content parts of a given `type` (0 if content isn't an array, e.g.
    // the empty string sent without parts). Avoids unwrap/expect (denied workspace-wide).
    fn count_type(body: &Value, ty: &str) -> usize {
        body["messages"][0]["content"]
            .as_array()
            .map(|parts| parts.iter().filter(|p| p["type"] == ty).count())
            .unwrap_or(0)
    }

    fn reply(finish: &str) -> anyhow::Result<VlCompletionResponse> {
        Ok(serde_json::from_str(&format!(
            r#"{{"choices":[{{"finish_reason":"{finish}"}}],
                "timings":{{"prompt_n":10,"prompt_ms":1.5,"predicted_ms":2.5,"predicted_n":64}}}}"#
        ))?)
    }

    fn read(reply: VlCompletionResponse) -> anyhow::Result<Generation> {
        let at = Instant::now();
        generation::generation(at, &reply.observations(at))
    }

    /// Stopping anywhere but the limit (EOS, say) means `ignore_eos` was not
    /// honored.
    #[rstest::rstest]
    #[case::at_the_limit("length", true)]
    #[case::early("stop", false)]
    fn only_a_stop_at_the_limit_is_accepted(
        #[case] finish: &str,
        #[case] accepted: bool,
    ) -> anyhow::Result<()> {
        assert_eq!(read(reply(finish)?)?.counts_at_limit().is_ok(), accepted);
        Ok(())
    }

    #[test]
    fn a_reply_reads_the_servers_clock() -> anyhow::Result<()> {
        let generation = read(reply("length")?)?;
        assert_eq!(
            generation.counts,
            Some(Counts {
                prompt_tokens: 10,
                completion_tokens: 64
            })
        );
        assert_eq!(
            generation.server_clock,
            Some(ServerClock {
                prefill_ms: 1.5,
                decode_ms: 2.5
            })
        );
        Ok(())
    }

    #[test]
    fn a_reply_without_timings_reads_without_them() -> anyhow::Result<()> {
        let generation = read(serde_json::from_str(
            r#"{"choices":[{"finish_reason":"length"}]}"#,
        )?)?;
        assert_eq!((generation.counts, generation.server_clock), (None, None));
        Ok(())
    }

    /// TensorRT Edge-LLM's server counts the image tokens; llama-server does not.
    #[rstest]
    #[case::counted(r#","image_n":128"#, Some(128))]
    #[case::uncounted("", None)]
    fn a_reply_carries_the_image_tokens_a_server_counts(
        #[case] image_n: &str,
        #[case] expected: Option<u32>,
    ) -> anyhow::Result<()> {
        let reply: VlCompletionResponse = serde_json::from_str(&format!(
            r#"{{"choices":[],"timings":{{"prompt_n":238,"prompt_ms":49.0,"predicted_n":2,"predicted_ms":20.0{image_n}}}}}"#
        ))?;
        assert_eq!(reply.timings.and_then(|t| t.image_n), expected);
        Ok(())
    }

    #[test]
    fn chat_body_without_images_is_empty_string_content() -> anyhow::Result<()> {
        // The whole body is pinned, so a `logit_bias` beside `ignore_eos` fails it.
        assert_eq!(
            build_chat_body("", "AAAA", 8, 0)?,
            json!({
                "messages": [{ "role": "user", "content": "" }],
                "temperature": 0.0,
                "max_tokens": 8,
                "ignore_eos": true,
                "cache_prompt": false,
            })
        );
        Ok(())
    }

    #[test]
    fn chat_body_single_image_has_one_image_part() -> anyhow::Result<()> {
        let body = build_chat_body("", "AAAA", 64, 1)?;
        assert_eq!(count_type(&body, "image_url"), 1);
        assert_eq!(count_type(&body, "text"), 0); // no text_prompt
        assert_eq!(body["max_tokens"], json!(64));
        Ok(())
    }

    #[test]
    fn chat_body_multi_image_has_n_image_parts() -> anyhow::Result<()> {
        let body = build_chat_body("hello", "AAAA", 16, 4)?;
        assert_eq!(count_type(&body, "image_url"), 4);
        assert_eq!(count_type(&body, "text"), 1); // text_prompt present
        // image parts carry the data URL.
        let has_data_url = body["messages"][0]["content"]
            .as_array()
            .and_then(|parts| parts.iter().find(|p| p["type"] == "image_url"))
            .and_then(|p| p["image_url"]["url"].as_str())
            .map(|u| u.starts_with("data:image/png;base64,"))
            .unwrap_or(false);
        assert!(has_data_url, "image part must carry a base64 data URL");
        Ok(())
    }
}
