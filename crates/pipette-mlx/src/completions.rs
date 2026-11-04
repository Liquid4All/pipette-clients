//! Client for `mlx_lm.server`'s `/v1/completions`. The timing cells stream it:
//! the server reports no per-phase rates, so prefill is the time to the first
//! streamed token and decode the remainder.
//!
//! The server sends a chunk only for a step with non-empty text, plus the
//! finish chunk once the last token is decoded (`server.py:1478-1490` in
//! 0.31.3, `1496-1511` in 0.32.0). A first token whose text the detokenizer
//! holds back (an incomplete UTF-8 sequence, a lone space) would stamp the
//! prefill late. Probed on eight fleet models at prompts of 16 to 8192 seed
//! tokens, the first token always carried text.

use std::collections::HashMap;
use std::time::Duration;

use serde::Serialize;

use pipette_completions::{Text, TimingRequest};

/// The server maps `"default_model"` to its `--model` path; any other id
/// would make it load that id from Hugging Face.
pub(crate) const DEFAULT_MODEL: &str = "default_model";

/// Request budget for timing and memory cells, and the idle bound on their
/// streams. No MLX text timing variant of `BenchmarkFlags` carries
/// `http_timeout`, so this bounds a wedged server; `vl_throughput` falls back
/// to it.
pub(crate) const TIMING_TIMEOUT: Duration = Duration::from_secs(3600);

/// How `mlx_lm.server` and `mlx_vlm.server` are kept past EOS: they take no
/// `ignore_eos`, so every EOS id is biased far negative.
#[derive(Debug, Serialize)]
pub(crate) struct EosBias {
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f32>>,
}

pub(crate) fn url(base_url: &str) -> String {
    format!("{base_url}/v1/completions")
}

pub(crate) fn completion_request<'a>(
    prompt: &'a str,
    max_tokens: u32,
    eos_ids: &[u32],
) -> TimingRequest<'a, Text<'a>, EosBias> {
    TimingRequest::new(
        Some(DEFAULT_MODEL),
        Text { prompt },
        max_tokens,
        eos_bias(eos_ids),
    )
}

pub(crate) fn eos_bias(eos_ids: &[u32]) -> EosBias {
    let logit_bias = (!eos_ids.is_empty()).then(|| {
        tracing::info!(
            token_ids = ?eos_ids,
            "suppressing EOS ids via logit_bias for timing cells"
        );
        eos_ids.iter().map(|id| (id.to_string(), -100.0)).collect()
    });
    EosBias { logit_bias }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use anyhow::Context;

    use pipette_http::HttpClient;
    use pipette_ops::generation::Counts;

    use super::*;

    /// Captured `mlx_lm.server` completions stream (LFM2-350M, `max_tokens=4`
    /// with `include_usage`): keepalive comments, per-token chunks, an empty
    /// length-finish chunk, then the usage trailer, whose `object` says
    /// `chat.completion`.
    const CAPTURED_STREAM: &str = ": keepalive 10/11\n\n: keepalive 11/11\n\ndata: {\"object\": \"text_completion\", \"choices\": [{\"index\": 0, \"finish_reason\": null, \"text\": \" This\"}]}\n\ndata: {\"object\": \"text_completion\", \"choices\": [{\"index\": 0, \"finish_reason\": null, \"text\": \" is\"}]}\n\ndata: {\"object\": \"text_completion\", \"choices\": [{\"index\": 0, \"finish_reason\": \"length\", \"text\": \"\"}]}\n\ndata: {\"object\": \"chat.completion\", \"choices\": [], \"usage\": {\"prompt_tokens\": 11, \"completion_tokens\": 4, \"total_tokens\": 15, \"prompt_tokens_details\": {\"cached_tokens\": 0}}}\n\ndata: [DONE]\n\n";

    /// The empty finish chunk is sent once the last token is decoded, text or
    /// not, so it is the last token and the end.
    #[test]
    fn the_first_text_is_the_first_token_and_the_finish_the_last() -> anyhow::Result<()> {
        let generation = pipette_completions::test_support::read_ticking(CAPTURED_STREAM)?;
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
                prompt_tokens: 11,
                completion_tokens: 4
            }
        );
        Ok(())
    }

    /// Silence after the first token fails on the client's timeout; an
    /// unbounded client would see the stream cut when this server hangs up,
    /// 10 s in.
    #[test]
    fn a_stalled_stream_fails_on_the_clients_timeout() -> anyhow::Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let url = format!("http://{}", listener.local_addr()?);
        std::thread::spawn(move || -> std::io::Result<()> {
            let (mut socket, _) = listener.accept()?;
            let _ = socket.read(&mut [0; 4096])?;
            socket.write_all(
                b"HTTP/1.0 200 OK\r\ncontent-type: text/event-stream\r\n\r\n\
                  data: {\"choices\": [{\"text\": \" a\"}]}\n\n",
            )?;
            socket.set_read_timeout(Some(Duration::from_secs(10)))?;
            while socket.read(&mut [0; 4096]).is_ok_and(|read| read > 0) {}
            Ok(())
        });
        let http = HttpClient::loopback("pipette", Duration::from_secs(1))?;
        let err = pipette_completions::stream(
            &http,
            &super::url(&url),
            &completion_request("hi", 2, &[]),
        )
        .err()
        .context("a stalled stream was read")?;
        // The body read timed out, not the wait for the headers.
        let message = format!("{err:#}");
        assert!(
            message.contains("failed to read SSE line") && message.contains("timed out"),
            "{message}"
        );
        Ok(())
    }

    #[test]
    fn a_request_is_greedy_on_the_default_model() -> anyhow::Result<()> {
        assert_eq!(
            serde_json::to_value(completion_request("hi", 4, &[]))?,
            serde_json::json!({
                "model": "default_model", "prompt": "hi", "max_tokens": 4, "temperature": 0.0,
            })
        );
        Ok(())
    }

    #[test]
    fn every_eos_id_is_biased_away() -> anyhow::Result<()> {
        let request = completion_request("hi", 4, &[2, 7]);
        assert_eq!(
            serde_json::to_value(&request)?["logit_bias"],
            serde_json::json!({ "2": -100.0, "7": -100.0 })
        );
        Ok(())
    }
}
