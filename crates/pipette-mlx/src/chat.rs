//! `mlx_vlm.server`'s `/v1/chat/completions` as the VL cells call it: the body
//! a cell sends, and the stream read on this client's clock.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use serde::Deserialize;
use serde_json::json;

use pipette_completions::{Messages, TimingRequest};
use pipette_http::HttpClient;
use pipette_ops::generation::{self, Arrived, Counts, Finish, Generation, Observation};
use pipette_ops::sse;

use crate::completions::EosBias;

pub(crate) fn body(
    model_dir: &Path,
    text: &str,
    image_url: &str,
    images: u32,
    max_tokens: u32,
    eos_bias: &EosBias,
) -> anyhow::Result<serde_json::Value> {
    let text_part = (!text.is_empty()).then(|| json!({ "type": "text", "text": text }));
    let image_parts =
        (0..images).map(|_| json!({ "type": "image_url", "image_url": { "url": image_url } }));
    let content: Vec<_> = text_part.into_iter().chain(image_parts).collect();
    // The served path: this server fetches any other `model` from the Hub.
    let model = model_dir.display().to_string();
    pipette_completions::streamed(&TimingRequest::new(
        Some(&model),
        Messages::user(json!(content)),
        max_tokens,
        eos_bias,
    ))
}

/// One streamed completion, on this client's clock.
pub(crate) fn stream(
    client: &HttpClient,
    url: &str,
    body: &serde_json::Value,
) -> anyhow::Result<Generation> {
    let endpoint = format!("{url}/v1/chat/completions");
    let request = client.client().post(&endpoint).json(body);
    let sent = Instant::now();
    let response = pipette_http::send(request).with_context(|| format!("POST {endpoint}"))?;
    read_stream(BufReader::new(response), sent, Instant::now)
}

/// Read a stream to `[DONE]`, each event stamped as it arrives and before it
/// is parsed; `now` is injected so a test can drive the clock. Lines are read
/// as they arrive, since a reader thread would stamp them late. The counts are
/// left to the callers.
fn read_stream(
    reader: impl BufRead,
    sent: Instant,
    mut now: impl FnMut() -> Instant,
) -> anyhow::Result<Generation> {
    let mut observed =
        sse::payloads(reader.lines()).try_fold(Vec::new(), |mut observed, data| {
            let data = data?;
            let at = now();
            let chunk: Chunk = serde_json::from_str(&data)
                .with_context(|| format!("failed to parse /v1/chat/completions chunk: {data}"))?;
            if let Some(error) = chunk.error {
                anyhow::bail!("/v1/chat/completions failed mid-stream: {error}");
            }
            observed.extend(chunk.arrivals().map(|what| Observation {
                prompt: 0,
                at,
                what,
            }));
            anyhow::Ok(observed)
        })?;
    observed.push(Observation {
        prompt: 0,
        at: now(),
        what: Arrived::End,
    });
    generation::generation(sent, &observed)
}

#[derive(Debug, Deserialize)]
struct Chunk {
    #[serde(default)]
    choices: Vec<Choice>,
    /// Complete on the trailing usage chunk alone; the others carry a rate.
    #[serde(default)]
    timings: Option<Timings>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

impl Chunk {
    fn arrivals(self) -> impl Iterator<Item = Arrived> {
        let choice = self.choices.into_iter().next();
        let token = choice
            .as_ref()
            .is_some_and(Choice::is_token)
            .then_some(Arrived::Token);
        let finish = choice
            .and_then(|choice| choice.finish_reason)
            .map(|reason| {
                Arrived::Finish(match reason.as_str() {
                    "length" => Finish::Limit,
                    "stop" => Finish::Stop,
                    _ => Finish::Other(reason),
                })
            });
        // The prompt less what a prefix cache served, so a cache shows as a
        // shorter prompt.
        let counts = match self.timings {
            Some(Timings {
                prompt_n: Some(prompt_tokens),
                predicted_n: Some(completion_tokens),
            }) => Some(Arrived::Counts(Counts {
                prompt_tokens,
                completion_tokens,
            })),
            _ => None,
        };
        token.into_iter().chain(finish).chain(counts)
    }
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

impl Choice {
    /// Text or reasoning, or the finish, which the server sends right after
    /// the last token: a last token with no text shows only there.
    fn is_token(&self) -> bool {
        let text = |part: &Option<String>| part.as_deref().is_some_and(|text| !text.is_empty());
        text(&self.delta.content) || text(&self.delta.reasoning) || self.finish_reason.is_some()
    }
}

#[derive(Debug, Default, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
}

/// `mlx_vlm.server`'s token counts, under `llama-server`'s names.
#[derive(Debug, Deserialize)]
struct Timings {
    prompt_n: Option<u32>,
    predicted_n: Option<u32>,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use httpmock::MockServer;
    use rstest::rstest;

    use super::*;
    use crate::completions;

    /// A `mlx_vlm.server` stream as it arrived (LFM2-VL-450M, `max_tokens` 3),
    /// trimmed to the fields read: three text deltas, a textless finish, then
    /// the usage chunk, the only one with complete `timings`.
    const CAPTURED_STREAM: &str = r#"data: {"choices":[{"finish_reason":null,"delta":{"role":"assistant","content":"The","reasoning":null}}],"usage":null,"timings":{"predicted_per_second":null}}

data: {"choices":[{"finish_reason":null,"delta":{"role":"assistant","content":" image","reasoning":null}}],"usage":null,"timings":{"predicted_per_second":463.04}}

data: {"choices":[{"finish_reason":null,"delta":{"role":"assistant","content":" you","reasoning":null}}],"usage":null,"timings":{"predicted_per_second":117.84}}

data: {"choices":[{"finish_reason":"length","delta":{"role":"assistant","content":null,"reasoning":null}}],"usage":null,"timings":{"predicted_per_second":187.87}}

data: {"choices":[],"usage":{"prompt_tokens":75,"completion_tokens":3,"total_tokens":78,"prompt_tokens_details":{"cached_tokens":0}},"timings":{"prompt_n":75,"cache_n":0,"predicted_n":3,"prompt_ms":56.23,"predicted_ms":15.97,"predicted_per_second":187.87}}

data: [DONE]

"#;

    fn captured(kept: &[usize]) -> String {
        let events: Vec<&str> = CAPTURED_STREAM.split("\n\n").collect();
        kept.iter().map(|&i| format!("{}\n\n", events[i])).collect()
    }

    /// On a clock that ticks a millisecond a call from 10 ms after the send,
    /// the three deltas and the finish stamp 10 to 13 ms, the generation ends
    /// at the finish, and the usage chunk is not a token. The first delta is
    /// text or, from a thinking model, reasoning.
    #[rstest]
    #[case::text(r#""content":"The","reasoning":null"#)]
    #[case::reasoning(r#""content":null,"reasoning":"The""#)]
    fn a_stream_is_timed_on_this_clients_clock(#[case] first_delta: &str) -> anyhow::Result<()> {
        let body = CAPTURED_STREAM.replacen(r#""content":"The","reasoning":null"#, first_delta, 1);
        anyhow::ensure!(body.contains(first_delta), "no first delta to replace");
        let started = Instant::now();
        let mut ms = 9;
        let generation = read_stream(body.as_bytes(), started, || {
            ms += 1;
            started + Duration::from_millis(ms)
        })?;
        assert!(
            (generation.first_token_ms - 10.0).abs() < 1e-9,
            "{generation:?}"
        );
        assert!(
            (generation.last_token_ms - 13.0).abs() < 1e-9,
            "{generation:?}"
        );
        assert!((generation.end_ms - 13.0).abs() < 1e-9, "{generation:?}");
        assert_eq!(
            generation.counts_at_limit()?,
            Counts {
                prompt_tokens: 75,
                completion_tokens: 3
            }
        );
        Ok(())
    }

    /// The captured reply is three tokens.
    #[rstest]
    #[case::long(1, false)]
    #[case::in_full(3, true)]
    #[case::short(4, false)]
    fn a_reply_is_held_to_the_cells_output_length(
        #[case] decode_tokens: u32,
        #[case] accepted: bool,
    ) -> anyhow::Result<()> {
        let generation = read_stream(CAPTURED_STREAM.as_bytes(), Instant::now(), Instant::now)?;
        assert_eq!(
            pipette_ops::measurement::decoded_in_full(&generation, decode_tokens).is_ok(),
            accepted
        );
        Ok(())
    }

    #[test]
    fn a_stream_without_counts_reads_without_them() -> anyhow::Result<()> {
        let body = captured(&[0, 1, 2, 3, 5]);
        let generation = read_stream(body.as_bytes(), Instant::now(), Instant::now)?;
        assert_eq!(generation.counts, None);
        Ok(())
    }

    #[rstest]
    #[case::no_token(captured(&[4, 5]), "ended without a token")]
    #[case::cut(captured(&[0, 1, 2, 3, 4]), "ended before its terminal event or [DONE]")]
    #[case::error(
        "data: {\"error\":{\"message\":\"out of memory\"}}\n\ndata: [DONE]\n\n".to_owned(),
        "out of memory"
    )]
    fn a_stream_short_of_a_reading_is_refused(
        #[case] body: String,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let err = read_stream(body.as_bytes(), Instant::now(), Instant::now)
            .err()
            .context("a stream short of a reading was read")?;
        assert!(format!("{err:#}").contains(expected), "{err:#}");
        Ok(())
    }

    /// A refusal quotes the server.
    #[test]
    fn a_rep_posts_its_body_and_reads_the_answer() -> anyhow::Result<()> {
        let server = MockServer::start();
        let body = json!({ "stream": true });
        let answered = server.mock(|when, then| {
            when.method("POST")
                .path("/v1/chat/completions")
                .json_body(body.clone());
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(CAPTURED_STREAM);
        });
        let client = HttpClient::loopback("pipette", Duration::from_secs(10))?;
        let generation = stream(&client, &server.base_url(), &body)?;
        answered.assert();
        assert_eq!(
            generation.counts,
            Some(Counts {
                prompt_tokens: 75,
                completion_tokens: 3
            })
        );

        let refused = server.mock(|when, then| {
            when.method("POST").path("/v1/chat/completions");
            then.status(400).body("{\"detail\":\"no image\"}");
        });
        let err = stream(&client, &server.base_url(), &json!({}))
            .err()
            .context("a refused request was read")?;
        refused.assert();
        assert!(format!("{err:#}").contains("no image"), "{err:#}");
        Ok(())
    }

    /// Greedy, EOS biased away, the served path as `model`, and the counts
    /// asked for.
    #[test]
    fn the_body_is_the_text_then_each_image() -> anyhow::Result<()> {
        let bias = completions::eos_bias(&[7]);
        let image = json!({ "type": "image_url", "image_url": { "url": "data:x" } });
        assert_eq!(
            body(Path::new("/models/vl"), "hi", "data:x", 2, 8, &bias)?,
            json!({
                "model": "/models/vl",
                "messages": [{
                    "role": "user",
                    "content": [{ "type": "text", "text": "hi" }, image, image],
                }],
                "max_tokens": 8,
                "temperature": 0.0,
                "logit_bias": { "7": -100.0 },
                "stream": true,
                "stream_options": { "include_usage": true },
            })
        );
        assert_eq!(
            body(Path::new("/models/vl"), "", "data:x", 1, 8, &bias)?["messages"][0]["content"],
            json!([image])
        );
        // No EOS ids, no bias: the key is left out rather than sent as null.
        let unbiased = body(
            Path::new("/models/vl"),
            "hi",
            "data:x",
            1,
            8,
            &completions::eos_bias(&[]),
        )?;
        assert_eq!(unbiased.get("logit_bias"), None);
        Ok(())
    }
}
