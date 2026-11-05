//! The text-completion endpoint, `/completions`: a request sent whole
//! ([`reply`]) or streamed ([`stream`]), and what came back read into a
//! [`pipette_ops::generation::Generation`].
//!
//! vLLM, SGLang, `mlx_lm.server`, OVMS (as `/v3/completions`) and pipette's
//! own Python servers answer it. They take one prompt per request, and none
//! sends a choice without a token behind it, so every chunk with a choice is a
//! token, text or not; the counts come from the `usage` trailer.
//!
//! Not llama-server's native `/completion`, which `pipette-llamacpp` reads
//! itself, and unrelated to an eval's completions
//! (`pipette_ops::eval_completions`).

use std::io::{BufRead, BufReader};
use std::time::Instant;

use anyhow::Context;
use reqwest::header::ACCEPT;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use pipette_http::HttpClient;
use pipette_ops::generation::{self, Arrived, Counts, Finish, Generation, Observation};
use pipette_ops::sse;

pub mod llama_server;

/// A greedy request that runs to `max_tokens`. `E`, the server's own keys,
/// must keep it generating past end-of-sequence; a server that stops early
/// anyway fails [`Generation::counts_at_limit`].
#[derive(Debug, Serialize)]
pub struct TimingRequest<'a, P = Text<'a>, E = IgnoreEos> {
    /// `None` for a server that serves one model and needs no name.
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    #[serde(flatten)]
    prompt: P,
    max_tokens: u32,
    temperature: f32,
    #[serde(flatten)]
    engine: E,
}

impl<'a, P: Serialize, E: Serialize> TimingRequest<'a, P, E> {
    pub fn new(model: Option<&'a str>, prompt: P, max_tokens: u32, engine: E) -> Self {
        Self {
            model,
            prompt,
            max_tokens,
            temperature: 0.0,
            engine,
        }
    }
}

impl<'a> TimingRequest<'a> {
    /// A text prompt to a server that takes `ignore_eos`.
    pub fn text(model: Option<&'a str>, prompt: &'a str, max_tokens: u32) -> Self {
        Self::new(model, Text { prompt }, max_tokens, IgnoreEos::default())
    }
}

/// A text prompt, for `/completions`.
#[derive(Debug, Serialize)]
pub struct Text<'a> {
    pub prompt: &'a str,
}

/// A conversation, for `/chat/completions`.
#[derive(Debug, Serialize)]
pub struct Messages {
    pub messages: Value,
}

impl Messages {
    /// One user turn.
    pub fn user(content: Value) -> Self {
        Self {
            messages: serde_json::json!([{ "role": "user", "content": content }]),
        }
    }
}

/// The `messages` of `/v1/chat/completions`, typed so an unsupported role or
/// part does not compile and a misspelled discriminant cannot reach a server.
#[derive(Debug, Serialize)]
pub struct ChatMessages<'a> {
    pub messages: Vec<ChatMessage<'a>>,
}

#[derive(Debug, Serialize)]
pub struct ChatMessage<'a> {
    pub role: ChatRole,
    pub content: MessageContent<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum MessageContent<'a> {
    Text(&'a str),
    Parts(Vec<ContentPart<'a>>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart<'a> {
    Text { text: &'a str },
    InputAudio { input_audio: InputAudio<'a> },
}

/// `data` is the file base64-encoded.
#[derive(Debug, Serialize)]
pub struct InputAudio<'a> {
    pub data: &'a str,
    pub format: AudioFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    Wav,
}

/// Generation past an end-of-sequence token, as vLLM, SGLang, OVMS,
/// llama-server and pipette's own servers take it.
#[derive(Debug, Serialize)]
pub struct IgnoreEos {
    ignore_eos: bool,
}

impl Default for IgnoreEos {
    fn default() -> Self {
        Self { ignore_eos: true }
    }
}

/// POST `body` to `url` and read the whole reply. A refusal keeps the
/// server's body.
pub fn reply(http: &HttpClient, url: &str, body: &impl Serialize) -> anyhow::Result<Generation> {
    // Built before the clock starts, so encoding the body is not timed.
    let request = http
        .client()
        .post(url)
        .header(ACCEPT, "application/json")
        .json(body);
    let sent = Instant::now();
    let reply: Reply = pipette_http::send(request)
        .with_context(|| format!("POST {url}"))?
        .json()
        .with_context(|| format!("failed to parse the reply from {url}"))?;
    generation::generation(sent, &reply.observations(Instant::now()))
        .with_context(|| format!("POST {url}"))
}

/// POST `body` to `url` streamed, with the token counts on the last chunk, and
/// [`read`] the stream as it arrives. A silent server fails on the client's
/// request timeout, which bounds each read, not the whole stream.
pub fn stream(http: &HttpClient, url: &str, body: &impl Serialize) -> anyhow::Result<Generation> {
    // Built before the clock starts, so encoding the body is not timed.
    let request = http.client().post(url).json(&streamed(body)?);
    let sent = Instant::now();
    let response = pipette_http::send(request).with_context(|| format!("POST {url}"))?;
    read(BufReader::new(response), sent, Instant::now)
        .with_context(|| format!("reading the stream from {url}"))
}

/// Read a stream sent at `sent` to `[DONE]`, each event stamped by `now` as it
/// arrives and before it is parsed, since parsing is this side's cost. Lines
/// are read directly: a pump thread's read-ahead would stamp them when
/// dequeued.
pub fn read(
    reader: impl BufRead,
    sent: Instant,
    mut now: impl FnMut() -> Instant,
) -> anyhow::Result<Generation> {
    let mut observed =
        sse::payloads(reader.lines()).try_fold(Vec::new(), |mut observed, data| {
            let data = data?;
            let at = now();
            let chunk: Chunk = serde_json::from_str(&data)
                .with_context(|| format!("failed to parse a chunk: {data}"))?;
            if let Some(error) = chunk.error {
                anyhow::bail!("the server failed mid-stream: {error}");
            }
            let choice = chunk.choices.into_iter().next();
            observed.extend(
                choice
                    .map(Choice::arrivals)
                    .into_iter()
                    .flatten()
                    .chain(chunk.usage.map(Usage::arrived))
                    .map(|what| Observation {
                        prompt: 0,
                        at,
                        what,
                    }),
            );
            anyhow::Ok(observed)
        })?;
    observed.push(Observation {
        prompt: 0,
        at: now(),
        what: Arrived::End,
    });
    generation::generation(sent, &observed)
}

/// `body` with streaming on; `include_usage` puts the token counts on the last
/// chunk, the only place a stream reports them.
pub fn streamed(body: &impl Serialize) -> anyhow::Result<Value> {
    let mut body = serde_json::to_value(body).context("failed to encode the request")?;
    let keys = body
        .as_object_mut()
        .context("the request did not encode to a JSON object")?;
    keys.insert("stream".into(), Value::Bool(true));
    keys.insert(
        "stream_options".into(),
        serde_json::json!({ "include_usage": true }),
    );
    Ok(body)
}

/// A prompt of exactly `target_tokens`, counted by the server's own
/// `/tokenize`, which encodes as `/v1/completions` does. Only pipette's own
/// servers answer `/tokenize` with a `count`.
pub fn build_prompt_text(
    http: &HttpClient,
    base_url: &str,
    model: &str,
    target_tokens: u32,
) -> anyhow::Result<String> {
    #[derive(Deserialize)]
    struct Tokens {
        count: usize,
    }
    let url = format!("{base_url}/tokenize");
    pipette_ops::prompt_seed::build_prompt_text(target_tokens, |text| {
        let body =
            serde_json::json!({ "model": model, "prompt": text, "add_special_tokens": true });
        let tokens: Tokens = http
            .json_request(reqwest::Method::POST, &url, None, Some(body))
            .with_context(|| format!("POST {url} while building the prompt"))?;
        Ok(tokens.count)
    })
}

/// A failure after the headers arrives as `error`; the usage trailer has an
/// empty `choices`.
#[derive(Debug, Deserialize)]
struct Chunk {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct Reply {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

impl Reply {
    fn observations(self, at: Instant) -> Vec<Observation> {
        let choice = self.choices.into_iter().next();
        choice
            .map(Choice::arrivals)
            .into_iter()
            .flatten()
            .chain(self.usage.map(Usage::arrived))
            .chain([Arrived::End])
            .map(|what| Observation {
                prompt: 0,
                at,
                what,
            })
            .collect()
    }
}

/// A choice is a token whatever its text: a server holding back an
/// incomplete UTF-8 sequence still sends the step's chunk, empty.
#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    finish_reason: Option<String>,
}

impl Choice {
    fn arrivals(self) -> Vec<Arrived> {
        let finish = self.finish_reason.map(|reason| match reason.as_str() {
            "length" => Finish::Limit,
            "stop" => Finish::Stop,
            _ => Finish::Other(reason),
        });
        [Arrived::Token]
            .into_iter()
            .chain(finish.map(Arrived::Finish))
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

impl Usage {
    fn arrived(self) -> Arrived {
        Arrived::Counts(Counts {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
        })
    }
}

/// Shared by the engine crates' tests, which `#[cfg(test)]` does not reach.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use std::time::{Duration, Instant};

    use pipette_ops::generation::Generation;

    /// `stream`'s bytes on a clock that advances 10 ms an event, so an event's
    /// stamp is its place in the stream times 10 ms.
    pub fn read_ticking(stream: &str) -> anyhow::Result<Generation> {
        let started = Instant::now();
        let mut at = started;
        super::read(stream.as_bytes(), started, || {
            at += Duration::from_millis(10);
            at
        })
    }

    /// One finish chunk; the counts come from the usage trailer.
    pub fn stream_at_limit(prompt_tokens: u32, completion_tokens: u32) -> String {
        let finish = serde_json::json!({
            "choices": [{"index": 0, "text": "x", "finish_reason": "length"}],
        });
        let usage = serde_json::json!({
            "choices": [],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        });
        format!("data: {finish}\n\ndata: {usage}\n\ndata: [DONE]\n\n")
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use httpmock::MockServer;
    use rstest::rstest;

    use super::test_support::read_ticking;
    use super::*;

    /// A one-token generation whose finish reason rides on the token's chunk.
    const ONE_CHUNK_STREAM: &str = "\
data: {\"choices\":[{\"index\":0,\"text\":\"A\",\"finish_reason\":\"length\"}]}\n\
data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\
data: [DONE]\n";

    /// A finish reason makes its chunk the end, so the usage trailer and
    /// `[DONE]` are not timed; a stream without one ends at `[DONE]`. An empty
    /// chunk is a token: a first token whose text is held back would otherwise
    /// land a chunk late. An empty `data:` event takes no stamp.
    #[rstest]
    #[case::one_chunk(ONE_CHUNK_STREAM, (10.0, 10.0, 10.0), Some(Finish::Limit))]
    #[case::empty_chunk(
        "data: {\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":null}]}\n\
         data: {\"choices\":[{\"index\":0,\"text\":\"A\",\"finish_reason\":\"length\"}]}\n\
         data: [DONE]\n",
        (10.0, 20.0, 20.0),
        Some(Finish::Limit)
    )]
    #[case::empty_data_event(
        "data: {\"choices\":[{\"index\":0,\"text\":\"A\",\"finish_reason\":null}]}\n\
         data:\n\
         data: {\"choices\":[{\"index\":0,\"text\":\"B\",\"finish_reason\":\"length\"}]}\n\
         data: [DONE]\n",
        (10.0, 20.0, 20.0),
        Some(Finish::Limit)
    )]
    #[case::trailing_space_after_done(
        "data: {\"choices\":[{\"index\":0,\"text\":\"AB\",\"finish_reason\":\"length\"}]}\n\
         data: [DONE] \n",
        (10.0, 10.0, 10.0),
        Some(Finish::Limit)
    )]
    #[case::no_finish(
        "data: {\"choices\":[{\"index\":0,\"text\":\"A\",\"finish_reason\":null}]}\n\
         data: [DONE]\n",
        (10.0, 10.0, 20.0),
        None
    )]
    #[case::in_house_servers(
        "data: {\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":null}]}\n\n\
         data: {\"choices\":[{\"index\":0,\"text\":\"A\",\"finish_reason\":null}]}\n\n\
         data: {\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"length\"}]}\n\n\
         data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n\
         data: [DONE]\n\n",
        (10.0, 30.0, 30.0),
        Some(Finish::Limit)
    )]
    fn stamps_each_event_as_it_arrives(
        #[case] stream: &str,
        #[case] stamps: (f64, f64, f64),
        #[case] finish: Option<Finish>,
    ) -> anyhow::Result<()> {
        let generation = read_ticking(stream)?;
        assert_eq!(
            (
                (
                    generation.first_token_ms,
                    generation.last_token_ms,
                    generation.end_ms
                ),
                generation.finish
            ),
            (stamps, finish)
        );
        Ok(())
    }

    #[rstest]
    #[case::cut(
        "data: {\"choices\":[{\"index\":0,\"text\":\"A\",\"finish_reason\":null}]}\n",
        "ended before its terminal event or [DONE]"
    )]
    #[case::no_token("data: {\"choices\":[]}\ndata: [DONE]\n", "ended without a token")]
    #[case::mid_stream_error(
        "data: {\"error\":{\"message\":\"device lost\"}}\ndata: [DONE]\n",
        "failed mid-stream"
    )]
    #[case::malformed("data: not json\ndata: [DONE]\n", "failed to parse")]
    fn a_stream_it_cannot_read_is_refused(
        #[case] stream: &str,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let err = read_ticking(stream).err().context("expected a refusal")?;
        assert!(format!("{err:#}").contains(expected), "{err:#}");
        Ok(())
    }

    /// Nothing else about the request changes.
    #[test]
    fn a_streamed_body_asks_for_the_usage_trailer() -> anyhow::Result<()> {
        let body = serde_json::json!({ "model": "m", "prompt": "p", "max_tokens": 3 });
        let streamed = streamed(&body)?;
        assert_eq!(
            streamed,
            serde_json::json!({
                "model": "m", "prompt": "p", "max_tokens": 3,
                "stream": true, "stream_options": { "include_usage": true },
            })
        );
        Ok(())
    }

    fn client() -> anyhow::Result<HttpClient> {
        Ok(HttpClient::loopback("pipette", Duration::from_secs(5))?)
    }

    #[test]
    fn a_text_request_is_greedy_past_eos() -> anyhow::Result<()> {
        assert_eq!(
            serde_json::to_value(TimingRequest::text(Some("m"), "p", 3))?,
            serde_json::json!({
                "model": "m", "prompt": "p", "max_tokens": 3,
                "temperature": 0.0, "ignore_eos": true,
            })
        );
        Ok(())
    }

    /// Each piece replaced: no model, a conversation, and the server's own keys.
    #[test]
    fn every_piece_can_be_the_engines() -> anyhow::Result<()> {
        #[derive(Serialize)]
        struct Keys {
            ignore_eos: bool,
            cache_prompt: bool,
        }
        let request = TimingRequest::new(
            None,
            Messages {
                messages: serde_json::json!([{ "role": "user", "content": "hi" }]),
            },
            2,
            Keys {
                ignore_eos: true,
                cache_prompt: false,
            },
        );
        assert_eq!(
            serde_json::to_value(request)?,
            serde_json::json!({
                "messages": [{ "role": "user", "content": "hi" }], "max_tokens": 2,
                "temperature": 0.0, "ignore_eos": true, "cache_prompt": false,
            })
        );
        Ok(())
    }

    #[test]
    fn a_stream_is_asked_for_and_read() -> anyhow::Result<()> {
        let server = MockServer::start();
        let completions = server.mock(|when, then| {
            when.method("POST")
                .path("/v1/completions")
                .json_body_includes(r#"{"stream":true,"stream_options":{"include_usage":true}}"#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(ONE_CHUNK_STREAM);
        });
        let url = server.url("/v1/completions");
        let body = serde_json::json!({ "model": "m", "prompt": "p", "max_tokens": 1 });
        let generation = stream(&client()?, &url, &body)?;
        completions.assert();
        assert_eq!(
            generation.counts_at_limit()?,
            Counts {
                prompt_tokens: 3,
                completion_tokens: 1,
            }
        );
        Ok(())
    }

    /// A whole reply: why it stopped and the counts, at one arrival.
    #[rstest]
    #[case::answered(
        r#"{"choices":[{"text":"ab","finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#,
        None
    )]
    #[case::no_choice(
        r#"{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#,
        Some("ended without a token")
    )]
    fn a_whole_reply_is_read_at_once(
        #[case] body: &str,
        #[case] refusal: Option<&str>,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v1/completions");
            then.status(200)
                .header("content-type", "application/json")
                .body(body);
        });
        let url = server.url("/v1/completions");
        let got = reply(&client()?, &url, &serde_json::json!({ "prompt": "p" }));
        match (got, refusal) {
            (Ok(generation), None) => {
                assert_eq!(generation.first_token_ms, generation.end_ms);
                let counts = generation.counts_at_limit()?;
                assert_eq!((counts.prompt_tokens, counts.completion_tokens), (5, 2));
            }
            (Err(err), Some(refusal)) => {
                assert!(format!("{err:#}").contains(refusal), "{err:#}");
            }
            (got, refusal) => anyhow::bail!("expected {refusal:?}, got {got:?}"),
        }
        Ok(())
    }

    /// The body OVMS sends for a token-id prompt, verbatim.
    const REFUSED_IDS: &str = r#"{"error":"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \nCalculator::Process() for node \"LLMExecutor\" failed: prompt is not a string"}"#;

    /// A refusal reaches the cell's error with the server's body intact, sent
    /// whole or streamed.
    #[rstest]
    #[case::whole(false)]
    #[case::streamed(true)]
    fn a_refused_request_surfaces_the_servers_body(#[case] streaming: bool) -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v3/completions");
            then.status(400).body(REFUSED_IDS);
        });
        let http = client()?;
        let url = server.url("/v3/completions");
        let body = serde_json::json!({ "prompt": "hi", "max_tokens": 4 });
        let got = if streaming {
            stream(&http, &url, &body)
        } else {
            reply(&http, &url, &body)
        };
        let message = format!("{:#}", got.err().context("expected a refusal")?);
        assert!(message.contains("400"), "{message}");
        assert!(message.contains("prompt is not a string"), "{message}");
        Ok(())
    }
}
