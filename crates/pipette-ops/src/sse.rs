//! Reading a streamed generation: the text as it arrives ([`read_text_stream`]),
//! and for an OpenAI-compatible chat stream also how it ended
//! ([`read_chat_stream`]). An engine contributes only where in its chunk the
//! delta sits.

use std::cell::{Cell, RefCell};
use std::io::BufRead;
use std::sync::mpsc::RecvTimeoutError;
use std::time::Duration;

use anyhow::Context;
use serde_json::Value;

use pipette_plan_types::BenchmarkEvalCompletionStopReason;

/// Events between hand-offs of the pending text to `on_flush`.
const FLUSH_EVERY: u64 = 10;

/// What one event carries, as the engine reads it off its own chunk shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Delta<'a> {
    /// The text this event adds; empty for a role-only or finish chunk.
    pub text: &'a str,
    /// The event that ends the generation, for a surface that says so in the
    /// event rather than with `[DONE]`. The stream stops after it.
    pub terminal: bool,
}

/// A stream read to its end.
#[derive(Debug, Default)]
pub struct TextStream {
    pub content: String,
    /// The event a [`Delta::terminal`] marked, whole, for the engine to read
    /// its stop metadata off. `None` when the stream ended on `[DONE]` or EOF.
    pub terminal: Option<Value>,
}

/// The `data:` payloads of an SSE body, one per event, up to `[DONE]`.
///
/// Comments, `retry:` lines and blank lines are skipped, and `data:` with or
/// without a space after the colon are the same event. A body that ends
/// before `[DONE]` is an error: a stream cut mid-generation must not read
/// as a short answer. A surface with no `[DONE]` ends on a terminal event
/// instead, and [`read_text_stream`] stops reading there.
///
/// `idle_timeout` bounds the silence between events, the only bound a
/// streamed generation can carry: an overall timeout would kill a long decode
/// that is making progress. Lines arrive on a pump thread so the gap can be
/// measured. Dropping the iterator does not interrupt the pump: it exits, and
/// drops the connection, on the next line it reads.
///
/// Not for a timing cell: the pump reads ahead, so a per-event clock stamps
/// when a line was dequeued, late under load. A cell that measures the stream
/// should drive [`payloads`] off `reader.lines()`.
pub fn data_lines<R>(
    reader: R,
    idle_timeout: Duration,
) -> impl Iterator<Item = anyhow::Result<String>>
where
    R: BufRead + Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        reader.lines().try_for_each(|line| {
            let failed = line.is_err();
            tx.send(line).map_err(|_| ())?;
            // A read error is the pump's last word; the receiver reports it.
            if failed { Err(()) } else { Ok(()) }
        })
    });
    let mut cut = false;
    payloads(std::iter::from_fn(move || {
        if cut {
            return None;
        }
        match rx.recv_timeout(idle_timeout) {
            Ok(line) => Some(line),
            // The channel closes when the pump exits, which is the body's
            // end: the rules below turn that into the cut-stream error.
            Err(RecvTimeoutError::Disconnected) => {
                cut = true;
                None
            }
            // Silence is this side's own bound, and says so.
            Err(RecvTimeoutError::Timeout) => {
                cut = true;
                Some(Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("no SSE stream event for {idle_timeout:?}"),
                )))
            }
        }
    }))
}

/// [`data_lines`]' rules without its pump, for an engine whose transport
/// needs its own.
pub fn payloads(
    lines: impl Iterator<Item = std::io::Result<String>>,
) -> impl Iterator<Item = anyhow::Result<String>> {
    let mut lines = lines;
    let mut done = false;
    std::iter::from_fn(move || {
        if done {
            return None;
        }
        loop {
            match lines.next() {
                None => {
                    done = true;
                    return Some(Err(anyhow::anyhow!(
                        "SSE stream ended before its terminal event or [DONE]"
                    )));
                }
                Some(Err(err)) => {
                    done = true;
                    return Some(Err(
                        anyhow::Error::from(err).context("failed to read SSE line")
                    ));
                }
                Some(Ok(line)) => {
                    let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                        continue;
                    };
                    // An empty `data:` event is legal SSE and carries nothing.
                    if data.is_empty() {
                        continue;
                    }
                    if data == "[DONE]" {
                        done = true;
                        return None;
                    }
                    return Some(Ok(data.to_owned()));
                }
            }
        }
    })
}

/// Read a generation off `events`, the `data:` payloads in order.
///
/// `delta` is the engine's: where the text sits in its chunk, and which
/// chunk is the last. Keepalives and comments never reach here, so a payload
/// that is not JSON, or that carries an `error` (how servers report a failure
/// after the headers), fails the stream, or its text would be lost from a
/// completion that then looks whole. `on_flush`
/// receives the text every [`FLUSH_EVERY`] events and at the end.
pub fn read_text_stream(
    events: impl IntoIterator<Item = anyhow::Result<String>>,
    delta: &dyn Fn(&Value) -> Delta<'_>,
    on_flush: &mut dyn FnMut(&str),
) -> anyhow::Result<TextStream> {
    let mut stream = TextStream::default();
    let mut pending = String::new();
    let mut chunks: u64 = 0;
    for data in events {
        let data = data?;
        let event = serde_json::from_str::<Value>(&data)
            .with_context(|| format!("failed to parse SSE data: {data}"))?;
        // `null` is not a failure: only a body is.
        if let Some(error) = event.get("error").filter(|error| !error.is_null()) {
            if !pending.is_empty() {
                on_flush(&pending);
            }
            anyhow::bail!("the stream failed mid-generation: {error}");
        }
        let Delta { text, terminal } = delta(&event);
        if !text.is_empty() {
            stream.content.push_str(text);
            pending.push_str(text);
        }
        chunks += 1;
        if chunks.is_multiple_of(FLUSH_EVERY) && !pending.is_empty() {
            on_flush(&pending);
            pending.clear();
        }
        if terminal {
            stream.terminal = Some(event);
            break;
        }
    }
    if !pending.is_empty() {
        on_flush(&pending);
    }
    Ok(stream)
}

/// The delta of an OpenAI-compatible `/v1/chat/completions` chunk:
/// `choices[0].delta.content`. The stream ends on `[DONE]`, so no chunk is
/// terminal.
pub fn chat_delta(event: &Value) -> Delta<'_> {
    Delta {
        text: event
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(Value::as_str)
            .unwrap_or(""),
        terminal: false,
    }
}

/// The reasoning of an OpenAI-compatible `/v1/chat/completions` chunk:
/// `choices[0].delta.reasoning`, where a vLLM server with a reasoning parser
/// streams what the model wrote before its answer, apart from `content`.
/// Empty on a chunk without it.
pub fn chat_reasoning(event: &Value) -> &str {
    event
        .pointer("/choices/0/delta/reasoning")
        .and_then(Value::as_str)
        .unwrap_or("")
}

/// A chat stream read to `[DONE]`.
#[derive(Debug, Default)]
pub struct ChatReply {
    pub content: String,
    /// Every chunk's [`chat_reasoning`], in order. Empty when the server split
    /// no reasoning off `content`.
    pub reasoning: String,
    /// How much of `content`, in bytes, had come when the first reasoning did:
    /// what the model wrote before it reasoned. Zero without reasoning.
    pub content_before_reasoning: usize,
    /// The finish chunk's `choices[0].finish_reason`.
    pub finish_reason: Option<String>,
    /// The usage trailer's count, EOS token included. Only a request that sets
    /// `stream_options.include_usage` gets the trailer.
    pub completion_tokens: Option<u64>,
}

impl ChatReply {
    /// The text the model wrote, with the reasoning the server split off put
    /// back in front of the answer as `<reasoning></think><answer>`, the shape
    /// a grader that cuts through the first `</think>` expects. With no answer
    /// (the stop came inside the reasoning) there is no tag. Content from
    /// before the first reasoning stays in front, and all of the reasoning
    /// comes before the answer even when the server interleaved them.
    pub fn completion_with_reasoning(self) -> String {
        if self.reasoning.is_empty() {
            return self.content;
        }
        let (before, answer) = self
            .content
            .split_at_checked(self.content_before_reasoning)
            .unwrap_or(("", &self.content));
        let close = if answer.is_empty() { "" } else { "</think>" };
        [before, &self.reasoning, close, answer].concat()
    }

    /// The row's stop reason and its detail. `stop` is EOS because eval sends
    /// no stop words; `repetition` is vLLM's opt-in repetition detection.
    pub fn stop(&self) -> (BenchmarkEvalCompletionStopReason, Option<String>) {
        match self.finish_reason.as_deref() {
            Some("stop") => (BenchmarkEvalCompletionStopReason::Eos, None),
            Some("length") => (BenchmarkEvalCompletionStopReason::Truncated, None),
            Some("repetition") => (
                BenchmarkEvalCompletionStopReason::DoomLoop,
                Some("finish_reason=repetition".to_owned()),
            ),
            Some(other) => (
                BenchmarkEvalCompletionStopReason::Unknown,
                Some(format!("unrecognized finish_reason={other}")),
            ),
            None => (
                BenchmarkEvalCompletionStopReason::Unknown,
                Some("the stream carried no finish_reason".to_owned()),
            ),
        }
    }
}

/// Read an OpenAI-compatible chat stream by [`read_text_stream`]'s rules, its
/// text by [`chat_delta`], noting the reasoning ([`chat_reasoning`]), the
/// finish chunk and the usage trailer as they pass. Only the content reaches
/// `on_flush`.
pub fn read_chat_stream(
    events: impl IntoIterator<Item = anyhow::Result<String>>,
    on_flush: &mut dyn FnMut(&str),
) -> anyhow::Result<ChatReply> {
    let ending = RefCell::new(ChatReply::default());
    let content_so_far = Cell::new(0);
    let stream = read_text_stream(
        events,
        &|event| {
            let mut ending = ending.borrow_mut();
            let delta = chat_delta(event);
            let reasoning = chat_reasoning(event);
            if !reasoning.is_empty() {
                // A chunk carrying both is the reasoning's end: its content
                // comes after.
                if ending.reasoning.is_empty() {
                    ending.content_before_reasoning = content_so_far.get();
                }
                ending.reasoning.push_str(reasoning);
            }
            content_so_far.set(content_so_far.get() + delta.text.len());
            if let Some(reason) = event
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str)
            {
                ending.finish_reason = Some(reason.to_owned());
            }
            if let Some(tokens) = event
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
            {
                ending.completion_tokens = Some(tokens);
            }
            delta
        },
        on_flush,
    )?;
    Ok(ChatReply {
        content: stream.content,
        ..ending.into_inner()
    })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use rstest::rstest;
    use serde_json::json;

    use super::*;

    /// A fixture body ends at EOF, never on silence, so the bound a test
    /// passes is unreachable.
    const NO_IDLE: Duration = Duration::from_secs(3600);

    /// The `llama-server` `/completion` shape: text at `content`, the end
    /// marked by `stop`.
    fn completion_delta(event: &Value) -> Delta<'_> {
        Delta {
            text: event.get("content").and_then(Value::as_str).unwrap_or(""),
            terminal: event.get("stop").and_then(Value::as_bool) == Some(true),
        }
    }

    fn completion_line(content: &str, stop: bool) -> String {
        format!("data: {}\n\n", json!({"content": content, "stop": stop}))
    }

    fn read(input: String) -> anyhow::Result<TextStream> {
        read_text_stream(
            data_lines(Cursor::new(input), NO_IDLE),
            &completion_delta,
            &mut |_| {},
        )
    }

    #[rstest]
    #[case::accumulates(format!("{}{}", completion_line("hello ", false), completion_line("world", true)), "hello world")]
    #[case::data_without_space("data:{\"content\":\"hi\",\"stop\":true}\n\n".to_owned(), "hi")]
    #[case::skips_non_data_lines(format!(": keepalive\nretry: 100\n{}", completion_line("ok", true)), "ok")]
    #[case::unicode(format!("{}{}{}", completion_line("こんにちは", false), completion_line("🌍", false), completion_line("", true)), "こんにちは🌍")]
    #[case::empty(completion_line("", true), "")]
    #[case::trailing_space_after_done(format!("{}data: [DONE] \n\n", completion_line("ok", true)), "ok")]
    #[case::empty_data_event(format!("data:\n\ndata: \n\n{}", completion_line("ok", true)), "ok")]
    #[case::null_error_is_not_a_failure("data: {\"content\":\"ok\",\"error\":null,\"stop\":true}\n\n".to_owned(), "ok")]
    fn the_text_is_the_deltas_in_order(
        #[case] input: String,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let stream = read(input)?;
        assert_eq!(stream.content, expected);
        assert!(stream.terminal.is_some(), "the stop chunk ends the stream");
        Ok(())
    }

    #[test]
    fn the_text_is_handed_on_every_ten_events_and_at_the_end() -> anyhow::Result<()> {
        let input: String = (0..15)
            .map(|i| completion_line(&format!("{i}"), i == 14))
            .collect();
        let mut flushes = Vec::new();
        read_text_stream(
            data_lines(Cursor::new(input), NO_IDLE),
            &completion_delta,
            &mut |text| flushes.push(text.to_owned()),
        )?;
        assert_eq!(flushes, ["0123456789", "1011121314"]);
        Ok(())
    }

    #[test]
    fn a_chat_stream_ends_on_done() -> anyhow::Result<()> {
        let input = "data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n\n\
                     data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}]}\n\n\
                     data: [DONE]\n\n\
                     data: {\"choices\": [{\"delta\": {\"content\": \"late\"}}]}\n\n";
        let stream = read_text_stream(
            data_lines(Cursor::new(input), NO_IDLE),
            &chat_delta,
            &mut |_| {},
        )?;
        assert_eq!(stream.content, "Hello world");
        assert!(stream.terminal.is_none());
        Ok(())
    }

    /// A completion missing text must not checkpoint as whole.
    #[rstest]
    #[case::read_error(vec![Ok("{\"content\":\"a\"}".to_owned()), Err(anyhow::anyhow!("cut"))], "cut")]
    #[case::malformed_chunk(vec![Ok("{\"content\":\"a\"".to_owned()), Ok("{\"stop\":true}".to_owned())], "failed to parse")]
    #[case::error_event(vec![Ok("{\"content\":\"a\"}".to_owned()), Ok("{\"error\":{\"code\":500,\"message\":\"slot lost\"}}".to_owned())], "slot lost")]
    fn a_damaged_stream_fails_the_read(
        #[case] events: Vec<anyhow::Result<String>>,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let err = read_text_stream(events, &completion_delta, &mut |_| {})
            .err()
            .context("expected the read to fail")?;
        assert!(format!("{err:#}").contains(expected), "{err:#}");
        Ok(())
    }

    /// A pipe with a live writer never yields a line, so the pump blocks.
    #[test]
    fn silence_past_the_idle_bound_is_an_error() -> anyhow::Result<()> {
        let (reader, _writer) = std::io::pipe()?;
        let err = read_text_stream(
            data_lines(std::io::BufReader::new(reader), Duration::from_millis(50)),
            &chat_delta,
            &mut |_| {},
        )
        .err()
        .context("expected the idle bound to fire")?;
        assert!(
            format!("{err:#}").contains("no SSE stream event"),
            "{err:#}"
        );
        Ok(())
    }

    /// Shaped after vLLM 0.31's chat stream (`chat_completion/serving.py`):
    /// the finish chunk carries the last token's text beside the reason, and
    /// the usage trailer, sent only on `include_usage`, has no choice.
    fn vllm_chat_stream(finish_reason: Option<&str>, usage: bool) -> String {
        let mut events = vec![
            json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}),
            json!({"choices": [{"index": 0, "delta": {"content": "The answer"}, "finish_reason": null}]}),
            json!({"choices": [{"index": 0, "delta": {"content": " is"}, "finish_reason": finish_reason, "stop_reason": null}]}),
        ];
        if usage {
            events.push(json!({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}}));
        }
        events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .chain(["data: [DONE]\n\n".to_owned()])
            .collect()
    }

    /// The text includes the finish chunk's content.
    #[rstest]
    #[case::stopped_on_eos(vllm_chat_stream(Some("stop"), false), Some("stop"), None)]
    #[case::cut_at_max_tokens(vllm_chat_stream(Some("length"), true), Some("length"), Some(3))]
    #[case::stopped_on_repetition(
        vllm_chat_stream(Some("repetition"), false),
        Some("repetition"),
        None
    )]
    #[case::no_finish_reason(vllm_chat_stream(None, false), None, None)]
    fn a_chat_stream_says_how_it_ended(
        #[case] body: String,
        #[case] finish_reason: Option<&str>,
        #[case] completion_tokens: Option<u64>,
    ) -> anyhow::Result<()> {
        let mut flushed = String::new();
        let reply = read_chat_stream(data_lines(Cursor::new(body), NO_IDLE), &mut |text| {
            flushed.push_str(text);
        })?;
        assert_eq!(
            (
                reply.content.as_str(),
                reply.finish_reason.as_deref(),
                reply.completion_tokens
            ),
            ("The answer is", finish_reason, completion_tokens)
        );
        assert_eq!(flushed, reply.content);
        Ok(())
    }

    #[rstest]
    #[case::eos(Some("stop"), BenchmarkEvalCompletionStopReason::Eos, None)]
    #[case::token_cap(Some("length"), BenchmarkEvalCompletionStopReason::Truncated, None)]
    #[case::vllm_repetition_detection(
        Some("repetition"),
        BenchmarkEvalCompletionStopReason::DoomLoop,
        Some("finish_reason=repetition")
    )]
    #[case::tool_call(
        Some("tool_calls"),
        BenchmarkEvalCompletionStopReason::Unknown,
        Some("unrecognized finish_reason=tool_calls")
    )]
    #[case::none(
        None,
        BenchmarkEvalCompletionStopReason::Unknown,
        Some("the stream carried no finish_reason")
    )]
    fn a_finish_reason_is_a_stop_reason(
        #[case] finish_reason: Option<&str>,
        #[case] stop_reason: BenchmarkEvalCompletionStopReason,
        #[case] stop_detail: Option<&str>,
    ) {
        let reply = ChatReply {
            finish_reason: finish_reason.map(str::to_owned),
            ..ChatReply::default()
        };
        let (got, detail) = reply.stop();
        assert_eq!((got, detail.as_deref()), (stop_reason, stop_detail));
    }

    /// A vLLM 0.31 chat stream from a server with a reasoning parser: the role
    /// chunk, then one chunk per `(reasoning, content)` delta carrying the
    /// fields that are set, the last beside `finish_reason`. A delta with
    /// neither is the empty `{}` vLLM sends when the last token had no text.
    fn vllm_reasoning_stream(
        deltas: &[(Option<&str>, Option<&str>)],
        finish_reason: &str,
    ) -> String {
        let last = deltas.len().saturating_sub(1);
        let chunks = deltas.iter().enumerate().map(|(at, (reasoning, content))| {
            let mut delta = serde_json::Map::new();
            if let Some(reasoning) = reasoning {
                delta.insert("reasoning".to_owned(), json!(reasoning));
            }
            if let Some(content) = content {
                delta.insert("content".to_owned(), json!(content));
            }
            let finish_reason = (at == last).then_some(finish_reason);
            json!({"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})
        });
        std::iter::once(json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}))
            .chain(chunks)
            .map(|event| format!("data: {event}\n\n"))
            .chain(["data: [DONE]\n\n".to_owned()])
            .collect()
    }

    /// `content` and what reaches `on_flush` stay the content alone.
    #[rstest]
    #[case::content_only(
        &[(None, Some("The answer")), (None, Some(" is 4."))],
        "stop",
        "The answer is 4."
    )]
    #[case::gemma4_reasoning_then_answer(
        &[(Some("Two and"), None), (Some(" two."), None), (None, Some("4.")), (None, None)],
        "stop",
        "Two and two.</think>4."
    )]
    // ernie45 leaves the <response> wrapper in the content.
    #[case::ernie45_response_wrapper(
        &[(Some("Two and two.\n"), None), (None, Some("\n\n<response>\n4.\n</response>\n"))],
        "stop",
        "Two and two.\n</think>\n\n<response>\n4.\n</response>\n"
    )]
    // Harmony: the analysis channel is the reasoning, the final channel the
    // content, and the `<|return|>` token's chunk is empty.
    #[case::gpt_oss_harmony(
        &[(Some("We need"), None), (Some(" 2+2."), None), (None, Some("4")), (None, None)],
        "stop",
        "We need 2+2.</think>4"
    )]
    #[case::cut_in_the_reasoning(
        &[(Some("Two and"), None), (Some(" two is"), None)],
        "length",
        "Two and two is"
    )]
    #[case::gpt_oss_cut_in_the_analysis(
        &[(Some("We need"), None), (Some(" to add"), None)],
        "length",
        "We need to add"
    )]
    #[case::repetition_in_the_reasoning(
        &[(Some("two two"), None), (Some(" two two"), None)],
        "repetition",
        "two two two two"
    )]
    #[case::eos_in_the_reasoning(&[(Some("Two and two."), None), (None, None)], "stop", "Two and two.")]
    #[case::cut_in_the_answer(
        &[(Some("Two and two."), None), (None, Some("The ans"))],
        "length",
        "Two and two.</think>The ans"
    )]
    #[case::one_chunk_ends_the_reasoning(
        &[(Some("Two and"), None), (Some(" two."), Some("4."))],
        "stop",
        "Two and two.</think>4."
    )]
    #[case::content_before_the_reasoning(
        &[(None, Some("\n")), (Some("Two and"), None)],
        "length",
        "\nTwo and"
    )]
    #[case::interleaved(
        &[(Some("Two."), None), (None, Some("4")), (Some(" Check."), None), (None, Some("."))],
        "stop",
        "Two. Check.</think>4."
    )]
    fn a_vllm_reply_puts_its_reasoning_before_the_answer(
        #[case] deltas: &[(Option<&str>, Option<&str>)],
        #[case] finish_reason: &str,
        #[case] completion: &str,
    ) -> anyhow::Result<()> {
        let content: String = deltas.iter().filter_map(|(_, content)| *content).collect();
        let mut flushed = String::new();
        let reply = read_chat_stream(
            data_lines(
                Cursor::new(vllm_reasoning_stream(deltas, finish_reason)),
                NO_IDLE,
            ),
            &mut |text| flushed.push_str(text),
        )?;
        assert_eq!(reply.content, content);
        assert_eq!(flushed, content);
        assert_eq!(reply.finish_reason.as_deref(), Some(finish_reason));
        assert_eq!(reply.completion_with_reasoning(), completion);
        Ok(())
    }

    /// A body that closes before `[DONE]` is a cut stream; a surface that ends
    /// on a terminal event never reads that far.
    #[rstest]
    #[case::chat_without_done("data: {\"choices\": [{\"delta\": {\"content\": \"Hi\"}}]}\n\n", &chat_delta, false)]
    #[case::completion_ends_on_stop(completion_line("Hi", true), &completion_delta, true)]
    fn an_early_eof_is_an_error(
        #[case] body: impl Into<String>,
        #[case] delta: &dyn Fn(&Value) -> Delta<'_>,
        #[case] ok: bool,
    ) {
        let result = read_text_stream(
            data_lines(Cursor::new(body.into()), NO_IDLE),
            delta,
            &mut |_| {},
        );
        assert_eq!(result.is_ok(), ok, "{result:?}");
    }
}
