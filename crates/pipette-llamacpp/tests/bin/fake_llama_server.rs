//! Fake `llama-server` for the eval, VL and ASR integration tests.
//!
//! Serves `GET /health`, `GET /props`, `POST /apply-template`,
//! `POST /completion` (SSE when the body streams, JSON otherwise),
//! `POST /v1/chat/completions` and `POST /tokenize`. Reads `--port <P>` and
//! `--ctx-size`, and ignores every other flag.
//!
//! `/apply-template` renders the last user message, as a template with no
//! thinking switch would, whatever `chat_template_kwargs` says.
//!
//! Test hooks, all env vars ("matching" means the `/completion` or
//! `/v1/chat/completions` body contains the var's value):
//! - `FAKE_LLAMA_CRASH_PROMPT_CONTAINS`: a matching request exits the process
//!   with status 139.
//! - `FAKE_LLAMA_DROP_PROMPT_CONTAINS`: a matching request has its connection
//!   closed with no response while the process stays alive, like Windows
//!   `os error 10054` (WSAECONNRESET), where `try_wait` never sees an exit.
//! - `FAKE_LLAMA_DROP_ONCE_PROMPT_CONTAINS`: as above, but only the first time
//!   each matching body arrives, as a race-closed connection would.
//! - `FAKE_LLAMA_LIMIT_PROMPT_CONTAINS`: a matching streamed `/completion`
//!   ends with `stop_type: "limit"` (the output-token cap) instead of `"eos"`.
//! - `FAKE_LLAMA_SSE_ERROR_PROMPT_CONTAINS`: a matching streamed chat request
//!   sends one content chunk, then an in-band `error` event, llama.cpp's shape
//!   for a failure after the 200.
//! - `FAKE_LLAMA_SSE_TRUNCATE_PROMPT_CONTAINS`: a matching streamed chat
//!   request sends one content chunk, then closes with no `finish_reason` and
//!   no `[DONE]`.
//! - `FAKE_LLAMA_CHAT_400`: `/v1/chat/completions` answers 400 with this body.
//! - `FAKE_LLAMA_PROMPT_TOKENS`: a one-response chat request carrying `n_beams`
//!   is answered as the beam search of pipette's ASR build of llama.cpp: the
//!   room left in `--ctx-size` after that many prompt tokens is split among the
//!   beams, and the reply generates `min(max_tokens, room)` tokens with
//!   `finish_reason: "length"` and a usage count.
//! - `FAKE_LLAMA_CHAT_DELAY_MS`: holds every `/v1/chat/completions` reply.
//! - `FAKE_LLAMA_CHAT_REPLY`: an unstreamed chat reply, which otherwise stops
//!   at the limit with `max_tokens` decoded, ends with `finish_reason: stop`
//!   (`early_stop`) or one completion token short (`short`).
//! - `FAKE_LLAMA_AUDIO_TOKENS`: each chat `input_audio` part counts this many
//!   prompt tokens (default 20).
//! - `FAKE_LLAMA_COUNT_TEXT`: a chat prompt also counts one token per word of
//!   every message's text, as `/tokenize` counts them.
//! - `FAKE_LLAMA_NEWLINE`: how a newline between two words counts, in
//!   `/tokenize` and the chat prompt alike: `own` makes every newline a token
//!   (Gemma 4's tokenizer), `merge` glues its neighbours into one token, and
//!   `own_in_chat` does `own` in the chat prompt only, a `/tokenize` that
//!   disagrees with the prompt the server renders.
//! - `FAKE_LLAMA_ENFORCE_CTX`: a chat prompt that does not fit `--ctx-size` is
//!   refused with llama-server's context-overflow 400.
//! - `FAKE_LLAMA_PROPS_TEMPLATE`: `GET /props` reports this `chat_template`, a
//!   server that fell back to another template. Otherwise it reports the
//!   `--chat-template` it was started with, or a stand-in for the GGUF's.
//! - `FAKE_LLAMA_THINKING_SWITCH`: the template gets a thinking switch, and
//!   `enable_thinking: false` renders an empty think block after the message,
//!   as Qwen3's does; set to `system`, only when a system message is present.
//! - `FAKE_LLAMA_TEMPLATE_DAY_TURNS_AFTER=N`: each render opens with the date,
//!   as gpt-oss's template does, and the day turns after N renders.
//! - `FAKE_LLAMA_TEMPLATE_COUNTER`: each render opens with its own count, so
//!   no two agree.
//! - `FAKE_LLAMA_TOKENIZE_BOS`: a `/tokenize` of non-empty content with
//!   `add_special: true` counts one more token, as a model that prepends a BOS
//!   (Gemma, LFM2) does.
//! - `FAKE_LLAMA_BODY_FILE` / `FAKE_LLAMA_TEMPLATE_BODY_FILE`: each generation
//!   / `/apply-template` body is appended as one line.
//! - `FAKE_LLAMA_CHAT_IMAGES_FILE` / `FAKE_LLAMA_CHAT_BODY_FILE`: each chat
//!   request that gets past the triggers appends its image count / its body.
//! - `FAKE_LLAMA_CALL_LOG`: each generation request appends its body as one
//!   line, so a test can tell which samples a resume re-inferred; the
//!   checkpoint cannot, since its append is last-write-wins.
//! - `FAKE_LLAMA_PID_FILE`: each process appends its PID at startup (one line
//!   per process). With `FAKE_LLAMA_ALLOCATE_FIRST_MIB=N`, the process that
//!   finds the file empty holds N MiB until exit, and with
//!   `FAKE_LLAMA_LATER_AUDIO_TOKENS=N`, every later process counts N tokens per
//!   clip: a later server that prefills another prompt than the first.
//! - `FAKE_LLAMA_ARGS_FILE`: each process appends its argv, `--port` value
//!   masked as `<port>`.
//! - `FAKE_LLAMA_ARGV_FILE`: each process appends its argv unmasked.
//! - `FAKE_LLAMA_RENDEZVOUS=N`: parks every generation request until N are in
//!   flight or `FAKE_LLAMA_RENDEZVOUS_TIMEOUT_MS` (default
//!   [`DEFAULT_RENDEZVOUS_TIMEOUT`]) elapses, so overlap is asserted without
//!   sleeps; the high-water mark goes to `FAKE_LLAMA_PEAK_INFLIGHT_FILE`.
//!   Tests with a deliberately serial tail shorten the timeout.
//! - `FAKE_LLAMA_RENDEZVOUS_GRACE_MS`: delays the response (not the crash
//!   trigger) after the rendezvous, so a crashing process beats its peers'
//!   replies.

use std::{
    env,
    fs::OpenOptions,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    process,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::Context;

/// Upper bound on a rendezvous wait, so a client that never reaches the
/// target fails its assertion instead of hanging the test run.
const DEFAULT_RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(3);

static INFLIGHT: AtomicUsize = AtomicUsize::new(0);
static PEAK_INFLIGHT: AtomicUsize = AtomicUsize::new(0);
/// Latched once the rendezvous target is first reached (see
/// [`enter_generation`]). One-shot per process: tests care about the first
/// batch, and a restarted server starts over.
static RENDEZVOUS_MET: AtomicBool = AtomicBool::new(false);
/// Serializes load-then-write of [`PEAK_INFLIGHT`], so the last write to the
/// file carries the maximum, not a stale read.
static PEAK_FILE_LOCK: Mutex<()> = Mutex::new(());

/// Serializes the log-file appends: concurrent requests would interleave lines.
static BODY_FILE_LOCK: Mutex<()> = Mutex::new(());

static DROPPED_ONCE: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// How a non-streamed chat reply ends (`FAKE_LLAMA_CHAT_REPLY`).
#[derive(Clone, Copy, Default)]
enum ChatReply {
    #[default]
    Full,
    EarlyStop,
    Short,
}
/// How many `/apply-template` requests this process has rendered.
static RENDERS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Default)]
struct Triggers {
    crash: Option<String>,
    drop: Option<String>,
    drop_once: Option<String>,
    limit: Option<String>,
    rendezvous: Option<usize>,
    rendezvous_timeout: Duration,
    rendezvous_grace: Duration,
    peak_file: Option<String>,
    chat_400: Option<String>,
    sse_error: Option<String>,
    sse_truncate: Option<String>,
    call_log: Option<String>,
    chat_images_file: Option<String>,
    chat_body_file: Option<String>,
    chat_delay: Duration,
    chat_reply: ChatReply,
    body_file: Option<String>,
    template_body_file: Option<String>,
    audio_tokens: u64,
    count_text: bool,
    newline: Newline,
    tokenize_bos: bool,
    ctx_size: Option<u64>,
    chat_template: String,
    thinking_switch: Option<String>,
    day_turns_after: Option<usize>,
    template_counter: bool,
    /// `--ctx-size` from argv, and the prompt tokens to bill each beam
    /// request (`FAKE_LLAMA_PROMPT_TOKENS`): with both, a one-response request
    /// carrying `n_beams` is capped as pipette's ASR build of llama.cpp caps a
    /// beam search.
    beam_ctx: Option<u64>,
    prompt_tokens: Option<u64>,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let port = parse_port(&args).context("--port <P> is required")?;
    let triggers = Triggers {
        crash: env::var("FAKE_LLAMA_CRASH_PROMPT_CONTAINS").ok(),
        drop: env::var("FAKE_LLAMA_DROP_PROMPT_CONTAINS").ok(),
        drop_once: env::var("FAKE_LLAMA_DROP_ONCE_PROMPT_CONTAINS").ok(),
        limit: env::var("FAKE_LLAMA_LIMIT_PROMPT_CONTAINS").ok(),
        rendezvous: env::var("FAKE_LLAMA_RENDEZVOUS")
            .ok()
            .and_then(|v| v.parse().ok()),
        rendezvous_timeout: env::var("FAKE_LLAMA_RENDEZVOUS_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .map_or(DEFAULT_RENDEZVOUS_TIMEOUT, Duration::from_millis),
        rendezvous_grace: env::var("FAKE_LLAMA_RENDEZVOUS_GRACE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .map_or(Duration::ZERO, Duration::from_millis),
        peak_file: env::var("FAKE_LLAMA_PEAK_INFLIGHT_FILE").ok(),
        chat_400: env::var("FAKE_LLAMA_CHAT_400").ok(),
        sse_error: env::var("FAKE_LLAMA_SSE_ERROR_PROMPT_CONTAINS").ok(),
        sse_truncate: env::var("FAKE_LLAMA_SSE_TRUNCATE_PROMPT_CONTAINS").ok(),
        call_log: env::var("FAKE_LLAMA_CALL_LOG").ok(),
        chat_images_file: env::var("FAKE_LLAMA_CHAT_IMAGES_FILE").ok(),
        chat_body_file: env::var("FAKE_LLAMA_CHAT_BODY_FILE").ok(),
        chat_delay: env::var("FAKE_LLAMA_CHAT_DELAY_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .map_or(Duration::ZERO, Duration::from_millis),
        chat_reply: match env::var("FAKE_LLAMA_CHAT_REPLY").as_deref() {
            Ok("early_stop") => ChatReply::EarlyStop,
            Ok("short") => ChatReply::Short,
            _ => ChatReply::Full,
        },
        body_file: env::var("FAKE_LLAMA_BODY_FILE").ok(),
        template_body_file: env::var("FAKE_LLAMA_TEMPLATE_BODY_FILE").ok(),
        audio_tokens: env::var("FAKE_LLAMA_AUDIO_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20),
        count_text: env::var_os("FAKE_LLAMA_COUNT_TEXT").is_some(),
        newline: match env::var("FAKE_LLAMA_NEWLINE").as_deref() {
            Ok("own") => Newline::Own,
            Ok("merge") => Newline::Merge,
            Ok("own_in_chat") => Newline::OwnInChat,
            _ => Newline::Word,
        },
        tokenize_bos: env::var_os("FAKE_LLAMA_TOKENIZE_BOS").is_some(),
        ctx_size: env::var_os("FAKE_LLAMA_ENFORCE_CTX")
            .and_then(|_| flag_value(&args, "--ctx-size")?.parse().ok()),
        chat_template: env::var("FAKE_LLAMA_PROPS_TEMPLATE")
            .ok()
            // As llama.cpp's jinja lexer keeps it: one trailing newline dropped.
            .or_else(|| {
                flag_value(&args, "--chat-template")
                    .map(|t| t.strip_suffix('\n').unwrap_or(t).to_owned())
            })
            .unwrap_or_else(|| "{{ the GGUF's template }}".to_owned()),
        thinking_switch: env::var("FAKE_LLAMA_THINKING_SWITCH").ok(),
        day_turns_after: env::var("FAKE_LLAMA_TEMPLATE_DAY_TURNS_AFTER")
            .ok()
            .and_then(|v| v.parse().ok()),
        template_counter: env::var_os("FAKE_LLAMA_TEMPLATE_COUNTER").is_some(),
        beam_ctx: flag_value(&args, "--ctx-size").and_then(|ctx| ctx.parse().ok()),
        prompt_tokens: env::var("FAKE_LLAMA_PROMPT_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok()),
    };

    let mut triggers = triggers;
    // Tests opt in to PID logging via env var; one append per process
    // lets them count restarts.
    if let Ok(path) = env::var("FAKE_LLAMA_PID_FILE") {
        let first = first_process(&path)?;
        if let Some(later) = env::var("FAKE_LLAMA_LATER_AUDIO_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            && !first
        {
            triggers.audio_tokens = later;
        }
        if let Some(mib) = env::var("FAKE_LLAMA_ALLOCATE_FIRST_MIB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            && first
        {
            // Non-zero, so every page is written; leaked, so held until exit.
            std::hint::black_box(vec![1u8; mib << 20].leak());
        }
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{}", process::id());
        }
    }
    if let Ok(path) = env::var("FAKE_LLAMA_ARGS_FILE")
        && let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path)
    {
        let masked = args
            .iter()
            .zip(args.iter().skip(1))
            .map(|(before, arg)| {
                if before == "--port" {
                    "<port>"
                } else {
                    arg.as_str()
                }
            })
            .collect::<Vec<_>>();
        let _ = writeln!(f, "{}", masked.join(" "));
    }

    // A restart that dropped `--mmproj` does not crash: it fails every
    // multimodal sample with a template error, which scores as a wrong answer,
    // so only the argv shows it.
    if let Ok(path) = env::var("FAKE_LLAMA_ARGV_FILE")
        && let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path)
    {
        let _ = writeln!(f, "{}", args.get(1..).unwrap_or_default().join(" "));
    }

    // Like the real binary, write to stderr so the crash path has a log to
    // capture; the pid says which process a kept log is.
    eprintln!(
        "fake-llama-server: listening on 127.0.0.1:{port}, pid {}",
        process::id()
    );

    let listener = TcpListener::bind(("127.0.0.1", port)).context("bind failed")?;
    for stream in listener.incoming().flatten() {
        let triggers = triggers.clone();
        thread::spawn(move || handle_connection(stream, triggers));
    }
    Ok(())
}

/// No process has logged its PID to `path` yet.
fn first_process(path: &str) -> anyhow::Result<bool> {
    match std::fs::read_to_string(path) {
        Ok(pids) => Ok(pids.is_empty()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(err) => Err(err).with_context(|| format!("read {path}")),
    }
}

fn parse_port(args: &[String]) -> Option<u16> {
    flag_value(args, "--port")?.parse().ok()
}

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == flag {
            return it.next().map(String::as_str);
        }
    }
    None
}

fn handle_connection(mut stream: TcpStream, triggers: Triggers) {
    let Ok(mirror) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(mirror);

    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("")
        .to_string();

    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            return;
        }
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 && reader.read_exact(&mut body).is_err() {
        return;
    }
    let body_s = std::str::from_utf8(&body).unwrap_or("");

    match path.as_str() {
        "/health" => write_json(&mut stream, r#"{"status":"ok"}"#),
        "/props" => write_json(
            &mut stream,
            &format!(
                r#"{{"chat_template":{}}}"#,
                json_escape(&triggers.chat_template)
            ),
        ),
        "/apply-template" => {
            if let Some(path) = triggers.template_body_file.as_deref()
                && let Ok(_guard) = BODY_FILE_LOCK.lock()
                && let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path)
            {
                let _ = writeln!(f, "{body_s}");
            }
            let prompt = render_template(body_s, &triggers);
            let body = format!(r#"{{"prompt":{}}}"#, json_escape(&prompt));
            write_json(&mut stream, &body);
        }
        "/completion" => {
            log_call(&triggers, body_s);
            enter_generation(&triggers);
            handle_completion(&mut stream, body_s, &triggers);
            INFLIGHT.fetch_sub(1, Ordering::SeqCst);
        }
        "/v1/chat/completions" => {
            if let Some(path) = triggers.body_file.as_deref()
                && let Ok(_guard) = BODY_FILE_LOCK.lock()
                && let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path)
            {
                let _ = writeln!(f, "{body_s}");
            }
            log_call(&triggers, body_s);
            enter_generation(&triggers);
            handle_chat(&mut stream, body_s, &triggers);
            INFLIGHT.fetch_sub(1, Ordering::SeqCst);
        }
        "/tokenize" => {
            // llama.cpp's native body is {"content": "..."}, not the chat
            // `messages[]` shape. One token id per word, newlines as
            // `FAKE_LLAMA_NEWLINE` counts them.
            let request = serde_json::from_str::<serde_json::Value>(body_s).unwrap_or_default();
            let content = request
                .get("content")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            let bos = triggers.tokenize_bos
                && !content.is_empty()
                && request.get("add_special") == Some(&serde_json::Value::Bool(true));
            let words = usize::try_from(text_tokens(content, triggers.newline.in_tokenize()))
                .unwrap_or(usize::MAX);
            let n = (words + usize::from(bos)).max(1);
            let ids = vec!["0"; n].join(",");
            write_json(&mut stream, &format!(r#"{{"tokens":[{ids}]}}"#));
        }
        _ => {
            let _ = write!(
                stream,
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
        }
    }
}

/// How a newline between two words counts ([`text_tokens`]).
#[derive(Clone, Copy, Default)]
enum Newline {
    #[default]
    Word,
    Own,
    Merge,
    OwnInChat,
}

impl Newline {
    fn in_chat(self) -> Self {
        match self {
            Self::OwnInChat => Self::Own,
            other @ (Self::Word | Self::Own | Self::Merge) => other,
        }
    }

    fn in_tokenize(self) -> Self {
        match self {
            Self::OwnInChat => Self::Word,
            other @ (Self::Word | Self::Own | Self::Merge) => other,
        }
    }
}

/// The tokens of `text` under the fake tokenizer: one per whitespace-separated
/// word, and its newlines as `newline` counts them.
fn text_tokens(text: &str, newline: Newline) -> u64 {
    let words = text.split_whitespace().count() as u64;
    let between_words = text
        .as_bytes()
        .windows(3)
        .filter(
            |w| matches!(w, [a, b'\n', c] if !a.is_ascii_whitespace() && !c.is_ascii_whitespace()),
        )
        .count() as u64;
    match newline {
        Newline::Word | Newline::OwnInChat => words,
        Newline::Own => words + text.matches('\n').count() as u64,
        Newline::Merge => words.saturating_sub(between_words),
    }
}

/// The tokens of every message's text in a chat body ([`text_tokens`]):
/// string contents and `text` parts alike.
fn message_words(body: &str, newline: Newline) -> u64 {
    let request = serde_json::from_str::<serde_json::Value>(body).unwrap_or_default();
    let words = |text: &str| text_tokens(text, newline);
    request
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .map(|message| match message.get("content") {
            Some(serde_json::Value::String(text)) => words(text),
            Some(serde_json::Value::Array(parts)) => parts
                .iter()
                .filter_map(|part| part.get("text")?.as_str())
                .map(words)
                .sum(),
            Some(
                serde_json::Value::Null
                | serde_json::Value::Bool(_)
                | serde_json::Value::Number(_)
                | serde_json::Value::Object(_),
            )
            | None => 0,
        })
        .sum()
}

/// Append the request body to `FAKE_LLAMA_CALL_LOG`, one line per generation
/// request, if a test opted in. Best-effort like the PID/argv logs: a write
/// failure here must never fail the request the harness is trying to serve.
fn log_call(triggers: &Triggers, body: &str) {
    let Some(path) = triggers.call_log.as_deref() else {
        return;
    };
    if let Ok(_guard) = BODY_FILE_LOCK.lock()
        && let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path)
    {
        let _ = writeln!(f, "{}", body.replace('\n', " "));
    }
}

/// Count this generation request as in flight, record the high-water mark, and
/// park until the rendezvous. Returns with the request still counted; the
/// caller decrements when it is done.
fn enter_generation(triggers: &Triggers) {
    let inflight = INFLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
    PEAK_INFLIGHT.fetch_max(inflight, Ordering::SeqCst);
    if let Some(path) = triggers.peak_file.as_deref()
        && let Ok(_guard) = PEAK_FILE_LOCK.lock()
    {
        let _ = std::fs::write(path, PEAK_INFLIGHT.load(Ordering::SeqCst).to_string());
    }
    let Some(target) = triggers.rendezvous else {
        return;
    };
    if inflight >= target {
        RENDEZVOUS_MET.store(true, Ordering::SeqCst);
        return;
    }
    // Wait on the latch, not a re-read of `INFLIGHT`: the request that
    // completes the batch can be answered and decrement within one poll (the
    // drop trigger answers in microseconds), leaving its peers to wait out the
    // whole timeout for a rendezvous that already happened.
    let deadline = Instant::now() + triggers.rendezvous_timeout;
    while !RENDEZVOUS_MET.load(Ordering::SeqCst) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
}

/// Split out so every exit path is covered by the caller's in-flight
/// accounting. Crash is checked before the grace delay and drop after it, as
/// in [`handle_completion`].
fn handle_chat(stream: &mut TcpStream, body_s: &str, triggers: &Triggers) {
    thread::sleep(triggers.chat_delay);
    if let Some(trigger) = triggers.crash.as_deref()
        && body_s.contains(trigger)
    {
        process::exit(139);
    }
    if !triggers.rendezvous_grace.is_zero() {
        thread::sleep(triggers.rendezvous_grace);
    }
    if let Some(trigger) = triggers.drop.as_deref()
        && body_s.contains(trigger)
    {
        let _ = stream.shutdown(std::net::Shutdown::Both);
        return;
    }
    if let Some(trigger) = triggers.drop_once.as_deref()
        && body_s.contains(trigger)
        && let Ok(mut dropped) = DROPPED_ONCE.lock()
        && !dropped.iter().any(|seen| seen == body_s)
    {
        dropped.push(body_s.to_owned());
        let _ = stream.shutdown(std::net::Shutdown::Both);
        return;
    }

    if let Some(body) = triggers.chat_400.as_deref() {
        let _ = write!(
            stream,
            "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body,
        );
        return;
    }
    // The unstreamed reply reports predicted_n == max_tokens so the
    // decode-count check passes, unless `chat_reply` says otherwise.
    let max_tokens = parse_u64_field(body_s, "max_tokens").unwrap_or(0);
    let n_images = body_s.matches("data:image/png").count() as u64;
    let n_clips = body_s.matches("\"input_audio\":{").count() as u64;
    let text_n = if triggers.count_text {
        message_words(body_s, triggers.newline.in_chat())
    } else {
        0
    };
    let prompt_n = 5 + 10 * n_images + triggers.audio_tokens * n_clips + text_n;
    if let Some(path) = triggers.chat_images_file.as_deref()
        && let Ok(_guard) = BODY_FILE_LOCK.lock()
        && let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path)
    {
        let _ = writeln!(f, "{n_images}");
    }
    if let Some(path) = triggers.chat_body_file.as_deref()
        && let Ok(_guard) = BODY_FILE_LOCK.lock()
        && let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path)
    {
        let _ = writeln!(f, "{body_s}");
    }
    if let Some(n_ctx) = triggers.ctx_size.filter(|&n_ctx| prompt_n >= n_ctx) {
        let body = format!(
            r#"{{"error":{{"code":400,"message":"request ({prompt_n} tokens) exceeds the available context size ({n_ctx} tokens), try increasing it","type":"exceed_context_size_error"}}}}"#
        );
        let _ = write!(
            stream,
            "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body,
        );
        return;
    }
    if body_s.contains("\"stream\":true") {
        // llama-server's chat SSE: a delta chunk, a finish chunk carrying
        // `timings`, then the `stream_options.include_usage` chunk.
        let truncate_hit = triggers
            .sse_truncate
            .as_deref()
            .is_some_and(|t| body_s.contains(t));
        if truncate_hit {
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                 Connection: close\r\n\r\n\
                 data: {{\"choices\":[{{\"delta\":{{\"content\":\"par\"}},\"finish_reason\":null}}]}}\n\n",
            );
            let _ = stream.shutdown(std::net::Shutdown::Both);
            return;
        }
        let error_hit = triggers
            .sse_error
            .as_deref()
            .is_some_and(|t| body_s.contains(t));
        let tail = if error_hit {
            "data: {\"error\":{\"code\":400,\"message\":\"image exceeds context window\"}}\n\n"
                .to_string()
        } else {
            format!(
                "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}],\
                 \"timings\":{{\"prompt_n\":{prompt_n},\"prompt_ms\":1.0,\
                 \"predicted_n\":3,\"predicted_ms\":2.0,\"cache_n\":0}}}}\n\n\
                 data: {{\"choices\":[],\"usage\":{{\"completion_tokens\":1,\
                 \"prompt_tokens\":{prompt_n}}}}}\n\n\
                 data: [DONE]\n\n"
            )
        };
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
             Connection: close\r\n\r\n\
             data: {{\"choices\":[{{\"delta\":{{\"content\":\"answer\"}},\"finish_reason\":null}}]}}\n\n\
             {tail}",
        );
        return;
    }
    // A beam search on pipette's ASR build of llama.cpp
    // (`beam_search_and_respond`):
    // the room after the prompt is split among the beams, and the best beam
    // stops at `min(max_tokens, room)` with an ordinary length stop.
    if let (Some(beams), Some(ctx), Some(prompt)) = (
        parse_u64_field(body_s, "n_beams").filter(|n| *n > 0),
        triggers.beam_ctx,
        triggers.prompt_tokens,
    ) {
        let room = ctx.saturating_sub(prompt) / beams;
        let generated = max_tokens.min(room);
        let out = format!(
            r#"{{"choices":[{{"message":{{"content":"answer"}},"finish_reason":"length"}}],
                "usage":{{"completion_tokens":{generated},"prompt_tokens":{prompt}}}}}"#
        );
        write_json(stream, &out);
        return;
    }
    let (finish, predicted_n) = match triggers.chat_reply {
        ChatReply::Full => ("length", max_tokens),
        ChatReply::EarlyStop => ("stop", max_tokens),
        ChatReply::Short => ("length", max_tokens.saturating_sub(1)),
    };
    // The message is what a one-response reader (asr_eval's beam search)
    // reads; vl_throughput reads only the timings.
    let out = format!(
        r#"{{"choices":[{{"message":{{"content":"answer"}},"finish_reason":"{finish}"}}],
            "timings":{{"prompt_n":{prompt_n},"prompt_ms":1.0,
            "predicted_n":{predicted_n},"predicted_ms":2.0}}}}"#
    );
    write_json(stream, &out);
}

/// Split out so every exit path is covered by the caller's in-flight accounting.
fn handle_completion(stream: &mut TcpStream, body_s: &str, triggers: &Triggers) {
    // Recorded before any trigger can divert this request, so a test asserting
    // what the client asked for sees the crashed and dropped ones too.
    if let Some(path) = triggers.body_file.as_deref()
        && let Ok(_guard) = BODY_FILE_LOCK.lock()
        && let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path)
    {
        let _ = writeln!(f, "{body_s}");
    }
    // Checked before the grace delay below: a crash test wants the exiting
    // process to beat its parked peers' replies, deterministically.
    if let Some(trigger) = triggers.crash.as_deref()
        && body_s.contains(trigger)
    {
        // 139 = 128 + SIGSEGV. `process::exit`, not `abort()`, avoids the
        // macOS CrashReporter dialog.
        process::exit(139);
    }
    if !triggers.rendezvous_grace.is_zero() {
        thread::sleep(triggers.rendezvous_grace);
    }
    if let Some(trigger) = triggers.drop.as_deref()
        && body_s.contains(trigger)
    {
        // The client sees early EOF / broken pipe: close enough to
        // WSAECONNRESET, since the eval loop only cares that `/completion`
        // returned an Err.
        let _ = stream.shutdown(std::net::Shutdown::Both);
        return;
    }
    if let Some(trigger) = triggers.drop_once.as_deref()
        && body_s.contains(trigger)
        && let Ok(mut dropped) = DROPPED_ONCE.lock()
        && !dropped.iter().any(|seen| seen == body_s)
    {
        dropped.push(body_s.to_owned());
        let _ = stream.shutdown(std::net::Shutdown::Both);
        return;
    }
    let is_stream = body_s.contains("\"stream\":true");
    if is_stream && body_s.contains("\"ignore_eos\":true") {
        // The end-to-end cell: `prompt_n` agrees with `/tokenize`, and the ends
        // arrive in reverse so the client must match each by its `index`. Like
        // llama-server from b9425, every prompt after the first opens with `null`.
        let prompts = serde_json::from_str::<serde_json::Value>(body_s)
            .ok()
            .and_then(|v| v.get("prompt")?.as_array().cloned())
            .unwrap_or_default();
        let n_predict = parse_n_predict(body_s).unwrap_or(0);
        let mut out = String::from(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
             Transfer-Encoding: chunked\r\n\r\n",
        );
        out.extend((0..prompts.len()).map(|index| {
            let opened = if index > 0 { "data: null\n\n" } else { "" };
            http_chunk(&format!(
                "{opened}data: {{\"index\":{index},\"content\":\"answer\",\"stop\":false}}\n\n"
            ))
        }));
        out.extend(prompts.iter().enumerate().rev().map(|(index, prompt)| {
            let prompt_n = prompt.as_str().map_or(0, |p| p.split_whitespace().count());
            http_chunk(&format!(
                "data: {{\"index\":{index},\"content\":\"\",\"stop\":true,\
                 \"stop_type\":\"limit\",\"timings\":{{\"prompt_n\":{prompt_n},\
                 \"predicted_n\":{n_predict}}}}}\n\n"
            ))
        }));
        out.push_str("0\r\n\r\n");
        let _ = stream.write_all(out.as_bytes());
        // Like llama-server, drop the connection the headers keep alive, but
        // only once a next request arrives, so reusing it always fails.
        let _ = stream.read(&mut [0; 1]);
    } else if is_stream {
        let limit_hit = triggers
            .limit
            .as_deref()
            .is_some_and(|t| body_s.contains(t));
        let n_predict = parse_n_predict(body_s).unwrap_or(0);
        let (stop_type, predicted_n) = if limit_hit {
            ("limit", n_predict)
        } else {
            // A natural stop is strictly below the cap, even for a tiny cap.
            ("eos", n_predict.saturating_sub(1).min(3))
        };
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
             Connection: close\r\n\r\n\
             data: {{\"content\":\"answer\",\"stop\":false}}\n\n\
             data: {{\"content\":\"\",\"stop\":true,\"stop_type\":\"{stop_type}\",\
             \"stopped_limit\":{limit_hit},\"timings\":{{\"predicted_n\":{predicted_n}}}}}\n\n",
        );
    } else {
        write_json(stream, r#"{"content":"answer"}"#);
    }
}

fn write_json(stream: &mut TcpStream, body: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body,
    );
}

fn http_chunk(data: &str) -> String {
    format!("{:x}\r\n{data}\r\n", data.len())
}

fn parse_n_predict(body: &str) -> Option<u64> {
    parse_u64_field(body, "n_predict")
}

fn parse_u64_field(body: &str, field: &str) -> Option<u64> {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()?
        .get(field)?
        .as_u64()
}

fn extract_last_user_content(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let messages = v.get("messages")?.as_array()?;
    let last = messages.last()?;
    let content = last.get("content")?.as_str()?;
    Some(content.to_string())
}

fn render_template(body: &str, triggers: &Triggers) -> String {
    let render = RENDERS.fetch_add(1, Ordering::SeqCst);
    let v = serde_json::from_str::<serde_json::Value>(body).unwrap_or_default();
    let mut prompt = String::new();
    if let Some(after) = triggers.day_turns_after {
        prompt.push_str(if render < after {
            "[day 1] "
        } else {
            "[day 2] "
        });
    }
    if triggers.template_counter {
        prompt.push_str(&format!("[render {render}] "));
    }
    prompt.push_str(&extract_last_user_content(body).unwrap_or_default());
    let has_system = v
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|messages| {
            messages
                .iter()
                .any(|m| m.get("role").and_then(serde_json::Value::as_str) == Some("system"))
        });
    let switch = match triggers.thinking_switch.as_deref() {
        None => false,
        Some("system") => has_system,
        Some(_) => true,
    };
    let thinking_off = v
        .get("chat_template_kwargs")
        .and_then(|kwargs| kwargs.get("enable_thinking"))
        .and_then(serde_json::Value::as_bool)
        == Some(false);
    if switch && thinking_off {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
    prompt
}

fn json_escape(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}
