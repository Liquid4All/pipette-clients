//! Fake `llama-server` used by `tests/eval_crash_recovery.rs`.
//!
//! Speaks just enough of llama.cpp's HTTP surface for the eval loop:
//! `GET /health`, `POST /apply-template`, `POST /completion` (streaming
//! SSE for free-form, JSON for MCQ). Parses `--port <P>` from argv and
//! ignores every other flag.
//!
//! When the env var `FAKE_LLAMA_CRASH_PROMPT_CONTAINS` is set and a
//! `/completion` body contains that substring, the process exits with
//! status 139 — the eval loop's `poll_child_exit` then sees the
//! dead child and drops onto the recovery branch.
//!
//! When `FAKE_LLAMA_DROP_PROMPT_CONTAINS` is set and a `/completion`
//! body contains its substring, the server closes the TCP connection
//! mid-request without sending a response and *stays alive*. This
//! simulates the Windows `os error 10054` / WSAECONNRESET case where
//! the parent's `try_wait` never sees an exit — the eval loop must
//! still recycle the server and record the sample as failed.
//!
//! When `FAKE_LLAMA_LIMIT_PROMPT_CONTAINS` is set and a `/completion`
//! body contains its substring, the streaming terminal chunk reports
//! `stop_type: "limit"` / `stopped_limit: true` (the model hit the
//! output-token cap) instead of the default `stop_type: "eos"` — so the
//! eval loop classifies the sample `truncated` rather than `eos`.
//!
//! When `FAKE_LLAMA_PID_FILE` is set, every fresh process appends its
//! PID + `\n` to that file on startup. Tests use this to assert
//! server-restart behavior across the run (one line == one process).

use std::{
    env,
    fs::OpenOptions,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    process, thread,
};

#[derive(Clone, Default)]
struct Triggers {
    crash: Option<String>,
    drop: Option<String>,
    limit: Option<String>,
}

fn main() -> anyhow::Result<()> {
    use anyhow::Context;

    let args: Vec<String> = env::args().collect();
    let port = parse_port(&args).context("--port <P> is required")?;
    let triggers = Triggers {
        crash: env::var("FAKE_LLAMA_CRASH_PROMPT_CONTAINS").ok(),
        drop: env::var("FAKE_LLAMA_DROP_PROMPT_CONTAINS").ok(),
        limit: env::var("FAKE_LLAMA_LIMIT_PROMPT_CONTAINS").ok(),
    };

    // Tests opt in to PID logging via env var; one append per process
    // lets them count restarts.
    if let Ok(path) = env::var("FAKE_LLAMA_PID_FILE") {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{}", process::id());
        }
    }

    // Mirror the real binary's habit of writing model-load progress
    // to stderr so `shutdown_and_collect_stderr` has something to
    // capture for the crash log.
    eprintln!("fake-llama-server: listening on 127.0.0.1:{port}");

    let listener = TcpListener::bind(("127.0.0.1", port)).context("bind failed")?;
    for stream in listener.incoming().flatten() {
        let triggers = triggers.clone();
        thread::spawn(move || handle_connection(stream, triggers));
    }
    Ok(())
}

fn parse_port(args: &[String]) -> Option<u16> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--port" {
            return it.next()?.parse().ok();
        }
    }
    None
}

fn handle_connection(mut stream: TcpStream, triggers: Triggers) {
    let mirror = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
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
        "/apply-template" => {
            let prompt = extract_last_user_content(body_s).unwrap_or_default();
            let body = format!(r#"{{"prompt":{}}}"#, json_escape(&prompt));
            write_json(&mut stream, &body);
        }
        "/completion" => {
            if let Some(trigger) = triggers.crash.as_deref() {
                if body_s.contains(trigger) {
                    // Status 139 = 128 + SIGSEGV, the canonical "the
                    // runtime died on us" exit. `process::exit` avoids
                    // the macOS CrashReporter dialog that `abort()`
                    // would otherwise produce in test runs.
                    process::exit(139);
                }
            }
            if let Some(trigger) = triggers.drop.as_deref() {
                if body_s.contains(trigger) {
                    // Close the connection without sending a response.
                    // The client sees this as a transport error (early
                    // EOF / broken pipe) — close enough to the Windows
                    // WSAECONNRESET case for the eval loop, which only
                    // cares that `/completion` returned an Err. Process
                    // stays alive on purpose.
                    let _ = stream.shutdown(std::net::Shutdown::Both);
                    drop(stream);
                    return;
                }
            }
            let is_stream = body_s.contains("\"stream\":true");
            if is_stream {
                // Mirror llama-server's streaming tail: a content chunk, then a
                // terminal chunk carrying `stop_type` + `timings` so the eval
                // loop can classify eos vs truncated. Default is a natural stop
                // (`eos`, tokens < cap); the limit trigger reports hitting the
                // `n_predict` cap (`truncated`, tokens == cap).
                let limit_hit = triggers
                    .limit
                    .as_deref()
                    .is_some_and(|t| body_s.contains(t));
                let n_predict = parse_n_predict(body_s).unwrap_or(0);
                let (stop_type, predicted_n) = if limit_hit {
                    ("limit", n_predict)
                } else {
                    // A natural stop is strictly below the cap; cap at a few
                    // tokens but never `>= n_predict` (robust for a tiny cap).
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
                write_json(&mut stream, r#"{"content":"answer"}"#);
            }
        }
        _ => {
            let _ = write!(
                stream,
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
        }
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

fn parse_n_predict(body: &str) -> Option<u64> {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()?
        .get("n_predict")?
        .as_u64()
}

fn extract_last_user_content(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let messages = v.get("messages")?.as_array()?;
    let last = messages.last()?;
    let content = last.get("content")?.as_str()?;
    Some(content.to_string())
}

fn json_escape(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}
