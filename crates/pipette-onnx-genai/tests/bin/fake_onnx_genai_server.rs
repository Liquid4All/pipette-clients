//! A stand-in for `pipette_onnx_genai_server.py`, launched in the interpreter's
//! place: `fake_onnx_genai_server <script> --port <port> [args...]`.
//!
//! Serves the same surface over plain `std` sockets. One token per
//! character, so `/tokenize` and `usage` agree by construction; each generated
//! token is its own SSE chunk, a millisecond apart, so the client's stamps are
//! ordered. A request without `ignore_eos` stops after one token, which is
//! what a cell that forgot to send it would see.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use serde_json::{Value, json};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // On Windows it stands in for the base interpreter, which finds the venv
    // only through what a launcher sets.
    if cfg!(windows) && std::env::var_os("__PYVENV_LAUNCHER__").is_none() {
        return Err("started without __PYVENV_LAUNCHER__, so outside its venv".into());
    }
    let args: Vec<String> = std::env::args().collect();
    let port = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .ok_or("missing --port")?;
    let listener = TcpListener::bind(format!("127.0.0.1:{port}"))?;
    listener.incoming().try_for_each(|stream| {
        // A client that hung up mid-request is its own problem, not the server's.
        let _ = serve(stream?);
        Ok::<_, std::io::Error>(())
    })?;
    Ok(())
}

fn serve(mut stream: TcpStream) -> Result<(), Box<dyn std::error::Error>> {
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut length = 0;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header)?;
        let header = header.trim();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse()?;
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    let body: Value = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body)?
    };

    let mut parts = request_line.split_whitespace();
    match (parts.next(), parts.next()) {
        (Some("GET"), Some("/v1/models")) => respond(
            &mut stream,
            &json!({"object": "list", "data": [{"id": "default"}]}),
        ),
        (Some("POST"), Some("/tokenize")) => {
            let count = prompt_tokens(&body);
            let tokens: Vec<usize> = (0..count).collect();
            respond(&mut stream, &json!({"tokens": tokens, "count": count}))
        }
        (Some("POST"), Some("/v1/completions")) => complete(&mut stream, &body),
        _ => Err("no such route".into()),
    }
}

fn complete(stream: &mut TcpStream, body: &Value) -> Result<(), Box<dyn std::error::Error>> {
    let prompt_tokens = prompt_tokens(body);
    let max_tokens = body
        .get("max_tokens")
        .and_then(Value::as_u64)
        .ok_or("no max_tokens")?;
    let generated = if flag(body, "ignore_eos") {
        max_tokens
    } else {
        1
    };
    let finish = if generated == max_tokens {
        "length"
    } else {
        "stop"
    };
    let usage = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": generated,
        "total_tokens": prompt_tokens as u64 + generated,
    });
    if !flag(body, "stream") {
        let text = "x".repeat(usize::try_from(generated)?);
        let choice = json!({"index": 0, "text": text, "finish_reason": finish});
        return respond(stream, &json!({"choices": [choice], "usage": usage}));
    }
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
    )?;
    (1..=generated).try_for_each(|token| {
        std::thread::sleep(Duration::from_millis(1));
        let reason = (token == generated).then_some(finish);
        let choice = json!({"index": 0, "text": "x", "finish_reason": reason});
        event(stream, &json!({"choices": [choice]}))
    })?;
    event(stream, &json!({"choices": [], "usage": usage}))?;
    stream.write_all(b"data: [DONE]\n\n")?;
    Ok(())
}

/// One token per character of a string prompt.
fn prompt_tokens(body: &Value) -> usize {
    body.get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .chars()
        .count()
}

fn flag(body: &Value, key: &str) -> bool {
    body.get(key) == Some(&Value::Bool(true))
}

fn event(stream: &mut TcpStream, chunk: &Value) -> std::io::Result<()> {
    stream.write_all(format!("data: {chunk}\n\n").as_bytes())?;
    stream.flush()
}

fn respond(stream: &mut TcpStream, payload: &Value) -> Result<(), Box<dyn std::error::Error>> {
    let body = payload.to_string();
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    Ok(())
}
