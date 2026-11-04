//! `pipette-ctl mcp serve`: the management routes as MCP tools over stdio.
//!
//! Newline-delimited JSON-RPC 2.0 on stdin and stdout, per the MCP stdio
//! transport. Only the lifecycle, `ping`, cancellation and the two `tools/*`
//! methods are handled: every tool is one row of [`ROUTES`], so there is no
//! resource or prompt to offer. stdout carries protocol only; diagnostics go
//! to stderr.
//!
//! A `tools/call` runs on a worker, so `ping` and the rest are answered while
//! a slow route is in flight.

use std::{
    borrow::Cow,
    collections::HashMap,
    fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard, PoisonError, mpsc},
    thread,
    time::SystemTime,
};

use anyhow::Context;
use serde_json::{Map, Value, json};

use pipette_mgmt_client::HttpMethod;

use super::{
    ALL, Session,
    catalog::{Field, Kind, ROUTES, Response, Route, param_help},
};

/// Newest first; an unknown version from the client is answered with the first.
const PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

/// How many tool calls run at once; later ones wait their turn.
const WORKERS: usize = 8;

/// The longest result returned inline; a longer one is saved to a file and
/// summarized. Claude Code moves a result of about 50 KB, JSON-escaped, into
/// a file of its own that a model without file tools cannot read.
const RESULT_MAX_CHARS: usize = 40_000;
/// How many saved answers are kept, the newest.
const RESULTS_KEPT: usize = 100;
/// How much of a saved answer its summary shows.
const PREVIEW_MAX_CHARS: usize = 8_000;
/// A value in an object's preview longer than this is named by its shape.
const PREVIEW_VALUE_MAX_CHARS: usize = 500;

/// Answer requests from `input` on `output` until `input` closes and every
/// call in flight has answered.
pub fn serve(
    session: &Session,
    input: impl BufRead,
    output: impl Write + Send,
) -> anyhow::Result<()> {
    let output = Mutex::new(output);
    // Each call in flight by its id, and whether its caller cancelled it.
    let calls = Mutex::new(HashMap::<String, bool>::new());
    let failed = Mutex::new(None::<io::Error>);
    let (queue, jobs) = mpsc::channel::<(Value, Value)>();
    let jobs = Mutex::new(jobs);
    let read = thread::scope(|scope| {
        for _ in 0..WORKERS {
            scope.spawn(|| {
                loop {
                    // Its own statement, so the queue is unlocked during the call.
                    let job = lock(&jobs).recv();
                    let Ok((id, params)) = job else {
                        break;
                    };
                    let key = id.to_string();
                    // One cancelled while it waited is not run.
                    let cancelled_early = lock(&calls).get(&key).copied().unwrap_or(false);
                    let reply = (!cancelled_early).then(|| respond(&id, call(session, &params)));
                    // A cancelled request is not answered.
                    let cancelled = lock(&calls).remove(&key).unwrap_or(false);
                    if let Some(reply) = reply.filter(|_| !cancelled) {
                        write(&output, &reply, &failed);
                    }
                }
            });
        }
        let read = read_requests(session, input, &output, &calls, &queue, &failed);
        drop(queue);
        read
    });
    read?;
    match failed.into_inner().unwrap_or_else(PoisonError::into_inner) {
        Some(error) => Err(anyhow::Error::new(error).context("to write to stdout")),
        None => Ok(()),
    }
}

fn read_requests(
    session: &Session,
    input: impl BufRead,
    output: &Mutex<impl Write>,
    calls: &Mutex<HashMap<String, bool>>,
    queue: &mpsc::Sender<(Value, Value)>,
    failed: &Mutex<Option<io::Error>>,
) -> anyhow::Result<()> {
    // Bytes rather than lines: a line that is not UTF-8 is a parse error to
    // answer, not the end of the session.
    for line in input.split(b'\n') {
        let line = line.context("to read stdin")?;
        if line.trim_ascii().is_empty() {
            continue;
        }
        let reply = match serde_json::from_slice::<Value>(&line) {
            Err(error) => Some(failure(&Value::Null, PARSE_ERROR, &error.to_string())),
            Ok(message) => match envelope(&message) {
                Err(reply) => reply,
                Ok(Message::Notification { method, params }) => {
                    if method == "notifications/cancelled" {
                        cancel(calls, params);
                    }
                    None
                }
                Ok(Message::Request {
                    id,
                    method: "tools/call",
                    params,
                }) => {
                    lock(calls).insert(id.to_string(), false);
                    queue
                        .send((id.clone(), params.clone()))
                        .context("the tool workers stopped")?;
                    None
                }
                Ok(Message::Request { id, method, params }) => {
                    Some(respond(id, answer(session, method, params)))
                }
            },
        };
        if let Some(reply) = reply {
            write(output, &reply, failed);
        }
    }
    Ok(())
}

/// One well-formed message.
enum Message<'a> {
    Request {
        id: &'a Value,
        method: &'a str,
        params: &'a Value,
    },
    Notification {
        method: &'a str,
        params: &'a Value,
    },
}

/// `message` as a request or a notification, or the reply that refuses it
/// (`None` for a response, which this server never asked for).
fn envelope(message: &Value) -> Result<Message<'_>, Option<Value>> {
    let Value::Object(fields) = message else {
        return Err(Some(failure(
            &Value::Null,
            INVALID_REQUEST,
            "not a JSON-RPC message: expected one object (batches are not supported)",
        )));
    };
    if !fields.contains_key("method")
        && (fields.contains_key("result") || fields.contains_key("error"))
    {
        return Err(None);
    }
    let id = fields.get("id");
    let usable = id.filter(|id| id.is_string() || id.is_i64() || id.is_u64());
    let invalid = |why: &str| {
        Some(failure(
            usable.unwrap_or(&Value::Null),
            INVALID_REQUEST,
            why,
        ))
    };
    if id.is_some() && usable.is_none() {
        return Err(invalid("`id` must be a string or an integer"));
    }
    let Some(method) = fields.get("method").and_then(Value::as_str) else {
        return Err(invalid("`method` must be a string"));
    };
    if fields.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        // A notification is never answered, even a malformed one.
        return Err(usable.and_then(|_| invalid("`jsonrpc` must be \"2.0\"")));
    }
    let params = fields.get("params").unwrap_or(&Value::Null);
    Ok(match usable {
        Some(id) => Message::Request { id, method, params },
        None => Message::Notification { method, params },
    })
}

/// A request answered on the reading thread: none of these waits on mgmt.
fn answer(session: &Session, method: &str, params: &Value) -> Result<Value, (i64, String)> {
    match method {
        "initialize" => Ok(initialize(session, params)),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": ROUTES.iter().map(tool).collect::<Vec<_>>() })),
        _ => Err((METHOD_NOT_FOUND, format!("method not found: {method}"))),
    }
}

fn cancel(calls: &Mutex<HashMap<String, bool>>, params: &Value) {
    if let Some(id) = params.get("requestId")
        && let Some(cancelled) = lock(calls).get_mut(&id.to_string())
    {
        *cancelled = true;
    }
}

fn respond(id: &Value, result: Result<Value, (i64, String)>) -> Value {
    match result {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        Err((code, message)) => failure(id, code, &message),
    }
}

fn failure(id: &Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// One message per line, whole: the writers share stdout.
fn write(output: &Mutex<impl Write>, reply: &Value, failed: &Mutex<Option<io::Error>>) {
    let mut output = lock(output);
    let written = serde_json::to_writer(&mut *output, reply)
        .map_err(io::Error::from)
        .and_then(|()| output.write_all(b"\n"))
        .and_then(|()| output.flush());
    if let Err(error) = written {
        lock(failed).get_or_insert(error);
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn initialize(session: &Session, params: &Value) -> Value {
    let asked = params.get("protocolVersion").and_then(Value::as_str);
    let version = PROTOCOL_VERSIONS
        .iter()
        .find(|version| Some(**version) == asked)
        .or(PROTOCOL_VERSIONS.first())
        .copied()
        .unwrap_or_default();
    json!({
        "protocolVersion": version,
        "capabilities": { "tools": {} },
        "serverInfo": { "name": "pipette-ctl", "version": crate::PLAN_VERSION },
        "instructions": instructions(session),
    })
}

/// What a model reads first: which server, as whom, and the rules every tool
/// shares.
fn instructions(session: &Session) -> String {
    let tools = |keep: fn(&Route) -> bool| {
        ROUTES
            .iter()
            .filter(|route| keep(route))
            .map(Route::tool)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let who = match session.identity() {
        Ok(identity) => format!(
            "as the account `{}`, signing with key `{}` (`users_me` reads the account back)",
            identity.user_id, identity.user_key_id
        ),
        Err(reason) => format!(
            "with no account: {reason}. Only the public tools ({}) work",
            tools(|route| route.public)
        ),
    };
    format!(
        "Operates the pipette management server at {server} {who}. Every tool is one HTTP \
         route; the server decides what the account may do and answers 403 otherwise, and \
         most tool descriptions name the permission they need. {paged} page with `limit` \
         and `cursor`, or `all: true` to walk every page from the first; plans_jobs pages \
         with `limit` and `offset`, or answers every job at once with `all: true`; the other \
         list tools answer in one response. A read tool's `fields` keeps only the named keys \
         of each row of a listing, or of the answer; a missing key comes back null. An \
         argument given as null counts as left out, except users_set_org's `org_id: null`, \
         which clears membership, and clients_set_owner's `owner: null`, which detaches the \
         device. An answer longer than {RESULT_MAX_CHARS} characters is saved to a file \
         under {results}, which keeps the newest {RESULTS_KEPT}, and summarized with its \
         path. Work is filed only through campaigns: campaigns_create with the first plan, \
         then campaigns_file, campaigns_revise (on the document campaigns_document reads), \
         campaigns_variants, campaigns_rerun or campaigns_recalculate; `plan` and `edit` \
         take TOML text or a path to a TOML file.",
        server = session.server(),
        paged = tools(|route| route.response == Response::Page),
        results = results_dir(session).display(),
    )
}

/// `tools/call`. A route that fails answers a result flagged `isError`, so the
/// model reads the server's refusal; only a call naming no tool is a protocol
/// error.
fn call(session: &Session, params: &Value) -> Result<Value, (i64, String)> {
    let name = params.get("name").and_then(Value::as_str).ok_or_else(|| {
        (
            INVALID_PARAMS,
            "`name` is required: the tool to call".into(),
        )
    })?;
    let route = ROUTES
        .iter()
        .find(|route| route.tool() == name)
        .ok_or_else(|| (INVALID_PARAMS, format!("unknown tool: {name}")))?;
    let args = match params.get("arguments") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(args)) => args.clone(),
        Some(_) => return Err((INVALID_PARAMS, "`arguments` must be an object".into())),
    };
    let (texts, is_error) = match session.invoke(route, &args) {
        Ok(answer) => {
            let mut texts = deliver(session, route, answer.value);
            texts.extend(answer.notes);
            (texts, false)
        }
        Err(error) => (vec![format!("{error:#}")], true),
    };
    let content = texts
        .into_iter()
        .map(|text| json!({ "type": "text", "text": text }))
        .collect::<Vec<_>>();
    Ok(json!({ "content": content, "isError": is_error }))
}

/// The answer as the model reads it: compact JSON, or a summary of the file
/// it was saved to when it is too long to read whole. The route has run, so a
/// save that fails cuts the answer and says why rather than report an error.
fn deliver(session: &Session, route: &Route, value: Value) -> Vec<String> {
    let text = match &value {
        Value::Null => return vec!["done (the server answered with no body)".to_string()],
        Value::String(text) => text.clone(),
        Value::Bool(_) | Value::Number(_) | Value::Array(_) | Value::Object(_) => value.to_string(),
    };
    if text.chars().count() <= RESULT_MAX_CHARS {
        return vec![text];
    }
    match save(session, route, &value).and_then(|saved_to| summary(route, &value, &saved_to)) {
        Ok(summary) => vec![summary.to_string()],
        Err(error) => vec![
            text.chars().take(RESULT_MAX_CHARS).collect(),
            format!(
                "The answer is cut at {RESULT_MAX_CHARS} characters: saving it whole failed: \
                 {error:#}"
            ),
        ],
    }
}

fn results_dir(session: &Session) -> PathBuf {
    crate::workspace::ctl_dir(session.work_dir()).join("mcp-results")
}

/// Write `value` to a new file under [`results_dir`] (a text answer as
/// itself, `.txt`, so it can be passed back as a file), readable only by this
/// user: an answer can hold other accounts' addresses and keys.
fn save(session: &Session, route: &Route, value: &Value) -> anyhow::Result<PathBuf> {
    let dir = results_dir(session);
    fs::create_dir_all(&dir).with_context(|| format!("to create {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("to narrow {} to 0700", dir.display()))?;
    }
    let (text, extension) = match value {
        Value::String(text) => (Cow::Borrowed(text.as_str()), ".txt"),
        other @ (Value::Null
        | Value::Bool(_)
        | Value::Number(_)
        | Value::Array(_)
        | Value::Object(_)) => (Cow::Owned(serde_json::to_string_pretty(other)?), ".json"),
    };
    // A name no other file has, created 0600 on Unix.
    let (mut file, path) = tempfile::Builder::new()
        .prefix(&format!("{}-", route.tool()))
        .suffix(extension)
        .tempfile_in(&dir)
        .and_then(|file| file.keep().map_err(|error| error.error))
        .with_context(|| format!("to create a file in {}", dir.display()))?;
    file.write_all(text.as_bytes())
        .with_context(|| format!("to write {}", path.display()))?;
    prune(&dir);
    Ok(path)
}

/// Remove all but the newest [`RESULTS_KEPT`] saved answers. Best effort: a
/// file another session removed first is fine.
fn prune(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut saved: Vec<(SystemTime, PathBuf)> = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
        .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
        .collect();
    saved.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
    for (_, path) in saved.iter().skip(RESULTS_KEPT) {
        let _ = fs::remove_file(path);
    }
}

/// What a saved answer holds: its size, its keys, the head of it, and how to
/// ask for less.
fn summary(route: &Route, value: &Value, saved_to: &Path) -> anyhow::Result<Value> {
    let mut summary = Map::new();
    summary.insert("saved_to".into(), json!(saved_to));
    summary.insert("bytes".into(), json!(fs::metadata(saved_to)?.len()));
    if let Value::String(text) = value {
        summary.insert(
            "preview".into(),
            Value::String(text.chars().take(PREVIEW_MAX_CHARS).collect()),
        );
        summary.insert(
            "hint".into(),
            json!("The whole answer is in `saved_to`, as the text the server sent."),
        );
        return Ok(Value::Object(summary));
    }
    let rows = match value {
        Value::Array(rows) => Some(rows),
        Value::Object(object) => route
            .rows_key()
            .and_then(|key| object.get(key))
            .and_then(Value::as_array),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => None,
    };
    let (keys, preview) = match (rows, value) {
        (Some(rows), _) => {
            summary.insert("rows".into(), json!(rows.len()));
            let mut keys: Vec<&str> = Vec::new();
            for key in rows.iter().filter_map(Value::as_object).flat_map(Map::keys) {
                if !keys.contains(&key.as_str()) {
                    keys.push(key);
                }
            }
            (keys, preview_rows(route, value, rows))
        }
        (None, Value::Object(object)) => {
            let pairs = object
                .iter()
                .map(|(key, value)| (key.clone(), shape(value)));
            let keys = object.keys().map(String::as_str).collect();
            (keys, Value::Object(fit(pairs).collect()))
        }
        (None, other) => (
            Vec::new(),
            Value::String(other.to_string().chars().take(PREVIEW_MAX_CHARS).collect()),
        ),
    };
    summary.insert("keys".into(), json!(fit(keys).collect::<Vec<_>>()));
    summary.insert("preview".into(), preview);
    summary.insert("hint".into(), json!(hint(route)));
    Ok(Value::Object(summary))
}

/// Where the whole answer is, and the arguments of this tool that ask for
/// less.
fn hint(route: &Route) -> String {
    // A write's query holds its options, not filters; a read's paging, order and
    // additions ask for no less.
    let reads = route.method == HttpMethod::Get;
    let narrowing: Vec<String> = route
        .fields()
        .filter(|field| {
            reads
                && !matches!(
                    field.name,
                    "cursor" | "offset" | "order" | "names" | "plan" | "object" | "results" | ALL
                )
        })
        .map(|field| format!("`{}`", field.name))
        .collect();
    if narrowing.is_empty() {
        "The whole answer is in `saved_to`.".to_string()
    } else {
        format!(
            "The whole answer is in `saved_to`; ask for less with {}.",
            narrowing.join(", ")
        )
    }
}

/// The leading items whose JSON fits in [`PREVIEW_MAX_CHARS`].
fn fit<T: serde::Serialize>(items: impl IntoIterator<Item = T>) -> impl Iterator<Item = T> {
    let mut used = 0_usize;
    items.into_iter().take_while(move |item| {
        let len = serde_json::to_string(item).map_or(0, |text| text.len());
        used = used.saturating_add(len);
        used <= PREVIEW_MAX_CHARS
    })
}

/// The first rows that fit the preview, in the answer's own envelope.
fn preview_rows(route: &Route, value: &Value, rows: &[Value]) -> Value {
    let head: Vec<Value> = fit(rows).cloned().collect();
    match (value, route.rows_key()) {
        (Value::Object(object), Some(key)) => {
            let mut object = object.clone();
            object.insert(key.to_string(), Value::Array(head));
            Value::Object(object)
        }
        _ => Value::Array(head),
    }
}

/// A short value as itself, a long one by what it is.
fn shape(value: &Value) -> Value {
    if value.to_string().len() <= PREVIEW_VALUE_MAX_CHARS {
        return value.clone();
    }
    Value::String(match value {
        Value::Array(items) => format!("(array of {})", items.len()),
        Value::Object(object) => format!("(object of {} keys)", object.len()),
        Value::String(text) => format!("(text of {} characters)", text.chars().count()),
        Value::Null | Value::Bool(_) | Value::Number(_) => value.to_string(),
    })
}

pub fn tool(route: &Route) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();
    for name in route.path_params() {
        properties.insert(
            name.to_string(),
            json!({ "type": "string", "description": param_help(name) }),
        );
        if !route.derives(name) {
            required.push(name);
        }
    }
    for field in route.fields() {
        properties.insert(field.name.to_string(), schema(field));
        if field.required {
            required.push(field.name);
        }
    }
    json!({
        "name": route.tool(),
        "title": route.summary(),
        "description": route.about,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
        },
        "annotations": {
            // A download writes a local file.
            "readOnlyHint": matches!(route.method, HttpMethod::Get | HttpMethod::Head)
                && route.response != Response::Stream,
            "destructiveHint": route.destructive,
            "openWorldHint": false,
        },
    })
}

fn schema(field: &Field) -> Value {
    let mut schema = match field.kind {
        Kind::String | Kind::Path | Kind::Toml => json!({ "type": "string" }),
        Kind::Clearable => json!({ "anyOf": [{ "type": "string" }, { "type": "null" }] }),
        Kind::Integer => json!({ "type": "integer", "minimum": 0 }),
        Kind::Boolean => json!({ "type": "boolean" }),
        Kind::Enum(allowed) => json!({ "type": "string", "enum": allowed }),
        Kind::Strings | Kind::CommaList => {
            json!({ "type": "array", "items": { "type": "string" } })
        }
        Kind::Enums(allowed) => {
            json!({ "type": "array", "items": { "type": "string", "enum": allowed } })
        }
        Kind::Keys => json!({ "type": "array", "items": { "type": "string" }, "minItems": 1 }),
        Kind::Pairs => json!({ "type": "object", "additionalProperties": { "type": "string" } }),
        Kind::JsonOrToml => json!({
            "anyOf": [
                { "type": "object" },
                {
                    "type": "string",
                    "description": "JSON or TOML text, or the path of a file holding either",
                },
            ],
        }),
        Kind::Owner { detach } => {
            let mut any_of = vec![
                json!({ "type": "string" }),
                json!({
                    "type": "object",
                    "description": "`{\"type\": \"org\", \"org_id\": \"org_…\"}` or \
                                    `{\"type\": \"user\", \"user_id\": \"user_…\"}`",
                    "properties": {
                        "type": { "enum": ["org", "user"] },
                        "org_id": { "type": "string" },
                        "user_id": { "type": "string" },
                    },
                    "required": ["type"],
                    "additionalProperties": false,
                }),
            ];
            if detach {
                any_of.push(json!({ "type": "null" }));
            }
            json!({ "anyOf": any_of })
        }
        Kind::VaultFile => json!({
            "anyOf": [
                { "type": "string", "description": "A local file path" },
                {
                    "type": "object",
                    "description": "A file already uploaded for the owner",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]*$",
                            "maxLength": 255,
                        },
                        "sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
                    },
                    "required": ["filename", "sha256"],
                    "additionalProperties": false,
                },
            ],
        }),
    };
    if let Value::Object(schema) = &mut schema {
        schema.insert("description".into(), Value::String(field.help.to_string()));
    }
    schema
}
