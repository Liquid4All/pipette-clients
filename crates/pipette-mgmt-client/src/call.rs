//! The request pipeline every endpoint call runs through: signing, encoding
//! and decoding, conditional requests, and status interpretation. Stateless:
//! the [`ServerUrl`] and the shared [`pipette_http::HttpClient`] are passed
//! into every call.

use serde::de::DeserializeOwned;

use pipette_http::HttpClient;
use pipette_http::range::ContentRange;

use crate::{
    auth::{SignsRequest, path_and_query},
    error::{Error, Result},
    httpsig::{self, Outgoing},
    transport::{HttpMethod, HttpRequest, HttpResponse, TransportError},
    types::{EntityTag, IfNoneMatch, ServerUrl},
};

#[derive(Debug, Clone)]
pub enum ConditionalResponse<T> {
    Modified { value: T, etag: Option<EntityTag> },
    NotModified { etag: Option<EntityTag> },
}

impl<T> ConditionalResponse<T> {
    pub fn etag(&self) -> Option<&EntityTag> {
        match self {
            Self::Modified { etag, .. } | Self::NotModified { etag } => etag.as_ref(),
        }
    }
}

/// An encoded request body and the media type that names it.
pub(crate) struct Payload {
    bytes: Vec<u8>,
    content_type: &'static str,
}

impl Payload {
    fn json(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            content_type: "application/json",
        }
    }
}

/// An untyped request body: JSON, or a TOML document (a plan, a variant edit).
#[derive(Debug, Clone, Copy)]
pub enum RequestBody<'a> {
    Json(&'a serde_json::Value),
    Toml(&'a str),
}

impl RequestBody<'_> {
    fn encode(self) -> Result<Payload> {
        match self {
            Self::Json(value) => serde_json::to_vec(value)
                .map(Payload::json)
                .map_err(Error::EncodeJson),
            Self::Toml(text) => Ok(Payload {
                bytes: text.as_bytes().to_vec(),
                content_type: "application/toml",
            }),
        }
    }
}

/// A call with a typed JSON body. Generic only here: everything below the
/// encode takes bytes.
pub(crate) fn send_json<T: DeserializeOwned, B: serde::Serialize>(
    server: &ServerUrl,
    http: &HttpClient,
    method: HttpMethod,
    path: &str,
    auth: Option<&dyn SignsRequest>,
    body: B,
) -> Result<T> {
    let body = serde_json::to_vec(&body).map_err(Error::EncodeJson)?;
    let (response, url, method_name) = dispatch(
        server,
        http,
        method,
        path,
        auth,
        None,
        Some(Payload::json(body)),
    )?;
    decode_success(response, url, method_name)
}

/// A call whose body and answer stay untyped, for a caller that relays them
/// rather than reading them.
pub(crate) fn send_value(
    server: &ServerUrl,
    http: &HttpClient,
    method: HttpMethod,
    path: &str,
    auth: Option<&dyn SignsRequest>,
    body: Option<RequestBody<'_>>,
) -> Result<serde_json::Value> {
    let body = body.map(RequestBody::encode).transpose()?;
    let (response, url, method_name) = dispatch(server, http, method, path, auth, None, body)?;
    decode_value(response, url, method_name)
}

/// The media type of the progress stream that an import reports on.
const NDJSON: &str = "application/x-ndjson";

/// A `progress` line of the progress stream gives the bytes of one file that
/// the server downloaded so far for an import.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportProgress {
    /// The HuggingFace source of the file, as `org/repo_name@commit`; a model
    /// with files in two repositories reports under each.
    pub hf_ref: String,
    /// The path of the file in the repository.
    pub file: String,
    pub bytes_done: u64,
    pub bytes_total: u64,
}

/// Send a call that files a plan with `Accept: application/x-ndjson` and read
/// its progress stream (`pipette-mgmt`'s `docs/api/campaigns.md`, "Filing a
/// plan document"). An `error` line becomes the [`Error::HttpStatus`] the
/// plain answer would give; a stream with no last line is
/// [`Error::StreamEnded`]. A server that answers without the stream gets the
/// [`send_value`] decode.
pub(crate) fn send_ingest(
    server: &ServerUrl,
    http: &HttpClient,
    method: HttpMethod,
    path: &str,
    auth: Option<&dyn SignsRequest>,
    body: RequestBody<'_>,
    on_progress: &mut dyn FnMut(&ImportProgress),
) -> Result<serde_json::Value> {
    let url = format!("{server}{path}");
    let method_name = method.as_str().to_string();
    let mut request = build_request(method, &url, auth, None, Some(body.encode()?))?;
    // `Accept` is outside the signed payload (`Outgoing`), so a new value
    // keeps the signature valid.
    request
        .headers
        .retain(|(name, _)| !name.eq_ignore_ascii_case("accept"));
    request
        .headers
        .push(("Accept".to_string(), NDJSON.to_string()));
    let transport = |source: TransportError| Error::Transport {
        method: method_name.clone(),
        url: url.clone(),
        message: source.message,
    };
    let response = open(http, request, None).map_err(transport)?;
    let streams = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| media_essence(value).eq_ignore_ascii_case(NDJSON));
    if !response.status().is_success() || !streams {
        let response = read_whole(response).map_err(transport)?;
        return decode_value(response, url, method_name);
    }
    tracing::debug!(method = %method_name, %url, "reading the import progress stream");
    read_ingest(
        std::io::BufReader::new(response),
        method_name,
        url,
        on_progress,
    )
}

/// Read the lines of a progress stream up to its `result` or `error` line.
fn read_ingest(
    mut reader: impl std::io::BufRead,
    method: String,
    url: String,
    on_progress: &mut dyn FnMut(&ImportProgress),
) -> Result<serde_json::Value> {
    use pipette_plan_types::IngestEvent;

    let ended = |method: String, url: String, cause: String| {
        tracing::warn!(%method, %url, %cause, "import progress stream ended before its last line");
        Error::StreamEnded { method, url, cause }
    };
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) => {
                return Err(ended(
                    method,
                    url,
                    "the server closed the connection".to_string(),
                ));
            }
            Ok(_) => {}
            Err(error) => return Err(ended(method, url, error.to_string())),
        }
        if line.trim_ascii().is_empty() {
            continue;
        }
        let value = match serde_json::from_slice::<serde_json::Value>(&line) {
            Ok(value) => value,
            // The server writes each line whole, with its newline, so a line
            // cut short is where the connection broke.
            Err(_) if !line.ends_with(b"\n") => {
                return Err(ended(
                    method,
                    url,
                    "the connection broke in the middle of a line".to_string(),
                ));
            }
            Err(source) => {
                return Err(Error::DecodeJson {
                    body: body_excerpt(&line, "; body: "),
                    method,
                    url,
                    source,
                });
            }
        };
        let kind = value
            .get("event")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if !matches!(kind, "progress" | "keepalive" | "result" | "error") {
            tracing::debug!(event = kind, "skipping an import progress event");
            continue;
        }
        let progress = kind == "progress";
        let event = match serde_json::from_value::<IngestEvent<serde_json::Value>>(value) {
            Ok(event) => event,
            // A `progress` line only tells the user how far the import is, so
            // a line that this build cannot read does not end the call.
            Err(source) if progress => {
                tracing::debug!(%source, "skipping an import progress line that does not decode");
                continue;
            }
            Err(source) => {
                return Err(Error::DecodeJson {
                    method,
                    url,
                    body: body_excerpt(&line, "; body: "),
                    source,
                });
            }
        };
        match event {
            IngestEvent::Progress {
                hf_ref,
                file,
                bytes_done,
                bytes_total,
            } => on_progress(&ImportProgress {
                hf_ref,
                file,
                bytes_done,
                bytes_total,
            }),
            IngestEvent::Keepalive => {}
            IngestEvent::Result { body, .. } => return Ok(body),
            IngestEvent::Error { status, error } => {
                // The body the plain answer carries, so that the message
                // reads the same with or without the stream.
                let body = serde_json::to_vec(&serde_json::json!({ "error": error }))
                    .map_err(Error::EncodeJson)?;
                return Err(Error::HttpStatus {
                    method,
                    url,
                    status,
                    body: body_excerpt(&body, ": "),
                });
            }
        }
    }
}

/// A success body as a JSON value: `null` when empty (a `204`), a string when
/// the server answered something other than JSON (a plan document).
fn decode_value(response: HttpResponse, url: String, method: String) -> Result<serde_json::Value> {
    if !(200..300).contains(&response.status) {
        return Err(status_error(response, url, method));
    }
    if response.body.trim_ascii().is_empty() {
        return Ok(serde_json::Value::Null);
    }
    let is_json = response.header("content-type").is_none_or(|value| {
        let essence = media_essence(value);
        essence.eq_ignore_ascii_case("application/json")
            || essence.to_ascii_lowercase().ends_with("+json")
    });
    if !is_json {
        return Ok(serde_json::Value::String(
            String::from_utf8_lossy(&response.body).into_owned(),
        ));
    }
    decode_success(response, url, method)
}

/// The media type of a `Content-Type` value, with its parameters removed.
fn media_essence(value: &str) -> &str {
    value.split(';').next().unwrap_or_default().trim()
}

/// A `HEAD`: whether the resource exists. `404` is `false`; any other refusal
/// is the error it would be for a `GET`.
pub(crate) fn exists(
    server: &ServerUrl,
    http: &HttpClient,
    path: &str,
    auth: Option<&dyn SignsRequest>,
) -> Result<bool> {
    let (response, url, method_name) =
        dispatch(server, http, HttpMethod::Head, path, auth, None, None)?;
    match response.status {
        200..=299 => Ok(true),
        404 => Ok(false),
        _ => Err(status_error(response, url, method_name)),
    }
}

/// A file sent as a request body: streamed from disk rather than read into
/// memory, since a vault blob is gigabytes.
pub(crate) struct FileBody {
    pub file: std::fs::File,
    pub len: u64,
    /// The file's digest, which a user's signature covers in place of the
    /// bytes it has not read.
    pub sha256: [u8; 32],
}

pub(crate) fn send_file(
    server: &ServerUrl,
    http: &HttpClient,
    method: HttpMethod,
    path: &str,
    auth: Option<&dyn SignsRequest>,
    body: FileBody,
) -> Result<serde_json::Value> {
    let url = format!("{server}{path}");
    let method_name = method.as_str().to_string();
    let mut request = build_signed(method, &url, auth, Some(body.sha256))?;
    request.headers.push((
        "Content-Type".to_string(),
        "application/octet-stream".to_string(),
    ));
    let response = execute_with(
        http,
        request,
        Some(reqwest::blocking::Body::sized(body.file, body.len)),
    )
    .map_err(|source| Error::Transport {
        method: method_name.clone(),
        url: url.clone(),
        message: source.message,
    })?;
    decode_value(response, url, method_name)
}

/// The request target for `segments` under the server root, each segment
/// percent-encoded as one path segment, followed by the encoded `query`.
pub(crate) fn encode_target(segments: &[&str], query: &[(String, String)]) -> Result<String> {
    // `extend` drops these rather than encoding them, so the request would
    // reach the parent route.
    if let Some(segment) = segments
        .iter()
        .find(|segment| matches!(**segment, "." | ".."))
    {
        return Err(Error::DotSegment {
            segment: (*segment).to_string(),
        });
    }
    let base = "http://target.invalid/";
    let mut url = url::Url::parse(base).map_err(|source| Error::InvalidUrl {
        url: base.to_string(),
        source,
    })?;
    url.path_segments_mut()
        .map_err(|()| Error::InvalidUrl {
            url: base.to_string(),
            source: url::ParseError::RelativeUrlWithCannotBeABaseBase,
        })?
        .pop_if_empty()
        .extend(segments);
    if !query.is_empty() {
        url.query_pairs_mut().extend_pairs(query);
    }
    request_target(url.as_str())
}

/// A `GET` with a JSON answer.
pub(crate) fn get_json<T: DeserializeOwned>(
    server: &ServerUrl,
    http: &HttpClient,
    path: &str,
    auth: Option<&dyn SignsRequest>,
) -> Result<T> {
    let (response, url, method_name) =
        dispatch(server, http, HttpMethod::Get, path, auth, None, None)?;
    decode_success(response, url, method_name)
}

/// [`get_json`] under an `If-None-Match`, so an unchanged resource costs a
/// `304` instead of a body.
pub(crate) fn get_json_conditional<T: DeserializeOwned>(
    server: &ServerUrl,
    http: &HttpClient,
    path: &str,
    auth: Option<&dyn SignsRequest>,
    if_none_match: Option<&IfNoneMatch>,
) -> Result<ConditionalResponse<T>> {
    let (response, url, method_name) = dispatch(
        server,
        http,
        HttpMethod::Get,
        path,
        auth,
        if_none_match,
        None,
    )?;
    let etag = response
        .header("etag")
        .and_then(EntityTag::from_header_value);
    if response.status == 304 {
        return Ok(ConditionalResponse::NotModified { etag });
    }
    let value = decode_success(response, url, method_name)?;
    Ok(ConditionalResponse::Modified { value, etag })
}

/// A bodiless request, its response handed back undecoded, for a caller that
/// reads the status first (`POST /plans/claim` tells a `200` from a `204`).
pub(crate) fn send(
    server: &ServerUrl,
    http: &HttpClient,
    method: HttpMethod,
    path: &str,
    auth: Option<&dyn SignsRequest>,
) -> Result<(HttpResponse, String, String)> {
    dispatch(server, http, method, path, auth, None, None)
}

/// A response body left on the wire, for an artifact rather than a record: a
/// vault object is gigabytes, and [`HttpResponse`] would hold it all in
/// memory. Dropping it mid-way closes the connection.
pub struct ByteStream {
    response: reqwest::blocking::Response,
}

impl ByteStream {
    /// The status the server answered with: `200` for the whole body, or `206`
    /// for the range in a [`ResumeFrom`].
    pub fn status(&self) -> u16 {
        self.response.status().as_u16()
    }

    /// The `Content-Range` of a `206`, if it names a range of one byte or more.
    pub fn content_range(&self) -> Option<ContentRange> {
        ContentRange::from_headers(self.response.headers())
    }

    /// The `ETag` as the server sent it, quotes included, or `None` if it is
    /// weak. A weak `ETag` does not promise the same bytes, so it cannot prove
    /// that a `206` continues a file.
    pub fn strong_etag(&self) -> Option<String> {
        pipette_http::range::strong_etag(self.response.headers())
    }

    /// The `ETag` without its quotes. A vault download's is the record's
    /// sha256, which the bytes are checked against.
    pub fn etag(&self) -> Option<String> {
        let value = self.response.headers().get(reqwest::header::ETAG)?;
        let value = value.to_str().ok()?.trim();
        let value = value.strip_prefix("W/").unwrap_or(value);
        Some(value.trim_matches('"').to_string())
    }
}

impl std::io::Read for ByteStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.response.read(buf)
    }
}

/// The byte where a download continues, and the `ETag` that the object must
/// still have on the server. The request asks for the bytes from `offset` to
/// the end. If the object changed, the server sends the whole object.
#[derive(Debug, Clone)]
pub struct ResumeFrom {
    pub offset: u64,
    pub if_range: EntityTag,
}

impl ResumeFrom {
    /// The `Range` and `If-Range` headers for the request.
    pub(crate) fn headers(&self) -> Vec<(String, String)> {
        vec![
            ("Range".to_string(), format!("bytes={}-", self.offset)),
            ("If-Range".to_string(), self.if_range.as_str().to_string()),
        ]
    }
}

/// A `GET` whose success body is streamed back rather than read. `extra`
/// holds request headers outside the signed payload, such as `Range`. A
/// non-`2xx` is the usual [`Error::HttpStatus`], body excerpt included.
pub(crate) fn get_stream(
    server: &ServerUrl,
    http: &HttpClient,
    path: &str,
    auth: Option<&dyn SignsRequest>,
    extra: Vec<(String, String)>,
) -> Result<ByteStream> {
    let url = format!("{server}{path}");
    let method = HttpMethod::Get;
    let method_name = method.as_str().to_string();
    let mut request = build_signed(method, &url, auth, None)?;
    request.headers.extend(extra);
    let response = open(http, request, None).map_err(|source| Error::Transport {
        method: method_name.clone(),
        url: url.clone(),
        message: source.message,
    })?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        let body = response
            .bytes()
            .map(|bytes| bytes.to_vec())
            .unwrap_or_default();
        return Err(status_error(
            HttpResponse {
                status,
                headers: Vec::new(),
                body,
            },
            url,
            method_name,
        ));
    }
    Ok(ByteStream { response })
}

/// Assemble, sign and execute a request whose body is already encoded.
fn dispatch(
    server: &ServerUrl,
    http: &HttpClient,
    method: HttpMethod,
    path: &str,
    auth: Option<&dyn SignsRequest>,
    if_none_match: Option<&IfNoneMatch>,
    body: Option<Payload>,
) -> Result<(HttpResponse, String, String)> {
    let url = format!("{server}{path}");
    let method_name = method.as_str().to_string();
    let request = build_request(method, &url, auth, if_none_match, body)?;
    let response = execute(http, request).map_err(|source| Error::Transport {
        method: method_name.clone(),
        url: url.clone(),
        message: source.message,
    })?;
    Ok((response, url, method_name))
}

fn build_request(
    method: HttpMethod,
    url: &str,
    auth: Option<&dyn SignsRequest>,
    if_none_match: Option<&IfNoneMatch>,
    body: Option<Payload>,
) -> Result<HttpRequest> {
    let body_sha256 = auth
        .and(body.as_ref())
        .map(|body| httpsig::sha256(&body.bytes));
    let mut request = build_signed(method, url, auth, body_sha256)?;
    if let Some(etag) = if_none_match {
        request
            .headers
            .push(("If-None-Match".to_string(), etag.header_value().to_string()));
    }
    if let Some(body) = body {
        request
            .headers
            .push(("Content-Type".to_string(), body.content_type.to_string()));
        request.body = Some(body.bytes);
    }
    Ok(request)
}

/// The request with its `Accept` and auth headers and no body yet.
/// `body_sha256` is the digest of the body that will follow, which a user's
/// signature covers.
fn build_signed(
    method: HttpMethod,
    url: &str,
    auth: Option<&dyn SignsRequest>,
    body_sha256: Option<[u8; 32]>,
) -> Result<HttpRequest> {
    // User-Agent comes from the shared reqwest client on `HttpClient`.
    let mut headers = vec![("Accept".to_string(), "application/json".to_string())];
    if let Some(auth) = auth {
        let parsed = url::Url::parse(url).map_err(|source| Error::InvalidUrl {
            url: url.to_string(),
            source,
        })?;
        let outgoing = Outgoing {
            method: method.as_str(),
            url: &parsed,
            body_sha256,
        };
        headers.extend(
            auth.signed_headers(&outgoing)?
                .into_iter()
                .map(|(name, value)| (name.as_str().to_string(), value)),
        );
    }
    Ok(HttpRequest {
        method,
        url: url.to_string(),
        headers,
        body: None,
    })
}

/// Read a successful response's body as `T`, or turn a non-2xx into the error
/// that names it. `pub(crate)` for the routes that read the status first.
pub(crate) fn decode_success<T: DeserializeOwned>(
    response: HttpResponse,
    url: String,
    method: String,
) -> Result<T> {
    if !(200..300).contains(&response.status) {
        return Err(status_error(response, url, method));
    }
    serde_json::from_slice(&response.body).map_err(|source| Error::DecodeJson {
        method,
        url,
        body: body_excerpt(&response.body, "; body: "),
        source,
    })
}

/// How much of a response body an error message carries, in characters so the
/// cut lands on a character boundary.
const EXCERPT_MAX_CHARS: usize = 512;

/// The body for an error message, behind `separator`. Empty for an empty body,
/// so the message does not end in a bare separator.
fn body_excerpt(body: &[u8], separator: &str) -> String {
    let text = String::from_utf8_lossy(body);
    let text = text.trim();
    if text.is_empty() {
        return String::new();
    }
    let excerpt: String = text.chars().take(EXCERPT_MAX_CHARS).collect();
    let ellipsis = if excerpt.chars().count() < text.chars().count() {
        "..."
    } else {
        ""
    };
    format!("{separator}{excerpt}{ellipsis}")
}

/// mgmt's `{"error": "..."}`, read from the raw body: the error excerpt may be
/// trimmed or cut, so it is no place to match a reason.
pub(crate) fn server_reason(body: &[u8]) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct ErrorBody {
        error: String,
    }
    serde_json::from_slice::<ErrorBody>(body)
        .ok()
        .map(|body| body.error)
}

/// A verb whose whole meaning is its URL: nothing to send, nothing to read
/// back, only a status to check.
pub(crate) fn send_empty(
    server: &ServerUrl,
    http: &HttpClient,
    method: HttpMethod,
    path: &str,
    auth: Option<&dyn SignsRequest>,
) -> Result<()> {
    let (response, url, method_name) = send(server, http, method, path, auth)?;
    if !(200..300).contains(&response.status) {
        return Err(status_error(response, url, method_name));
    }
    Ok(())
}

pub(crate) fn status_error(response: HttpResponse, url: String, method: String) -> Error {
    Error::HttpStatus {
        method,
        url,
        status: response.status,
        body: body_excerpt(&response.body, ": "),
    }
}

/// The request target the server sees: path plus query. Parsed out of the
/// assembled URL rather than concatenated, so it carries a base-URL path prefix
/// and the parser's percent-encoding, as reqwest's wire request does.
fn request_target(url: &str) -> Result<String> {
    let parsed = url::Url::parse(url).map_err(|source| Error::InvalidUrl {
        url: url.to_string(),
        source,
    })?;
    Ok(path_and_query(&parsed))
}

/// Run one request on the shared client's pool that refuses redirects: the
/// signature covers the request target, so a followed redirect would earn a
/// `401` and forward the signature headers to the redirect host. A redirecting
/// deployment surfaces its `3xx` instead.
fn execute(
    http: &HttpClient,
    request: HttpRequest,
) -> std::result::Result<HttpResponse, TransportError> {
    execute_with(http, request, None)
}

/// [`execute`], with a streamed body in place of the request's own.
fn execute_with(
    http: &HttpClient,
    request: HttpRequest,
    stream: Option<reqwest::blocking::Body>,
) -> std::result::Result<HttpResponse, TransportError> {
    let method = request.method;
    let url = request.url.clone();
    let started = std::time::Instant::now();
    let response = open(http, request, stream)?;
    let status = response.status().as_u16();
    tracing::debug!(
        method = %method.as_str(),
        %url,
        status,
        elapsed = ?started.elapsed(),
        "response"
    );
    read_whole(response)
}

/// The status, the headers and the whole body of `response`.
fn read_whole(
    response: reqwest::blocking::Response,
) -> std::result::Result<HttpResponse, TransportError> {
    let status = response.status().as_u16();
    let headers = response
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect();
    let body = response
        .bytes()
        .map_err(|source| TransportError::new(with_causes(&source)))?
        .to_vec();
    Ok(HttpResponse {
        status,
        headers,
        body,
    })
}

/// Send one request and hand back the response with its body unread.
#[expect(
    clippy::disallowed_methods,
    reason = "every status is an answer the caller reads, a refusal included"
)]
fn open(
    http: &HttpClient,
    request: HttpRequest,
    stream: Option<reqwest::blocking::Body>,
) -> std::result::Result<reqwest::blocking::Response, TransportError> {
    let method = match request.method {
        HttpMethod::Get => reqwest::Method::GET,
        HttpMethod::Post => reqwest::Method::POST,
        HttpMethod::Put => reqwest::Method::PUT,
        HttpMethod::Patch => reqwest::Method::PATCH,
        HttpMethod::Delete => reqwest::Method::DELETE,
        HttpMethod::Head => reqwest::Method::HEAD,
    };
    let mut builder = request.headers.iter().fold(
        http.client_no_redirects().request(method, &request.url),
        |builder, (name, value)| builder.header(name, value),
    );
    let body_len = request.body.as_ref().map_or(0, Vec::len);
    if let Some(body) = request.body {
        builder = builder.body(body);
    }
    if let Some(stream) = stream {
        builder = builder.body(stream);
    }
    // The one line every management exchange leaves at debug. Headers stay
    // out: the signed set carries the request's authentication.
    tracing::debug!(
        method = %request.method.as_str(),
        url = %request.url,
        body_bytes = body_len,
        "sending request"
    );
    let started = std::time::Instant::now();
    builder.send().map_err(|source| {
        tracing::debug!(
            method = %request.method.as_str(),
            url = %request.url,
            elapsed = ?started.elapsed(),
            error = %source,
            "request failed"
        );
        TransportError::new(with_causes(&source))
    })
}

/// `error` and every cause under it: reqwest's own text is the same for a
/// refused connection, a failed lookup and a TLS failure.
fn with_causes(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        let text = cause.to_string();
        if !message.ends_with(&text) {
            message.push_str(": ");
            message.push_str(&text);
        }
        source = cause.source();
    }
    message
}

/// Test fixtures shared by [`crate::client_api`] and [`crate::user_api`].
#[cfg(test)]
pub(crate) mod test_support {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};
    use httpmock::MockServer;

    use pipette_http::HttpClient;
    use pipette_plan_types::{ClientId, UserId, UserKeyId};

    use crate::{
        auth::{ClientIdentity, SIGNATURE_LEN, UserIdentity, generate_keypair},
        error::{Error, Result},
        types::{PublicKey, ServerUrl},
    };

    pub(crate) fn auth() -> Result<ClientIdentity> {
        let (private_key_hex, _) = generate_keypair()?;
        Ok(ClientIdentity {
            client_id: client_id("ev1_test")?,
            private_key_hex,
        })
    }

    /// A fixed key id, so a test can assert the `keyid` that reaches the wire.
    pub(crate) fn user_auth() -> Result<UserIdentity> {
        Ok(user_auth_and_key()?.0)
    }

    /// [`user_auth`] and the public key its signatures verify against.
    pub(crate) fn user_auth_and_key() -> Result<(UserIdentity, PublicKey)> {
        let (private_key_hex, public_key) = generate_keypair()?;
        let identity = UserIdentity::with_private_key(
            UserId::try_new("user_0192f0c1").map_err(|e| Error::BuildClient(e.to_string()))?,
            UserKeyId::try_new("userkey_9f2c").map_err(|e| Error::BuildClient(e.to_string()))?,
            private_key_hex,
        );
        Ok((identity, public_key))
    }

    pub(crate) fn client_id(raw: &str) -> Result<ClientId> {
        ClientId::try_new(raw).map_err(|e| Error::BuildClient(e.to_string()))
    }

    pub(crate) fn http_ctx() -> Result<HttpClient> {
        HttpClient::new("pipette-test/0").map_err(|e| Error::BuildClient(e.to_string()))
    }

    pub(crate) fn server_url(server: &MockServer) -> Result<ServerUrl> {
        ServerUrl::try_new(server.base_url()).map_err(|e| Error::BuildClient(e.to_string()))
    }

    /// Verify `signature_hex` over `payload` as the server does. Failures
    /// borrow `BuildClient`.
    pub(crate) fn verify_signature(
        public_key: &PublicKey,
        payload: &str,
        signature_hex: &str,
    ) -> Result<()> {
        let public: [u8; 32] = hex::decode(public_key.as_ref())
            .map_err(|e| Error::BuildClient(e.to_string()))?
            .try_into()
            .map_err(|_| Error::BuildClient("public key must be 32 bytes".into()))?;
        let signature: [u8; SIGNATURE_LEN] = hex::decode(signature_hex)
            .map_err(|e| Error::BuildClient(e.to_string()))?
            .try_into()
            .map_err(|_| Error::BuildClient("signature must be 64 bytes".into()))?;
        VerifyingKey::from_bytes(&public)
            .map_err(|e| Error::BuildClient(e.to_string()))?
            .verify(payload.as_bytes(), &Signature::from_bytes(&signature))
            .map_err(|e| Error::BuildClient(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{client_id, verify_signature};
    use super::*;
    use crate::auth::{AuthHeader, ClientIdentity, generate_keypair};

    /// A base-URL path prefix is in the signed payload, or every request to
    /// such a deployment would 401.
    #[test]
    fn signed_headers_cover_the_method_and_the_full_request_target() -> Result<()> {
        let (private_key_hex, public_key_hex) = generate_keypair()?;
        let auth = ClientIdentity {
            client_id: client_id("ev1_a3f8")?,
            private_key_hex,
        };
        let server = ServerUrl::try_new("https://mgmt.example.com/api/")
            .map_err(|e| Error::BuildClient(e.to_string()))?;

        // As `dispatch` builds it, so the `/api` prefix comes from the server URL.
        let url = format!("{server}/clients/me");
        let request = build_request(HttpMethod::Patch, &url, Some(&auth), None, None)?;

        let header = |name: AuthHeader| {
            request
                .headers
                .iter()
                .find(|(key, _)| key == name.as_str())
                .map(|(_, value)| value.clone())
                .ok_or_else(|| Error::BuildClient(format!("missing {}", name.as_str())))
        };
        let timestamp = header(AuthHeader::Timestamp)?;
        // Read off the header: the server verifies against the nonce the
        // request carried.
        let nonce = header(AuthHeader::Nonce)?;
        assert_eq!(header(AuthHeader::ClientId)?, "ev1_a3f8");

        let payload = format!("v1\nPATCH\n/api/clients/me\n{timestamp}\nev1_a3f8\n{nonce}");
        verify_signature(&public_key_hex, &payload, &header(AuthHeader::Signature)?)
    }

    /// The server's replay cache rejects a repeated nonce.
    #[test]
    fn each_signed_request_carries_a_fresh_nonce() -> Result<()> {
        let (private_key_hex, _) = generate_keypair()?;
        let auth = ClientIdentity {
            client_id: client_id("ev1_a3f8")?,
            private_key_hex,
        };

        let nonce = |request: &HttpRequest| {
            request
                .headers
                .iter()
                .find(|(key, _)| key == AuthHeader::Nonce.as_str())
                .map(|(_, value)| value.clone())
                .ok_or_else(|| Error::BuildClient("missing X-Nonce".into()))
        };
        let build = || {
            build_request(
                HttpMethod::Get,
                "https://mgmt.example.com/clients/me",
                Some(&auth),
                None,
                None,
            )
        };

        assert_ne!(nonce(&build()?)?, nonce(&build()?)?);
        Ok(())
    }

    #[test]
    fn content_type_is_sent_with_a_body_and_only_with_one() -> Result<()> {
        let content_type = |body: Option<RequestBody<'_>>| -> Result<Option<String>> {
            Ok(build_request(
                HttpMethod::Post,
                "https://mgmt.example.com/plans/claim",
                None,
                None,
                body.map(RequestBody::encode).transpose()?,
            )?
            .headers
            .into_iter()
            .find(|(name, _)| name == "Content-Type")
            .map(|(_, value)| value))
        };
        let json = serde_json::json!({});
        assert_eq!(
            content_type(Some(RequestBody::Json(&json)))?.as_deref(),
            Some("application/json")
        );
        assert_eq!(
            content_type(Some(RequestBody::Toml("x = 1")))?.as_deref(),
            Some("application/toml")
        );
        assert_eq!(content_type(None)?, None);
        Ok(())
    }

    /// A `/` or `?` in an id cannot reach another route.
    #[test]
    fn encode_target_keeps_each_segment_whole() -> Result<()> {
        assert_eq!(
            encode_target(
                &["orgs", "a/b?c", "users"],
                &[("cursor".into(), "x y&z".into())]
            )?,
            "/orgs/a%2Fb%3Fc/users?cursor=x+y%26z"
        );
        assert_eq!(encode_target(&["plans"], &[])?, "/plans");
        Ok(())
    }

    #[test]
    fn encode_target_refuses_a_dot_segment() {
        for segment in [".", ".."] {
            assert!(
                matches!(
                    encode_target(&["clients", segment], &[]),
                    Err(Error::DotSegment { .. })
                ),
                "{segment}"
            );
        }
    }

    /// A refused connection says so, rather than only that sending failed.
    #[test]
    fn a_transport_error_carries_its_cause() -> Result<()> {
        let server = ServerUrl::try_new("http://127.0.0.1:1")
            .map_err(|e| Error::BuildClient(e.to_string()))?;
        let http = test_support::http_ctx()?;
        let error = send_value(&server, &http, HttpMethod::Get, "/users/me", None, None)
            .err()
            .ok_or_else(|| Error::BuildClient("port 1 answered".into()))?;
        let message = error.to_string();
        assert!(message.contains("connect"), "{message}");
        Ok(())
    }

    #[test]
    fn request_target_keeps_the_query_string() -> Result<()> {
        assert_eq!(
            request_target("https://mgmt.example.com/clients/me?page=2")?,
            "/clients/me?page=2"
        );
        Ok(())
    }

    #[test]
    fn a_decode_error_quotes_the_body_that_would_not_parse() -> Result<()> {
        #[derive(serde::Deserialize)]
        struct Wanted {
            #[expect(dead_code, reason = "the test only needs the decode to fail")]
            key_id: String,
        }

        let response = HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: br#"{"keys":[{"user_key_id":"userkey_9f2c"}]}"#.to_vec(),
        };
        let error = decode_success::<Wanted>(response, "u".into(), "GET".into())
            .err()
            .ok_or_else(|| Error::BuildClient("expected a decode error".into()))?;

        let message = error.to_string();
        assert!(message.contains("missing field `key_id`"), "{message}");
        assert!(
            message.contains(r#""user_key_id":"userkey_9f2c""#),
            "{message}"
        );
        Ok(())
    }

    /// Counted in characters, so a multi-byte one is never split.
    #[test]
    fn a_long_body_is_truncated_and_marked() {
        let excerpt = body_excerpt("é".repeat(EXCERPT_MAX_CHARS + 10).as_bytes(), "; body: ");
        assert!(excerpt.ends_with("..."));
        assert_eq!(excerpt.matches('é').count(), EXCERPT_MAX_CHARS);
    }

    #[test]
    fn an_empty_body_adds_nothing_to_the_message() {
        assert_eq!(body_excerpt(b"", "; body: "), "");
        assert_eq!(body_excerpt(b"   \n  ", ": "), "");
    }

    #[test]
    fn http_status_5xx_is_transient_4xx_is_not() {
        let transient = Error::HttpStatus {
            method: "POST".into(),
            url: "u".into(),
            status: 503,
            body: String::new(),
        };
        let definitive = Error::HttpStatus {
            method: "POST".into(),
            url: "u".into(),
            status: 404,
            body: String::new(),
        };
        assert!(transient.is_transient());
        assert!(!definitive.is_transient());
        assert!(
            Error::Transport {
                method: "GET".into(),
                url: "u".into(),
                message: "reset".into(),
            }
            .is_transient()
        );
    }
}
