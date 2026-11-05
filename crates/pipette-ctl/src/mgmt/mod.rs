//! The management server's account-facing HTTP surface, driven from
//! [`catalog::ROUTES`].
//!
//! Two front ends share one invoker: typed CLI verbs ([`cli`]) and an MCP
//! server on stdio ([`mcp`]). Both reach mgmt directly over signed HTTP as the
//! account in `.pipette-ctl/identity/`, and nothing else: no server-host
//! shell, no `pipette-mgmt` binary.

pub mod catalog;
pub mod cli;
pub mod mcp;

use std::{
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
    str::FromStr as _,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use anyhow::Context;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use pipette_http::HttpClient;
use pipette_mgmt_client::{
    HttpMethod, ImportProgress, RequestBody, ServerUrl, UserIdentity, user_api,
};
use pipette_plan_types::{
    ClientId, ClientPreauthKeyId, ListPage, Named, UserKeyId, UserPreauthKeyId, VaultPrincipalId,
};
use pipette_workspace::storage_root;

use self::catalog::{Body, Field, Kind, PAGE_QUERY, ROUTES, Response, Route};
use crate::identity::{self, RegisteredIdentity};

/// Arguments to one route, keyed by field name, typed as the catalog says.
pub type Args = Map<String, Value>;

/// Follow every page of a [`Response::Page`] route.
pub const ALL: &str = "all";
/// Where a [`Response::Stream`] route writes its bytes.
pub const OUT: &str = "out";
/// Lets a [`Response::Stream`] route replace an existing `out`.
pub const OVERWRITE: &str = "overwrite";
/// The top-level keys to keep of a JSON answer, or of each row of a listing.
pub const FIELDS: &str = "fields";

/// How long one JSON exchange may take, so a server that stops answering
/// ends the call instead of hanging it.
pub(crate) const JSON_TIMEOUT: Duration = Duration::from_secs(120);

/// mgmt's static routes the catalog does not carry. Each sits where a catalog
/// route takes a parameter, so a value spelled like one reaches it instead.
const UNCATALOGUED: &[&str] = &["/api/v1/users/register", "/benchmarks/batch"];

/// Where requests go and who signs them.
pub struct Session {
    server: ServerUrl,
    /// JSON calls and downloads. [`JSON_TIMEOUT`] bounds the wait for the
    /// headers, then each read of the body, so a stalled download fails and a
    /// long one does not.
    http: HttpClient,
    /// Uploads: a blocking client's timeout bounds the whole request body,
    /// which a large file outlasts.
    bulk: HttpClient,
    /// The reason there is no identity, reported only by a route that needs one.
    identity: Result<UserIdentity, String>,
    work_dir: PathBuf,
}

/// A route's answer, and what the invoker did on the way to it.
#[derive(Debug)]
pub struct Answer {
    pub value: Value,
    /// Said beside the answer rather than in it, e.g. which files an upload
    /// sent and which were already there.
    pub notes: Vec<String>,
}

impl Session {
    /// The account stored under `work_dir`, against `server` when named.
    ///
    /// A working directory with no account still opens: the public catalog
    /// reads need none, and every other route reports why it is missing. One
    /// whose stored server is unusable does not, rather than fall back to a
    /// server nobody named.
    pub fn open(work_dir: &Path, server: Option<ServerUrl>) -> anyhow::Result<Self> {
        // Absolute, since the paths it yields are read by a model that does
        // not know this process's working directory.
        let work_dir = std::path::absolute(work_dir)
            .with_context(|| format!("to resolve {}", work_dir.display()))?;
        let stored = RegisteredIdentity::read(&work_dir)?;
        let server = stored.server(server)?;
        let dir = identity::identity_dir(&work_dir);
        let identity = if stored.is_empty() {
            Err(format!(
                "{} holds no user identity. Run `pipette-ctl auth register` first",
                dir.display()
            ))
        } else if !stored.has_key() {
            Err(format!(
                "{} names an account but holds no key to sign with: no `id_ed25519`, and \
                 `user.json` names no ssh key",
                dir.display()
            ))
        } else {
            stored
                .resolve(Some(server.clone()))
                .and_then(|target| {
                    Ok(UserIdentity {
                        user_id: target.user_id,
                        user_key_id: target.user_key_id,
                        signer: identity::signer(&target.key)?,
                    })
                })
                .map_err(|error| {
                    let context = format!("{} holds an unusable user identity", dir.display());
                    format!("{:#}", error.context(context))
                })
        };
        Ok(Self {
            server,
            http: HttpClient::with_request_timeout("pipette-ctl", JSON_TIMEOUT)
                .context("to build the HTTP client")?,
            bulk: crate::http_client()?,
            identity,
            work_dir,
        })
    }

    pub fn server(&self) -> &ServerUrl {
        &self.server
    }

    pub fn work_dir(&self) -> &Path {
        &self.work_dir
    }

    /// Who this session signs as, or why it signs as nobody.
    pub fn identity(&self) -> Result<&UserIdentity, &str> {
        self.identity.as_ref().map_err(String::as_str)
    }

    /// The key registration this session signs with.
    pub fn user_key_id(&self) -> Option<&UserKeyId> {
        self.identity().ok().map(|identity| &identity.user_key_id)
    }

    /// Call `route` with `args`, every argument checked before anything is
    /// read, hashed or sent. This method discards the progress of an import.
    pub fn invoke(&self, route: &Route, args: &Args) -> anyhow::Result<Answer> {
        self.invoke_reporting(route, args, &mut |_| {})
    }

    /// Call `route` as [`Session::invoke`] does. Each `progress` line of a
    /// [`Response::Ingest`] route goes to `on_progress`.
    pub fn invoke_reporting(
        &self,
        route: &Route,
        args: &Args,
        on_progress: &mut dyn FnMut(&ImportProgress),
    ) -> anyhow::Result<Answer> {
        check_names(route, args)?;
        let identity = match (&self.identity, route.public) {
            (Ok(identity), _) => Some(identity),
            (Err(_), true) => None,
            (Err(reason), false) => anyhow::bail!("{reason}"),
        };
        let mut args = prepared(route, args)?;
        if route.derives("client_id") && !args.contains_key("client_id") {
            args.insert(
                "client_id".into(),
                Value::String(local_catalog_device_id(&self.work_dir, &self.server)?),
            );
        }
        let args = &args;
        let mut notes = Vec::new();
        if let Body::Blob(field) = route.body {
            let value = self.blob_put(identity, args, &path_arg(field, args)?, &mut notes)?;
            return Ok(Answer { value, notes });
        }
        let args = &self.stage(route, args, identity, &mut notes)?;
        let segments = segments(route, args)?;
        let mut query = query_pairs(route.query, args)?;
        if route.response == Response::Page {
            query.extend(query_pairs(PAGE_QUERY, args)?);
        }
        let body = match route.body {
            Body::None | Body::Blob(_) => None,
            Body::Fields(fields) => Some(Sent::Json(body(fields, args)?)),
            Body::Whole(field) => match checked(field, args)? {
                Some(Value::String(value)) => Some(json_or_toml(route, field.name, &value)?),
                value => value.map(|value| Sent::Json(without_path_keys(route, value))),
            },
            Body::Document(field) => Some(Sent::Toml(document(field, args)?)),
        };
        let body = body.as_ref().map(Sent::as_request);

        let value = match route.response {
            Response::Json | Response::Text => user_api::call(
                &self.server,
                &self.http,
                identity,
                route.method,
                &segments,
                &query,
                body,
            )?,
            Response::Page if flag(args, ALL) => {
                self.every_page(route, identity, &segments, query)?
            }
            Response::Page => user_api::call(
                &self.server,
                &self.http,
                identity,
                route.method,
                &segments,
                &query,
                body,
            )?,
            Response::Stream => self.download(identity, &segments, &query, args)?,
            Response::Ingest => user_api::call_ingest(
                &self.server,
                &self.http,
                identity,
                route.method,
                &segments,
                &query,
                body.context("a route that files a plan sends a document")?,
                on_progress,
            )
            .map_err(|error| {
                // Both errors come after the server accepted the call, so the
                // server can still file the plan.
                let unsure = matches!(
                    error,
                    pipette_mgmt_client::Error::StreamEnded { .. }
                        | pipette_mgmt_client::Error::DecodeJson { .. }
                );
                let error = anyhow::Error::new(error);
                if unsure {
                    error.context(
                        "the server can still file the plan. Read the campaign \
                         (`campaigns list`, `campaigns plans`) before you file the plan again",
                    )
                } else {
                    error
                }
            })?,
            Response::Exists => {
                let exists = user_api::exists(&self.server, &self.http, identity, &segments)
                    .map_err(|error| blob_refusal(error, text_arg(args, "principal_id")))?;
                json!({ "exists": exists })
            }
        };
        let value = match fields(args) {
            Some(fields) => {
                let (value, note) = project(route, value, &fields)?;
                notes.extend(note);
                value
            }
            None => value,
        };
        if let Some(own) = self
            .user_key_id()
            .filter(|own| revokes_own_key(route, args, own))
        {
            notes.push(format!(
                "{own} was this session's signing key, so every later call will fail with 401. \
                 To sign again, run `pipette-ctl auth login` here to approve this key again \
                 from the management app; or delete {}, then run `auth register` with a \
                 pre-auth key (for a person's account, one minted on the server host), or \
                 `auth adopt` here and `auth keys add` (the `users_add_key` tool) on another \
                 machine.",
                identity::identity_dir(&self.work_dir).display()
            ));
        }
        Ok(Answer { value, notes })
    }

    /// The arguments with each vault file given as a local path uploaded
    /// under the upload's owner, unless already there, and replaced by
    /// `{filename, sha256}`.
    fn stage(
        &self,
        route: &Route,
        args: &Args,
        identity: Option<&UserIdentity>,
        notes: &mut Vec<String>,
    ) -> anyhow::Result<Args> {
        let mut args = args.clone();
        let files = route
            .body_fields()
            .iter()
            .filter(|field| matches!(field.kind, Kind::VaultFile));
        for field in files {
            let Some(Value::String(path)) = args.get(field.name).cloned() else {
                continue;
            };
            let identity = identity.context("uploading a file needs an identity")?;
            let owner = match args.get("owner") {
                Some(Value::String(owner)) => owner.clone(),
                _ => identity.user_id.to_string(),
            };
            let path = Path::new(&path);
            let filename = file_name(field.name, path)?;
            let digest = sha256_file(path)?;
            let sent = self.put_blob(Some(identity), &owner, &digest, path)?;
            notes.push(if sent.is_some() {
                format!(
                    "`{}`: {filename} uploaded ({} bytes)",
                    field.name, digest.len
                )
            } else {
                format!(
                    "`{}`: {filename} was already uploaded; not sent again",
                    field.name
                )
            });
            args.insert(
                field.name.to_string(),
                json!({ "filename": filename, "sha256": digest.hex() }),
            );
        }
        Ok(args)
    }

    /// `PUT /vault/blobs/…` with the file's own digest.
    fn blob_put(
        &self,
        identity: Option<&UserIdentity>,
        args: &Args,
        file: &Path,
        notes: &mut Vec<String>,
    ) -> anyhow::Result<Value> {
        let principal = text_arg(args, "principal_id");
        let digest = sha256_file(file)?;
        if let Some(Value::String(given)) = args.get("sha256") {
            anyhow::ensure!(
                *given == digest.hex(),
                "{} hashes to {}, not {given}",
                file.display(),
                digest.hex()
            );
        }
        if let Some(answer) = self.put_blob(identity, principal, &digest, file)? {
            return Ok(answer);
        }
        notes.push(format!(
            "{} was already uploaded for {principal}; not sent again",
            file.display()
        ));
        Ok(json!({ "sha256": digest.hex(), "bytes": digest.len }))
    }

    /// Upload `path` as `principal`'s blob `digest`, unless the server holds
    /// it (`None`): mgmt answers that before reading the body, which a
    /// streamed upload loses.
    fn put_blob(
        &self,
        identity: Option<&UserIdentity>,
        principal: &str,
        digest: &FileDigest,
        path: &Path,
    ) -> anyhow::Result<Option<Value>> {
        let hex = digest.hex();
        let segments = ["api", "v1", "vault", "blobs", principal, &hex];
        let there = user_api::exists(&self.server, &self.http, identity, &segments)
            .map_err(|error| blob_refusal(error, principal))?;
        if there {
            return Ok(None);
        }
        let (file, len) = open_file(path)?;
        eprintln!("uploading {} ({len} bytes)", path.display());
        user_api::upload(
            &self.server,
            &self.bulk,
            identity,
            &segments,
            file,
            len,
            digest.sha256,
        )
        .map(Some)
        .map_err(|error| {
            if matches!(error, pipette_mgmt_client::Error::Transport { .. }) {
                anyhow::Error::new(error).context(
                    "the upload broke off; the server may have answered before reading the \
                         file, as it does for one over its size limit (413) or while too many \
                         uploads are in progress (429)",
                )
            } else {
                error.into()
            }
        })
    }

    /// Stream a resource into `out`, checked against the record's digest,
    /// which mgmt sends as the ETag.
    fn download(
        &self,
        identity: Option<&UserIdentity>,
        segments: &[&str],
        query: &[(String, String)],
        args: &Args,
    ) -> anyhow::Result<Value> {
        let out = std::path::absolute(text_arg(args, OUT))
            .with_context(|| format!("to resolve `{OUT}`"))?;
        let overwrite = flag(args, OVERWRITE);
        anyhow::ensure!(
            !out.is_dir(),
            "{} is a directory; name the file to write",
            out.display()
        );
        anyhow::ensure!(
            overwrite || !out.exists(),
            "{} already exists; pass `{OVERWRITE}` to replace it",
            out.display()
        );
        let staging = Staging::beside(&out)?;
        let stream = user_api::download(&self.server, &self.http, identity, segments, query)?;
        let expected = stream
            .etag()
            .filter(|etag| is_sha256(etag))
            .context("the server sent no sha256 ETag to check the bytes against")?;
        let (bytes, sha256) = write_stream(stream, &staging, &out, &expected, overwrite)?;
        Ok(json!({ "path": out, "bytes": bytes, "sha256": sha256 }))
    }

    /// Request pages until `next_cursor` is null, merged into one envelope.
    fn every_page(
        &self,
        route: &Route,
        identity: Option<&UserIdentity>,
        segments: &[&str],
        mut base: Vec<(String, String)>,
    ) -> anyhow::Result<Value> {
        // mgmt reads its whole listing for each newest-first page, so walk
        // oldest-first and turn the rows around.
        let newest = ("order".to_string(), "newest".to_string());
        let reversed = base.contains(&newest);
        base.retain(|pair| *pair != newest);
        let mut merged = Named {
            body: ListPage::whole(Vec::new(), 0),
            names: None,
        };
        let mut cursor: Option<String> = None;
        loop {
            let mut query = base.clone();
            query.extend(cursor.clone().map(|cursor| ("cursor".to_string(), cursor)));
            let page = user_api::call(
                &self.server,
                &self.http,
                identity,
                route.method,
                segments,
                &query,
                None,
            )?;
            let page: Named<ListPage<Value>> = serde_json::from_value(page)
                .with_context(|| format!("{} answered a body that is not a page", route.path))?;
            merged.body.items.extend(page.body.items);
            merged.body.unreadable += page.body.unreadable;
            // Keyed by id, so two pages naming one id agree.
            if let Some(named) = page.names {
                merged.names.get_or_insert_default().extend(named);
            }
            let next = page.body.next_cursor;
            // A server that hands back the cursor it was given would loop forever.
            if next.is_none() || next == cursor {
                break;
            }
            cursor = next;
        }
        if reversed {
            merged.body.items.reverse();
        }
        Ok(serde_json::to_value(merged)?)
    }
}

/// A download's own `<out>.<pid>-<n>.part`, so two downloads of one file
/// never write the same bytes, removed unless it lands. Not `tempfile`, whose
/// 0600 the downloaded file would keep.
struct Staging {
    path: PathBuf,
    file: fs::File,
}

/// Numbers this process's staging files.
static STAGED: AtomicU64 = AtomicU64::new(1);

impl Staging {
    fn beside(out: &Path) -> anyhow::Result<Self> {
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        loop {
            let mut path = out.as_os_str().to_owned();
            let n = STAGED.fetch_add(1, Ordering::Relaxed);
            path.push(format!(".{}-{n}.part", std::process::id()));
            let path = PathBuf::from(path);
            match options.open(&path) {
                Ok(file) => return Ok(Self { path, file }),
                // Left by an earlier process with this id.
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "to stage the download of {} in {}",
                            out.display(),
                            path.display()
                        )
                    });
                }
            }
        }
    }
}

impl Drop for Staging {
    /// A no-op once it landed: nothing else is ever staged under this name.
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Refuse an argument the route does not take, so a misspelled optional
/// field is an error rather than silently unsent.
fn check_names(route: &Route, args: &Args) -> anyhow::Result<()> {
    let unknown = args
        .keys()
        .filter(|name| {
            !route.path_params().any(|param| param == name.as_str())
                && !route.fields().any(|field| field.name == name.as_str())
        })
        .map(String::as_str)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        unknown.is_empty(),
        "`{}` takes no argument named {}",
        route.tool(),
        unknown.join(", ")
    );
    Ok(())
}

/// The registration file of the pipette client in the work directory is a
/// convenience for catalog reads. The management request still signs with this
/// session's user key.
fn local_catalog_device_id(work_dir: &Path, server: &ServerUrl) -> anyhow::Result<String> {
    let path = storage_root(work_dir, "pipette").join("identity/registration.json");
    let hint = "pass `client_id` explicitly";
    let bytes =
        fs::read(&path).with_context(|| format!("cannot read {}: {hint}", path.display()))?;
    let registration: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid {}: {hint}", path.display()))?;
    let id = registration
        .get("client_id")
        .and_then(serde_json::Value::as_str)
        .filter(|id| !id.is_empty())
        .with_context(|| format!("{} has no client_id: {hint}", path.display()))?;
    let id = ClientId::try_new(id)
        .with_context(|| format!("{} has an invalid client_id: {hint}", path.display()))?;
    let registered_server = registration
        .get("server_url")
        .and_then(serde_json::Value::as_str)
        .and_then(|raw| ServerUrl::try_new(raw).ok())
        .with_context(|| format!("{} has no valid server_url: {hint}", path.display()))?;
    anyhow::ensure!(
        &registered_server == server,
        "{} names a different server: {hint}",
        path.display()
    );
    Ok(id.into_inner())
}

/// `args` checked whole, before anything is read, hashed or sent: a mistake
/// in the last argument must not cost an upload. Digests come back lowercased.
fn prepared(route: &Route, args: &Args) -> anyhow::Result<Args> {
    let mut args = args.clone();
    path_params(route)
        .try_for_each(|(index, name)| prepare_param(route, &mut args, index, name))?;
    query_pairs(route.query, &args)?;
    for field in route.added_fields() {
        checked(field, &args)?;
    }
    let cursor = matches!(args.get("cursor"), Some(value) if !value.is_null());
    anyhow::ensure!(
        !(flag(&args, ALL) && cursor),
        "`{ALL}` walks every page from the first; drop `cursor`"
    );
    match route.body {
        Body::None => {}
        Body::Fields(fields) => {
            body(fields, &args)?;
        }
        Body::Whole(field) | Body::Document(field) | Body::Blob(field) => {
            checked(field, &args)?;
        }
    }
    for field in route.body_fields() {
        match (field.kind, args.get(field.name)) {
            (Kind::VaultFile, Some(Value::String(path))) => {
                let path = Path::new(path);
                anyhow::ensure!(
                    path.is_file(),
                    "`{}`: {} is not a file",
                    field.name,
                    path.display()
                );
                file_name(field.name, path)?;
            }
            (Kind::Path, Some(Value::String(path))) => {
                anyhow::ensure!(
                    Path::new(path).is_file(),
                    "`{}`: {path} is not a file",
                    field.name
                );
            }
            _ => {}
        }
    }
    let uploads = route
        .body_fields()
        .iter()
        .any(|field| matches!(field.kind, Kind::VaultFile));
    if uploads && let Some(Value::String(owner)) = args.get("owner") {
        VaultPrincipalId::from_str(owner)
            .map_err(|error| anyhow::anyhow!("`owner`: {error} (an id, not a handle)"))?;
    }
    Ok(args)
}

/// The path parameter `name`, at `index` among the path's segments, checked
/// in `args`: a derived one left out is dropped, and a digest lowercased.
fn prepare_param(route: &Route, args: &mut Args, index: usize, name: &str) -> anyhow::Result<()> {
    let value = match args.get(name) {
        None | Some(Value::Null) if route.derives(name) => {
            args.remove(name);
            return Ok(());
        }
        Some(Value::String(value)) if !value.is_empty() => value.clone(),
        _ => anyhow::bail!("`{name}` is required and must be a non-empty string"),
    };
    if let Some(sibling) = sibling_segments(route, index)
        .into_iter()
        .find(|sibling| *sibling == value)
    {
        anyhow::bail!(
            "`{name}` cannot be `{sibling}`: `{}` is a different route",
            path_prefix(route, index, sibling)
        );
    }
    match name {
        "principal_id" => {
            VaultPrincipalId::from_str(&value).map_err(|error| {
                anyhow::anyhow!("`principal_id`: {error} (an id, not a handle)")
            })?;
        }
        // A pasted token would put its secret in the URL and mgmt's audit log.
        "preauth_key_id" => {
            let parsed = match route.area {
                "users" => UserPreauthKeyId::from_str(&value).map(|_| ()),
                _ => ClientPreauthKeyId::from_str(&value).map(|_| ()),
            };
            parsed.map_err(|error| {
                anyhow::anyhow!("`preauth_key_id`: {error} (the listing's `key_id`, not the token)")
            })?;
        }
        "sha256" => {
            let digest = value.to_ascii_lowercase();
            anyhow::ensure!(
                is_sha256(&digest),
                "`sha256` must be 64 hex characters, got `{value}`"
            );
            args.insert(name.to_string(), Value::String(digest));
        }
        _ => {}
    }
    Ok(())
}

/// Each `{name}` placeholder with its position among the path's segments.
fn path_params(route: &Route) -> impl Iterator<Item = (usize, &'static str)> {
    path_segments(route.path)
        .enumerate()
        .filter_map(|(index, segment)| Some((index, param_name(segment)?)))
}

fn path_segments(path: &'static str) -> impl Iterator<Item = &'static str> {
    path.split('/').filter(|segment| !segment.is_empty())
}

fn param_name(segment: &str) -> Option<&str> {
    segment.strip_prefix('{')?.strip_suffix('}')
}

/// The static segments other routes put at `index` under the same prefix,
/// each a word the parameter there cannot be.
fn sibling_segments(route: &Route, index: usize) -> Vec<&'static str> {
    let own: Vec<_> = path_segments(route.path).collect();
    ROUTES
        .iter()
        .map(|other| other.path)
        .chain(UNCATALOGUED.iter().copied())
        .filter_map(|path| {
            let other: Vec<_> = path_segments(path).collect();
            let segment = *other.get(index)?;
            let same_prefix = own
                .iter()
                .zip(&other)
                .take(index)
                .all(|(a, b)| a == b || (param_name(a).is_some() && param_name(b).is_some()));
            (same_prefix && param_name(segment).is_none()).then_some(segment)
        })
        .collect()
}

/// `/…/{sibling}`: the route a sibling word at `index` would reach.
fn path_prefix(route: &Route, index: usize, sibling: &'static str) -> String {
    path_segments(route.path)
        .take(index)
        .chain([sibling])
        .fold(String::new(), |path, segment| format!("{path}/{segment}"))
}

/// The segments of `route`'s path, each placeholder filled from `args`.
fn segments<'a>(route: &Route, args: &'a Args) -> anyhow::Result<Vec<&'a str>> {
    path_segments(route.path)
        .map(|segment| match param_name(segment) {
            Some(name) => args
                .get(name)
                .and_then(Value::as_str)
                .with_context(|| format!("`{name}` is required")),
            None => Ok(segment),
        })
        .collect()
}

/// A string argument [`prepared`] already required.
fn text_arg<'a>(args: &'a Args, name: &str) -> &'a str {
    args.get(name).and_then(Value::as_str).unwrap_or_default()
}

/// A boolean [`prepared`] checked; absent is false.
fn flag(args: &Args, name: &str) -> bool {
    matches!(args.get(name), Some(Value::Bool(true)))
}

/// The account path needs no check of `user_id`: another account's path
/// answers 404 for this session's key.
fn revokes_own_key(route: &Route, args: &Args, own: &UserKeyId) -> bool {
    route.method == HttpMethod::Delete
        && matches!(
            route.path,
            "/api/v1/users/me/keys/{key_id}" | "/api/v1/users/{user_id}/keys/{key_id}"
        )
        // mgmt trims the id it parses, so a padded one still revokes this key.
        && UserKeyId::try_new(text_arg(args, "key_id")).is_ok_and(|given| given == *own)
}

/// The keys `fields` names, which [`prepared`] checked.
fn fields(args: &Args) -> Option<Vec<String>> {
    let names = args.get(FIELDS)?.as_array()?;
    Some(
        names
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
    )
}

/// Keep only `fields` of each row of a listing, else of the record. A key
/// a record lacks comes back `null`, since the server omits empty keys, and
/// a note names it. A record holding none of them is refused; a listing is
/// not, since a later page may hold them.
fn project(
    route: &Route,
    mut value: Value,
    fields: &[String],
) -> anyhow::Result<(Value, Option<String>)> {
    let key = route.rows_key();
    let (records, listing): (Vec<&mut Map<String, Value>>, bool) = match &mut value {
        Value::Array(rows) => (
            rows.iter_mut().filter_map(Value::as_object_mut).collect(),
            true,
        ),
        Value::Object(object)
            if key.is_some_and(|key| object.get(key).is_some_and(Value::is_array)) =>
        {
            let rows = key
                .and_then(|key| object.get_mut(key))
                .and_then(Value::as_array_mut)
                .into_iter()
                .flatten()
                .filter_map(Value::as_object_mut)
                .collect();
            (rows, true)
        }
        Value::Object(object) => (vec![object], false),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => anyhow::bail!(
            "`{FIELDS}` picks keys of a JSON object or of a listing's rows, and this answer is \
             neither"
        ),
    };
    let mut available: Vec<String> = Vec::new();
    for key in records.iter().flat_map(|record| record.keys()) {
        if !available.contains(key) {
            available.push(key.clone());
        }
    }
    let absent: Vec<&str> = fields
        .iter()
        .filter(|field| !available.contains(field))
        .map(String::as_str)
        .collect();
    anyhow::ensure!(
        listing || absent.len() < fields.len(),
        "none of {} is in the answer; its keys are: {}",
        fields.join(", "),
        available.join(", ")
    );
    let note = (!records.is_empty() && !absent.is_empty()).then(|| {
        format!(
            "no record here has {}; the keys here are: {}",
            absent.join(", "),
            available.join(", ")
        )
    });
    for record in records {
        record.retain(|key, _| fields.contains(key));
        for field in fields {
            record.entry(field.clone()).or_insert(Value::Null);
        }
    }
    Ok((value, note))
}

/// The present fields of `fields`, checked and rendered as query pairs.
fn query_pairs(fields: &[Field], args: &Args) -> anyhow::Result<Vec<(String, String)>> {
    fields.iter().try_fold(Vec::new(), |mut pairs, field| {
        if let Some(value) = checked(field, args)? {
            pairs.extend(field_pairs(field, value));
        }
        Ok(pairs)
    })
}

/// A list repeats its key, but a [`Kind::CommaList`] or [`Kind::Enums`] goes
/// as one value.
fn field_pairs(field: &Field, value: Value) -> Vec<(String, String)> {
    let name = field.name.to_string();
    match (field.kind, value) {
        (Kind::CommaList | Kind::Enums(_), Value::Array(values)) => {
            let joined: Vec<String> = values.iter().map(scalar_text).collect();
            vec![(name, joined.join(","))]
        }
        (_, Value::Array(values)) => values
            .iter()
            .map(|value| (name.clone(), scalar_text(value)))
            .collect(),
        (_, value) => vec![(name, scalar_text(&value))],
    }
}

fn scalar_text(value: &Value) -> String {
    value
        .as_str()
        .map_or_else(|| value.to_string(), str::to_string)
}

fn body(fields: &[Field], args: &Args) -> anyhow::Result<Value> {
    let mut object = Map::new();
    for field in fields {
        if let Some(value) = checked(field, args)? {
            object.insert(field.name.to_string(), value);
        }
    }
    Ok(Value::Object(object))
}

/// `field`'s value from `args`, checked against its kind, or `None` when an
/// optional field is absent. A `null` counts as absent: a query string cannot
/// carry one, and most body fields mgmt takes read it as a missing key. An
/// owner and a [`Kind::Clearable`] are the exceptions: their `null` is sent,
/// though an owner that cannot be detached refuses it.
fn checked(field: &Field, args: &Args) -> anyhow::Result<Option<Value>> {
    let name = field.name;
    let value = match (field.kind, args.get(name)) {
        (Kind::Owner { detach }, Some(value)) => return owner_value(name, value, detach).map(Some),
        (Kind::Clearable, Some(value)) if value.is_null() || value == "none" => {
            return Ok(Some(Value::Null));
        }
        (_, None | Some(Value::Null)) => {
            anyhow::ensure!(!field.required, "`{name}` is required");
            return Ok(None);
        }
        (_, Some(value)) => value,
    };
    let ok = match (field.kind, value) {
        (Kind::String | Kind::Clearable, Value::String(_))
        | (Kind::Boolean, Value::Bool(_))
        | (Kind::JsonOrToml, Value::Object(_))
        // Paths and text are read later.
        | (Kind::JsonOrToml | Kind::Path | Kind::Toml | Kind::VaultFile, Value::String(_)) => true,
        (Kind::Integer, Value::Number(number)) => number.is_u64(),
        (Kind::Enum(allowed), Value::String(text)) => allowed.contains(&text.as_str()),
        (Kind::Strings | Kind::CommaList, Value::Array(values)) => {
            values.iter().all(Value::is_string)
        }
        (Kind::Pairs, Value::Object(pairs)) => pairs.values().all(Value::is_string),
        (Kind::Keys, Value::Array(values)) => {
            !values.is_empty() && values.iter().all(Value::is_string)
        }
        (Kind::Enums(allowed), Value::Array(values)) => values
            .iter()
            .all(|value| value.as_str().is_some_and(|text| allowed.contains(&text))),
        (Kind::VaultFile, Value::Object(file)) => return vault_file(name, file).map(Some),
        _ => false,
    };
    anyhow::ensure!(
        ok,
        "`{name}` must be {}, got {value}",
        kind_name(field.kind)
    );
    Ok(Some(value.clone()))
}

pub fn kind_name(kind: Kind) -> String {
    match kind {
        Kind::String => "a string".into(),
        Kind::Clearable => "a string, or `none`".into(),
        Kind::Integer => "a non-negative integer".into(),
        Kind::Boolean => "a boolean".into(),
        Kind::Enum(allowed) => format!("one of {}", allowed.join(", ")),
        Kind::Strings | Kind::CommaList => "a list of strings".into(),
        Kind::Enums(allowed) => format!("a list of {}", allowed.join(", ")),
        Kind::Keys => "a non-empty list of keys".into(),
        Kind::Pairs => "an object of string values".into(),
        Kind::JsonOrToml => {
            "a JSON object, JSON or TOML text, or the path of a file holding either".into()
        }
        Kind::Owner { detach: true } => {
            "a `user_…` or `org_…` id, an owner object, or `none`".into()
        }
        Kind::Owner { detach: false } => "a `user_…` or `org_…` id, or an owner object".into(),
        Kind::Path => "a file path".into(),
        Kind::Toml => "TOML text or a path to a TOML file".into(),
        Kind::VaultFile => "a file path or `{filename, sha256}`".into(),
    }
}

/// The owner object an id stands for, by its prefix; `null` for a detach.
fn owner_value(name: &str, value: &Value, detach: bool) -> anyhow::Result<Value> {
    let id = match value {
        Value::Object(_) => return Ok(value.clone()),
        Value::Null => None,
        Value::String(text) if text == "none" => None,
        Value::String(text) => Some(text.as_str()),
        other @ (Value::Bool(_) | Value::Number(_) | Value::Array(_)) => anyhow::bail!(
            "`{name}` must be {}, got {other}",
            kind_name(Kind::Owner { detach })
        ),
    };
    match id {
        None if detach => Ok(Value::Null),
        None => anyhow::bail!("`{name}` cannot be cleared here: name a `user_…` or `org_…` id"),
        Some(id) if id.starts_with("user_") => Ok(json!({ "type": "user", "user_id": id })),
        Some(id) if id.starts_with("org_") => Ok(json!({ "type": "org", "org_id": id })),
        Some(id) => anyhow::bail!(
            "`{name}` `{id}` is neither a `user_…` nor an `org_…` id (an organization is named \
             by its id, not its handle)"
        ),
    }
}

/// A vault file named by an earlier upload: exactly `{filename, sha256}`.
fn vault_file(name: &str, file: &Map<String, Value>) -> anyhow::Result<Value> {
    let (Some(Value::String(filename)), Some(Value::String(sha256)), 2) =
        (file.get("filename"), file.get("sha256"), file.len())
    else {
        anyhow::bail!("`{name}` as an object must be exactly `{{filename, sha256}}`");
    };
    check_file_name(name, filename)?;
    let sha256 = sha256.to_ascii_lowercase();
    anyhow::ensure!(
        is_sha256(&sha256),
        "`{name}`: `sha256` must be 64 hex characters"
    );
    Ok(json!({ "filename": filename, "sha256": sha256 }))
}

/// `path`'s name, as a vault upload files it.
fn file_name<'a>(field: &str, path: &'a Path) -> anyhow::Result<&'a str> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("`{field}`: {} has no usable file name", path.display()))?;
    check_file_name(field, name)?;
    Ok(name)
}

/// mgmt's rule for a vault file's name: one path segment, checked here so a
/// bad one costs no upload.
fn check_file_name(field: &str, name: &str) -> anyhow::Result<()> {
    let mut chars = name.chars();
    let ok = name.len() <= 255
        && chars
            .next()
            .is_some_and(|first| first.is_ascii_alphanumeric())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    anyhow::ensure!(
        ok,
        "`{field}`: file name `{name}` must start with a letter or digit and hold only letters, \
         digits, `.`, `_` and `-`, at most 255; rename the file, or upload it with \
         `vault_blob_put` and pass `{{\"filename\": …, \"sha256\": …}}`"
    );
    Ok(())
}

fn is_sha256(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// A refused `HEAD /vault/blobs/…` arrives without the server's reason, since
/// a `HEAD` answer has no body; say what the status means there.
fn blob_refusal(error: pipette_mgmt_client::Error, principal: &str) -> anyhow::Error {
    let meaning = match error.http_status() {
        Some(400) => format!(
            "the server refused `{principal}` or the digest (a HEAD answer carries no reason); \
             both are well formed, so most likely no account or organization `{principal}` \
             exists"
        ),
        Some(401) => "the server refused this session's signature (a HEAD answer carries no \
                      reason): the key may be revoked or the clock skewed; `users_me` shows \
                      the server's reason"
            .to_string(),
        Some(403) => format!(
            "`{principal}` is outside this account's reach: it may upload only for itself, an \
             organization it administers, or a member of one"
        ),
        _ => return error.into(),
    };
    anyhow::Error::new(error).context(meaning)
}

/// `value` without keys named after `route`'s path parameters: an entry read
/// back carries them, the path already does, and the server refuses both.
fn without_path_keys(route: &Route, mut value: Value) -> Value {
    if let Value::Object(object) = &mut value {
        for name in route.path_params() {
            object.remove(name);
        }
    }
    value
}

/// A request body the invoker built, owned until it is sent.
enum Sent {
    Json(Value),
    Toml(String),
}

impl Sent {
    fn as_request(&self) -> RequestBody<'_> {
        match self {
            Self::Json(value) => RequestBody::Json(value),
            Self::Toml(text) => RequestBody::Toml(text),
        }
    }
}

/// A required path argument.
fn path_arg(field: &Field, args: &Args) -> anyhow::Result<PathBuf> {
    match checked(field, args)? {
        Some(Value::String(path)) => Ok(PathBuf::from(path)),
        _ => anyhow::bail!("`{}` is required: a file path", field.name),
    }
}

/// A TOML document argument: the file it names, else the text itself.
fn document(field: &Field, args: &Args) -> anyhow::Result<String> {
    let name = field.name;
    let value = match checked(field, args)? {
        Some(Value::String(value)) => value,
        None if !field.required => return Ok(String::new()),
        _ => anyhow::bail!("`{name}` is required: TOML text or a path to a TOML file"),
    };
    let unread = match fs::read_to_string(&value) {
        Ok(text) => return Ok(text),
        Err(error) => error,
    };
    match toml::from_str::<toml::Table>(&value) {
        Ok(table) if !table.is_empty() || !field.required => Ok(value),
        Ok(_) => anyhow::bail!("`{name}` names no readable file ({unread}) and holds no TOML"),
        Err(error) => anyhow::bail!(
            "`{name}` names no readable file ({unread}) and is not TOML text either: {}",
            error.message()
        ),
    }
}

/// A [`Kind::JsonOrToml`] string: the file it names, else the text. JSON must
/// be an object; anything else must be TOML, sent as it is.
fn json_or_toml(route: &Route, name: &str, value: &str) -> anyhow::Result<Sent> {
    let (text, source) = match fs::read_to_string(value) {
        Ok(text) => (text, format!("`{name}` file {value}")),
        Err(error) => (
            value.to_string(),
            format!("`{name}` names no readable file ({error}), and its text"),
        ),
    };
    let json_error = match serde_json::from_str::<Value>(&text) {
        Ok(value @ Value::Object(_)) => return Ok(Sent::Json(without_path_keys(route, value))),
        Ok(value) => anyhow::bail!("`{name}` must be a JSON object, got {value}"),
        Err(error) => error,
    };
    // No TOML document starts with `{`, so the JSON error is the useful one.
    anyhow::ensure!(
        !text.trim_start().starts_with('{'),
        "{source} is not JSON: {json_error}"
    );
    match toml::from_str::<toml::Table>(&text) {
        Ok(table) if !table.is_empty() => Ok(Sent::Toml(text)),
        Ok(_) => anyhow::bail!("{source} holds no TOML"),
        Err(error) => anyhow::bail!("{source} is neither JSON nor TOML: {}", error.message()),
    }
}

fn open_file(path: &Path) -> anyhow::Result<(fs::File, u64)> {
    let file = fs::File::open(path).with_context(|| format!("to open {}", path.display()))?;
    let len = file.metadata()?.len();
    Ok((file, len))
}

/// Hashes what passes through on its way to `inner`.
struct Hashing<W> {
    inner: W,
    hasher: Sha256,
}

impl<W: io::Write> io::Write for Hashing<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.hasher.update(buf.get(..written).unwrap_or_default());
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// A file's SHA-256, whose lowercase hex is the key a vault blob is stored
/// under, and its length.
struct FileDigest {
    sha256: [u8; 32],
    len: u64,
}

impl FileDigest {
    fn hex(&self) -> String {
        hex::encode(self.sha256)
    }
}

fn sha256_file(path: &Path) -> anyhow::Result<FileDigest> {
    let (mut file, _) = open_file(path)?;
    let mut hashing = Hashing {
        inner: io::sink(),
        hasher: Sha256::new(),
    };
    let len =
        io::copy(&mut file, &mut hashing).with_context(|| format!("to hash {}", path.display()))?;
    Ok(FileDigest {
        sha256: hashing.hasher.finalize().into(),
        len,
    })
}

/// Stream into the staging file while hashing, and move it onto `out` only
/// when the bytes hash to `expected`, so neither a broken nor an altered
/// download lands under the requested name.
fn write_stream(
    mut stream: impl io::Read,
    staging: &Staging,
    out: &Path,
    expected: &str,
    overwrite: bool,
) -> anyhow::Result<(u64, String)> {
    let mut hashing = Hashing {
        inner: &staging.file,
        hasher: Sha256::new(),
    };
    let bytes = io::copy(&mut stream, &mut hashing)
        .with_context(|| format!("to download {}", out.display()))?;
    hashing.flush()?;
    let digest = hex::encode(hashing.hasher.finalize());
    anyhow::ensure!(
        digest == expected,
        "the bytes hash to {digest}, not the record's {expected}: the stored file no longer \
         matches its record"
    );
    anyhow::ensure!(
        overwrite || !out.exists(),
        "{} appeared during the download; pass `{OVERWRITE}` to replace it",
        out.display()
    );
    fs::rename(&staging.path, out)
        .with_context(|| format!("to move the download to {}", out.display()))?;
    Ok((bytes, digest))
}

#[cfg(test)]
mod tests;
