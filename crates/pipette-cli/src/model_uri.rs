//! The compact `--model` model URI: [`parse_model_uri`] and its inverse
//! [`model_to_uri`], over a shared key vocabulary so the two directions can't
//! drift. `models pull` takes one self-contained argument — a JSON `Model`
//! object or the URI below — so a model is fetched without naming a runtime.
//!
//! # Grammar
//!
//! ```text
//! uri    ::= scheme "://" body            ; split on the FIRST "://"
//! scheme ::= "gguf-text" | "gguf-vision" | "mlx" | "torch" | "openvino"
//! body   ::= "" | pair ("&" pair)*        ; keys unordered, each at most once
//! pair   ::= key "=" value                ; first "=" splits; value may contain "="
//! ```
//! `&` separates pairs, so no value may contain `&`; a URL value additionally may
//! not contain `?` (a query string can't survive the split — use the JSON form).
//!
//! # Keys → variant (parsing is deterministic on key *presence*, not order)
//!
//! | scheme        | discriminant       | keys (required; *optional*)                          |
//! |---------------|--------------------|------------------------------------------------------|
//! | `gguf-text`   | `repo` xor `url`   | `repo`+`path` \| `url`; *`rev`, `sha256`*             |
//! | `gguf-vision` | `repo` present?    | `repo`+`model`+`mmproj` \| `model`+`mmproj` (URLs); *`rev`, `model_sha256`, `mmproj_sha256`* |
//! | `mlx`,`torch`,`openvino` | the scheme | `repo`; *`prefix`, `rev`*                       |
//!
//! # Not representable (both directions error)
//!
//! - `apple-foundation`, `local`, and on-disk sources → [`ModelUriError::NotImportable`].
//! - a URL carrying a query string → [`ModelUriError::QueryUrl`].
//! - a repo carrying an auth token → [`ModelUriError::SecretNotRepresentable`]
//!   (a URI can't hold a secret). [`ModelUri`] serializes such a model
//!   structurally instead, so the token is never silently dropped.

use std::fmt;
use std::str::FromStr;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use pipette_plan_types::{
    GgufText, GgufTextSource, GgufVision, GgufVisionSource, HfRepo, HfRevision, Mlx, Model,
    ModelSource, Openvino, RepoSubpath, ResourceUrl, Sha256, Torch,
};

// Key names, shared by the parser and [`model_to_uri`] so the two directions
// reference one spelling each — a rename can't desync them. Schemes get the
// same guarantee, plus exhaustiveness, from the [`Scheme`] enum below.
const KEY_REPO: &str = "repo";
const KEY_REV: &str = "rev";
const KEY_PATH: &str = "path";
const KEY_URL: &str = "url";
const KEY_SHA256: &str = "sha256";
const KEY_MODEL: &str = "model";
const KEY_MMPROJ: &str = "mmproj";
const KEY_MODEL_SHA256: &str = "model_sha256";
const KEY_MMPROJ_SHA256: &str = "mmproj_sha256";
const KEY_PREFIX: &str = "prefix";

/// The importable model kinds, one per URI scheme. Both parsing and
/// [`model_to_uri`] route through this; the exhaustive match means neither
/// direction can silently drop a kind. Scheme strings are the kebab-case of the
/// variant (`GgufText` → `gguf-text`), matching the runtime/type enums.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr)]
#[strum(serialize_all = "kebab-case")]
enum Scheme {
    GgufText,
    GgufVision,
    Mlx,
    Torch,
    Openvino,
}

impl Scheme {
    fn as_str(self) -> &'static str {
        self.into()
    }

    /// The scheme for `s`. `apple-foundation`/`local`/`file` name real but
    /// non-importable sources (OS-bundled or on-disk), so they get a distinct
    /// [`ModelUriError::NotImportable`] rather than "unknown" — a taxonomy the
    /// derived `FromStr` can't express, so the mapping stays explicit. A
    /// `file://` *value* is unaffected: the scheme is the prefix before the first
    /// `://`, so `gguf-text://url=file:///m.gguf` still schemes as `gguf-text`.
    fn parse(s: &str) -> Result<Self, ModelUriError> {
        Scheme::from_str(s).map_err(|_| match s {
            "apple-foundation" | "local" | "file" => ModelUriError::NotImportable(s.to_owned()),
            _ => ModelUriError::UnknownScheme(s.to_owned()),
        })
    }
}

/// Everything that can go wrong turning a compact model URI into a typed
/// [`Model`]. Converts into `anyhow` at the CLI boundary via `?`.
#[derive(Debug, thiserror::Error)]
pub enum ModelUriError {
    #[error("model URI must be `<scheme>://<key>=<value>[&<key>=<value>...]`")]
    MissingSchemeSeparator,

    #[error(
        "unknown model URI scheme `{0}` (expected `gguf-text`, `gguf-vision`, `mlx`, \
         `torch`, or `openvino`)"
    )]
    UnknownScheme(String),

    #[error("model source `{0}` is not importable")]
    NotImportable(String),

    #[error("malformed `key=value` pair `{0}` in model URI")]
    MalformedPair(String),

    #[error("empty key in model URI")]
    EmptyKey,

    #[error("duplicate key `{0}` in model URI")]
    DuplicateKey(String),

    #[error("unknown key `{key}` for scheme `{scheme}`")]
    UnknownKey { scheme: &'static str, key: String },

    #[error("missing required key `{key}` for scheme `{scheme}`")]
    MissingKey {
        scheme: &'static str,
        key: &'static str,
    },

    #[error("scheme `{scheme}` requires exactly one of {expected}")]
    MutuallyExclusive {
        scheme: &'static str,
        expected: &'static str,
    },

    #[error(
        "query-string URL in key `{key}` is unsupported in the URI form \
         (the `&` separator can't carry query params). Pass a JSON `--model` object instead"
    )]
    QueryUrl { key: &'static str },

    #[error(
        "a model with an auth token has no URI form (a URI can't carry a secret). \
         Pass a JSON `--model` object instead"
    )]
    SecretNotRepresentable,

    #[error("invalid value for key `{key}`: {message}")]
    InvalidValue { key: &'static str, message: String },
}

/// The parsed `key=value` pairs of a URI body. `&` is the pair separator; the
/// first `=` in each segment splits key from value, so a value keeps any later
/// `=` (e.g. a URL path). URL values are barred from carrying a query string
/// (see [`resource_url`]), so a raw `&` can never be part of a value — every
/// `&` is a genuine pair boundary and any leftover key is a real typo.
struct Pairs<'a> {
    scheme: Scheme,
    items: Vec<(&'a str, &'a str)>,
}

impl<'a> Pairs<'a> {
    fn parse(scheme: Scheme, body: &'a str) -> Result<Self, ModelUriError> {
        // An empty body carries no pairs; the per-scheme `require` calls then
        // produce a precise MissingKey rather than a MalformedPair(""). Otherwise
        // fold the pairs, rejecting a duplicate against the accumulator so far.
        let items = if body.is_empty() {
            Vec::new()
        } else {
            body.split('&')
                .try_fold(Vec::new(), |mut items: Vec<(&str, &str)>, seg| {
                    let (key, value) = seg
                        .split_once('=')
                        .ok_or_else(|| ModelUriError::MalformedPair(seg.to_owned()))?;
                    if key.is_empty() {
                        return Err(ModelUriError::EmptyKey);
                    }
                    if items.iter().any(|(existing, _)| *existing == key) {
                        return Err(ModelUriError::DuplicateKey(key.to_owned()));
                    }
                    items.push((key, value));
                    Ok(items)
                })?
        };
        Ok(Self { scheme, items })
    }

    fn take(&mut self, key: &str) -> Option<&'a str> {
        self.items
            .iter()
            .position(|(k, _)| *k == key)
            .map(|pos| self.items.remove(pos).1)
    }

    fn require(&mut self, key: &'static str) -> Result<&'a str, ModelUriError> {
        self.take(key).ok_or(ModelUriError::MissingKey {
            scheme: self.scheme.as_str(),
            key,
        })
    }

    /// Consume the tokenizer, rejecting any key the scheme didn't claim.
    fn finish(self) -> Result<(), ModelUriError> {
        match self.items.first() {
            None => Ok(()),
            Some((key, _)) => Err(ModelUriError::UnknownKey {
                scheme: self.scheme.as_str(),
                key: (*key).to_owned(),
            }),
        }
    }
}

fn hf_repo(org_repo: &str, rev: Option<&str>) -> Result<HfRepo, ModelUriError> {
    let mut repo = HfRepo::parse_org_repo(org_repo).map_err(|e| ModelUriError::InvalidValue {
        key: KEY_REPO,
        message: e.to_string(),
    })?;
    if let Some(rev) = rev {
        repo.revision =
            Some(
                HfRevision::try_new(rev).map_err(|e| ModelUriError::InvalidValue {
                    key: KEY_REV,
                    message: e.to_string(),
                })?,
            );
    }
    Ok(repo)
}

fn repo_subpath(key: &'static str, v: &str) -> Result<RepoSubpath, ModelUriError> {
    RepoSubpath::try_new(v).map_err(|e| ModelUriError::InvalidValue {
        key,
        message: e.to_string(),
    })
}

fn resource_url(key: &'static str, v: &str) -> Result<ResourceUrl, ModelUriError> {
    // A query string can't survive the `&` split: `url=…?a=x&b=y` fractures into
    // `url=…?a=x` plus a bogus `b=y` pair (which, if `b` is a consumed key like
    // `sha256`, would silently truncate the URL and lift a fake value). Reject it
    // up front; query/signed URLs go through the JSON `--model` form.
    if v.contains('?') {
        return Err(ModelUriError::QueryUrl { key });
    }
    ResourceUrl::try_new(v).map_err(|e| ModelUriError::InvalidValue {
        key,
        message: e.to_string(),
    })
}

fn opt_sha256(key: &'static str, v: Option<&str>) -> Result<Option<Sha256>, ModelUriError> {
    v.map(|s| {
        Sha256::try_new(s).map_err(|e| ModelUriError::InvalidValue {
            key,
            message: e.to_string(),
        })
    })
    .transpose()
}

fn parse_gguf_text(mut p: Pairs) -> Result<Model, ModelUriError> {
    let repo = p.take(KEY_REPO);
    let url = p.take(KEY_URL);
    let source = match (repo, url) {
        // HF: `repo`+`path`, the file addressed repo-relative.
        (Some(repo), None) => {
            let rev = p.take(KEY_REV);
            let path = p.require(KEY_PATH)?;
            let sha256 = p.take(KEY_SHA256);
            GgufTextSource::HuggingFace {
                repo: hf_repo(repo, rev)?,
                path: repo_subpath(KEY_PATH, path)?,
                sha256: opt_sha256(KEY_SHA256, sha256)?,
            }
        }
        // URL: a single direct download; `rev` has no meaning for a raw URL.
        (None, Some(url)) => {
            let sha256 = p.take(KEY_SHA256);
            GgufTextSource::Url {
                url: resource_url(KEY_URL, url)?,
                sha256: opt_sha256(KEY_SHA256, sha256)?,
            }
        }
        _ => {
            return Err(ModelUriError::MutuallyExclusive {
                scheme: p.scheme.as_str(),
                expected: "`repo`+`path` or `url`",
            })
        }
    };
    p.finish()?;
    Ok(Model::GgufText(GgufText { source }))
}

fn parse_gguf_vision(mut p: Pairs) -> Result<Model, ModelUriError> {
    let repo = p.take(KEY_REPO);
    let source = match repo {
        // HF iff `repo` present: model/mmproj are repo-relative paths.
        Some(repo) => {
            let rev = p.take(KEY_REV);
            let model = p.require(KEY_MODEL)?;
            let mmproj = p.require(KEY_MMPROJ)?;
            let model_sha256 = p.take(KEY_MODEL_SHA256);
            let mmproj_sha256 = p.take(KEY_MMPROJ_SHA256);
            GgufVisionSource::HuggingFace {
                repo: hf_repo(repo, rev)?,
                model: repo_subpath(KEY_MODEL, model)?,
                model_sha256: opt_sha256(KEY_MODEL_SHA256, model_sha256)?,
                mmproj: repo_subpath(KEY_MMPROJ, mmproj)?,
                mmproj_sha256: opt_sha256(KEY_MMPROJ_SHA256, mmproj_sha256)?,
            }
        }
        // No `repo`: model/mmproj are each a URL.
        None => {
            let model = p.require(KEY_MODEL)?;
            let mmproj = p.require(KEY_MMPROJ)?;
            let model_sha256 = p.take(KEY_MODEL_SHA256);
            let mmproj_sha256 = p.take(KEY_MMPROJ_SHA256);
            GgufVisionSource::Url {
                model: resource_url(KEY_MODEL, model)?,
                model_sha256: opt_sha256(KEY_MODEL_SHA256, model_sha256)?,
                mmproj: resource_url(KEY_MMPROJ, mmproj)?,
                mmproj_sha256: opt_sha256(KEY_MMPROJ_SHA256, mmproj_sha256)?,
            }
        }
    };
    p.finish()?;
    Ok(Model::GgufVision(GgufVision { source }))
}

/// mlx and torch share the directory-shaped HF source (importable = HF only).
fn parse_dir_source(mut p: Pairs) -> Result<ModelSource, ModelUriError> {
    let repo = p.require(KEY_REPO)?;
    let prefix = p.take(KEY_PREFIX);
    let rev = p.take(KEY_REV);
    let source = ModelSource::HuggingFace {
        repo: hf_repo(repo, rev)?,
        prefix: prefix.map(|s| repo_subpath(KEY_PREFIX, s)).transpose()?,
    };
    p.finish()?;
    Ok(source)
}

/// Parse a compact model URI into a typed [`Model`]. Only importable sources
/// have a scheme; on-disk/local and Apple-Foundation sources are rejected (see
/// [`Scheme::parse`]).
pub fn parse_model_uri(input: &str) -> Result<Model, ModelUriError> {
    let (scheme, body) = input
        .split_once("://")
        .ok_or(ModelUriError::MissingSchemeSeparator)?;
    let scheme = Scheme::parse(scheme)?;
    let pairs = Pairs::parse(scheme, body)?;
    match scheme {
        Scheme::GgufText => parse_gguf_text(pairs),
        Scheme::GgufVision => parse_gguf_vision(pairs),
        Scheme::Mlx => Ok(Model::Mlx(Mlx {
            source: parse_dir_source(pairs)?,
        })),
        Scheme::Torch => Ok(Model::Torch(Torch {
            source: parse_dir_source(pairs)?,
        })),
        Scheme::Openvino => Ok(Model::Openvino(Openvino {
            source: parse_dir_source(pairs)?,
        })),
    }
}

/// Accumulates a rendered URI's `key=value` pairs in push order, then joins them.
struct Body {
    scheme: Scheme,
    pairs: Vec<String>,
}

impl Body {
    fn new(scheme: Scheme) -> Self {
        Self {
            scheme,
            pairs: Vec::new(),
        }
    }

    fn push(&mut self, key: &str, value: &str) {
        self.pairs.push(format!("{key}={value}"));
    }

    /// `repo=org/name` — `HfRepo`'s Display is the bare `org/repo_name` (the
    /// revision is emitted separately by [`Body::rev`]). Refuses a repo carrying
    /// an auth token: a URI can't hold a secret, so rather than silently drop it
    /// we error, and `ModelUri::Serialize` falls back to the lossless structured
    /// form.
    fn repo(&mut self, repo: &HfRepo) -> Result<(), ModelUriError> {
        if repo.auth_token.is_some() {
            return Err(ModelUriError::SecretNotRepresentable);
        }
        self.push(KEY_REPO, &repo.to_string());
        Ok(())
    }

    fn rev(&mut self, repo: &HfRepo) {
        if let Some(rev) = &repo.revision {
            self.push(KEY_REV, rev.as_ref());
        }
    }

    fn sha(&mut self, key: &str, sha: &Option<Sha256>) {
        if let Some(sha) = sha {
            self.push(key, sha.as_ref());
        }
    }

    /// A URL value that would fracture on re-parse — `&` (the separator) or `?`
    /// (a query string) — can't be represented; reject it symmetrically with the
    /// parser's [`resource_url`] check.
    fn url(&mut self, key: &'static str, url: &ResourceUrl) -> Result<(), ModelUriError> {
        let value = url.as_ref();
        if value.contains('&') || value.contains('?') {
            return Err(ModelUriError::QueryUrl { key });
        }
        self.push(key, value);
        Ok(())
    }

    fn finish(self) -> String {
        format!("{}://{}", self.scheme.as_str(), self.pairs.join("&"))
    }
}

fn not_importable_local() -> ModelUriError {
    ModelUriError::NotImportable("local".to_owned())
}

/// Render a [`Model`] as its compact URI — the inverse of [`parse_model_uri`],
/// mirroring it arm-for-arm with a fixed key order (required, then `rev`, then
/// optional) for stable output. Non-representable sources error with the same
/// variants the parser uses: on-disk / Apple-Foundation → [`ModelUriError::NotImportable`];
/// a URL a query string or `&` would fracture → [`ModelUriError::QueryUrl`]. Auth
/// is never emitted, so round-trip is total only on the importable, auth-free,
/// query-free subset.
pub fn model_to_uri(model: &Model) -> Result<String, ModelUriError> {
    match model {
        Model::GgufText(m) => gguf_text_to_uri(&m.source),
        Model::GgufVision(m) => gguf_vision_to_uri(&m.source),
        Model::Mlx(m) => dir_to_uri(Scheme::Mlx, &m.source),
        Model::Torch(m) => dir_to_uri(Scheme::Torch, &m.source),
        Model::Openvino(m) => dir_to_uri(Scheme::Openvino, &m.source),
        Model::AppleFoundationText => {
            Err(ModelUriError::NotImportable("apple-foundation".to_owned()))
        }
    }
}

fn gguf_text_to_uri(source: &GgufTextSource) -> Result<String, ModelUriError> {
    let mut body = Body::new(Scheme::GgufText);
    match source {
        GgufTextSource::HuggingFace { repo, path, sha256 } => {
            body.repo(repo)?;
            body.push(KEY_PATH, path.as_ref());
            body.rev(repo);
            body.sha(KEY_SHA256, sha256);
        }
        GgufTextSource::Url { url, sha256 } => {
            body.url(KEY_URL, url)?;
            body.sha(KEY_SHA256, sha256);
        }
        GgufTextSource::AbsoluteFile { .. } | GgufTextSource::RelativeFile { .. } => {
            return Err(not_importable_local())
        }
    }
    Ok(body.finish())
}

fn gguf_vision_to_uri(source: &GgufVisionSource) -> Result<String, ModelUriError> {
    let mut body = Body::new(Scheme::GgufVision);
    match source {
        GgufVisionSource::HuggingFace {
            repo,
            model,
            model_sha256,
            mmproj,
            mmproj_sha256,
        } => {
            body.repo(repo)?;
            body.push(KEY_MODEL, model.as_ref());
            body.push(KEY_MMPROJ, mmproj.as_ref());
            body.rev(repo);
            body.sha(KEY_MODEL_SHA256, model_sha256);
            body.sha(KEY_MMPROJ_SHA256, mmproj_sha256);
        }
        GgufVisionSource::Url {
            model,
            model_sha256,
            mmproj,
            mmproj_sha256,
        } => {
            body.url(KEY_MODEL, model)?;
            body.url(KEY_MMPROJ, mmproj)?;
            body.sha(KEY_MODEL_SHA256, model_sha256);
            body.sha(KEY_MMPROJ_SHA256, mmproj_sha256);
        }
        GgufVisionSource::AbsoluteFiles { .. } | GgufVisionSource::RelativeFiles { .. } => {
            return Err(not_importable_local())
        }
    }
    Ok(body.finish())
}

fn dir_to_uri(scheme: Scheme, source: &ModelSource) -> Result<String, ModelUriError> {
    match source {
        ModelSource::HuggingFace { repo, prefix } => {
            let mut body = Body::new(scheme);
            body.repo(repo)?;
            if let Some(prefix) = prefix {
                body.push(KEY_PREFIX, prefix.as_ref());
            }
            body.rev(repo);
            Ok(body.finish())
        }
        ModelSource::AbsoluteDir { .. } | ModelSource::RelativeDir { .. } => {
            Err(not_importable_local())
        }
    }
}

/// The `--model` dispatch: a JSON object (reusing `Model`'s `Deserialize`) when
/// the trimmed arg starts with `{`, else the compact URI grammar.
pub fn parse_model_arg(arg: &str) -> anyhow::Result<Model> {
    let trimmed = arg.trim();
    if trimmed.starts_with('{') {
        Ok(serde_json::from_str::<Model>(trimmed)?)
    } else {
        Ok(parse_model_uri(trimmed)?)
    }
}

/// A [`Model`] wrapper that serde-round-trips through the compact URI. It reads
/// the URI string *or* the structured object, and writes the URI when the model
/// is representable (else the structured form) — so a serde field (config/plan
/// JSON or TOML) can use either: `"gguf-text://repo=o/r&path=Q4.gguf"` or the
/// full `Model` table.
#[derive(Debug, Clone)]
pub struct ModelUri(pub Model);

impl<'de> Deserialize<'de> for ModelUri {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ModelUriVisitor;

        impl<'de> Visitor<'de> for ModelUriVisitor {
            type Value = Model;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a model URI string or a Model object")
            }

            fn visit_str<E>(self, uri: &str) -> Result<Model, E>
            where
                E: de::Error,
            {
                parse_model_uri(uri).map_err(de::Error::custom)
            }

            // A map is the structured `Model` — defer to its derived Deserialize.
            fn visit_map<A>(self, map: A) -> Result<Model, A::Error>
            where
                A: MapAccess<'de>,
            {
                Model::deserialize(de::value::MapAccessDeserializer::new(map))
            }
        }

        deserializer.deserialize_any(ModelUriVisitor).map(ModelUri)
    }
}

impl Serialize for ModelUri {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Emit the compact URI when the model is representable, else the
        // structured form — the mirror of the URI-or-struct Deserialize.
        match model_to_uri(&self.0) {
            Ok(uri) => serializer.serialize_str(&uri),
            Err(_) => self.0.serialize(serializer),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::ModelType;

    use super::*;

    // A syntactically valid 64-hex digest for the `sha256` value slots.
    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn gguf_text_hf_minimal() -> anyhow::Result<()> {
        let model = parse_model_uri("gguf-text://repo=org/repo&path=Q4_K_M.gguf")?;
        let Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace { repo, path, sha256 },
        }) = &model
        else {
            anyhow::bail!("expected gguf-text HuggingFace, got {model:?}");
        };
        assert_eq!(path.as_ref(), "Q4_K_M.gguf");
        assert!(sha256.is_none());
        assert!(repo.revision.is_none());
        assert_eq!(model.to_string(), "org/repo:Q4_K_M.gguf");
        Ok(())
    }

    #[test]
    fn gguf_text_hf_full() -> anyhow::Result<()> {
        let uri = format!("gguf-text://repo=org/repo&path=Q4_K_M.gguf&rev=v2&sha256={SHA}");
        let model = parse_model_uri(&uri)?;
        let Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace { repo, sha256, .. },
        }) = &model
        else {
            anyhow::bail!("expected gguf-text HuggingFace, got {model:?}");
        };
        let Some(rev) = &repo.revision else {
            anyhow::bail!("expected a pinned revision");
        };
        assert_eq!(rev.as_ref(), "v2");
        assert!(sha256.is_some());
        Ok(())
    }

    #[test]
    fn gguf_text_url_minimal() -> anyhow::Result<()> {
        let model = parse_model_uri("gguf-text://url=https://cdn.example/m.gguf")?;
        let Model::GgufText(GgufText {
            source: GgufTextSource::Url { sha256, .. },
        }) = &model
        else {
            anyhow::bail!("expected gguf-text Url, got {model:?}");
        };
        assert!(sha256.is_none());
        Ok(())
    }

    #[test]
    fn gguf_text_url_with_sha256() -> anyhow::Result<()> {
        let uri = format!("gguf-text://url=https://cdn.example/m.gguf&sha256={SHA}");
        let model = parse_model_uri(&uri)?;
        let Model::GgufText(GgufText {
            source: GgufTextSource::Url { sha256, .. },
        }) = &model
        else {
            anyhow::bail!("expected gguf-text Url, got {model:?}");
        };
        assert!(sha256.is_some());
        Ok(())
    }

    #[test]
    fn gguf_vision_hf() -> anyhow::Result<()> {
        let model = parse_model_uri("gguf-vision://repo=org/repo&model=a.gguf&mmproj=mm.gguf")?;
        let Model::GgufVision(_) = &model else {
            anyhow::bail!("expected gguf-vision, got {model:?}");
        };
        assert_eq!(model.to_string(), "org/repo:a.gguf+org/repo:mm.gguf");

        // With a revision and both per-file digests set.
        let uri = format!(
            "gguf-vision://repo=org/repo&model=a.gguf&mmproj=mm.gguf&rev=v2\
             &model_sha256={SHA}&mmproj_sha256={SHA}"
        );
        let full = parse_model_uri(&uri)?;
        let Model::GgufVision(GgufVision {
            source:
                GgufVisionSource::HuggingFace {
                    repo,
                    model_sha256,
                    mmproj_sha256,
                    ..
                },
        }) = &full
        else {
            anyhow::bail!("expected gguf-vision HuggingFace, got {full:?}");
        };
        assert!(repo.revision.is_some());
        assert!(model_sha256.is_some());
        assert!(mmproj_sha256.is_some());
        Ok(())
    }

    #[test]
    fn gguf_vision_url() -> anyhow::Result<()> {
        let model =
            parse_model_uri("gguf-vision://model=https://x/a.gguf&mmproj=https://x/mm.gguf")?;
        let Model::GgufVision(GgufVision {
            source: GgufVisionSource::Url { .. },
        }) = &model
        else {
            anyhow::bail!("expected gguf-vision Url, got {model:?}");
        };
        assert_eq!(model.to_string(), "https://x/a.gguf+https://x/mm.gguf");

        let uri = format!(
            "gguf-vision://model=https://x/a.gguf&mmproj=https://x/mm.gguf\
             &model_sha256={SHA}&mmproj_sha256={SHA}"
        );
        let full = parse_model_uri(&uri)?;
        let Model::GgufVision(GgufVision {
            source:
                GgufVisionSource::Url {
                    model_sha256,
                    mmproj_sha256,
                    ..
                },
        }) = &full
        else {
            anyhow::bail!("expected gguf-vision Url, got {full:?}");
        };
        assert!(model_sha256.is_some());
        assert!(mmproj_sha256.is_some());
        Ok(())
    }

    /// The three directory-shaped schemes share one grammar (`repo`, optional
    /// `prefix`/`rev`), so they share one table. Both halves matter: the scheme
    /// has to pick the right variant, and `prefix`/`rev` have to fold into the
    /// identity string — the revision pins the repo (`@v2`), the prefix names
    /// the subdirectory (`:4bit`). For openvino the prefix is the precision
    /// variant, so one repo can carry a whole export matrix without its
    /// members colliding.
    #[rstest]
    #[case("mlx://repo=org/repo", ModelType::Mlx, "org/repo")]
    #[case(
        "mlx://repo=org/repo&prefix=4bit&rev=v2",
        ModelType::Mlx,
        "org/repo@v2:4bit"
    )]
    #[case("torch://repo=org/repo", ModelType::Torch, "org/repo")]
    #[case("torch://repo=org/repo&prefix=sub", ModelType::Torch, "org/repo:sub")]
    #[case("openvino://repo=org/repo", ModelType::Openvino, "org/repo")]
    #[case(
        "openvino://repo=LiquidAI/LFM2.5-350M-ov&prefix=int4-sym-cw",
        ModelType::Openvino,
        "LiquidAI/LFM2.5-350M-ov:int4-sym-cw"
    )]
    fn dir_scheme_parses_to_variant_and_identity(
        #[case] uri: &str,
        #[case] expected_type: ModelType,
        #[case] expected_identity: &str,
    ) -> anyhow::Result<()> {
        let model = parse_model_uri(uri)?;
        assert_eq!(ModelType::of(&model), expected_type);
        assert_eq!(model.to_string(), expected_identity);
        Ok(())
    }

    // A `{`-leading arg routes to JSON; the surrounding whitespace is trimmed.
    #[rstest]
    #[case(r#"{"type":"gguf_text","source":"huggingface","org":"o","repo_name":"r","path":"Q4_K_M.gguf"}"#)]
    #[case(r#"  {"type":"gguf_text","source":"huggingface","org":"o","repo_name":"r","path":"Q4_K_M.gguf"}  "#)]
    fn dispatch_json_object(#[case] arg: &str) -> anyhow::Result<()> {
        assert!(matches!(parse_model_arg(arg)?, Model::GgufText(_)));
        Ok(())
    }

    #[rstest]
    #[case("mlx://repo=o/r")]
    #[case("   mlx://repo=o/r   ")]
    fn dispatch_uri(#[case] arg: &str) -> anyhow::Result<()> {
        assert!(matches!(parse_model_arg(arg)?, Model::Mlx(_)));
        Ok(())
    }

    #[test]
    fn unknown_scheme() -> anyhow::Result<()> {
        assert!(matches!(
            parse_model_uri("foo://repo=o/r"),
            Err(ModelUriError::UnknownScheme(_))
        ));
        Ok(())
    }

    #[rstest]
    #[case("apple-foundation://")]
    #[case("local://dir=/x")]
    #[case("file:///models/m.gguf")]
    fn not_importable(#[case] uri: &str) -> anyhow::Result<()> {
        assert!(matches!(
            parse_model_uri(uri),
            Err(ModelUriError::NotImportable(_))
        ));
        Ok(())
    }

    #[rstest]
    #[case("mlx:repo=o/r")]
    #[case("just-text")]
    fn missing_scheme_separator(#[case] uri: &str) -> anyhow::Result<()> {
        assert!(matches!(
            parse_model_uri(uri),
            Err(ModelUriError::MissingSchemeSeparator)
        ));
        Ok(())
    }

    #[rstest]
    #[case("gguf-text://repo=o/r", "path")]
    #[case("gguf-vision://repo=o/r&model=a.gguf", "mmproj")]
    #[case("mlx://", "repo")]
    fn missing_required_key(#[case] uri: &str, #[case] key: &str) -> anyhow::Result<()> {
        assert!(matches!(
            parse_model_uri(uri),
            Err(ModelUriError::MissingKey { key: got, .. }) if got == key
        ));
        Ok(())
    }

    #[rstest]
    #[case(&format!("gguf-text://sha256={SHA}"))]
    #[case("gguf-text://repo=o/r&url=https://x")]
    fn gguf_text_repo_url_exclusive(#[case] uri: &str) -> anyhow::Result<()> {
        assert!(matches!(
            parse_model_uri(uri),
            Err(ModelUriError::MutuallyExclusive { .. })
        ));
        Ok(())
    }

    #[test]
    fn duplicate_key() -> anyhow::Result<()> {
        assert!(matches!(
            parse_model_uri("mlx://repo=a/b&repo=c/d"),
            Err(ModelUriError::DuplicateKey(_))
        ));
        Ok(())
    }

    #[test]
    fn unknown_key() -> anyhow::Result<()> {
        assert!(matches!(
            parse_model_uri("mlx://repo=o/r&foo=bar"),
            Err(ModelUriError::UnknownKey { key, .. }) if key == "foo"
        ));
        Ok(())
    }

    #[test]
    fn malformed_pair() -> anyhow::Result<()> {
        assert!(matches!(
            parse_model_uri("mlx://repo"),
            Err(ModelUriError::MalformedPair(_))
        ));
        Ok(())
    }

    #[test]
    fn empty_key() -> anyhow::Result<()> {
        assert!(matches!(
            parse_model_uri("mlx://=abc"),
            Err(ModelUriError::EmptyKey)
        ));
        Ok(())
    }

    // Rejected per `resource_url`. Case 2 is the silent-truncation trap: the
    // `&sha256=` would otherwise fracture off as a real digest for a cut URL.
    #[rstest]
    #[case("gguf-text://url=https://cdn.example/m.gguf?sig=a&exp=b", "url")]
    #[case(&format!("gguf-text://url=https://host/m.gguf?token=x&sha256={SHA}"), "url")]
    #[case(
        "gguf-vision://model=https://x/a?a=1&b=2&mmproj=https://x/mm.gguf",
        "model"
    )]
    fn query_url_rejected(#[case] uri: &str, #[case] key: &str) -> anyhow::Result<()> {
        let err = parse_model_uri(uri)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected `{uri}` to be rejected"))?;
        let ModelUriError::QueryUrl { key: got } = err else {
            anyhow::bail!("expected QueryUrl, got {err:?}");
        };
        assert_eq!(got, key);
        Ok(())
    }

    // A leftover key beside a query-free URL is a plain unknown key — not
    // misattributed to the URL.
    #[test]
    fn url_leftover_key_is_unknown_key() -> anyhow::Result<()> {
        assert!(matches!(
            parse_model_uri("gguf-text://url=https://x/m.gguf&foo=bar"),
            Err(ModelUriError::UnknownKey { key, .. }) if key == "foo"
        ));
        Ok(())
    }

    // One bad value per newtype path — assert the variant + `key` only; the
    // message is nutype-generated and brittle.
    #[rstest]
    #[case("mlx://repo=noslash", "repo")]
    #[case("gguf-text://url=https://x&sha256=nothex", "sha256")]
    #[case("mlx://repo=o/r&prefix=../escape", "prefix")]
    #[case("mlx://repo=o/r&rev=-bad", "rev")]
    fn invalid_value(#[case] uri: &str, #[case] key: &str) -> anyhow::Result<()> {
        let err = parse_model_uri(uri)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected `{uri}` to be rejected"))?;
        let ModelUriError::InvalidValue { key: got, .. } = err else {
            anyhow::bail!("expected InvalidValue, got {err:?}");
        };
        assert_eq!(got, key);
        Ok(())
    }

    #[test]
    fn model_uri_deserializes_from_a_uri_string() -> anyhow::Result<()> {
        // A JSON string value routes through the URI grammar.
        let ModelUri(model) = serde_json::from_str(r#""gguf-text://repo=o/r&path=Q4_K_M.gguf""#)?;
        assert!(matches!(model, Model::GgufText(_)));
        Ok(())
    }

    #[test]
    fn model_uri_deserializes_from_a_struct() -> anyhow::Result<()> {
        // A JSON object routes through Model's structured Deserialize.
        let ModelUri(model) = serde_json::from_str(
            r#"{"type":"gguf_text","source":"huggingface","org":"o","repo_name":"r","path":"Q4_K_M.gguf"}"#,
        )?;
        assert!(matches!(model, Model::GgufText(_)));
        Ok(())
    }

    #[test]
    fn model_uri_propagates_parse_errors() -> anyhow::Result<()> {
        assert!(serde_json::from_str::<ModelUri>(r#""bogus://x=y""#).is_err());
        Ok(())
    }

    #[test]
    fn model_uri_serializes_a_representable_model_as_the_uri() -> anyhow::Result<()> {
        let model: ModelUri = serde_json::from_str(r#""mlx://repo=o/r&prefix=4bit""#)?;
        assert_eq!(
            serde_json::to_string(&model)?,
            r#""mlx://repo=o/r&prefix=4bit""#
        );
        Ok(())
    }

    #[test]
    fn model_uri_serializes_a_non_representable_model_structurally() -> anyhow::Result<()> {
        // A local source has no URI form, so serde falls back to the object.
        let local = ModelUri(Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: pipette_plan_types::AbsolutePath::try_new("/on/disk/w.gguf")?,
            },
        }));
        assert!(serde_json::to_string(&local)?.starts_with('{'));
        Ok(())
    }

    /// A gguf-text HF model with an auth token — starts from the parsed
    /// (auth-free) model and adds the token, since the URI grammar can't express it.
    fn hf_text_with_auth() -> anyhow::Result<Model> {
        let mut model = parse_model_uri("gguf-text://repo=o/r&path=Q4_K_M.gguf")?;
        let Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace { repo, .. },
        }) = &mut model
        else {
            anyhow::bail!("expected gguf-text HuggingFace");
        };
        repo.auth_token = Some(pipette_plan_types::AuthToken::try_new("hf_secret")?);
        Ok(model)
    }

    #[test]
    fn render_rejects_a_repo_with_an_auth_token() -> anyhow::Result<()> {
        assert!(matches!(
            model_to_uri(&hf_text_with_auth()?),
            Err(ModelUriError::SecretNotRepresentable)
        ));
        Ok(())
    }

    #[test]
    fn model_uri_keeps_auth_by_serializing_structurally() -> anyhow::Result<()> {
        let model = hf_text_with_auth()?;
        let json = serde_json::to_string(&ModelUri(model.clone()))?;
        assert!(
            json.starts_with('{'),
            "auth-bearing model serializes as the object, not a URI"
        );
        // The token survives the serde round-trip (it would be lost via a URI).
        let ModelUri(back) = serde_json::from_str(&json)?;
        assert_eq!(back, model);
        Ok(())
    }

    // Canonical URIs (keys already in render order) — the round-trip must be an
    // identity: parse → render → parse yields the same Model, and since these are
    // canonical, the rendered string equals the input.
    #[rstest]
    #[case("gguf-text://repo=org/repo&path=Q4_K_M.gguf")]
    #[case(&format!("gguf-text://repo=org/repo&path=Q4_K_M.gguf&rev=v2&sha256={SHA}"))]
    #[case("gguf-text://url=https://cdn.example/m.gguf")]
    #[case(&format!("gguf-text://url=https://cdn.example/m.gguf&sha256={SHA}"))]
    #[case("gguf-vision://repo=org/repo&model=m.gguf&mmproj=p.gguf")]
    #[case("gguf-vision://model=https://x/m.gguf&mmproj=https://x/p.gguf")]
    #[case("mlx://repo=org/repo")]
    #[case("mlx://repo=org/repo&prefix=4bit&rev=v2")]
    #[case("torch://repo=org/repo&prefix=fp16")]
    #[case("openvino://repo=org/repo")]
    #[case("openvino://repo=org/repo&prefix=int4-sym-cw&rev=v2")]
    fn round_trips(#[case] uri: &str) -> anyhow::Result<()> {
        let model = parse_model_uri(uri)?;
        let rendered = model_to_uri(&model)?;
        assert_eq!(rendered, uri, "render of a canonical URI is byte-identical");
        assert_eq!(
            parse_model_uri(&rendered)?,
            model,
            "re-parse recovers the model"
        );
        Ok(())
    }

    #[test]
    fn render_rejects_apple_foundation() -> anyhow::Result<()> {
        assert!(matches!(
            model_to_uri(&Model::AppleFoundationText),
            Err(ModelUriError::NotImportable(_))
        ));
        Ok(())
    }

    #[test]
    fn render_rejects_a_local_source() -> anyhow::Result<()> {
        let local = Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: pipette_plan_types::AbsolutePath::try_new("/on/disk/w.gguf")?,
            },
        });
        assert!(matches!(
            model_to_uri(&local),
            Err(ModelUriError::NotImportable(_))
        ));
        Ok(())
    }

    #[test]
    fn render_rejects_a_query_url() -> anyhow::Result<()> {
        let model = Model::GgufText(GgufText {
            source: GgufTextSource::Url {
                url: ResourceUrl::try_new("https://x/m.gguf?token=y")?,
                sha256: None,
            },
        });
        assert!(matches!(
            model_to_uri(&model),
            Err(ModelUriError::QueryUrl { .. })
        ));
        Ok(())
    }
}
