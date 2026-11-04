//! The compact `--model` URI: [`parse_model_uri`] and its inverse
//! [`model_to_uri`], over shared key constants so the two directions cannot
//! drift. The grammar and keys are in `docs/pipette-cli/models-and-runtimes.md`.
//!
//! `&` separates pairs, so a URL with a query string has no URI form. Neither
//! does a repo with an auth token (a URI cannot hold a secret) nor a `vault_*`
//! source (no key for the principal, and dropping it would name a public fetch
//! of a private artifact); [`ModelUri`] serializes those structurally.

use std::fmt;
use std::str::FromStr;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use pipette_plan_types::{
    AppleCoreAi, GgufText, GgufTextSource, GgufVision, GgufVisionHfFiles, GgufVisionSource,
    GgufVisionSpeculative, GgufVisionSpeculativeSource, GgufVisionUrlFiles, HfRepo, HfRevision,
    Mlx, Model, ModelSource, OnnxGenai, Openvino, RepoSubpath, ResourceUrl, Sha256,
    SpeculativePair, Torch,
};

// Shared by the parser and [`model_to_uri`], so a rename cannot desync them.
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
const KEY_DRAFT: &str = "draft";
const KEY_DRAFT_SHA256: &str = "draft_sha256";
// `mlx-speculative` and `torch-speculative` name two whole repositories, so the
// drafter's keys mirror the target's; `gguf-vision-speculative`'s `draft` is one
// file from the same source as the weights.
const KEY_DRAFT_REPO: &str = "draft_repo";
const KEY_DRAFT_PREFIX: &str = "draft_prefix";
const KEY_DRAFT_REV: &str = "draft_rev";

/// The importable model kinds, one per URI scheme (the variant in kebab-case).
/// The parser matches on it exhaustively and [`model_to_uri`] on [`Model`], so
/// neither direction can drop a kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr)]
#[strum(serialize_all = "kebab-case")]
enum Scheme {
    GgufText,
    GgufVision,
    GgufVisionSpeculative,
    Mlx,
    #[strum(serialize = "core-ai")]
    AppleCoreAi,
    MlxSpeculative,
    Torch,
    TorchSpeculative,
    Openvino,
    OnnxGenai,
}

impl Scheme {
    fn as_str(self) -> &'static str {
        self.into()
    }

    /// `apple-foundation`, `local` and `file` name real but non-importable
    /// sources, so they get [`ModelUriError::NotImportable`] rather than
    /// "unknown". A `file://` value is unaffected: the scheme is what precedes
    /// the first `://`.
    fn parse(s: &str) -> Result<Self, ModelUriError> {
        Scheme::from_str(s).map_err(|_| match s {
            "apple-foundation" | "local" | "file" => ModelUriError::NotImportable(s.to_owned()),
            _ => ModelUriError::UnknownScheme(s.to_owned()),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelUriError {
    #[error("model URI must be `<scheme>://<key>=<value>[&<key>=<value>...]`")]
    MissingSchemeSeparator,

    #[error(
        "unknown model URI scheme `{0}` (expected `gguf-text`, `gguf-vision`, `mlx`, \
         `core-ai`, `torch`, `openvino`, or `onnx-genai`)"
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

    #[error(
        "a vault source has no URI form (the grammar has no key for its principal). \
         Pass a JSON `--model` object instead"
    )]
    VaultNotRepresentable,

    #[error("invalid value for key `{key}`: {message}")]
    InvalidValue { key: &'static str, message: String },
}

/// The `key=value` pairs of a URI body. The first `=` splits, so a value keeps
/// any later `=`. URL values carry no query string ([`resource_url`]), so every
/// `&` is a pair boundary and a leftover key is a real typo.
struct Pairs<'a> {
    scheme: Scheme,
    items: Vec<(&'a str, &'a str)>,
}

impl<'a> Pairs<'a> {
    fn parse(scheme: Scheme, body: &'a str) -> Result<Self, ModelUriError> {
        // An empty body carries no pairs, so `require` reports a precise
        // MissingKey rather than a MalformedPair("").
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

    /// Reject any key the scheme didn't claim.
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
    // `url=…?a=x&sha256=y` would fracture into a truncated URL and a fake
    // digest. Query and signed URLs go through the JSON `--model` form.
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
        // `rev` has no meaning for a raw URL.
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
            });
        }
    };
    p.finish()?;
    Ok(Model::GgufText(GgufText { source }))
}

fn parse_gguf_vision(mut p: Pairs) -> Result<Model, ModelUriError> {
    let source = parse_gguf_vision_files(&mut p)?;
    p.finish()?;
    Ok(Model::GgufVision(GgufVision { source }))
}

/// `gguf-vision`'s two files plus the optional drafter. The drafter is resolved
/// against the same source as the weights (a repo-relative path under `repo`,
/// a URL without one), because it is part of the same model.
fn parse_gguf_vision_speculative(mut p: Pairs) -> Result<Model, ModelUriError> {
    let files = parse_gguf_vision_files(&mut p)?;
    let draft = p.take(KEY_DRAFT);
    let draft_sha256 = opt_sha256(KEY_DRAFT_SHA256, p.take(KEY_DRAFT_SHA256))?;
    let source = match files {
        GgufVisionSource::HuggingFace(files) => GgufVisionSpeculativeSource::HuggingFace {
            files,
            draft: draft.map(|d| repo_subpath(KEY_DRAFT, d)).transpose()?,
            draft_sha256,
        },
        GgufVisionSource::Url(files) => GgufVisionSpeculativeSource::Url {
            files,
            draft: draft.map(|d| resource_url(KEY_DRAFT, d)).transpose()?,
            draft_sha256,
        },
        // `parse_gguf_vision_files` builds only the two importable arms.
        GgufVisionSource::AbsoluteFiles(_) | GgufVisionSource::RelativeFiles(_) => {
            return Err(not_importable_local());
        }
        GgufVisionSource::VaultUrl(_) => return Err(ModelUriError::VaultNotRepresentable),
    };
    p.finish()?;
    Ok(Model::GgufVisionSpeculative(GgufVisionSpeculative {
        source,
    }))
}

fn parse_gguf_vision_files(p: &mut Pairs) -> Result<GgufVisionSource, ModelUriError> {
    let repo = p.take(KEY_REPO);
    let source = match repo {
        Some(repo) => {
            let rev = p.take(KEY_REV);
            let model = p.require(KEY_MODEL)?;
            let mmproj = p.require(KEY_MMPROJ)?;
            let model_sha256 = p.take(KEY_MODEL_SHA256);
            let mmproj_sha256 = p.take(KEY_MMPROJ_SHA256);
            GgufVisionSource::HuggingFace(GgufVisionHfFiles {
                repo: hf_repo(repo, rev)?,
                model: repo_subpath(KEY_MODEL, model)?,
                model_sha256: opt_sha256(KEY_MODEL_SHA256, model_sha256)?,
                mmproj: repo_subpath(KEY_MMPROJ, mmproj)?,
                mmproj_sha256: opt_sha256(KEY_MMPROJ_SHA256, mmproj_sha256)?,
            })
        }
        None => {
            let model = p.require(KEY_MODEL)?;
            let mmproj = p.require(KEY_MMPROJ)?;
            let model_sha256 = p.take(KEY_MODEL_SHA256);
            let mmproj_sha256 = p.take(KEY_MMPROJ_SHA256);
            GgufVisionSource::Url(GgufVisionUrlFiles {
                model: resource_url(KEY_MODEL, model)?,
                model_sha256: opt_sha256(KEY_MODEL_SHA256, model_sha256)?,
                mmproj: resource_url(KEY_MMPROJ, mmproj)?,
                mmproj_sha256: opt_sha256(KEY_MMPROJ_SHA256, mmproj_sha256)?,
            })
        }
    };
    Ok(source)
}

/// The directory-shaped models share one HF source (importable = HF only).
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

/// `mlx-speculative`'s and `torch-speculative`'s two repositories: the target
/// under the plain dir keys, the drafter under `draft_`-prefixed twins. Such a
/// drafter is published as its own repository, and `draft_rev` pins it apart.
fn parse_speculative_pair(mut p: Pairs) -> Result<SpeculativePair, ModelUriError> {
    let repo = p.require(KEY_REPO)?;
    let prefix = p.take(KEY_PREFIX);
    let rev = p.take(KEY_REV);
    let draft_repo = p.require(KEY_DRAFT_REPO)?;
    let draft_prefix = p.take(KEY_DRAFT_PREFIX);
    let draft_rev = p.take(KEY_DRAFT_REV);
    let pair = SpeculativePair {
        target: ModelSource::HuggingFace {
            repo: hf_repo(repo, rev)?,
            prefix: prefix.map(|s| repo_subpath(KEY_PREFIX, s)).transpose()?,
        },
        draft: ModelSource::HuggingFace {
            repo: hf_repo(draft_repo, draft_rev)?,
            prefix: draft_prefix
                .map(|s| repo_subpath(KEY_DRAFT_PREFIX, s))
                .transpose()?,
        },
    };
    p.finish()?;
    Ok(pair)
}

pub fn parse_model_uri(input: &str) -> Result<Model, ModelUriError> {
    let (scheme, body) = input
        .split_once("://")
        .ok_or(ModelUriError::MissingSchemeSeparator)?;
    let scheme = Scheme::parse(scheme)?;
    let pairs = Pairs::parse(scheme, body)?;
    match scheme {
        Scheme::GgufText => parse_gguf_text(pairs),
        Scheme::GgufVision => parse_gguf_vision(pairs),
        Scheme::GgufVisionSpeculative => parse_gguf_vision_speculative(pairs),
        Scheme::AppleCoreAi => Ok(Model::AppleCoreAi(AppleCoreAi {
            source: parse_dir_source(pairs)?,
        })),
        Scheme::Mlx => Ok(Model::Mlx(Mlx {
            source: parse_dir_source(pairs)?,
        })),
        Scheme::MlxSpeculative => parse_speculative_pair(pairs).map(Model::MlxSpeculative),
        Scheme::TorchSpeculative => parse_speculative_pair(pairs).map(Model::TorchSpeculative),
        Scheme::Torch => Ok(Model::Torch(Torch {
            source: parse_dir_source(pairs)?,
        })),
        Scheme::Openvino => Ok(Model::Openvino(Openvino {
            source: parse_dir_source(pairs)?,
        })),
        Scheme::OnnxGenai => Ok(Model::OnnxGenai(OnnxGenai {
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

    /// `repo=org/name`; the revision is [`Body::rev`]. Refuses an auth token
    /// rather than dropping it, so `ModelUri` falls back to the structured form.
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

    /// A URL with `&` or `?` would fracture on re-parse (see [`resource_url`]).
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

/// The inverse of [`parse_model_uri`], with a fixed key order (required, then
/// `rev`, then optional) for stable output.
pub fn model_to_uri(model: &Model) -> Result<String, ModelUriError> {
    match model {
        Model::GgufText(m) => gguf_text_to_uri(&m.source),
        Model::GgufVision(m) => gguf_vision_to_uri(&m.source),
        Model::GgufVisionSpeculative(m) => gguf_vision_speculative_to_uri(&m.source),
        Model::Mlx(m) => dir_to_uri(Scheme::Mlx, &m.source),
        Model::AppleCoreAi(m) => dir_to_uri(Scheme::AppleCoreAi, &m.source),
        Model::MlxSpeculative(m) => speculative_pair_to_uri(Scheme::MlxSpeculative, m),
        Model::Torch(m) => dir_to_uri(Scheme::Torch, &m.source),
        Model::TorchSpeculative(m) => speculative_pair_to_uri(Scheme::TorchSpeculative, m),
        Model::Openvino(m) => dir_to_uri(Scheme::Openvino, &m.source),
        Model::OnnxGenai(m) => dir_to_uri(Scheme::OnnxGenai, &m.source),
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
            return Err(not_importable_local());
        }
        GgufTextSource::VaultUrl { .. } => return Err(ModelUriError::VaultNotRepresentable),
    }
    Ok(body.finish())
}

fn gguf_vision_to_uri(source: &GgufVisionSource) -> Result<String, ModelUriError> {
    let mut body = Body::new(Scheme::GgufVision);
    match source {
        GgufVisionSource::HuggingFace(GgufVisionHfFiles {
            repo,
            model,
            model_sha256,
            mmproj,
            mmproj_sha256,
        }) => {
            body.repo(repo)?;
            body.push(KEY_MODEL, model.as_ref());
            body.push(KEY_MMPROJ, mmproj.as_ref());
            body.rev(repo);
            body.sha(KEY_MODEL_SHA256, model_sha256);
            body.sha(KEY_MMPROJ_SHA256, mmproj_sha256);
        }
        GgufVisionSource::Url(GgufVisionUrlFiles {
            model,
            model_sha256,
            mmproj,
            mmproj_sha256,
        }) => {
            body.url(KEY_MODEL, model)?;
            body.url(KEY_MMPROJ, mmproj)?;
            body.sha(KEY_MODEL_SHA256, model_sha256);
            body.sha(KEY_MMPROJ_SHA256, mmproj_sha256);
        }
        GgufVisionSource::AbsoluteFiles(_) | GgufVisionSource::RelativeFiles(_) => {
            return Err(not_importable_local());
        }
        GgufVisionSource::VaultUrl(_) => return Err(ModelUriError::VaultNotRepresentable),
    }
    Ok(body.finish())
}

fn gguf_vision_speculative_to_uri(
    source: &GgufVisionSpeculativeSource,
) -> Result<String, ModelUriError> {
    let mut body = Body::new(Scheme::GgufVisionSpeculative);
    match source {
        GgufVisionSpeculativeSource::HuggingFace {
            files:
                GgufVisionHfFiles {
                    repo,
                    model,
                    model_sha256,
                    mmproj,
                    mmproj_sha256,
                },
            draft,
            draft_sha256,
        } => {
            body.repo(repo)?;
            body.push(KEY_MODEL, model.as_ref());
            body.push(KEY_MMPROJ, mmproj.as_ref());
            if let Some(draft) = draft {
                body.push(KEY_DRAFT, draft.as_ref());
            }
            body.rev(repo);
            body.sha(KEY_MODEL_SHA256, model_sha256);
            body.sha(KEY_MMPROJ_SHA256, mmproj_sha256);
            body.sha(KEY_DRAFT_SHA256, draft_sha256);
        }
        GgufVisionSpeculativeSource::Url {
            files:
                GgufVisionUrlFiles {
                    model,
                    model_sha256,
                    mmproj,
                    mmproj_sha256,
                },
            draft,
            draft_sha256,
        } => {
            body.url(KEY_MODEL, model)?;
            body.url(KEY_MMPROJ, mmproj)?;
            if let Some(draft) = draft {
                body.url(KEY_DRAFT, draft)?;
            }
            body.sha(KEY_MODEL_SHA256, model_sha256);
            body.sha(KEY_MMPROJ_SHA256, mmproj_sha256);
            body.sha(KEY_DRAFT_SHA256, draft_sha256);
        }
        GgufVisionSpeculativeSource::AbsoluteFiles { .. }
        | GgufVisionSpeculativeSource::RelativeFiles { .. } => return Err(not_importable_local()),
        GgufVisionSpeculativeSource::VaultUrl { .. } => {
            return Err(ModelUriError::VaultNotRepresentable);
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
        ModelSource::VaultArchive { .. } => Err(ModelUriError::VaultNotRepresentable),
        ModelSource::AbsoluteDir { .. } | ModelSource::RelativeDir { .. } => {
            Err(not_importable_local())
        }
    }
}

/// [`parse_speculative_pair`]'s inverse: the target's keys, then the drafter's.
fn speculative_pair_to_uri(
    scheme: Scheme,
    model: &SpeculativePair,
) -> Result<String, ModelUriError> {
    let mut body = Body::new(scheme);
    let dir = |body: &mut Body,
               source: &ModelSource,
               repo_key: &'static str,
               prefix_key: &'static str,
               rev_key: &'static str|
     -> Result<(), ModelUriError> {
        let (repo, prefix) = match source {
            ModelSource::HuggingFace { repo, prefix } => (repo, prefix),
            ModelSource::VaultArchive { .. } => return Err(ModelUriError::VaultNotRepresentable),
            ModelSource::AbsoluteDir { .. } | ModelSource::RelativeDir { .. } => {
                return Err(not_importable_local());
            }
        };
        if repo.auth_token.is_some() {
            return Err(ModelUriError::SecretNotRepresentable);
        }
        body.push(repo_key, &repo.to_string());
        if let Some(prefix) = prefix {
            body.push(prefix_key, prefix.as_ref());
        }
        if let Some(rev) = &repo.revision {
            body.push(rev_key, rev.as_ref());
        }
        Ok(())
    };
    dir(&mut body, &model.target, KEY_REPO, KEY_PREFIX, KEY_REV)?;
    dir(
        &mut body,
        &model.draft,
        KEY_DRAFT_REPO,
        KEY_DRAFT_PREFIX,
        KEY_DRAFT_REV,
    )?;
    Ok(body.finish())
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

/// A [`Model`] that reads a URI string or the structured object, and writes the
/// URI when the model has one, else the structured form.
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

        let uri = format!(
            "gguf-vision://repo=org/repo&model=a.gguf&mmproj=mm.gguf&rev=v2\
             &model_sha256={SHA}&mmproj_sha256={SHA}"
        );
        let full = parse_model_uri(&uri)?;
        let Model::GgufVision(GgufVision {
            source:
                GgufVisionSource::HuggingFace(GgufVisionHfFiles {
                    repo,
                    model_sha256,
                    mmproj_sha256,
                    ..
                }),
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
            source: GgufVisionSource::Url(_),
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
                GgufVisionSource::Url(GgufVisionUrlFiles {
                    model_sha256,
                    mmproj_sha256,
                    ..
                }),
        }) = &full
        else {
            anyhow::bail!("expected gguf-vision Url, got {full:?}");
        };
        assert!(model_sha256.is_some());
        assert!(mmproj_sha256.is_some());
        Ok(())
    }

    /// `prefix` and `rev` have to fold into the identity string (`@v2`, `:4bit`),
    /// or one repo's variants would collide.
    #[rstest]
    #[case("mlx://repo=org/repo", ModelType::Mlx, "org/repo")]
    #[case(
        "mlx://repo=org/repo&prefix=4bit&rev=v2",
        ModelType::Mlx,
        "org/repo@v2:4bit"
    )]
    #[case("torch://repo=org/repo", ModelType::Torch, "org/repo")]
    #[case("torch://repo=org/repo&prefix=sub", ModelType::Torch, "org/repo:sub")]
    #[case("core-ai://repo=org/repo", ModelType::AppleCoreAi, "org/repo")]
    #[case(
        "core-ai://repo=org/repo&prefix=gpu-pipelined/int4",
        ModelType::AppleCoreAi,
        "org/repo:gpu-pipelined/int4"
    )]
    #[case("openvino://repo=org/repo", ModelType::Openvino, "org/repo")]
    #[case(
        "openvino://repo=LiquidAI/LFM2.5-350M-ov&prefix=int4-sym-cw",
        ModelType::Openvino,
        "LiquidAI/LFM2.5-350M-ov:int4-sym-cw"
    )]
    #[case("onnx-genai://repo=org/repo", ModelType::OnnxGenai, "org/repo")]
    // An ONNX repo publishes one subdirectory per precision and provider, so
    // `prefix` is what names the variant being measured.
    #[case(
        "onnx-genai://repo=microsoft/Phi-3-mini-4k-instruct-onnx&prefix=cpu_and_mobile/cpu-int4-rtn-block-32",
        ModelType::OnnxGenai,
        "microsoft/Phi-3-mini-4k-instruct-onnx:cpu_and_mobile/cpu-int4-rtn-block-32"
    )]
    // The pair's identity is `{target}+{drafter}`, the same `+` join
    // `gguf_vision_speculative` uses, so one reading covers both.
    #[case(
        "mlx-speculative://repo=org/target&draft_repo=org/draft",
        ModelType::MlxSpeculative,
        "org/target+org/draft"
    )]
    #[case(
        "mlx-speculative://repo=org/target&rev=v2&draft_repo=org/draft&draft_rev=v9",
        ModelType::MlxSpeculative,
        "org/target@v2+org/draft@v9"
    )]
    #[case(
        "torch-speculative://repo=LiquidAI/LFM2.5-VL-3B&draft_repo=LiquidAI/LFM2.5-VL-3B-DSpark",
        ModelType::TorchSpeculative,
        "LiquidAI/LFM2.5-VL-3B+LiquidAI/LFM2.5-VL-3B-DSpark"
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
    // There is no merged form, so a pair without a drafter has nothing to
    // measure and must fail here rather than at fetch.
    #[case("mlx-speculative://repo=o/r", "draft_repo")]
    #[case("mlx-speculative://draft_repo=o/d", "repo")]
    #[case("torch-speculative://repo=o/r", "draft_repo")]
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

    // Case 2 is the truncation trap: `&sha256=` would fracture off as a digest.
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

    // Not misattributed to the URL.
    #[test]
    fn url_leftover_key_is_unknown_key() -> anyhow::Result<()> {
        assert!(matches!(
            parse_model_uri("gguf-text://url=https://x/m.gguf&foo=bar"),
            Err(ModelUriError::UnknownKey { key, .. }) if key == "foo"
        ));
        Ok(())
    }

    // The variant and `key` only: the message is nutype's and brittle.
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
        let ModelUri(model) = serde_json::from_str(r#""gguf-text://repo=o/r&path=Q4_K_M.gguf""#)?;
        assert!(matches!(model, Model::GgufText(_)));
        Ok(())
    }

    #[test]
    fn model_uri_deserializes_from_a_struct() -> anyhow::Result<()> {
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
        let local = ModelUri(Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: pipette_plan_types::AbsolutePath::try_new("/on/disk/w.gguf")?,
            },
        }));
        assert!(serde_json::to_string(&local)?.starts_with('{'));
        Ok(())
    }

    /// The URI grammar cannot express a token, so it is added after parsing.
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

    /// The principal is never dropped.
    #[rstest]
    #[case::gguf_text(r#"{"type":"gguf_text","source":"vault_url","url":"/vault/resources/vault_1/content","principal_id":"org_1"}"#)]
    #[case::gguf_vision(r#"{"type":"gguf_vision","source":"vault_url","model":"/vault/resources/vault_1/content","mmproj":"/vault/resources/vault_2/content","principal_id":"org_1"}"#)]
    #[case::gguf_vision_speculative(r#"{"type":"gguf_vision_speculative","source":"vault_url","model":"/vault/resources/vault_1/content","mmproj":"/vault/resources/vault_2/content","principal_id":"org_1"}"#)]
    #[case::mlx(r#"{"type":"mlx","source":"vault_archive","url":"/vault/resources/vault_1/content","principal_id":"user_1"}"#)]
    #[case::torch_speculative(r#"{"type":"torch_speculative","target":{"source":"vault_archive","url":"/vault/resources/vault_1/content","principal_id":"org_1"},"draft":{"source":"huggingface","org":"o","repo_name":"r"}}"#)]
    fn a_vault_model_has_no_uri_and_serializes_structurally(
        #[case] json: &str,
    ) -> anyhow::Result<()> {
        let model: Model = serde_json::from_str(json)?;
        assert!(matches!(
            model_to_uri(&model),
            Err(ModelUriError::VaultNotRepresentable)
        ));
        let out = serde_json::to_string(&ModelUri(model.clone()))?;
        assert!(out.starts_with('{'), "got {out}");
        let ModelUri(back) = serde_json::from_str(&out)?;
        assert_eq!(back, model);
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
        let ModelUri(back) = serde_json::from_str(&json)?;
        assert_eq!(back, model);
        Ok(())
    }

    // Canonical URIs (keys in render order), so the render equals the input.
    #[rstest]
    #[case("gguf-text://repo=org/repo&path=Q4_K_M.gguf")]
    #[case(&format!("gguf-text://repo=org/repo&path=Q4_K_M.gguf&rev=v2&sha256={SHA}"))]
    #[case("gguf-text://url=https://cdn.example/m.gguf")]
    #[case(&format!("gguf-text://url=https://cdn.example/m.gguf&sha256={SHA}"))]
    #[case("gguf-vision://repo=org/repo&model=m.gguf&mmproj=p.gguf")]
    #[case("gguf-vision://model=https://x/m.gguf&mmproj=https://x/p.gguf")]
    #[case("gguf-vision-speculative://repo=org/repo&model=m.gguf&mmproj=p.gguf")]
    #[case("gguf-vision-speculative://repo=org/repo&model=m.gguf&mmproj=p.gguf&draft=d.gguf")]
    #[case(&format!(
        "gguf-vision-speculative://repo=org/repo&model=m.gguf&mmproj=p.gguf&draft=d.gguf\
         &rev=v2&draft_sha256={SHA}"
    ))]
    #[case("gguf-vision-speculative://model=https://x/m.gguf&mmproj=https://x/p.gguf")]
    #[case(
        "gguf-vision-speculative://model=https://x/m.gguf&mmproj=https://x/p.gguf\
         &draft=https://x/d.gguf"
    )]
    #[case("mlx://repo=org/repo")]
    #[case("mlx://repo=org/repo&prefix=4bit&rev=v2")]
    #[case("torch://repo=org/repo&prefix=fp16")]
    #[case("openvino://repo=org/repo")]
    #[case("openvino://repo=org/repo&prefix=int4-sym-cw&rev=v2")]
    #[case("onnx-genai://repo=org/repo")]
    #[case("onnx-genai://repo=org/repo&prefix=cpu_and_mobile/cpu-int4-rtn-block-32&rev=v2")]
    #[case("mlx-speculative://repo=org/target&draft_repo=org/draft")]
    #[case(
        "mlx-speculative://repo=org/target&prefix=8bit&rev=v2&draft_repo=org/draft&draft_prefix=4L&draft_rev=v9"
    )]
    #[case("torch-speculative://repo=org/target&draft_repo=org/draft&draft_rev=v9")]
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
