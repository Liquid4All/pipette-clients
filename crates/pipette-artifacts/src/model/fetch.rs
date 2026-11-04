//! Materialize a model into its store entry's `blobs/`.
//!
//! [`parts`] is the one place a model type is taken apart; fetching
//! ([`fetch_model`]) and sizing ([`declared_size_bytes`]) both work from its
//! parts. A local source is copied, so the store owns a self-contained copy.

use std::fs;
use std::path::Path;

use reqwest::StatusCode;
use reqwest::blocking::Response;
use reqwest::header::{IF_RANGE, RANGE};
use serde::Deserialize;

use pipette_http::HttpClient;
use pipette_http::retry::{Disposition, RetryPolicy, retrying};
use pipette_mgmt_client::EntityTag;
use pipette_plan_types::{
    AbsolutePath, AuthToken, GgufTextSource, GgufVisionAbsoluteFiles, GgufVisionHfFiles,
    GgufVisionSource, GgufVisionSpeculativeSource, GgufVisionUrlFiles, GgufVisionVaultUrlFiles,
    HfRepo, Mlx, Model, ModelSource, OnnxGenai, Openvino, RepoSubpath, ResourceUrl, Sha256, Torch,
    VaultPath,
};

use super::stored::{ModelStoredError, to_stored};
use crate::io_kind::names_the_host;
use crate::progress::Reporter;
use crate::resume::{self, Expect, Partial, RangeBody, ResumePoint, Source};
use crate::vault::{self, VaultAccess, VaultFetchError};

/// Why [`fetch_model`] couldn't materialize a model.
#[derive(Debug, thiserror::Error)]
pub enum ModelFetchError {
    /// The model's parts can't be planned: a store-relative source, a bound form
    /// that is not absolute, or a declared/bound kind mismatch.
    #[error("{0}")]
    NotFetchable(String),
    #[error("invalid download URL: {0}")]
    InvalidUrl(String),
    #[error("invalid destination path: {0}")]
    InvalidDestination(String),
    #[error(transparent)]
    UnresolvedDestination(#[from] ModelStoredError),
    #[error("{0}")]
    NothingToFetch(String),
    #[error("sha256 mismatch for {dest}: expected {expected}, got {actual}")]
    Sha256Mismatch {
        dest: String,
        expected: String,
        actual: String,
    },
    /// A resumed file didn't match the declared `sha256`. `resumed_at` is the
    /// byte where a second response first joins the file.
    #[error(
        "sha256 mismatch for {dest} resumed at byte {resumed_at}: expected {expected}, got {actual}"
    )]
    ResumedSha256Mismatch {
        dest: String,
        resumed_at: u64,
        expected: String,
        actual: String,
    },
    #[error(
        "length mismatch for {dest} resumed at byte {resumed_at}: expected {expected} bytes, got {actual}"
    )]
    ResumedLengthMismatch {
        dest: String,
        resumed_at: u64,
        expected: u64,
        actual: u64,
    },
    #[error("HTTP request to {url} failed")]
    Http {
        url: String,
        #[source]
        source: pipette_http::Error,
    },
    #[error("reading the repo listing from {url} failed")]
    Listing {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("{message}")]
    Io {
        message: String,
        /// The authoring path for a local import; `None` for a download write.
        src: Option<String>,
        dest: String,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Vault(#[from] VaultFetchError),
}

impl ModelFetchError {
    /// A resumed file that fails its length or digest check is transient: the
    /// next attempt downloads it from zero. A digest mismatch in a file from one
    /// response fails the same way every time.
    fn disposition(&self) -> Disposition {
        match self {
            Self::Http { source, .. } => source.disposition(),
            Self::ResumedSha256Mismatch { .. } | Self::ResumedLengthMismatch { .. } => {
                Disposition::TRANSIENT
            }
            Self::Listing { source, .. } if !is_malformed_json(source) => Disposition::TRANSIENT,
            Self::Io {
                src: None, source, ..
            } if !names_the_host(source) => Disposition::TRANSIENT,
            Self::NotFetchable(_)
            | Self::InvalidUrl(_)
            | Self::InvalidDestination(_)
            | Self::UnresolvedDestination(_)
            | Self::NothingToFetch(_)
            | Self::Sha256Mismatch { .. }
            | Self::Listing { .. }
            | Self::Io { .. }
            // A vault fetch retries inside `vault`, under its own classification.
            | Self::Vault(_) => Disposition::Permanent,
        }
    }
}

/// Whether a listing failed on its JSON rather than on the body that carried
/// it. reqwest reports both as a decode error; only the cause tells them apart.
fn is_malformed_json(error: &reqwest::Error) -> bool {
    std::iter::successors(
        Some(error as &(dyn std::error::Error + 'static)),
        |&cause| cause.source(),
    )
    .any(|cause| cause.is::<serde_json::Error>())
}

fn io_err(src: Option<&Path>, dest: &Path, source: std::io::Error) -> ModelFetchError {
    let dest = dest.display().to_string();
    let src = src.map(|p| p.display().to_string());
    let message = match &src {
        Some(s) => format!("importing `{s}` → `{dest}` failed"),
        None => format!("writing {dest} failed"),
    };
    ModelFetchError::Io {
        message,
        src,
        dest,
        source,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Download {
    origin: Origin,
    /// The plan's pin. For a public origin it is what the bytes are checked
    /// against; for a vault one the record's pin is, and this must agree with it.
    sha256: Option<Sha256>,
    dest: AbsolutePath,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Origin {
    /// A plain HTTP `GET`, with a bearer token when the repo carries one.
    Public {
        url: ResourceUrl,
        auth: Option<AuthToken>,
    },
    /// A vault resource, fetched through mgmt signed as the device.
    Vault { path: VaultPath },
    /// A file on this host, copied.
    Local { path: AbsolutePath },
}

#[cfg(test)]
impl Download {
    fn url(&self) -> &str {
        match &self.origin {
            Origin::Public { url, .. } => url.as_ref(),
            Origin::Vault { path } => path.resource_id().as_ref(),
            Origin::Local { path } => path.as_ref(),
        }
    }

    fn auth(&self) -> Option<&AuthToken> {
        match &self.origin {
            Origin::Public { auth, .. } => auth.as_ref(),
            Origin::Vault { .. } | Origin::Local { .. } => None,
        }
    }
}

fn public(url: ResourceUrl, auth: Option<AuthToken>) -> Origin {
    Origin::Public { url, auth }
}

fn local(path: &AbsolutePath) -> Origin {
    Origin::Local { path: path.clone() }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Part {
    File(Download),
    HfSnapshot {
        repo: HfRepo,
        prefix: Option<RepoSubpath>,
        dest: AbsolutePath,
    },
    VaultArchive {
        path: VaultPath,
        sha256: Option<Sha256>,
        dest: AbsolutePath,
    },
    LocalDir {
        src: AbsolutePath,
        dest: AbsolutePath,
    },
}

/// What `declared` consists of, each part landing at its path in `into` (see
/// [`to_stored`]). Network-free.
fn parts(declared: &Model, into: &Model) -> Result<Vec<Part>, ModelFetchError> {
    let files =
        |downloads: Vec<Download>| -> Vec<Part> { downloads.into_iter().map(Part::File).collect() };
    Ok(match (declared, into) {
        (Model::GgufText(d), Model::GgufText(i)) => {
            files(gguf_text_downloads(&d.source, &i.source)?)
        }
        (Model::GgufVision(d), Model::GgufVision(i)) => {
            files(gguf_vision_downloads(&d.source, &i.source)?)
        }
        (Model::GgufVisionSpeculative(d), Model::GgufVisionSpeculative(i)) => {
            files(gguf_vision_speculative_downloads(&d.source, &i.source)?)
        }
        (Model::Mlx(Mlx { source: d }), Model::Mlx(Mlx { source: i }))
        | (
            Model::AppleCoreAi(pipette_plan_types::AppleCoreAi { source: d }),
            Model::AppleCoreAi(pipette_plan_types::AppleCoreAi { source: i }),
        )
        | (Model::Torch(Torch { source: d }), Model::Torch(Torch { source: i }))
        | (Model::Openvino(Openvino { source: d }), Model::Openvino(Openvino { source: i }))
        | (Model::OnnxGenai(OnnxGenai { source: d }), Model::OnnxGenai(OnnxGenai { source: i })) => {
            vec![dir_part(d, i)?]
        }
        // The drafter is fetched with the target rather than as a model of its
        // own, because the two together are what the row measured.
        (Model::MlxSpeculative(d), Model::MlxSpeculative(i))
        | (Model::TorchSpeculative(d), Model::TorchSpeculative(i)) => vec![
            dir_part(&d.target, &i.target)?,
            dir_part(&d.draft, &i.draft)?,
        ],
        (Model::AppleFoundationText, Model::AppleFoundationText) => Vec::new(),
        // Named rather than `_`, so a new model type does not compile until it
        // has an arm above.
        (
            Model::GgufText(_)
            | Model::GgufVision(_)
            | Model::GgufVisionSpeculative(_)
            | Model::Mlx(_)
            | Model::MlxSpeculative(_)
            | Model::TorchSpeculative(_)
            | Model::Torch(_)
            | Model::Openvino(_)
            | Model::OnnxGenai(_)
            | Model::AppleFoundationText
            | Model::AppleCoreAi(_),
            _,
        ) => {
            return Err(ModelFetchError::NotFetchable(
                "declared and localized model kinds disagree".to_owned(),
            ));
        }
    })
}

fn dir_part(declared: &ModelSource, into: &ModelSource) -> Result<Part, ModelFetchError> {
    let ModelSource::AbsoluteDir { dir: dest } = into else {
        return Err(ModelFetchError::NotFetchable(
            "bound directory model is not an Absolute source".to_owned(),
        ));
    };
    let dest = dest.clone();
    Ok(match declared {
        ModelSource::HuggingFace { repo, prefix } => Part::HfSnapshot {
            repo: repo.clone(),
            prefix: prefix.clone(),
            dest,
        },
        ModelSource::VaultArchive { url, sha256, .. } => Part::VaultArchive {
            path: url.clone(),
            sha256: sha256.clone(),
            dest,
        },
        ModelSource::AbsoluteDir { dir } => Part::LocalDir {
            src: dir.clone(),
            dest,
        },
        ModelSource::RelativeDir { .. } => {
            return Err(ModelFetchError::NotFetchable(
                "a store-relative directory model has no source to fetch".to_owned(),
            ));
        }
    })
}

fn gguf_text_downloads(
    declared: &GgufTextSource,
    into: &GgufTextSource,
) -> Result<Vec<Download>, ModelFetchError> {
    let dest = match into {
        GgufTextSource::AbsoluteFile { path } => path,
        GgufTextSource::RelativeFile { path: _ } => {
            return Err(ModelFetchError::NotFetchable(
                "gguf-text fetch dest must be abs Local, got Relative".to_owned(),
            ));
        }
        GgufTextSource::HuggingFace { .. }
        | GgufTextSource::Url { .. }
        | GgufTextSource::VaultUrl { .. } => {
            return Err(ModelFetchError::NotFetchable(
                "bound gguf-text is not an Absolute source".to_owned(),
            ));
        }
    };
    let download = match declared {
        GgufTextSource::HuggingFace { repo, path, sha256 } => Download {
            origin: public(hf_file_url(repo, path)?, repo.auth_token.clone()),
            sha256: sha256.clone(),
            dest: dest.clone(),
        },
        GgufTextSource::Url { url, sha256 } => Download {
            origin: public(url.clone(), None),
            sha256: sha256.clone(),
            dest: dest.clone(),
        },
        GgufTextSource::VaultUrl { url, sha256, .. } => Download {
            origin: Origin::Vault { path: url.clone() },
            sha256: sha256.clone(),
            dest: dest.clone(),
        },
        GgufTextSource::AbsoluteFile { path } => Download {
            origin: local(path),
            sha256: None,
            dest: dest.clone(),
        },
        GgufTextSource::RelativeFile { .. } => {
            return Err(ModelFetchError::NotFetchable(
                "store-relative gguf-text has no source to fetch".to_owned(),
            ));
        }
    };
    Ok(vec![download])
}

fn gguf_vision_downloads(
    declared: &GgufVisionSource,
    into: &GgufVisionSource,
) -> Result<Vec<Download>, ModelFetchError> {
    let GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles {
        model: model_dest,
        mmproj: mmproj_dest,
    }) = into
    else {
        return Err(ModelFetchError::NotFetchable(
            "gguf-vision fetch dest must be abs Local".to_owned(),
        ));
    };
    let downloads = match declared {
        GgufVisionSource::HuggingFace(GgufVisionHfFiles {
            repo,
            model,
            model_sha256,
            mmproj,
            mmproj_sha256,
        }) => vec![
            Download {
                origin: public(hf_file_url(repo, model)?, repo.auth_token.clone()),
                sha256: model_sha256.clone(),
                dest: model_dest.clone(),
            },
            Download {
                origin: public(hf_file_url(repo, mmproj)?, repo.auth_token.clone()),
                sha256: mmproj_sha256.clone(),
                dest: mmproj_dest.clone(),
            },
        ],
        GgufVisionSource::Url(GgufVisionUrlFiles {
            model,
            model_sha256,
            mmproj,
            mmproj_sha256,
        }) => vec![
            Download {
                origin: public(model.clone(), None),
                sha256: model_sha256.clone(),
                dest: model_dest.clone(),
            },
            Download {
                origin: public(mmproj.clone(), None),
                sha256: mmproj_sha256.clone(),
                dest: mmproj_dest.clone(),
            },
        ],
        GgufVisionSource::VaultUrl(GgufVisionVaultUrlFiles {
            model,
            model_sha256,
            mmproj,
            mmproj_sha256,
            ..
        }) => vec![
            Download {
                origin: Origin::Vault {
                    path: model.clone(),
                },
                sha256: model_sha256.clone(),
                dest: model_dest.clone(),
            },
            Download {
                origin: Origin::Vault {
                    path: mmproj.clone(),
                },
                sha256: mmproj_sha256.clone(),
                dest: mmproj_dest.clone(),
            },
        ],
        GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles { model, mmproj }) => vec![
            Download {
                origin: local(model),
                sha256: None,
                dest: model_dest.clone(),
            },
            Download {
                origin: local(mmproj),
                sha256: None,
                dest: mmproj_dest.clone(),
            },
        ],
        GgufVisionSource::RelativeFiles(_) => {
            return Err(ModelFetchError::NotFetchable(
                "store-relative gguf-vision has no source to fetch".to_owned(),
            ));
        }
    };
    Ok(downloads)
}

/// The two vision files as [`gguf_vision_downloads`] plans them, plus the
/// drafter when it has a file of its own. Delegating keeps the pair fetched
/// exactly as a plain VL model's: same URLs, digests and auth.
fn gguf_vision_speculative_downloads(
    declared: &GgufVisionSpeculativeSource,
    into: &GgufVisionSpeculativeSource,
) -> Result<Vec<Download>, ModelFetchError> {
    let mut downloads = gguf_vision_downloads(&declared.vision(), &into.vision())?;
    let GgufVisionSpeculativeSource::AbsoluteFiles { draft: dest, .. } = into else {
        return Err(ModelFetchError::NotFetchable(
            "gguf-vision-speculative fetch dest must be abs Local".to_owned(),
        ));
    };
    let draft = match (declared, dest) {
        // Merged into the weights.
        (_, None) => None,
        (
            GgufVisionSpeculativeSource::HuggingFace {
                files,
                draft,
                draft_sha256,
            },
            Some(dest),
        ) => draft.as_ref().map(|path| {
            Ok::<_, ModelFetchError>(Download {
                origin: public(
                    hf_file_url(&files.repo, path)?,
                    files.repo.auth_token.clone(),
                ),
                sha256: draft_sha256.clone(),
                dest: dest.clone(),
            })
        }),
        (
            GgufVisionSpeculativeSource::Url {
                draft,
                draft_sha256,
                ..
            },
            Some(dest),
        ) => draft.as_ref().map(|url| {
            Ok(Download {
                origin: public(url.clone(), None),
                sha256: draft_sha256.clone(),
                dest: dest.clone(),
            })
        }),
        (
            GgufVisionSpeculativeSource::VaultUrl {
                draft,
                draft_sha256,
                ..
            },
            Some(dest),
        ) => draft.as_ref().map(|url| {
            Ok(Download {
                origin: Origin::Vault { path: url.clone() },
                sha256: draft_sha256.clone(),
                dest: dest.clone(),
            })
        }),
        (GgufVisionSpeculativeSource::AbsoluteFiles { draft, .. }, Some(dest)) => {
            draft.as_ref().map(|path| {
                Ok(Download {
                    origin: local(path),
                    sha256: None,
                    dest: dest.clone(),
                })
            })
        }
        (GgufVisionSpeculativeSource::RelativeFiles { .. }, Some(_)) => {
            return Err(ModelFetchError::NotFetchable(
                "store-relative gguf-vision-speculative has no source to fetch".to_owned(),
            ));
        }
    }
    .transpose()?;
    downloads.extend(draft);
    Ok(downloads)
}

fn hf_file_url(repo: &HfRepo, path: &RepoSubpath) -> Result<ResourceUrl, ModelFetchError> {
    let revision = repo.revision.as_ref().map_or("main", AsRef::as_ref);
    hf_resolve_url(repo, revision, path.as_ref())
}

/// Takes the path as a raw string so it also serves repo-listing entries, which
/// aren't [`RepoSubpath`]s.
fn hf_resolve_url(
    repo: &HfRepo,
    revision: &str,
    path: &str,
) -> Result<ResourceUrl, ModelFetchError> {
    let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{path}");
    ResourceUrl::try_new(url.clone()).map_err(|_| ModelFetchError::InvalidUrl(url))
}

/// The files to pull for a directory model. A `prefix` keeps only its subtree,
/// stripped so the files land directly under `dest_dir`. Every file resolves
/// against `info.sha`, not the requested ref, so a push mid-download cannot mix
/// two commits. An LFS file is verified against the `sha256` the listing gives
/// it; a small non-LFS file is not.
fn plan_dir_downloads(
    repo: &HfRepo,
    prefix: Option<&RepoSubpath>,
    dest_dir: &AbsolutePath,
    info: &HfModelInfo,
) -> Result<Vec<Download>, ModelFetchError> {
    let prefix_slash = prefix.map(|prefix| format!("{}/", prefix.as_ref()));
    let downloads: Vec<Download> = info
        .siblings
        .iter()
        .filter_map(|sibling| {
            // A prefix keeps only its subtree; the stripped tail is the local layout.
            let relative = match &prefix_slash {
                Some(prefix) => sibling.rfilename.strip_prefix(prefix)?,
                None => &sibling.rfilename,
            };
            Some((sibling, relative))
        })
        .map(|(sibling, relative)| {
            Ok(Download {
                origin: public(
                    hf_resolve_url(repo, &info.sha, &sibling.rfilename)?,
                    repo.auth_token.clone(),
                ),
                sha256: sibling.lfs.as_ref().map(|lfs| lfs.sha256.clone()),
                dest: local_join(dest_dir, relative)?,
            })
        })
        .collect::<Result<_, ModelFetchError>>()?;
    // Usually a mistyped `prefix`. Fetching it would publish an empty model dir
    // that every later `ensure` resolves as valid.
    if downloads.is_empty() {
        return Err(ModelFetchError::NothingToFetch(match prefix {
            Some(prefix) => format!(
                "repo `{repo}` has no files under prefix `{}`",
                prefix.as_ref()
            ),
            None => format!("repo `{repo}` has no files to fetch"),
        }));
    }
    Ok(downloads)
}

fn local_join(dir: &AbsolutePath, relative: &str) -> Result<AbsolutePath, ModelFetchError> {
    let joined = format!("{}/{}", dir.as_ref(), relative);
    AbsolutePath::try_new(joined.clone()).map_err(|_| ModelFetchError::InvalidDestination(joined))
}

#[derive(Deserialize)]
struct HfModelInfo {
    /// The commit the requested revision resolved to; required so downloads
    /// never fall back to the ref.
    sha: String,
    siblings: Vec<HfSibling>,
}

#[derive(Deserialize)]
struct HfSibling {
    rfilename: String,
    /// Only returned when the listing is requested with `?blobs=true`.
    size: Option<u64>,
    /// Likewise, and only for a file stored in LFS.
    lfs: Option<HfLfs>,
}

#[derive(Deserialize)]
struct HfLfs {
    sha256: Sha256,
}

const HF_ENDPOINT: &str = "https://huggingface.co";

/// Materialize `declared` at the paths in `into`. A failure may leave a partial
/// file; the store stages into a dir it discards on error.
///
/// A local copy reports no progress: `fs::copy` gives no count to observe, so
/// the caller's [`Reporter::finish`] counts it.
pub(crate) fn fetch_model(
    http: &HttpClient,
    retry: &RetryPolicy,
    vault: Option<&VaultAccess>,
    declared: &Model,
    into: &Model,
    reporter: &mut Reporter,
) -> Result<(), ModelFetchError> {
    fetch_model_with_hf_endpoint(http, retry, vault, HF_ENDPOINT, declared, into, reporter)
}

/// [`fetch_model`] with the HF listing at `hf_endpoint`, for mock servers.
fn fetch_model_with_hf_endpoint(
    http: &HttpClient,
    retry: &RetryPolicy,
    vault: Option<&VaultAccess>,
    hf_endpoint: &str,
    declared: &Model,
    into: &Model,
    reporter: &mut Reporter,
) -> Result<(), ModelFetchError> {
    let parts = parts(declared, into)?;
    // Every snapshot is listed before any part is fetched, so a drafter repo
    // that cannot be listed fails before the target downloads.
    let listings = parts
        .iter()
        .map(|part| match part {
            Part::HfSnapshot { repo, prefix, dest } => {
                snapshot_downloads(http, retry, hf_endpoint, repo, prefix.as_ref(), dest)
            }
            Part::File(_) | Part::VaultArchive { .. } | Part::LocalDir { .. } => Ok(Vec::new()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    parts
        .iter()
        .zip(listings)
        .try_for_each(|(part, listing)| match part {
            Part::File(download) => fetch_file(http, retry, vault, download, reporter),
            Part::HfSnapshot { .. } => listing
                .iter()
                .try_for_each(|download| fetch_file(http, retry, vault, download, reporter)),
            Part::VaultArchive { path, sha256, dest } => {
                vault::fetch_and_unpack(
                    vault,
                    http,
                    retry,
                    path,
                    sha256.as_ref(),
                    Path::new(dest.as_ref()),
                    reporter,
                )?;
                Ok(())
            }
            Part::LocalDir { src, dest } => {
                copy_tree(Path::new(src.as_ref()), Path::new(dest.as_ref()))
            }
        })
}

fn snapshot_downloads(
    http: &HttpClient,
    retry: &RetryPolicy,
    hf_endpoint: &str,
    repo: &HfRepo,
    prefix: Option<&RepoSubpath>,
    dest: &AbsolutePath,
) -> Result<Vec<Download>, ModelFetchError> {
    let info = repo_info(http, retry, hf_endpoint, repo)?;
    let requested_revision = repo.revision.as_ref().map_or("main", AsRef::as_ref);
    tracing::info!(
        %repo,
        revision = %requested_revision,
        commit = %info.sha,
        "resolved the Hugging Face repo revision"
    );
    plan_dir_downloads(repo, prefix, dest, &info)
}

/// The HuggingFace model-info record for `repo`; an unpinned repo reads `main`.
/// Not paginated: one response carries the full `siblings` list. `blobs=true`
/// adds per-file sizes and LFS digests: the quota pre-flight sums the one, the
/// fetch verifies against the other.
fn repo_info(
    http: &HttpClient,
    retry: &RetryPolicy,
    hf_endpoint: &str,
    repo: &HfRepo,
) -> Result<HfModelInfo, ModelFetchError> {
    let revision = repo.revision.as_ref().map_or("main", AsRef::as_ref);
    let url = format!("{hf_endpoint}/api/models/{repo}/revision/{revision}?blobs=true");
    retrying(
        retry,
        &url,
        |_| {
            let mut request = http.client().get(&url);
            if let Some(token) = &repo.auth_token {
                request = request.bearer_auth(token.as_ref());
            }
            let response = pipette_http::send(request).map_err(|source| ModelFetchError::Http {
                url: url.clone(),
                source,
            })?;
            response.json().map_err(|source| ModelFetchError::Listing {
                url: url.clone(),
                source,
            })
        },
        ModelFetchError::disposition,
    )
}

/// `Content-Length` for `url`, from a HEAD. A failed HEAD is logged, not fatal:
/// some servers refuse HEAD and serve the GET anyway.
pub(crate) fn content_length(
    http: &HttpClient,
    url: &str,
    auth: Option<&AuthToken>,
) -> Option<u64> {
    let mut request = http.client().head(url);
    if let Some(token) = auth {
        request = request.bearer_auth(token.as_ref());
    }
    // Read the header rather than `Response::content_length()`: a HEAD reply has
    // no body, so the body-length hint is always 0.
    pipette_http::send(request)
        .inspect_err(|source| {
            tracing::warn!(
                %url,
                error = %source,
                "sizing failed; the storage quota cannot be checked before the fetch"
            );
        })
        .ok()?
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

/// Bytes `declared` will occupy once fetched, when knowable before the fetch.
/// `Ok(None)` leaves the post-publish sweep as the only enforcement. `Err` means
/// the fetch could not be planned: sizing uses the same [`to_stored`] and
/// [`parts`], so it is the fetch's own failure surfacing early.
pub(crate) fn declared_size_bytes(
    http: &HttpClient,
    retry: &RetryPolicy,
    vault: Option<&VaultAccess>,
    declared: &Model,
) -> Result<Option<u64>, ModelFetchError> {
    let into = to_stored(declared, Path::new(QUOTA_PROBE_BASE))?;
    // One unknowable part makes the total unknowable rather than an undercount
    // the quota pre-flight would enforce against.
    parts(declared, &into)?
        .iter()
        .try_fold(Some(0u64), |total, part| {
            let size = part_size_bytes(http, retry, vault, part)?;
            Ok(total
                .zip(size)
                .map(|(total, size)| total.saturating_add(size)))
        })
}

fn part_size_bytes(
    http: &HttpClient,
    retry: &RetryPolicy,
    vault: Option<&VaultAccess>,
    part: &Part,
) -> Result<Option<u64>, ModelFetchError> {
    Ok(match part {
        Part::File(Download {
            origin: Origin::Public { url, auth },
            ..
        }) => content_length(http, url.as_ref(), auth.as_ref()),
        Part::File(Download {
            origin: Origin::Vault { path },
            sha256,
            ..
        }) => Some(vault::resource(vault, http, retry, path, sha256.as_ref())?.bytes),
        Part::File(Download {
            origin: Origin::Local { path: src },
            ..
        })
        | Part::LocalDir { src, .. } => {
            Some(crate::entry::entry_size_bytes(Path::new(src.as_ref())))
        }
        Part::HfSnapshot { repo, prefix, .. } => {
            hf_dir_size_bytes(http, retry, HF_ENDPOINT, repo, prefix.as_ref())
        }
        // The record sizes the packed archive, an undercount of the unpacked tree.
        // It is still read, so a refusal surfaces here as the fetch's would.
        Part::VaultArchive { path, sha256, .. } => {
            vault::resource(vault, http, retry, path, sha256.as_ref())?;
            None
        }
    })
}

/// Never written to: it exists so [`to_stored`] yields the `Absolute*` arms
/// [`parts`] requires. Platform-shaped because a driveless `/quota-probe` is
/// relative on Windows.
#[cfg(windows)]
const QUOTA_PROBE_BASE: &str = r"C:\quota-probe";
#[cfg(not(windows))]
const QUOTA_PROBE_BASE: &str = "/quota-probe";

/// `None` if the listing fails or any file omits its size.
fn hf_dir_size_bytes(
    http: &HttpClient,
    retry: &RetryPolicy,
    hf_endpoint: &str,
    repo: &HfRepo,
    prefix: Option<&RepoSubpath>,
) -> Option<u64> {
    let prefix_slash = prefix.map(|p| format!("{}/", p.as_ref()));
    // Not fatal, for the same reason a failed HEAD isn't: the fetch makes its
    // own listing request and will report the failure itself if it persists.
    let info = repo_info(http, retry, hf_endpoint, repo)
        .inspect_err(|source| {
            tracing::warn!(
                %repo,
                error = %source,
                "sizing failed; the storage quota cannot be checked before the fetch"
            );
        })
        .ok()?;
    let under_prefix: Vec<_> = info
        .siblings
        .into_iter()
        .filter(|sibling| {
            prefix_slash
                .as_ref()
                .is_none_or(|prefix| sibling.rfilename.starts_with(prefix.as_str()))
        })
        .collect();
    // An empty match is "no such subtree", not a zero-byte one.
    if under_prefix.is_empty() {
        return None;
    }
    under_prefix.into_iter().try_fold(0u64, |total, sibling| {
        Some(total.saturating_add(sibling.size?))
    })
}

fn fetch_file(
    http: &HttpClient,
    retry: &RetryPolicy,
    vault: Option<&VaultAccess>,
    download: &Download,
    reporter: &mut Reporter,
) -> Result<(), ModelFetchError> {
    let dest = Path::new(download.dest.as_ref());
    let (url, auth) = match &download.origin {
        Origin::Public { url, auth } => (url, auth),
        Origin::Vault { path } => {
            vault::fetch_to_file(
                vault,
                http,
                retry,
                path,
                download.sha256.as_ref(),
                dest,
                reporter,
            )?;
            return Ok(());
        }
        Origin::Local { path } => return copy_file(Path::new(path.as_ref()), dest),
    };
    let start = reporter.mark();
    let mut partial = None;
    retrying(
        retry,
        url.as_ref(),
        |_| {
            fetch_public_file(
                http,
                url,
                auth.as_ref(),
                download,
                &mut partial,
                start,
                reporter,
            )
        },
        ModelFetchError::disposition,
    )
}

struct PublicFile<'a> {
    http: &'a HttpClient,
    url: &'a ResourceUrl,
    auth: Option<&'a AuthToken>,
    dest: &'a Path,
}

impl Source for PublicFile<'_> {
    type Body = Response;
    type Error = ModelFetchError;

    fn get(&self, resume: Option<&ResumePoint>) -> Result<Response, ModelFetchError> {
        let mut request = self.http.client().get(self.url.as_ref());
        if let Some(token) = self.auth {
            request = request.bearer_auth(token.as_ref());
        }
        if let Some(point) = resume {
            request = request
                .header(RANGE, format!("bytes={}-", point.offset))
                .header(IF_RANGE, point.etag.as_str());
        }
        pipette_http::send(request).map_err(|source| ModelFetchError::Http {
            url: self.url.to_string(),
            source,
        })
    }

    fn range_refused(&self, error: &ModelFetchError) -> bool {
        matches!(
            error,
            ModelFetchError::Http {
                source: pipette_http::Error::Status { status, .. },
                ..
            } if *status == StatusCode::RANGE_NOT_SATISFIABLE
        )
    }

    fn expect(&self, body: &Response) -> Expect {
        Expect {
            etag: body
                .strong_etag()
                .and_then(|etag| EntityTag::try_new(etag).ok()),
            total: body.content_length(),
        }
    }

    fn io_error(&self, source: std::io::Error) -> ModelFetchError {
        io_err(None, self.dest, source)
    }
}

/// One attempt to stream `url` into the download's destination and verify it.
/// It continues `partial` when the server agrees the file is the same.
/// `start` is the progress count a restart rewinds to.
fn fetch_public_file(
    http: &HttpClient,
    url: &ResourceUrl,
    auth: Option<&AuthToken>,
    download: &Download,
    partial: &mut Option<Partial>,
    start: u64,
    reporter: &mut Reporter,
) -> Result<(), ModelFetchError> {
    let _span = tracing::info_span!("public download", %url).entered();
    let dest = Path::new(download.dest.as_ref());
    let source = PublicFile {
        http,
        url,
        auth,
        dest,
    };
    let file = resume::attempt(&source, dest, partial, start, reporter)?;

    // The body read makes a file from one response as long as its
    // `Content-Length`. A joined file can be shorter or longer than the total.
    if let (Some(resumed_at), Some(expected)) = (file.resumed_at, file.expect.total)
        && file.writer.written() != expected
    {
        return Err(ModelFetchError::ResumedLengthMismatch {
            dest: download.dest.to_string(),
            resumed_at,
            expected,
            actual: file.writer.written(),
        });
    }
    let actual = file.writer.hex_digest();
    if let Some(expected) = &download.sha256
        && actual != expected.as_ref()
    {
        let (dest, expected) = (download.dest.to_string(), expected.to_string());
        return Err(match file.resumed_at {
            Some(resumed_at) => ModelFetchError::ResumedSha256Mismatch {
                dest,
                resumed_at,
                expected,
                actual,
            },
            None => ModelFetchError::Sha256Mismatch {
                dest,
                expected,
                actual,
            },
        });
    }
    Ok(())
}

fn copy_file(src: &Path, dest: &Path) -> Result<(), ModelFetchError> {
    let to_io = |source| io_err(Some(src), dest, source);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(to_io)?;
    }
    fs::copy(src, dest).map_err(to_io)?;
    Ok(())
}

fn copy_tree(src: &Path, dest: &Path) -> Result<(), ModelFetchError> {
    let to_io = |source| io_err(Some(src), dest, source);
    fs::create_dir_all(dest).map_err(to_io)?;
    fs::read_dir(src).map_err(to_io)?.try_for_each(|entry| {
        let entry = entry.map_err(to_io)?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            copy_tree(&src_path, &dest_path)
        } else {
            copy_file(&src_path, &dest_path)
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use httpmock::prelude::*;
    use rstest::rstest;

    use pipette_plan_types::{
        GgufText, GgufVision, GgufVisionSpeculative, HfOrg, HfRepoName, HfRevision, RelativePath,
        SpeculativePair,
    };

    use super::*;
    use crate::progress::{FetchProgress, ProgressSink};
    use crate::scripted_server::{Reply, ScriptedServer};

    const SHA_A: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const HF_COMMIT_A: &str = "0123456789abcdef0123456789abcdef01234567";
    const HF_COMMIT_B: &str = "resolved-v2-snapshot";

    /// Platform-shaped like [`QUOTA_PROBE_BASE`]; forward slashes on both
    /// because `to_stored` normalizes separators to `/`.
    #[cfg(windows)]
    const BASE: &str = "C:/store/entry/blobs";
    #[cfg(not(windows))]
    const BASE: &str = "/store/entry/blobs";

    fn base() -> &'static Path {
        Path::new(BASE)
    }

    fn dest(relative: &str) -> String {
        format!("{BASE}/{relative}")
    }

    fn hf_repo(revision: Option<&str>, auth: Option<&str>) -> anyhow::Result<HfRepo> {
        Ok(HfRepo {
            org: HfOrg::try_new("meta".to_owned())?,
            repo_name: HfRepoName::try_new("llama".to_owned())?,
            revision: revision
                .map(|r| HfRevision::try_new(r.to_owned()))
                .transpose()?,
            auth_token: auth.map(|t| AuthToken::try_new(t.to_owned())).transpose()?,
        })
    }

    fn hf_info(sha: &str, files: &[&str]) -> HfModelInfo {
        HfModelInfo {
            sha: sha.to_owned(),
            siblings: files
                .iter()
                .map(|rfilename| HfSibling {
                    rfilename: (*rfilename).to_owned(),
                    size: None,
                    lfs: None,
                })
                .collect(),
        }
    }

    fn hf_text(revision: Option<&str>, auth: Option<&str>) -> anyhow::Result<Model> {
        Ok(Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: hf_repo(revision, auth)?,
                path: RepoSubpath::try_new("Q4.gguf")?,
                sha256: Some(Sha256::try_new(SHA_A.to_owned())?),
            },
        }))
    }

    fn planned(declared: &Model) -> anyhow::Result<Vec<Part>> {
        Ok(parts(declared, &to_stored(declared, base())?)?)
    }

    fn downloads(declared: &Model) -> anyhow::Result<Vec<Download>> {
        planned(declared)?
            .into_iter()
            .map(|part| match part {
                Part::File(download) => Ok(download),
                dir @ (Part::HfSnapshot { .. }
                | Part::VaultArchive { .. }
                | Part::LocalDir { .. }) => anyhow::bail!("expected a file, got {dir:?}"),
            })
            .collect()
    }

    fn only_download(declared: &Model) -> anyhow::Result<Download> {
        let mut downloads = downloads(declared)?;
        anyhow::ensure!(downloads.len() == 1, "expected a single download");
        Ok(downloads.remove(0))
    }

    /// The weights and projector plan exactly as a plain VL model's.
    #[rstest]
    #[case::separate_drafter(
        Some("draft-4L.gguf"),
        vec!["Q4.gguf", "mm.gguf", "draft-4L.gguf"]
    )]
    // Merged into the weights, as DSpark's release ships it.
    #[case::merged_drafter(None, vec!["Q4.gguf", "mm.gguf"])]
    fn hf_speculative_pulls_the_vision_pair_plus_any_drafter(
        #[case] draft: Option<&str>,
        #[case] expected: Vec<&str>,
    ) -> anyhow::Result<()> {
        let declared = Model::GgufVisionSpeculative(GgufVisionSpeculative {
            source: GgufVisionSpeculativeSource::HuggingFace {
                files: GgufVisionHfFiles {
                    repo: hf_repo(None, Some("hf_tok"))?,
                    model: RepoSubpath::try_new("Q4.gguf")?,
                    model_sha256: None,
                    mmproj: RepoSubpath::try_new("mm.gguf")?,
                    mmproj_sha256: None,
                },
                draft: draft
                    .map(|d| RepoSubpath::try_new(d.to_owned()))
                    .transpose()?,
                draft_sha256: draft
                    .is_some()
                    .then(|| Sha256::try_new(SHA_A.to_owned()))
                    .transpose()?,
            },
        });
        let downloads = downloads(&declared)?;

        assert_eq!(
            downloads
                .iter()
                .map(|d| d.url().to_owned())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|f| format!("https://huggingface.co/meta/llama/resolve/main/{f}"))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            downloads
                .iter()
                .map(|d| d.dest.as_ref().to_owned())
                .collect::<Vec<_>>(),
            expected.iter().map(|f| dest(f)).collect::<Vec<_>>()
        );
        assert!(
            downloads
                .iter()
                .all(|d| d.auth().map(AsRef::as_ref) == Some("hf_tok"))
        );
        Ok(())
    }

    #[test]
    fn a_merged_speculative_model_plans_like_its_plain_vl_twin() -> anyhow::Result<()> {
        let files = GgufVisionHfFiles {
            repo: hf_repo(None, None)?,
            model: RepoSubpath::try_new("Q4.gguf")?,
            model_sha256: None,
            mmproj: RepoSubpath::try_new("mm.gguf")?,
            mmproj_sha256: None,
        };
        let vision = Model::GgufVision(GgufVision {
            source: GgufVisionSource::HuggingFace(files.clone()),
        });
        let speculative = Model::GgufVisionSpeculative(GgufVisionSpeculative {
            source: GgufVisionSpeculativeSource::HuggingFace {
                files,
                draft: None,
                draft_sha256: None,
            },
        });
        assert_eq!(planned(&vision)?, planned(&speculative)?);
        Ok(())
    }

    #[test]
    fn hf_text_resolves_to_a_resolve_url_with_auth_and_sha() -> anyhow::Result<()> {
        let download = only_download(&hf_text(None, Some("hf_tok"))?)?;
        assert_eq!(
            download.url(),
            "https://huggingface.co/meta/llama/resolve/main/Q4.gguf"
        );
        assert_eq!(download.auth().map(AsRef::as_ref), Some("hf_tok"));
        assert_eq!(
            download.sha256.as_ref().map(ToString::to_string).as_deref(),
            Some(SHA_A)
        );
        assert_eq!(download.dest.as_ref(), dest("Q4.gguf").as_str());
        Ok(())
    }

    #[rstest]
    #[case(None, "main")]
    #[case(Some("v1.0"), "v1.0")]
    fn revision_is_used_when_pinned_else_main(
        #[case] revision: Option<&str>,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let download = only_download(&hf_text(revision, None)?)?;
        assert_eq!(
            download.url(),
            format!("https://huggingface.co/meta/llama/resolve/{expected}/Q4.gguf")
        );
        assert!(download.auth().is_none(), "no token on a public repo");
        Ok(())
    }

    #[test]
    fn url_source_passes_through_without_auth() -> anyhow::Result<()> {
        let declared = Model::GgufText(GgufText {
            source: GgufTextSource::Url {
                url: ResourceUrl::try_new("https://ex.com/dir/w.gguf")?,
                sha256: None,
            },
        });
        let download = only_download(&declared)?;
        assert_eq!(download.url(), "https://ex.com/dir/w.gguf");
        assert!(download.auth().is_none());
        assert!(download.sha256.is_none());
        Ok(())
    }

    #[test]
    fn hf_vision_plans_model_then_mmproj() -> anyhow::Result<()> {
        let declared = Model::GgufVision(GgufVision {
            source: GgufVisionSource::HuggingFace(GgufVisionHfFiles {
                repo: hf_repo(None, None)?,
                model: RepoSubpath::try_new("model.gguf")?,
                model_sha256: None,
                mmproj: RepoSubpath::try_new("mmproj.gguf")?,
                mmproj_sha256: None,
            }),
        });
        let downloads = downloads(&declared)?;
        let urls: Vec<&str> = downloads.iter().map(|d| d.url()).collect();
        assert_eq!(
            urls,
            [
                "https://huggingface.co/meta/llama/resolve/main/model.gguf",
                "https://huggingface.co/meta/llama/resolve/main/mmproj.gguf",
            ]
        );
        let dests: Vec<&str> = downloads.iter().map(|d| d.dest.as_ref()).collect();
        assert_eq!(dests, [dest("model.gguf"), dest("mmproj.gguf")]);
        Ok(())
    }

    #[test]
    fn url_vision_plans_both_files_without_auth() -> anyhow::Result<()> {
        let declared = Model::GgufVision(GgufVision {
            source: GgufVisionSource::Url(GgufVisionUrlFiles {
                model: ResourceUrl::try_new("https://ex.com/model.gguf")?,
                model_sha256: None,
                mmproj: ResourceUrl::try_new("https://ex.com/mmproj.gguf")?,
                mmproj_sha256: None,
            }),
        });
        let downloads = downloads(&declared)?;
        let urls: Vec<&str> = downloads.iter().map(|d| d.url()).collect();
        assert_eq!(
            urls,
            ["https://ex.com/model.gguf", "https://ex.com/mmproj.gguf"]
        );
        assert!(
            downloads.iter().all(|d| d.auth().is_none()),
            "URL sources carry no token"
        );
        let dests: Vec<&str> = downloads.iter().map(|d| d.dest.as_ref()).collect();
        assert_eq!(dests, [dest("model.gguf"), dest("mmproj.gguf")]);
        Ok(())
    }

    #[test]
    fn apple_foundation_has_no_parts() -> anyhow::Result<()> {
        let declared = Model::AppleFoundationText;
        assert!(parts(&declared, &declared)?.is_empty());
        assert_eq!(
            declared_size_bytes(&test_http()?, &RetryPolicy::IMMEDIATE, None, &declared)?,
            Some(0)
        );
        Ok(())
    }

    #[test]
    fn declared_and_bound_kinds_that_disagree_are_refused() -> anyhow::Result<()> {
        let declared = hf_text(None, None)?;
        let into = to_stored(&Model::AppleFoundationText, base())?;
        assert!(matches!(
            parts(&declared, &into),
            Err(ModelFetchError::NotFetchable(_))
        ));
        Ok(())
    }

    #[test]
    fn a_directory_source_plans_as_a_part_of_its_kind() -> anyhow::Result<()> {
        let mlx = |source| Model::Mlx(Mlx { source });
        let repo = hf_repo(None, None)?;
        let prefix = RepoSubpath::try_new("4bit")?;
        assert_eq!(
            planned(&mlx(ModelSource::HuggingFace {
                repo: repo.clone(),
                prefix: Some(prefix.clone()),
            }))?,
            [Part::HfSnapshot {
                repo,
                prefix: Some(prefix),
                dest: AbsolutePath::try_new(dest("4bit"))?,
            }]
        );

        let path = crate::vault::test_support::content_path("vault_m")?;
        assert_eq!(
            planned(&mlx(ModelSource::VaultArchive {
                url: path.clone(),
                sha256: None,
                principal_id: "org_1".parse()?,
                custom_label: None,
            }))?,
            [Part::VaultArchive {
                path,
                sha256: None,
                dest: AbsolutePath::try_new(BASE.to_owned())?,
            }]
        );

        let src = AbsolutePath::try_new(dest("authored"))?;
        assert_eq!(
            planned(&mlx(ModelSource::AbsoluteDir { dir: src.clone() }))?,
            [Part::LocalDir {
                src,
                dest: AbsolutePath::try_new(BASE.to_owned())?,
            }]
        );

        let relative = mlx(ModelSource::RelativeDir {
            dir: RelativePath::try_new("weights".to_owned())?,
        });
        assert!(matches!(
            parts(&relative, &to_stored(&relative, base())?),
            Err(ModelFetchError::NotFetchable(_))
        ));
        Ok(())
    }

    fn ensure_parent(path: &Path) -> anyhow::Result<()> {
        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("missing parent for {}", path.display()))?;
        fs::create_dir_all(parent)?;
        Ok(())
    }

    #[test]
    fn fetch_copies_a_local_gguf_text_into_store_paths() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let src = tmp.path().join("src/w.gguf");
        ensure_parent(&src)?;
        fs::write(&src, b"gguf-bytes")?;
        let dest = tmp.path().join("store/blobs/w.gguf");

        let declared = Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: AbsolutePath::try_new(src.to_string_lossy().into_owned())?,
            },
        });
        let into = Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: AbsolutePath::try_new(dest.to_string_lossy().into_owned())?,
            },
        });
        fetch_model(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            &into,
            &mut Reporter::silent(),
        )?;
        assert_eq!(fs::read(&dest)?, b"gguf-bytes");
        assert!(src.exists());
        Ok(())
    }

    #[test]
    fn fetch_copies_a_local_gguf_vision_into_store_paths() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let src_model = tmp.path().join("src/model.gguf");
        let src_mmproj = tmp.path().join("src/mmproj.gguf");
        ensure_parent(&src_model)?;
        fs::write(&src_model, b"model")?;
        fs::write(&src_mmproj, b"mmproj")?;
        let dest_model = tmp.path().join("store/blobs/model.gguf");
        let dest_mmproj = tmp.path().join("store/blobs/mmproj.gguf");

        let declared = Model::GgufVision(GgufVision {
            source: GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles {
                model: AbsolutePath::try_new(src_model.to_string_lossy().into_owned())?,
                mmproj: AbsolutePath::try_new(src_mmproj.to_string_lossy().into_owned())?,
            }),
        });
        let into = Model::GgufVision(GgufVision {
            source: GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles {
                model: AbsolutePath::try_new(dest_model.to_string_lossy().into_owned())?,
                mmproj: AbsolutePath::try_new(dest_mmproj.to_string_lossy().into_owned())?,
            }),
        });
        fetch_model(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            &into,
            &mut Reporter::silent(),
        )?;
        assert_eq!(fs::read(&dest_model)?, b"model");
        assert_eq!(fs::read(&dest_mmproj)?, b"mmproj");
        Ok(())
    }

    #[rstest]
    #[case::separate_drafter(Some("draft.gguf"))]
    #[case::merged_drafter(None)]
    fn fetch_copies_a_local_gguf_vision_speculative_into_store_paths(
        #[case] draft: Option<&str>,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let src = tmp.path().join("src");
        fs::create_dir_all(&src)?;
        let authored = |name: &str| -> anyhow::Result<AbsolutePath> {
            fs::write(src.join(name), name)?;
            Ok(AbsolutePath::try_new(
                src.join(name).to_string_lossy().into_owned(),
            )?)
        };
        let declared = Model::GgufVisionSpeculative(GgufVisionSpeculative {
            source: GgufVisionSpeculativeSource::AbsoluteFiles {
                files: GgufVisionAbsoluteFiles {
                    model: authored("model.gguf")?,
                    mmproj: authored("mmproj.gguf")?,
                },
                draft: draft.map(authored).transpose()?,
            },
        });
        let blobs = tmp.path().join("store/blobs");

        fetch_model(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            &to_stored(&declared, &blobs)?,
            &mut Reporter::silent(),
        )?;
        ["model.gguf", "mmproj.gguf"]
            .into_iter()
            .chain(draft)
            .try_for_each(|name| {
                assert_eq!(fs::read(blobs.join(name))?, name.as_bytes());
                Ok(())
            })
    }

    #[test]
    fn fetch_local_missing_source_names_src_in_error() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let src = tmp.path().join("missing/w.gguf");
        let dest = tmp.path().join("store/blobs/w.gguf");
        let declared = Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: AbsolutePath::try_new(src.to_string_lossy().into_owned())?,
            },
        });
        let into = Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: AbsolutePath::try_new(dest.to_string_lossy().into_owned())?,
            },
        });
        match fetch_model(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            &into,
            &mut Reporter::silent(),
        ) {
            Ok(()) => anyhow::bail!("expected an Io error"),
            Err(ModelFetchError::Io {
                src: Some(err_src),
                dest: err_dest,
                ..
            }) => {
                assert_eq!(err_src.as_str(), src.to_string_lossy());
                assert_eq!(err_dest.as_str(), dest.to_string_lossy());
            }
            other => anyhow::bail!("expected Io with src, got {other:?}"),
        }
        Ok(())
    }

    #[rstest]
    #[case::mlx(|source| Model::Mlx(Mlx { source }))]
    #[case::core_ai(|source| Model::AppleCoreAi(pipette_plan_types::AppleCoreAi { source }))]
    fn fetch_copies_a_local_dir_into_store_paths(
        #[case] make: fn(ModelSource) -> Model,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let src = tmp.path().join("src/weights");
        fs::create_dir_all(src.join("sub"))?;
        fs::write(src.join("config.json"), b"{}")?;
        fs::write(src.join("sub/w.safetensors"), b"tensor")?;
        let dest = tmp.path().join("store/blobs");

        let source = ModelSource::AbsoluteDir {
            dir: AbsolutePath::try_new(src.to_string_lossy().into_owned())?,
        };
        let target = ModelSource::AbsoluteDir {
            dir: AbsolutePath::try_new(dest.to_string_lossy().into_owned())?,
        };
        let declared = make(source);
        let into = make(target);
        fetch_model(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            &into,
            &mut Reporter::silent(),
        )?;
        assert_eq!(fs::read(dest.join("config.json"))?, b"{}");
        assert_eq!(fs::read(dest.join("sub/w.safetensors"))?, b"tensor");
        Ok(())
    }

    #[rstest]
    #[case::both_local(true, true)]
    #[case::both_from_the_vault(false, false)]
    #[case::local_target_and_vault_drafter(true, false)]
    fn a_pair_lands_each_half_in_its_role_dir(
        #[case] local_target: bool,
        #[case] local_draft: bool,
    ) -> anyhow::Result<()> {
        use crate::model::stored::{MLX_DRAFT_DIR, MLX_TARGET_DIR};
        use crate::vault::test_support::{access, content_path, serve_bytes, tar_with};

        let server = MockServer::start();
        let tmp = tempfile::tempdir()?;
        let half = |local: bool, role: &str| -> anyhow::Result<ModelSource> {
            Ok(if local {
                let dir = tmp.path().join("src").join(role);
                fs::create_dir_all(&dir)?;
                fs::write(dir.join("config.json"), role)?;
                ModelSource::AbsoluteDir {
                    dir: AbsolutePath::try_new(dir.to_string_lossy().into_owned())?,
                }
            } else {
                let id = format!("vault_{role}");
                serve_bytes(&server, &id, &tar_with("config.json", role.as_bytes())?);
                ModelSource::VaultArchive {
                    url: content_path(&id)?,
                    sha256: None,
                    principal_id: "org_1".parse()?,
                    custom_label: None,
                }
            })
        };
        let declared = Model::MlxSpeculative(SpeculativePair {
            target: half(local_target, MLX_TARGET_DIR)?,
            draft: half(local_draft, MLX_DRAFT_DIR)?,
        });
        let blobs = tmp.path().join("store/blobs");

        fetch_model(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            Some(&access(&server)?),
            &declared,
            &to_stored(&declared, &blobs)?,
            &mut Reporter::silent(),
        )?;
        [MLX_TARGET_DIR, MLX_DRAFT_DIR]
            .into_iter()
            .try_for_each(|role| {
                assert_eq!(
                    fs::read(blobs.join(role).join("config.json"))?,
                    role.as_bytes()
                );
                Ok(())
            })
    }

    #[test]
    fn a_pair_lists_its_drafter_before_fetching_its_target() -> anyhow::Result<()> {
        use crate::model::stored::MLX_TARGET_DIR;

        let server = MockServer::start();
        let listing = server.mock(|when, then| {
            when.method(GET)
                .path("/api/models/meta/llama/revision/main");
            then.status(404);
        });
        let tmp = tempfile::tempdir()?;
        let src_target = tmp.path().join("src/target");
        fs::create_dir_all(&src_target)?;
        fs::write(src_target.join("config.json"), b"target")?;
        let declared = Model::MlxSpeculative(SpeculativePair {
            target: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(src_target.to_string_lossy().into_owned())?,
            },
            draft: ModelSource::HuggingFace {
                repo: hf_repo(None, None)?,
                prefix: None,
            },
        });
        let blobs = tmp.path().join("store/blobs");

        let err = fetch_model_with_hf_endpoint(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            None,
            &server.base_url(),
            &declared,
            &to_stored(&declared, &blobs)?,
            &mut Reporter::silent(),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected the drafter listing to fail"))?;

        listing.assert();
        assert!(matches!(err, ModelFetchError::Http { .. }), "{err:?}");
        assert!(
            !blobs.join(MLX_TARGET_DIR).exists(),
            "the target was fetched before the drafter was listed"
        );
        Ok(())
    }

    #[test]
    fn dir_downloads_map_the_whole_snapshot() -> anyhow::Result<()> {
        let repo = hf_repo(None, Some("hf_tok"))?;
        let dest = AbsolutePath::try_new("/store/entry/blobs".to_owned())?;
        let mut info = hf_info(
            HF_COMMIT_A,
            &["config.json", "model.safetensors", "tokenizer/vocab.json"],
        );
        if let Some(weights) = info
            .siblings
            .iter_mut()
            .find(|s| s.rfilename == "model.safetensors")
        {
            weights.lfs = Some(HfLfs {
                sha256: Sha256::try_new(SHA_A.to_owned())?,
            });
        }
        let downloads = plan_dir_downloads(&repo, None, &dest, &info)?;
        let urls: Vec<&str> = downloads.iter().map(|d| d.url()).collect();
        assert_eq!(
            urls,
            [
                "https://huggingface.co/meta/llama/resolve/0123456789abcdef0123456789abcdef01234567/config.json",
                "https://huggingface.co/meta/llama/resolve/0123456789abcdef0123456789abcdef01234567/model.safetensors",
                "https://huggingface.co/meta/llama/resolve/0123456789abcdef0123456789abcdef01234567/tokenizer/vocab.json",
            ]
        );
        assert!(
            downloads
                .iter()
                .all(|d| d.auth().map(AsRef::as_ref) == Some("hf_tok")),
            "the repo token rides every file"
        );
        let digests: Vec<Option<&str>> = downloads
            .iter()
            .map(|d| d.sha256.as_ref().map(AsRef::as_ref))
            .collect();
        assert_eq!(
            digests,
            [None, Some(SHA_A), None],
            "only an LFS file carries the listed sha256"
        );
        let dests: Vec<&str> = downloads.iter().map(|d| d.dest.as_ref()).collect();
        assert_eq!(
            dests,
            [
                "/store/entry/blobs/config.json",
                "/store/entry/blobs/model.safetensors",
                "/store/entry/blobs/tokenizer/vocab.json",
            ]
        );
        Ok(())
    }

    #[test]
    fn dir_downloads_keep_only_the_prefix_subtree_and_strip_it() -> anyhow::Result<()> {
        let repo = hf_repo(Some("v2"), None)?;
        let prefix = RepoSubpath::try_new("4bit")?;
        // `to_stored` already ends `dest_dir` in the prefix.
        let dest = AbsolutePath::try_new("/store/entry/blobs/4bit".to_owned())?;
        let info = hf_info(
            HF_COMMIT_B,
            &[
                "README.md",
                "4bit/config.json",
                "4bit/weights/model.safetensors",
            ],
        );
        let downloads = plan_dir_downloads(&repo, Some(&prefix), &dest, &info)?;
        let urls: Vec<&str> = downloads.iter().map(|d| d.url()).collect();
        assert_eq!(
            urls,
            [
                "https://huggingface.co/meta/llama/resolve/resolved-v2-snapshot/4bit/config.json",
                "https://huggingface.co/meta/llama/resolve/resolved-v2-snapshot/4bit/weights/model.safetensors",
            ]
        );
        let dests: Vec<&str> = downloads.iter().map(|d| d.dest.as_ref()).collect();
        assert_eq!(
            dests,
            [
                "/store/entry/blobs/4bit/config.json",
                "/store/entry/blobs/4bit/weights/model.safetensors",
            ]
        );
        Ok(())
    }

    #[test]
    fn a_prefix_matching_nothing_is_rejected() -> anyhow::Result<()> {
        let repo = hf_repo(None, None)?;
        let prefix = RepoSubpath::try_new("8bit")?;
        let dest = AbsolutePath::try_new("/store/entry/blobs/8bit".to_owned())?;
        let info = hf_info(HF_COMMIT_A, &["README.md", "4bit/config.json"]);
        assert!(matches!(
            plan_dir_downloads(&repo, Some(&prefix), &dest, &info),
            Err(ModelFetchError::NothingToFetch(_))
        ));
        Ok(())
    }

    #[test]
    fn an_empty_repo_listing_is_rejected() -> anyhow::Result<()> {
        let repo = hf_repo(None, None)?;
        let dest = AbsolutePath::try_new("/store/entry/blobs".to_owned())?;
        assert!(matches!(
            plan_dir_downloads(&repo, None, &dest, &hf_info(HF_COMMIT_A, &[])),
            Err(ModelFetchError::NothingToFetch(_))
        ));
        Ok(())
    }

    #[test]
    fn a_traversal_filename_cannot_escape_the_model_dir() -> anyhow::Result<()> {
        // `AbsolutePath` rejects `..` segments.
        let repo = hf_repo(None, None)?;
        let dest = AbsolutePath::try_new("/store/entry/blobs".to_owned())?;
        let info = hf_info(HF_COMMIT_A, &["../../etc/passwd"]);
        assert!(matches!(
            plan_dir_downloads(&repo, None, &dest, &info),
            Err(ModelFetchError::InvalidDestination(_))
        ));
        Ok(())
    }

    // ---- network path: the real reqwest HTTP against a mock server ----------

    fn test_http() -> anyhow::Result<HttpClient> {
        Ok(HttpClient::new("pipette-test")?)
    }

    /// `(declared, into)` for a `Url` source, which needs no endpoint override.
    fn url_text_model(
        url: &str,
        sha256: Option<&str>,
        dest: &Path,
    ) -> anyhow::Result<(Model, Model)> {
        let declared = Model::GgufText(GgufText {
            source: GgufTextSource::Url {
                url: ResourceUrl::try_new(url.to_owned())?,
                sha256: sha256.map(|s| Sha256::try_new(s.to_owned())).transpose()?,
            },
        });
        let into = Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: AbsolutePath::try_new(dest.to_string_lossy().into_owned())?,
            },
        });
        Ok((declared, into))
    }

    #[test]
    fn fetch_downloads_a_url_source_and_verifies_sha256() -> anyhow::Result<()> {
        let server = MockServer::start();
        let hit = server.mock(|when, then| {
            when.method(GET).path("/model.gguf");
            then.status(200).body("abc"); // sha256("abc") == SHA_A
        });
        let tmp = tempfile::tempdir()?;
        let dest = tmp.path().join("model.gguf");
        let (declared, into) = url_text_model(&server.url("/model.gguf"), Some(SHA_A), &dest)?;

        fetch_model(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            &into,
            &mut Reporter::silent(),
        )?;

        hit.assert();
        assert_eq!(std::fs::read_to_string(&dest)?, "abc");
        Ok(())
    }

    #[test]
    fn fetch_rejects_a_sha256_mismatch() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/w.gguf");
            then.status(200).body("xyz");
        });
        let tmp = tempfile::tempdir()?;
        let dest = tmp.path().join("w.gguf");
        let (declared, into) = url_text_model(&server.url("/w.gguf"), Some(SHA_A), &dest)?;

        match fetch_model(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            &into,
            &mut Reporter::silent(),
        ) {
            Ok(()) => anyhow::bail!("expected a sha256 mismatch"),
            Err(ModelFetchError::Sha256Mismatch { .. }) => {}
            other => anyhow::bail!("unexpected error: {other:?}"),
        }
        Ok(())
    }

    /// A `404` is not retried.
    #[test]
    fn fetch_surfaces_a_non_success_status() -> anyhow::Result<()> {
        let server = MockServer::start();
        let hit = server.mock(|when, then| {
            when.method(GET).path("/missing.gguf");
            then.status(404);
        });
        let tmp = tempfile::tempdir()?;
        let dest = tmp.path().join("missing.gguf");
        let (declared, into) = url_text_model(&server.url("/missing.gguf"), None, &dest)?;

        match fetch_model(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            &into,
            &mut Reporter::silent(),
        ) {
            Ok(()) => anyhow::bail!("expected an HTTP error"),
            Err(ModelFetchError::Http { .. }) => {}
            other => anyhow::bail!("unexpected error: {other:?}"),
        }
        hit.assert_calls(1);
        Ok(())
    }

    fn model_body() -> anyhow::Result<(Vec<u8>, String)> {
        use sha2::Digest;

        let body: Vec<u8> = (0..=255).collect();
        let sha = hex::encode(sha2::Sha256::digest(&body));
        Ok((body, sha))
    }

    struct Fetched {
        file: Vec<u8>,
        requests: Vec<String>,
        progress: Vec<u64>,
    }

    impl Fetched {
        fn request(&self, n: usize) -> &str {
            self.requests.get(n).map(String::as_str).unwrap_or_default()
        }

        /// Progress counts the file once, not once per attempt.
        fn assert_counts_file_once(&self) {
            assert_eq!(self.progress.last(), Some(&256), "{:?}", self.progress);
            assert!(
                self.progress.iter().all(|done| *done <= 256),
                "{:?}",
                self.progress
            );
        }
    }

    fn fetch_scripted(script: Vec<Reply>, pinned: bool) -> anyhow::Result<Fetched> {
        let server = ScriptedServer::start(script)?;
        fetch_url(&server.url("/model.gguf"), pinned, &server)
    }

    fn fetch_url(url: &str, pinned: bool, server: &ScriptedServer) -> anyhow::Result<Fetched> {
        let (_, sha) = model_body()?;
        let tmp = tempfile::tempdir()?;
        let dest = tmp.path().join("model.gguf");
        let pin = pinned.then_some(sha.as_str());
        let (declared, into) = url_text_model(url, pin, &dest)?;
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink_seen = Arc::clone(&seen);
        let sink: ProgressSink = Arc::new(move |p: FetchProgress<'_>| {
            if let Ok(mut seen) = sink_seen.lock() {
                seen.push(p.done_bytes);
            }
        });

        fetch_model(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            &into,
            &mut Reporter::new(Some(sink), "m".to_owned(), None),
        )?;

        let progress = seen
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned"))?
            .clone();
        Ok(Fetched {
            file: std::fs::read(&dest)?,
            requests: server
                .requests()
                .iter()
                .map(|head| head.to_ascii_lowercase())
                .collect(),
            progress,
        })
    }

    #[rstest]
    #[case::dropped_connections(vec![Reply::Drop, Reply::Drop])]
    #[case::unavailable(vec![Reply::status(503), Reply::status(503)])]
    #[case::cut_body(vec![Reply::cut(&model_body()?.0, 100)])]
    fn a_transient_download_failure_is_retried(#[case] failures: Vec<Reply>) -> anyhow::Result<()> {
        let (body, _) = model_body()?;
        let attempts = failures.len() + 1;
        let fetched = fetch_scripted(
            failures.into_iter().chain([Reply::ok(&body)]).collect(),
            true,
        )?;
        assert_eq!(fetched.file, body);
        assert_eq!(fetched.requests.len(), attempts);
        fetched.assert_counts_file_once();
        Ok(())
    }

    const ETAG_A: &str = "\"etag-a\"";

    fn cut_at_100(etag: &str) -> anyhow::Result<Reply> {
        Ok(Reply::Cut {
            status: 200,
            headers: vec![("etag", etag.to_owned())],
            body: model_body()?.0,
            sent: 100,
        })
    }

    /// Chunked, so no `Content-Length`.
    fn chunked_cut_at_100() -> anyhow::Result<Reply> {
        Ok(Reply::CutChunked {
            headers: vec![("etag", ETAG_A.to_owned())],
            body: model_body()?.0,
            sent: 100,
        })
    }

    struct RangeReply {
        first: usize,
        last: usize,
        etag: &'static str,
        content_range: Option<&'static str>,
        zeroed: bool,
        sent: Option<usize>,
    }

    fn range_reply(first: usize, last: usize) -> RangeReply {
        RangeReply {
            first,
            last,
            etag: ETAG_A,
            content_range: None,
            zeroed: false,
            sent: None,
        }
    }

    impl RangeReply {
        fn etag(self, etag: &'static str) -> Self {
            Self { etag, ..self }
        }

        fn claiming(self, content_range: &'static str) -> Self {
            Self {
                content_range: Some(content_range),
                ..self
            }
        }

        fn zeroed(self) -> Self {
            Self {
                zeroed: true,
                ..self
            }
        }

        fn cut_after(self, sent: usize) -> Self {
            Self {
                sent: Some(sent),
                ..self
            }
        }

        fn reply(self) -> anyhow::Result<Reply> {
            let body = model_body()?.0;
            let bytes = body.get(self.first..=self.last).unwrap_or_default();
            let body = if self.zeroed {
                vec![0; bytes.len()]
            } else {
                bytes.to_vec()
            };
            let content_range = self.content_range.map_or_else(
                || format!("bytes {}-{}/256", self.first, self.last),
                str::to_owned,
            );
            let headers = vec![
                ("etag", self.etag.to_owned()),
                ("content-range", content_range),
            ];
            Ok(match self.sent {
                Some(sent) => Reply::Cut {
                    status: 206,
                    headers,
                    body,
                    sent,
                },
                None => Reply::Full {
                    status: 206,
                    headers,
                    body,
                },
            })
        }
    }

    /// Only a `206` for the same file under a strong `ETag` continues; any other
    /// answer restarts from zero.
    #[rstest]
    #[case::resumed(vec![cut_at_100(ETAG_A)?, range_reply(100, 255).reply()?], true)]
    #[case::whole_file_sent(vec![cut_at_100(ETAG_A)?, Reply::ok(&model_body()?.0)], true)]
    #[case::range_refused(
        vec![cut_at_100(ETAG_A)?, Reply::status(416), Reply::ok(&model_body()?.0)],
        true
    )]
    #[case::whole_file_request_dropped(
        vec![cut_at_100(ETAG_A)?, Reply::status(416), Reply::Drop, Reply::ok(&model_body()?.0)],
        true
    )]
    #[case::another_file(
        vec![
            cut_at_100(ETAG_A)?,
            range_reply(100, 255).etag("\"etag-b\"").reply()?,
            Reply::ok(&model_body()?.0),
        ],
        true
    )]
    #[case::another_offset(
        vec![cut_at_100(ETAG_A)?, range_reply(0, 255).reply()?, Reply::ok(&model_body()?.0)],
        true
    )]
    #[case::another_length(
        vec![
            cut_at_100(ETAG_A)?,
            range_reply(100, 255).claiming("bytes 100-255/300").reply()?,
            Reply::ok(&model_body()?.0),
        ],
        true
    )]
    #[case::unknown_length(
        vec![
            cut_at_100(ETAG_A)?,
            range_reply(100, 255).claiming("bytes 100-255/*").reply()?,
            Reply::ok(&model_body()?.0),
        ],
        true
    )]
    #[case::no_etag(vec![Reply::cut(&model_body()?.0, 100), Reply::ok(&model_body()?.0)], false)]
    #[case::weak_etag(vec![cut_at_100("W/\"weak\"")?, Reply::ok(&model_body()?.0)], false)]
    fn a_broken_body_resumes_only_where_the_server_continues_it(
        #[case] script: Vec<Reply>,
        #[case] asks_for_a_range: bool,
    ) -> anyhow::Result<()> {
        let requests = script.len();
        let fetched = fetch_scripted(script, true)?;
        assert_eq!(fetched.file, model_body()?.0);
        assert_eq!(fetched.requests.len(), requests, "{:?}", fetched.requests);
        let second = fetched.request(1);
        assert_eq!(
            second.contains("range: bytes=100-"),
            asks_for_a_range,
            "{second}"
        );
        assert_eq!(
            second.contains("if-range: \"etag-a\""),
            asks_for_a_range,
            "{second}"
        );
        // After a restart, no request asks for the refused range again.
        assert!(
            fetched
                .requests
                .iter()
                .skip(2)
                .all(|request| !request.contains("range:")),
            "{:?}",
            fetched.requests
        );
        fetched.assert_counts_file_once();
        Ok(())
    }

    #[rstest]
    #[case::range_request_dropped(
        vec![cut_at_100(ETAG_A)?, Reply::Drop, range_reply(100, 255).reply()?],
        "range: bytes=100-"
    )]
    #[case::resume_cut(
        vec![
            cut_at_100(ETAG_A)?,
            range_reply(100, 255).cut_after(50).reply()?,
            range_reply(150, 255).reply()?,
        ],
        "range: bytes=150-"
    )]
    #[case::range_answered_short(
        vec![cut_at_100(ETAG_A)?, range_reply(100, 199).reply()?, range_reply(200, 255).reply()?],
        "range: bytes=200-"
    )]
    #[case::range_answered_short_after_chunked(
        vec![chunked_cut_at_100()?, range_reply(100, 199).reply()?, range_reply(200, 255).reply()?],
        "range: bytes=200-"
    )]
    fn a_resume_that_breaks_again_continues_where_it_stopped(
        #[case] script: Vec<Reply>,
        #[case] last_range: &str,
    ) -> anyhow::Result<()> {
        let fetched = fetch_scripted(script, true)?;
        assert_eq!(fetched.file, model_body()?.0);
        let last = fetched.request(fetched.requests.len().saturating_sub(1));
        assert!(last.contains(last_range), "{last}");
        fetched.assert_counts_file_once();
        Ok(())
    }

    /// A `206` can pass the resume checks and still not continue the file.
    #[rstest]
    #[case::wrong_bytes(cut_at_100(ETAG_A)?, range_reply(100, 255).zeroed().reply()?, true)]
    #[case::short_pinned(
        cut_at_100(ETAG_A)?,
        range_reply(100, 199).claiming("bytes 100-255/256").reply()?,
        true
    )]
    #[case::short_unpinned(
        cut_at_100(ETAG_A)?,
        range_reply(100, 199).claiming("bytes 100-255/256").reply()?,
        false
    )]
    #[case::short_unpinned_chunked(
        chunked_cut_at_100()?,
        range_reply(100, 199).claiming("bytes 100-255/256").reply()?,
        false
    )]
    fn a_resumed_file_that_fails_its_check_downloads_again_from_zero(
        #[case] cut: Reply,
        #[case] join: Reply,
        #[case] pinned: bool,
    ) -> anyhow::Result<()> {
        let body = model_body()?.0;
        let fetched = fetch_scripted(vec![cut, join, Reply::ok(&body)], pinned)?;
        assert_eq!(fetched.file, body);
        assert_eq!(fetched.requests.len(), 3, "{:?}", fetched.requests);
        let third = fetched.request(2);
        assert!(!third.contains("range:"), "{third}");
        fetched.assert_counts_file_once();
        Ok(())
    }

    #[test]
    fn a_digest_mismatch_names_the_first_join() -> anyhow::Result<()> {
        let script = vec![
            cut_at_100(ETAG_A)?,
            range_reply(100, 255).cut_after(50).reply()?,
            range_reply(150, 255).cut_after(50).reply()?,
            range_reply(200, 255).cut_after(50).reply()?,
            range_reply(250, 255).zeroed().reply()?,
        ];
        let Err(error) = fetch_scripted(script, true) else {
            anyhow::bail!("expected a sha256 mismatch");
        };
        match error.downcast_ref::<ModelFetchError>() {
            Some(ModelFetchError::ResumedSha256Mismatch {
                resumed_at: 100, ..
            }) => Ok(()),
            _ => anyhow::bail!("unexpected error: {error:?}"),
        }
    }

    #[rstest]
    #[case::one_response(
        ModelFetchError::Sha256Mismatch {
            dest: "model.gguf".to_owned(),
            expected: SHA_A.to_owned(),
            actual: "0".repeat(64),
        },
        Disposition::Permanent
    )]
    #[case::resumed_digest(
        ModelFetchError::ResumedSha256Mismatch {
            dest: "model.gguf".to_owned(),
            resumed_at: 100,
            expected: SHA_A.to_owned(),
            actual: "0".repeat(64),
        },
        Disposition::TRANSIENT
    )]
    #[case::resumed_length(
        ModelFetchError::ResumedLengthMismatch {
            dest: "model.gguf".to_owned(),
            resumed_at: 100,
            expected: 256,
            actual: 200,
        },
        Disposition::TRANSIENT
    )]
    fn a_check_is_retried_only_in_a_resumed_file(
        #[case] error: ModelFetchError,
        #[case] disposition: Disposition,
    ) -> anyhow::Result<()> {
        assert_eq!(error.disposition(), disposition, "{error}");
        Ok(())
    }

    /// HuggingFace redirects every file to a CDN on another host.
    #[test]
    fn a_resume_keeps_its_range_across_a_redirect() -> anyhow::Result<()> {
        let cdn = ScriptedServer::start(vec![cut_at_100(ETAG_A)?, range_reply(100, 255).reply()?])?;
        let to_cdn = || Reply::Full {
            status: 302,
            headers: vec![("location", cdn.url("/blob"))],
            body: Vec::new(),
        };
        let hub = ScriptedServer::start(vec![to_cdn(), to_cdn()])?;
        let fetched = fetch_url(&hub.url("/model.gguf"), true, &cdn)?;
        assert_eq!(fetched.file, model_body()?.0);
        let resumed = fetched.request(1);
        assert!(resumed.contains("range: bytes=100-"), "{resumed}");
        fetched.assert_counts_file_once();
        Ok(())
    }

    #[test]
    fn fetch_downloads_a_url_carrying_a_query_string() -> anyhow::Result<()> {
        // Not representable in the model URI grammar, but a valid download.
        let server = MockServer::start();
        let hit = server.mock(|when, then| {
            when.method(GET)
                .path("/file.gguf")
                .query_param("token", "abc");
            then.status(200).body("abc");
        });
        let tmp = tempfile::tempdir()?;
        let dest = tmp.path().join("file.gguf");
        let (declared, into) = url_text_model(&server.url("/file.gguf?token=abc"), None, &dest)?;

        fetch_model(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            None,
            &declared,
            &into,
            &mut Reporter::silent(),
        )?;

        hit.assert();
        Ok(())
    }

    // ---- vault sources ------------------------------------------------------

    #[test]
    fn a_vault_gguf_is_fetched_through_mgmt_and_verified() -> anyhow::Result<()> {
        use crate::vault::test_support::{
            BODY, BODY_SHA, RESOURCE_ID, access, content_path, serve,
        };

        let server = MockServer::start();
        let (record, content) = serve(&server, RESOURCE_ID, BODY_SHA, BODY);
        let declared = Model::GgufText(GgufText {
            source: GgufTextSource::VaultUrl {
                url: content_path(RESOURCE_ID)?,
                sha256: Some(Sha256::try_new(BODY_SHA.to_owned())?),
                principal_id: "org_1".parse()?,
                custom_label: None,
            },
        });
        let download = only_download(&declared)?;
        assert!(matches!(download.origin, Origin::Vault { .. }));
        assert_eq!(download.dest.as_ref(), dest(&format!("{RESOURCE_ID}.gguf")));

        let tmp = tempfile::tempdir()?;
        let into = to_stored(&declared, tmp.path())?;
        fetch_model(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            Some(&access(&server)?),
            &declared,
            &into,
            &mut Reporter::silent(),
        )?;
        record.assert();
        content.assert();
        assert_eq!(
            std::fs::read(tmp.path().join(format!("{RESOURCE_ID}.gguf")))?,
            BODY
        );
        Ok(())
    }

    fn vault_vision_files() -> anyhow::Result<GgufVisionVaultUrlFiles> {
        Ok(GgufVisionVaultUrlFiles {
            model: "/vault/resources/vault_m/content".parse()?,
            model_sha256: Some(Sha256::try_new(SHA_A.to_owned())?),
            mmproj: "/vault/resources/vault_p/content".parse()?,
            mmproj_sha256: None,
            principal_id: "org_1".parse()?,
            custom_label: None,
        })
    }

    #[rstest]
    #[case::vision(
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::VaultUrl(vault_vision_files()?),
        }),
        vec!["vault_m", "vault_p"]
    )]
    #[case::speculative(
        Model::GgufVisionSpeculative(GgufVisionSpeculative {
            source: GgufVisionSpeculativeSource::VaultUrl {
                files: vault_vision_files()?,
                draft: Some("/vault/resources/vault_d/content".parse()?),
                draft_sha256: None,
            },
        }),
        vec!["vault_m", "vault_p", "vault_d"]
    )]
    fn a_vault_vision_pair_and_its_drafter_plan_as_vault_downloads(
        #[case] model: Model,
        #[case] expected: Vec<&str>,
    ) -> anyhow::Result<()> {
        let downloads = downloads(&model)?;
        assert!(
            downloads
                .iter()
                .all(|d| matches!(d.origin, Origin::Vault { .. }))
        );
        assert_eq!(
            downloads.iter().map(Download::url).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            downloads
                .iter()
                .map(|d| d.dest.as_ref().to_owned())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|id| dest(&format!("{id}.gguf")))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            downloads
                .first()
                .and_then(|d| d.sha256.as_ref())
                .map(ToString::to_string),
            Some(SHA_A.to_owned())
        );
        Ok(())
    }

    /// Also proves its size is unknown up front (the record sizes the packed
    /// form) and a refusal on the record is permanent.
    #[test]
    fn a_vault_archive_model_is_unpacked_in_place() -> anyhow::Result<()> {
        use crate::vault::test_support::{
            RESOURCE_ID, access, content_path, serve_bytes, tar_with,
        };

        let server = MockServer::start();
        serve_bytes(&server, RESOURCE_ID, &tar_with("config.json", b"{}")?);
        let declared = Model::Mlx(Mlx {
            source: ModelSource::VaultArchive {
                url: content_path(RESOURCE_ID)?,
                sha256: None,
                principal_id: "org_1".parse()?,
                custom_label: None,
            },
        });
        let http = test_http()?;
        let device = access(&server)?;
        assert_eq!(
            declared_size_bytes(&http, &RetryPolicy::IMMEDIATE, Some(&device), &declared)?,
            None
        );

        let tmp = tempfile::tempdir()?;
        let into = to_stored(&declared, tmp.path())?;
        fetch_model(
            &http,
            &RetryPolicy::IMMEDIATE,
            Some(&device),
            &declared,
            &into,
            &mut Reporter::silent(),
        )?;
        assert_eq!(std::fs::read(tmp.path().join("config.json"))?, b"{}");

        // The fetch goes to the registered server, not the URL's host.
        let refusing = MockServer::start();
        refusing.mock(|when, then| {
            when.method(GET)
                .path(format!("/vault/resources/{RESOURCE_ID}"));
            then.status(403).body("not yours");
        });
        let err = fetch_model(
            &http,
            &RetryPolicy::IMMEDIATE,
            Some(&access(&refusing)?),
            &declared,
            &into,
            &mut Reporter::silent(),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected a refusal"))?;
        assert!(
            matches!(&err, ModelFetchError::Vault(v) if v.is_permanent()),
            "{err}"
        );
        Ok(())
    }

    #[test]
    fn a_vault_gguf_is_sized_from_its_record() -> anyhow::Result<()> {
        use crate::vault::test_support::{
            BODY, BODY_SHA, RESOURCE_ID, access, content_path, serve,
        };

        let server = MockServer::start();
        serve(&server, RESOURCE_ID, BODY_SHA, BODY);
        let declared = Model::GgufText(GgufText {
            source: GgufTextSource::VaultUrl {
                url: content_path(RESOURCE_ID)?,
                sha256: None,
                principal_id: "org_1".parse()?,
                custom_label: None,
            },
        });
        assert_eq!(
            declared_size_bytes(
                &test_http()?,
                &RetryPolicy::IMMEDIATE,
                Some(&access(&server)?),
                &declared
            )?,
            Some(3)
        );
        Ok(())
    }

    #[rstest]
    #[case::default_revision(None, "main", HF_COMMIT_A)]
    #[case::named_revision(Some("v2"), "v2", HF_COMMIT_B)]
    fn dir_plan_pins_downloads_to_the_listed_commit(
        #[case] revision: Option<&str>,
        #[case] requested_revision: &str,
        #[case] resolved_revision: &str,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        let hit = server.mock(|when, then| {
            when.method(GET).path(format!(
                "/api/models/meta/llama/revision/{requested_revision}"
            ));
            then.status(200).json_body(serde_json::json!({
                "sha": resolved_revision,
                "siblings": [
                    { "rfilename": "config.json" },
                    { "rfilename": "model.safetensors" },
                ]
            }));
        });

        let downloads = snapshot_downloads(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            &server.base_url(),
            &hf_repo(revision, None)?,
            None,
            &AbsolutePath::try_new(BASE.to_owned())?,
        )?;

        hit.assert();
        let urls: Vec<&str> = downloads.iter().map(|d| d.url()).collect();
        assert_eq!(
            urls,
            [
                format!(
                    "https://huggingface.co/meta/llama/resolve/{resolved_revision}/config.json"
                ),
                format!(
                    "https://huggingface.co/meta/llama/resolve/{resolved_revision}/model.safetensors"
                ),
            ]
        );
        assert!(
            urls.iter()
                .all(|url| !url.contains(&format!("/resolve/{requested_revision}/"))),
            "the requested movable ref must not be reused for downloads"
        );
        Ok(())
    }

    #[test]
    fn dir_plan_refuses_a_listing_without_a_sha() -> anyhow::Result<()> {
        let server = MockServer::start();
        let hit = server.mock(|when, then| {
            when.method(GET)
                .path("/api/models/meta/llama/revision/main");
            then.status(200).json_body(serde_json::json!({
                "siblings": [{ "rfilename": "config.json" }]
            }));
        });

        let result = snapshot_downloads(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            &server.base_url(),
            &hf_repo(None, None)?,
            None,
            &AbsolutePath::try_new(BASE.to_owned())?,
        );

        hit.assert();
        match result {
            Ok(downloads) => anyhow::bail!("a listing without a sha must not plan: {downloads:?}"),
            Err(e) => anyhow::ensure!(
                matches!(e, ModelFetchError::Listing { .. }),
                "unexpected error: {e:?}"
            ),
        }
        Ok(())
    }

    #[rstest]
    #[case::unavailable(Reply::status(503), 2)]
    #[case::cut_body(Reply::cut(br#"{"sha": "abc", "siblings": []}"#, 5), 2)]
    #[case::malformed(Reply::ok(b"{not json"), 1)]
    fn a_listing_retries_only_a_transient_failure(
        #[case] first: Reply,
        #[case] attempts: usize,
    ) -> anyhow::Result<()> {
        let listing = serde_json::json!({ "sha": HF_COMMIT_A, "siblings": [] }).to_string();
        let server = ScriptedServer::start(vec![first, Reply::ok(listing.as_bytes())])?;
        let info = repo_info(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            &server.url(""),
            &hf_repo(None, None)?,
        );
        assert_eq!(server.requests().len(), attempts);
        assert_eq!(
            info.as_ref().map(|info| info.sha.as_str()).ok(),
            (attempts == 2).then_some(HF_COMMIT_A),
            "{:?}",
            info.as_ref().err()
        );
        Ok(())
    }

    #[test]
    fn repo_info_sends_the_repo_bearer_token() -> anyhow::Result<()> {
        let server = MockServer::start();
        let hit = server.mock(|when, then| {
            when.method(GET)
                .path("/api/models/meta/llama/revision/main")
                .header("authorization", "Bearer hf_tok");
            then.status(200).json_body(serde_json::json!({
                "sha": HF_COMMIT_A,
                "siblings": [{ "rfilename": "config.json" }]
            }));
        });

        let info = repo_info(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            &server.base_url(),
            &hf_repo(None, Some("hf_tok"))?,
        )?;

        hit.assert();
        assert_eq!(info.sha, HF_COMMIT_A);
        let files: Vec<&str> = info.siblings.iter().map(|s| s.rfilename.as_str()).collect();
        assert_eq!(files, ["config.json"]);
        Ok(())
    }

    #[test]
    fn declared_size_bytes_walks_a_local_import() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let src = tmp.path().join("w.gguf");
        fs::write(&src, [0u8; 9000])?;
        let declared = Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: AbsolutePath::try_new(src.to_string_lossy().into_owned())?,
            },
        });

        let size = declared_size_bytes(&test_http()?, &RetryPolicy::IMMEDIATE, None, &declared)?;

        assert!(size.is_some_and(|bytes| bytes >= 9000), "{size:?}");
        Ok(())
    }

    /// httpmock rewrites `Content-Length` to 0 on a bodyless reply, so the size
    /// probe needs a server that says exactly what it is told to.
    fn serve(response: String) -> anyhow::Result<String> {
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let Ok(head) = stream.try_clone() else {
                    continue;
                };
                let mut reader = BufReader::new(head);
                let mut line = String::new();
                while reader.read_line(&mut line).is_ok_and(|read| read > 0) {
                    if line == "\r\n" {
                        break;
                    }
                    line.clear();
                }
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        Ok(format!("http://{addr}"))
    }

    #[test]
    fn the_quota_probe_base_is_absolute_on_this_platform() {
        assert!(
            Path::new(QUOTA_PROBE_BASE).is_absolute(),
            "{QUOTA_PROBE_BASE} is not absolute here"
        );
    }

    #[test]
    fn declared_size_bytes_sums_content_length_over_every_planned_download() -> anyhow::Result<()> {
        let base = serve(
            "HTTP/1.1 200 OK\r\nContent-Length: 700\r\nConnection: close\r\n\r\n".to_owned(),
        )?;
        let declared = Model::GgufVision(GgufVision {
            source: GgufVisionSource::Url(GgufVisionUrlFiles {
                model: ResourceUrl::try_new(format!("{base}/model.gguf"))?,
                model_sha256: None,
                mmproj: ResourceUrl::try_new(format!("{base}/mmproj.gguf"))?,
                mmproj_sha256: None,
            }),
        });

        assert_eq!(
            declared_size_bytes(&test_http()?, &RetryPolicy::IMMEDIATE, None, &declared)?,
            Some(1400)
        );
        Ok(())
    }

    #[test]
    fn declared_size_bytes_is_unknown_when_the_probe_fails() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::HEAD).path("/w.gguf");
            then.status(404);
        });
        let (declared, _into) =
            url_text_model(&server.url("/w.gguf"), None, Path::new("/tmp/w.gguf"))?;

        // Not a planning failure: the fetch itself will surface the 404.
        assert_eq!(
            declared_size_bytes(&test_http()?, &RetryPolicy::IMMEDIATE, None, &declared)?,
            None
        );
        Ok(())
    }

    /// `None` would skip the quota pre-flight silently and leave a post-publish
    /// sweep that cannot evict the entry it just pinned.
    #[test]
    fn declared_size_bytes_fails_when_no_plan_can_be_formed() -> anyhow::Result<()> {
        let declared = Model::GgufText(GgufText {
            source: GgufTextSource::Url {
                url: ResourceUrl::try_new("https://example.com/models/".to_owned())?,
                sha256: None,
            },
        });

        let err = declared_size_bytes(&test_http()?, &RetryPolicy::IMMEDIATE, None, &declared)
            .err()
            .ok_or_else(|| anyhow::anyhow!("an unplannable fetch must not size as unknown"))?;

        assert!(
            matches!(err, ModelFetchError::UnresolvedDestination(_)),
            "{err:?}"
        );
        Ok(())
    }

    #[test]
    fn hf_dir_size_bytes_sums_the_prefix_subtree() -> anyhow::Result<()> {
        let server = MockServer::start();
        let hit = server.mock(|when, then| {
            when.method(GET)
                .path("/api/models/meta/llama/revision/main")
                .query_param("blobs", "true");
            then.status(200).json_body(serde_json::json!({
                "sha": HF_COMMIT_A,
                "siblings": [
                    { "rfilename": "README.md", "size": 1 },
                    { "rfilename": "4bit/config.json", "size": 40 },
                    { "rfilename": "4bit/model.safetensors", "size": 4000 },
                ]
            }));
        });

        let size = hf_dir_size_bytes(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            &server.base_url(),
            &hf_repo(None, None)?,
            Some(&RepoSubpath::try_new("4bit")?),
        );

        hit.assert();
        assert_eq!(size, Some(4040), "only the prefix subtree counts");
        Ok(())
    }

    #[test]
    fn hf_dir_size_bytes_is_unknown_when_a_file_omits_its_size() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path_includes("/api/models/");
            then.status(200).json_body(serde_json::json!({
                "sha": HF_COMMIT_A,
                "siblings": [
                    { "rfilename": "config.json", "size": 40 },
                    { "rfilename": "model.safetensors" },
                ]
            }));
        });

        assert_eq!(
            hf_dir_size_bytes(
                &test_http()?,
                &RetryPolicy::IMMEDIATE,
                &server.base_url(),
                &hf_repo(None, None)?,
                None
            ),
            None
        );
        Ok(())
    }

    #[test]
    fn hf_dir_size_bytes_is_unknown_without_a_sha() -> anyhow::Result<()> {
        let server = MockServer::start();
        let hit = server.mock(|when, then| {
            when.method(GET)
                .path("/api/models/meta/llama/revision/main")
                .query_param("blobs", "true");
            then.status(200).json_body(serde_json::json!({
                "siblings": [{ "rfilename": "config.json", "size": 40 }]
            }));
        });

        let size = hf_dir_size_bytes(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            &server.base_url(),
            &hf_repo(None, None)?,
            None,
        );

        hit.assert();
        assert_eq!(size, None);
        Ok(())
    }

    #[test]
    fn repo_info_surfaces_a_refusal_and_its_reason() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path_includes("/api/models/");
            then.status(403)
                .body(r#"{"error":"Access to model meta/llama is restricted."}"#);
        });

        match repo_info(
            &test_http()?,
            &RetryPolicy::IMMEDIATE,
            &server.base_url(),
            &hf_repo(None, None)?,
        ) {
            Ok(_) => anyhow::bail!("expected an HTTP error"),
            Err(e) => anyhow::ensure!(
                matches!(&e, ModelFetchError::Http {
                    source: pipette_http::Error::Status { status, body, .. },
                    ..
                } if *status == reqwest::StatusCode::FORBIDDEN
                    && body.contains("is restricted")),
                "unexpected error: {e:?}"
            ),
        }
        Ok(())
    }
}
