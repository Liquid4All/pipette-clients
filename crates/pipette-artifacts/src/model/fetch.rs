//! Model materialization into store entries (`models/<key>/blobs/`).
//!
//! The entry point is [`fetch_model`]: it takes only the download arguments
//! (`HttpClient`, declared model, bound dest). No fetcher object.
//!
//! Both remote and local sources end at the same place: paths under the entry's
//! `blobs/`. How bytes get there differs:
//!
//! - **Remote** (URL / HuggingFace): stream HTTP into those paths.
//! - **Local**: the plan already names a path on disk outside the store
//!   (e.g. `/Users/me/w.gguf`). That file is **copied** into `blobs/` so the
//!   store owns a self-contained copy; the authoring path is left untouched.

use std::fs;
use std::io::Write;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256 as Sha256Hasher};

use pipette_http::HttpClient;
use pipette_plan_types::{
    AbsolutePath, AuthToken, GgufText, GgufTextSource, GgufVision, GgufVisionSource, HfRepo, Mlx,
    Model, ModelSource, Openvino, RepoSubpath, ResourceUrl, Sha256, Torch,
};

use super::stored::{to_stored, ModelStoredError};
use crate::progress::{copy_reporting, Reporter};

/// Why [`fetch_model`] couldn't materialize a model.
#[derive(Debug, thiserror::Error)]
pub enum ModelFetchError {
    /// This fetcher can't pull the model — a directory snapshot reaching the pure
    /// planner (local import and HTTP download are separate paths), Apple
    /// Foundation, or a declared/bound mismatch. The message is a complete,
    /// self-contained explanation.
    #[error("{0}")]
    NotFetchable(String),
    /// A resolved HuggingFace download URL wasn't a valid [`ResourceUrl`].
    #[error("invalid download URL: {0}")]
    InvalidUrl(String),
    /// A resolved on-disk destination wasn't a valid [`AbsolutePath`].
    #[error("invalid destination path: {0}")]
    InvalidDestination(String),
    /// The store-relative destinations for a model couldn't be derived at all,
    /// so there is no plan to run — a `Url` source naming no file, colliding
    /// vision leaves, a base that doesn't join into a valid path.
    #[error(transparent)]
    UnresolvedDestination(#[from] ModelStoredError),
    /// A directory model's repo (or `prefix` subtree) held no files to fetch —
    /// an empty repo or, more often, a mistyped `prefix`.
    #[error("{0}")]
    NothingToFetch(String),
    /// The downloaded bytes didn't match the declared `sha256`.
    #[error("sha256 mismatch for {dest}: expected {expected}, got {actual}")]
    Sha256Mismatch {
        dest: String,
        expected: String,
        actual: String,
    },
    /// An HTTP request (repo listing or file download) failed or returned a
    /// non-success status.
    #[error("HTTP request to {url} failed")]
    Http {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    /// Filesystem failure while materializing a model: download write (`src` is
    /// `None`) or local import (`src` is the authoring path).
    #[error("{message}")]
    Io {
        message: String,
        /// Present for local import; absent for pure download writes.
        src: Option<String>,
        dest: String,
        #[source]
        source: std::io::Error,
    },
}

/// Build [`ModelFetchError::Io`] for a write or a copy (`src` when importing).
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

/// One file to pull: where from, how to authenticate, what to verify, where to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Download {
    pub url: ResourceUrl,
    pub auth: Option<AuthToken>,
    pub sha256: Option<Sha256>,
    pub dest: AbsolutePath,
}

/// The files to pull for `declared`, landing at the `Local` paths in `into` (the
/// store's bound form, see [`to_stored`](crate::model::to_stored)).
/// Network-free — every HTTP decision is captured here so the download loop stays
/// dumb and the mapping stays testable.
pub fn plan_downloads(declared: &Model, into: &Model) -> Result<Vec<Download>, ModelFetchError> {
    match (declared, into) {
        (Model::GgufText(d), Model::GgufText(i)) => gguf_text_downloads(&d.source, &i.source),
        (Model::GgufVision(d), Model::GgufVision(i)) => gguf_vision_downloads(&d.source, &i.source),
        (Model::Mlx(_), _) | (Model::Torch(_), _) | (Model::Openvino(_), _) => {
            Err(ModelFetchError::NotFetchable(
                "directory snapshots need a network repo listing; planned by the fetcher, not here"
                    .to_owned(),
            ))
        }
        (Model::AppleFoundationText, _) => Err(ModelFetchError::NotFetchable(
            "the Apple Foundation model has no fetchable files".to_owned(),
        )),
        _ => Err(ModelFetchError::NotFetchable(
            "declared and localized model kinds disagree".to_owned(),
        )),
    }
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
        _ => {
            return Err(ModelFetchError::NotFetchable(
                "bound gguf-text is not an Absolute source".to_owned(),
            ));
        }
    };
    let download = match declared {
        GgufTextSource::HuggingFace { repo, path, sha256 } => Download {
            url: hf_file_url(repo, path)?,
            auth: repo.auth_token.clone(),
            sha256: sha256.clone(),
            dest: dest.clone(),
        },
        GgufTextSource::Url { url, sha256 } => Download {
            url: url.clone(),
            auth: None,
            sha256: sha256.clone(),
            dest: dest.clone(),
        },
        GgufTextSource::AbsoluteFile { .. } | GgufTextSource::RelativeFile { .. } => {
            return Err(ModelFetchError::NotFetchable(
                "local/relative gguf-text is imported, not downloaded".to_owned(),
            ));
        }
    };
    Ok(vec![download])
}

fn gguf_vision_downloads(
    declared: &GgufVisionSource,
    into: &GgufVisionSource,
) -> Result<Vec<Download>, ModelFetchError> {
    let (model_dest, mmproj_dest) = match into {
        GgufVisionSource::AbsoluteFiles {
            model: model_dest,
            mmproj: mmproj_dest,
        } => (model_dest, mmproj_dest),
        _ => {
            return Err(ModelFetchError::NotFetchable(
                "gguf-vision fetch dest must be abs Local".to_owned(),
            ));
        }
    };
    let downloads = match declared {
        GgufVisionSource::HuggingFace {
            repo,
            model,
            model_sha256,
            mmproj,
            mmproj_sha256,
        } => vec![
            Download {
                url: hf_file_url(repo, model)?,
                auth: repo.auth_token.clone(),
                sha256: model_sha256.clone(),
                dest: model_dest.clone(),
            },
            Download {
                url: hf_file_url(repo, mmproj)?,
                auth: repo.auth_token.clone(),
                sha256: mmproj_sha256.clone(),
                dest: mmproj_dest.clone(),
            },
        ],
        GgufVisionSource::Url {
            model,
            model_sha256,
            mmproj,
            mmproj_sha256,
        } => vec![
            Download {
                url: model.clone(),
                auth: None,
                sha256: model_sha256.clone(),
                dest: model_dest.clone(),
            },
            Download {
                url: mmproj.clone(),
                auth: None,
                sha256: mmproj_sha256.clone(),
                dest: mmproj_dest.clone(),
            },
        ],
        GgufVisionSource::AbsoluteFiles { .. } | GgufVisionSource::RelativeFiles { .. } => {
            return Err(ModelFetchError::NotFetchable(
                "local/relative gguf-vision is imported, not downloaded".to_owned(),
            ));
        }
    };
    Ok(downloads)
}

/// The public file-download URL for a repo-relative `path`. An unpinned repo
/// resolves against `main`.
fn hf_file_url(repo: &HfRepo, path: &RepoSubpath) -> Result<ResourceUrl, ModelFetchError> {
    let revision = repo.revision.as_ref().map_or("main", AsRef::as_ref);
    hf_resolve_url(repo, revision, path.as_ref())
}

/// `https://huggingface.co/<org>/<repo>/resolve/<revision>/<path>` — takes the
/// path as a raw string so it also serves repo-listing entries (which aren't
/// [`RepoSubpath`]s).
fn hf_resolve_url(
    repo: &HfRepo,
    revision: &str,
    path: &str,
) -> Result<ResourceUrl, ModelFetchError> {
    let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{path}");
    ResourceUrl::try_new(url.clone()).map_err(|_| ModelFetchError::InvalidUrl(url))
}

/// Map a HuggingFace repo `listing` to the files to pull for a directory model.
/// With a `prefix`, only that subtree is pulled and the prefix is stripped so the
/// files land directly under `dest_dir` (which already ends in the prefix);
/// without one, the whole snapshot lands under `dest_dir`. Directory models carry
/// no per-file `sha256`, so none is verified. Network-free — the caller supplies
/// the listing.
fn plan_dir_downloads(
    repo: &HfRepo,
    prefix: Option<&RepoSubpath>,
    dest_dir: &AbsolutePath,
    listing: &[String],
) -> Result<Vec<Download>, ModelFetchError> {
    let revision = repo.revision.as_ref().map_or("main", AsRef::as_ref);
    let prefix_slash = prefix.map(|prefix| format!("{}/", prefix.as_ref()));
    let downloads: Vec<Download> = listing
        .iter()
        .filter_map(|rfilename| {
            // A prefix keeps only its subtree; the stripped tail is the local layout.
            let relative = match &prefix_slash {
                Some(prefix) => rfilename.strip_prefix(prefix)?,
                None => rfilename.as_str(),
            };
            Some((rfilename.as_str(), relative.to_owned()))
        })
        .map(|(rfilename, relative)| {
            Ok(Download {
                url: hf_resolve_url(repo, revision, rfilename)?,
                auth: repo.auth_token.clone(),
                sha256: None,
                dest: local_join(dest_dir, &relative)?,
            })
        })
        .collect::<Result<_, ModelFetchError>>()?;
    // An empty plan means an empty repo or — far more likely — a `prefix` that
    // matches nothing (a typo). Fetching it would publish an empty model dir that
    // every later `ensure` resolves as valid, so reject it loudly here.
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

/// `dir/relative` as a [`AbsolutePath`].
fn local_join(dir: &AbsolutePath, relative: &str) -> Result<AbsolutePath, ModelFetchError> {
    let joined = format!("{}/{}", dir.as_ref(), relative);
    AbsolutePath::try_new(joined.clone()).map_err(|_| ModelFetchError::InvalidDestination(joined))
}

/// The projection of the HuggingFace model-info API this fetcher needs: the
/// repo's file list.
#[derive(Deserialize)]
struct HfModelInfo {
    siblings: Vec<HfSibling>,
}

#[derive(Deserialize)]
struct HfSibling {
    rfilename: String,
    /// Only returned when the listing is requested with `?blobs=true`.
    size: Option<u64>,
}

/// The public HuggingFace host the repo-listing request targets.
const HF_ENDPOINT: &str = "https://huggingface.co";

/// Materialize `declared` at the store-bound paths in `into`.
///
/// Local plan paths are copied; remote sources are downloaded through `http`.
/// On failure (network or `sha256` mismatch) a partial file may remain at the
/// dest; the store stages into a dir it discards on error, so nothing corrupt is
/// published.
///
/// `reporter` hears every byte the download path writes. A local import reports
/// nothing: `fs::copy` moves the file in one call with no count to observe, and a
/// copy that finishes before the first redraw would only flash.
pub(crate) fn fetch_model(
    http: &HttpClient,
    declared: &Model,
    into: &Model,
    reporter: &mut Reporter,
) -> Result<(), ModelFetchError> {
    fetch_model_with_hf_endpoint(http, HF_ENDPOINT, declared, into, reporter)
}

/// Like [`fetch_model`], but HF repo listing hits `hf_endpoint` (mock servers).
fn fetch_model_with_hf_endpoint(
    http: &HttpClient,
    hf_endpoint: &str,
    declared: &Model,
    into: &Model,
    reporter: &mut Reporter,
) -> Result<(), ModelFetchError> {
    // Local plan paths live outside the store. Copy them into `into`
    // (absolute paths under the staged entry's blobs/) — same endpoint as
    // a download, just disk→disk instead of network→disk.
    // e.g. /Users/me/w.gguf → …/models/.staging/…/blobs/w.gguf
    match (declared, into) {
        (
            Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile { path: src },
            }),
            Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile { path: dest },
            }),
        ) => {
            copy_file(Path::new(src.as_ref()), Path::new(dest.as_ref()))?;
            return Ok(());
        }
        (
            Model::GgufVision(GgufVision {
                source:
                    GgufVisionSource::AbsoluteFiles {
                        model: src_model,
                        mmproj: src_mmproj,
                    },
            }),
            Model::GgufVision(GgufVision {
                source:
                    GgufVisionSource::AbsoluteFiles {
                        model: dest_model,
                        mmproj: dest_mmproj,
                    },
            }),
        ) => {
            copy_file(
                Path::new(src_model.as_ref()),
                Path::new(dest_model.as_ref()),
            )?;
            copy_file(
                Path::new(src_mmproj.as_ref()),
                Path::new(dest_mmproj.as_ref()),
            )?;
            return Ok(());
        }
        (
            Model::Mlx(Mlx {
                source: ModelSource::AbsoluteDir { dir: src },
            }),
            Model::Mlx(Mlx {
                source: ModelSource::AbsoluteDir { dir: dest },
            }),
        )
        | (
            Model::Torch(Torch {
                source: ModelSource::AbsoluteDir { dir: src },
            }),
            Model::Torch(Torch {
                source: ModelSource::AbsoluteDir { dir: dest },
            }),
        )
        | (
            Model::Openvino(Openvino {
                source: ModelSource::AbsoluteDir { dir: src },
            }),
            Model::Openvino(Openvino {
                source: ModelSource::AbsoluteDir { dir: dest },
            }),
        ) => {
            copy_tree(Path::new(src.as_ref()), Path::new(dest.as_ref()))?;
            return Ok(());
        }
        (Model::AppleFoundationText, _) => {
            return Err(ModelFetchError::NotFetchable(
                "the Apple Foundation model has no importable files".to_owned(),
            ));
        }
        // Remote declared + Local into: download path below.
        _ => {}
    }
    plan(http, hf_endpoint, declared, into)?
        .iter()
        .try_for_each(|download| download_one(http, download, reporter))?;
    Ok(())
}

/// The files to pull for `declared`. Single-file sources map purely;
/// directory snapshots (`Mlx`/`Torch`/`Openvino`) list the HuggingFace repo
/// first.
fn plan(
    http: &HttpClient,
    hf_endpoint: &str,
    declared: &Model,
    into: &Model,
) -> Result<Vec<Download>, ModelFetchError> {
    match (declared, into) {
        (Model::Mlx(d), Model::Mlx(i)) => plan_dir(http, hf_endpoint, &d.source, &i.source),
        (Model::Torch(d), Model::Torch(i)) => plan_dir(http, hf_endpoint, &d.source, &i.source),
        (Model::Openvino(d), Model::Openvino(i)) => {
            plan_dir(http, hf_endpoint, &d.source, &i.source)
        }
        _ => plan_downloads(declared, into),
    }
}

fn plan_dir(
    http: &HttpClient,
    hf_endpoint: &str,
    declared: &ModelSource,
    into: &ModelSource,
) -> Result<Vec<Download>, ModelFetchError> {
    let ModelSource::AbsoluteDir { dir: dest_dir } = into else {
        return Err(ModelFetchError::NotFetchable(
            "bound directory model is not an Absolute source".to_owned(),
        ));
    };
    match declared {
        ModelSource::HuggingFace { repo, prefix } => {
            let listing = list_repo_files(http, hf_endpoint, repo)?;
            plan_dir_downloads(repo, prefix.as_ref(), dest_dir, &listing)
        }
        ModelSource::AbsoluteDir { .. } | ModelSource::RelativeDir { .. } => {
            Err(ModelFetchError::NotFetchable(
                "local/relative directory models are imported, not downloaded".to_owned(),
            ))
        }
    }
}

/// The HuggingFace model-info record for `repo`. An unpinned repo reads `main`.
/// The endpoint returns the full `siblings` list in one response (not
/// paginated), so a single request is complete regardless of repo size.
/// `with_blobs` asks for per-file sizes, which only the quota pre-flight needs.
fn repo_info(
    http: &HttpClient,
    hf_endpoint: &str,
    repo: &HfRepo,
    with_blobs: bool,
) -> Result<HfModelInfo, ModelFetchError> {
    let revision = repo.revision.as_ref().map_or("main", AsRef::as_ref);
    let blobs = if with_blobs { "?blobs=true" } else { "" };
    let url = format!("{hf_endpoint}/api/models/{repo}/revision/{revision}{blobs}");
    let mut request = http.client().get(&url);
    if let Some(token) = &repo.auth_token {
        request = request.bearer_auth(token.as_ref());
    }
    request
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .and_then(|response| response.json())
        .map_err(|source| ModelFetchError::Http { url, source })
}

/// The repo-relative paths of every file in `repo`.
fn list_repo_files(
    http: &HttpClient,
    hf_endpoint: &str,
    repo: &HfRepo,
) -> Result<Vec<String>, ModelFetchError> {
    Ok(repo_info(http, hf_endpoint, repo, false)?
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .collect())
}

/// `Content-Length` for `url`, from a HEAD. `None` whenever the server declines
/// to say — the caller then has no size to enforce against.
///
/// A failed HEAD is not fatal: servers exist that refuse HEAD and serve the GET
/// the fetch will make anyway, so a probe failure must not decide the resolve.
/// It is logged, though — it is the one `None` here that isn't the server
/// answering, and it is why the quota pre-flight has nothing to check.
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
    request
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .inspect_err(|source| {
            log::warn!(
                "sizing {url} failed ({source}); the storage quota cannot be checked before the \
                 fetch"
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

/// Bytes `declared` will occupy once fetched, when that is knowable before the
/// fetch: an exact walk for a local import, `Content-Length` for single-file
/// downloads, the HuggingFace blob listing for a directory snapshot.
///
/// `Ok(None)` is "nobody could say how big this is" — the server declined, the
/// listing omitted sizes, the source has no installer to size — and leaves the
/// post-publish sweep as the only enforcement. `Err` is the narrower case where
/// the fetch could not even be *planned*, and it is an error rather than another
/// `None` because the two are not interchangeable to the caller: a `None` skips
/// the quota pre-flight silently, letting the fetch run unchecked into a sweep
/// that cannot evict the entry it just pinned. Planning here goes through the
/// same [`to_stored`] and [`plan_downloads`] the fetch will, so an `Err` is the
/// fetch's own failure surfacing one step early.
pub(crate) fn declared_size_bytes(
    http: &HttpClient,
    declared: &Model,
) -> Result<Option<u64>, ModelFetchError> {
    let walk = |path: &AbsolutePath| crate::entry::entry_size_bytes(Path::new(path.as_ref()));
    Ok(match declared {
        // OS-bundled: nothing lands in the store.
        Model::AppleFoundationText => Some(0),
        Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile { path },
        }) => Some(walk(path)),
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::AbsoluteFiles { model, mmproj },
        }) => Some(walk(model).saturating_add(walk(mmproj))),
        Model::Mlx(Mlx {
            source: ModelSource::AbsoluteDir { dir },
        })
        | Model::Torch(Torch {
            source: ModelSource::AbsoluteDir { dir },
        })
        | Model::Openvino(Openvino {
            source: ModelSource::AbsoluteDir { dir },
        }) => Some(walk(dir)),
        Model::Mlx(Mlx { source })
        | Model::Torch(Torch { source })
        | Model::Openvino(Openvino { source }) => hf_dir_size_bytes(http, HF_ENDPOINT, source),
        Model::GgufText(_) | Model::GgufVision(_) => remote_files_size_bytes(http, declared)?,
    })
}

/// Absolute base for the size probe's throwaway dest.
///
/// Never written to: it exists only so [`to_stored`] resolves to its `Absolute*`
/// arms, which `plan_downloads` requires and refuses a relative dest for. What
/// decides that is `Path::is_absolute`, and its answer is per-platform — a bare
/// `/quota-probe` has a root but no drive prefix, so Windows reads it as
/// relative. Spelled per platform rather than Unix-shaped for that reason: the
/// Unix form made every remote size probe on Windows fail to plan, dropping the
/// pre-fetch quota check to the post-publish sweep.
#[cfg(windows)]
const QUOTA_PROBE_BASE: &str = r"C:\quota-probe";
#[cfg(not(windows))]
const QUOTA_PROBE_BASE: &str = "/quota-probe";

/// Total `Content-Length` over every file a single-file source would download.
fn remote_files_size_bytes(
    http: &HttpClient,
    declared: &Model,
) -> Result<Option<u64>, ModelFetchError> {
    // Only the URLs matter here — the probe dest is never written. Going
    // through `plan_downloads` keeps URL and auth derivation single-sourced,
    // and it is why both failures propagate: the fetch plans the same way
    // against the staging base, so anything that stops a plan forming here
    // stops one forming there. Swallowing it would trade a real error for a
    // skipped quota check.
    let into = to_stored(declared, Path::new(QUOTA_PROBE_BASE))?;
    Ok(plan_downloads(declared, &into)?
        .iter()
        .try_fold(0u64, |total, download| {
            content_length(http, download.url.as_ref(), download.auth.as_ref())
                .map(|bytes| total.saturating_add(bytes))
        }))
}

/// Total blob size of the repo subtree a directory model would snapshot.
/// `None` if the repo isn't a HuggingFace source or any file omits its size.
fn hf_dir_size_bytes(http: &HttpClient, hf_endpoint: &str, source: &ModelSource) -> Option<u64> {
    let ModelSource::HuggingFace { repo, prefix } = source else {
        return None;
    };
    let prefix_slash = prefix.as_ref().map(|p| format!("{}/", p.as_ref()));
    // Not fatal, for the same reason a failed HEAD isn't: the fetch makes its
    // own listing request and will report the failure itself if it persists.
    let info = repo_info(http, hf_endpoint, repo, true)
        .inspect_err(|source| {
            log::warn!(
                "sizing {repo} failed ({source}); the storage quota cannot be checked before the \
                 fetch"
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

fn download_one(
    http: &HttpClient,
    download: &Download,
    reporter: &mut Reporter,
) -> Result<(), ModelFetchError> {
    let mut request = http.client().get(download.url.as_ref());
    if let Some(token) = &download.auth {
        request = request.bearer_auth(token.as_ref());
    }
    let response = request
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|source| ModelFetchError::Http {
            url: download.url.to_string(),
            source,
        })?;

    let dest = Path::new(download.dest.as_ref());
    let to_io = |source| io_err(None, dest, source);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(to_io)?;
    }
    let actual = stream_to_file(response, dest, reporter).map_err(to_io)?;

    if let Some(expected) = &download.sha256 {
        if actual != expected.as_ref() {
            return Err(ModelFetchError::Sha256Mismatch {
                dest: download.dest.to_string(),
                expected: expected.to_string(),
                actual,
            });
        }
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

/// Recursively copy `src` into `dest` (creating `dest` if needed).
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

/// Stream `response` into a new file at `dest`, returning the hex SHA-256 of the
/// bytes written. Hashes inline so a multi-GB model never buffers in memory.
fn stream_to_file(
    mut response: reqwest::blocking::Response,
    dest: &Path,
    reporter: &mut Reporter,
) -> std::io::Result<String> {
    let mut writer = HashingWriter::new(fs::File::create(dest)?);
    // Named for the reader, not the store: the staged destination is a
    // content-addressed directory nobody asked about.
    let file = dest
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    copy_reporting(&mut response, &mut writer, &file, reporter)?;
    writer.flush()?;
    Ok(writer.hex_digest())
}

/// A [`Write`] that hashes every byte it forwards, so a streamed download can be
/// verified without a second pass over the file.
struct HashingWriter<W> {
    inner: W,
    hasher: Sha256Hasher,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256Hasher::new(),
        }
    }

    fn hex_digest(self) -> String {
        hex::encode(self.hasher.finalize())
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // Hash only the bytes the inner writer accepted, so a short write can't
        // desync the digest from what landed on disk.
        let written = self.inner.write(buf)?;
        self.hasher.update(&buf[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;
    use rstest::rstest;

    use pipette_plan_types::{GgufText, GgufVision, HfOrg, HfRepoName, HfRevision, Mlx};

    use super::*;

    const SHA_A: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    /// Store base the planning tests re-home into. Platform-shaped for the same
    /// reason [`QUOTA_PROBE_BASE`] is: `to_stored` picks its `Absolute*` arms off
    /// `Path::is_absolute`, whose answer is per-platform, so a bare
    /// `/store/entry/blobs` is read as relative on Windows and every re-home
    /// through it fails `RelativePath` validation instead of testing anything.
    ///
    /// Note the asymmetry that makes this easy to miss: `AbsolutePath`'s own
    /// validator is a platform-independent string check that accepts the Unix
    /// spelling everywhere, so tests handing a literal straight to
    /// `AbsolutePath::try_new` pass on Windows — only the ones routed through
    /// `to_stored` break.
    ///
    /// Spelled with forward slashes on both platforms because `to_stored`
    /// normalizes separators to `/`; only the drive prefix differs, which is what
    /// lets [`dest`] compose expectations from one string.
    #[cfg(windows)]
    const BASE: &str = "C:/store/entry/blobs";
    #[cfg(not(windows))]
    const BASE: &str = "/store/entry/blobs";

    fn base() -> &'static Path {
        Path::new(BASE)
    }

    /// The dest a planned download carries for `relative` under [`BASE`], spelled
    /// the way the planner spells it — separators normalized to `/`.
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

    fn hf_text(revision: Option<&str>, auth: Option<&str>) -> anyhow::Result<Model> {
        Ok(Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: hf_repo(revision, auth)?,
                path: RepoSubpath::try_new("Q4.gguf")?,
                sha256: Some(Sha256::try_new(SHA_A.to_owned())?),
            },
        }))
    }

    /// Plans exactly one download for `declared`, or fails the test.
    fn only_download(declared: &Model) -> anyhow::Result<Download> {
        let into = to_stored(declared, base())?;
        let mut downloads = plan_downloads(declared, &into)?;
        anyhow::ensure!(downloads.len() == 1, "expected a single download");
        Ok(downloads.remove(0))
    }

    #[test]
    fn hf_text_resolves_to_a_resolve_url_with_auth_and_sha() -> anyhow::Result<()> {
        let download = only_download(&hf_text(None, Some("hf_tok"))?)?;
        assert_eq!(
            download.url.as_ref(),
            "https://huggingface.co/meta/llama/resolve/main/Q4.gguf"
        );
        assert_eq!(download.auth.as_ref().map(AsRef::as_ref), Some("hf_tok"));
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
            download.url.as_ref(),
            format!("https://huggingface.co/meta/llama/resolve/{expected}/Q4.gguf")
        );
        assert!(download.auth.is_none(), "no token on a public repo");
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
        assert_eq!(download.url.as_ref(), "https://ex.com/dir/w.gguf");
        assert!(download.auth.is_none());
        assert!(download.sha256.is_none());
        Ok(())
    }

    #[test]
    fn hf_vision_plans_model_then_mmproj() -> anyhow::Result<()> {
        let declared = Model::GgufVision(GgufVision {
            source: GgufVisionSource::HuggingFace {
                repo: hf_repo(None, None)?,
                model: RepoSubpath::try_new("model.gguf")?,
                model_sha256: None,
                mmproj: RepoSubpath::try_new("mmproj.gguf")?,
                mmproj_sha256: None,
            },
        });
        let into = to_stored(&declared, base())?;
        let downloads = plan_downloads(&declared, &into)?;
        let urls: Vec<&str> = downloads.iter().map(|d| d.url.as_ref()).collect();
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
            source: GgufVisionSource::Url {
                model: ResourceUrl::try_new("https://ex.com/model.gguf")?,
                model_sha256: None,
                mmproj: ResourceUrl::try_new("https://ex.com/mmproj.gguf")?,
                mmproj_sha256: None,
            },
        });
        let into = to_stored(&declared, base())?;
        let downloads = plan_downloads(&declared, &into)?;
        let urls: Vec<&str> = downloads.iter().map(|d| d.url.as_ref()).collect();
        assert_eq!(
            urls,
            ["https://ex.com/model.gguf", "https://ex.com/mmproj.gguf"]
        );
        assert!(
            downloads.iter().all(|d| d.auth.is_none()),
            "URL sources carry no token"
        );
        let dests: Vec<&str> = downloads.iter().map(|d| d.dest.as_ref()).collect();
        assert_eq!(dests, [dest("model.gguf"), dest("mmproj.gguf")]);
        Ok(())
    }

    #[test]
    fn apple_foundation_is_unsupported() -> anyhow::Result<()> {
        let declared = Model::AppleFoundationText;
        assert!(matches!(
            plan_downloads(&declared, &declared),
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
        fetch_model(&test_http()?, &declared, &into, &mut Reporter::silent())?;
        assert_eq!(fs::read(&dest)?, b"gguf-bytes");
        // Authoring path is left in place (copy, not move).
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
            source: GgufVisionSource::AbsoluteFiles {
                model: AbsolutePath::try_new(src_model.to_string_lossy().into_owned())?,
                mmproj: AbsolutePath::try_new(src_mmproj.to_string_lossy().into_owned())?,
            },
        });
        let into = Model::GgufVision(GgufVision {
            source: GgufVisionSource::AbsoluteFiles {
                model: AbsolutePath::try_new(dest_model.to_string_lossy().into_owned())?,
                mmproj: AbsolutePath::try_new(dest_mmproj.to_string_lossy().into_owned())?,
            },
        });
        fetch_model(&test_http()?, &declared, &into, &mut Reporter::silent())?;
        assert_eq!(fs::read(&dest_model)?, b"model");
        assert_eq!(fs::read(&dest_mmproj)?, b"mmproj");
        Ok(())
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
        match fetch_model(&test_http()?, &declared, &into, &mut Reporter::silent()) {
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

    #[test]
    fn fetch_copies_a_local_dir_into_store_paths() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let src = tmp.path().join("src/weights");
        fs::create_dir_all(src.join("sub"))?;
        fs::write(src.join("config.json"), b"{}")?;
        fs::write(src.join("sub/w.safetensors"), b"tensor")?;
        let dest = tmp.path().join("store/blobs");

        let declared = Model::Mlx(Mlx {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(src.to_string_lossy().into_owned())?,
            },
        });
        let into = Model::Mlx(Mlx {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(dest.to_string_lossy().into_owned())?,
            },
        });
        fetch_model(&test_http()?, &declared, &into, &mut Reporter::silent())?;
        assert_eq!(fs::read(dest.join("config.json"))?, b"{}");
        assert_eq!(fs::read(dest.join("sub/w.safetensors"))?, b"tensor");
        Ok(())
    }

    #[test]
    fn the_pure_planner_defers_dir_snapshots() -> anyhow::Result<()> {
        // A `Mlx`/`Torch` repo needs a network listing, so the pure `plan_downloads`
        // can't handle it — the fetcher's `plan_dir` path does (see below).
        let declared = Model::Mlx(Mlx {
            source: ModelSource::HuggingFace {
                repo: hf_repo(None, None)?,
                prefix: None,
            },
        });
        let into = to_stored(&declared, base())?;
        assert!(matches!(
            plan_downloads(&declared, &into),
            Err(ModelFetchError::NotFetchable(_))
        ));
        Ok(())
    }

    #[test]
    fn dir_downloads_map_the_whole_snapshot() -> anyhow::Result<()> {
        let repo = hf_repo(None, Some("hf_tok"))?;
        let dest = AbsolutePath::try_new("/store/entry/blobs".to_owned())?;
        let listing = [
            "config.json".to_owned(),
            "model.safetensors".to_owned(),
            "tokenizer/vocab.json".to_owned(),
        ];
        let downloads = plan_dir_downloads(&repo, None, &dest, &listing)?;
        let urls: Vec<&str> = downloads.iter().map(|d| d.url.as_ref()).collect();
        assert_eq!(
            urls,
            [
                "https://huggingface.co/meta/llama/resolve/main/config.json",
                "https://huggingface.co/meta/llama/resolve/main/model.safetensors",
                "https://huggingface.co/meta/llama/resolve/main/tokenizer/vocab.json",
            ]
        );
        assert!(
            downloads
                .iter()
                .all(|d| d.auth.as_ref().map(AsRef::as_ref) == Some("hf_tok")),
            "the repo token rides every file"
        );
        assert!(
            downloads.iter().all(|d| d.sha256.is_none()),
            "dir models declare no per-file sha256"
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
        // `dest_dir` already ends in the prefix (that's what `to_stored` produces).
        let dest = AbsolutePath::try_new("/store/entry/blobs/4bit".to_owned())?;
        let listing = [
            "README.md".to_owned(), // outside the subtree — dropped
            "4bit/config.json".to_owned(),
            "4bit/weights/model.safetensors".to_owned(),
        ];
        let downloads = plan_dir_downloads(&repo, Some(&prefix), &dest, &listing)?;
        let urls: Vec<&str> = downloads.iter().map(|d| d.url.as_ref()).collect();
        assert_eq!(
            urls,
            [
                "https://huggingface.co/meta/llama/resolve/v2/4bit/config.json",
                "https://huggingface.co/meta/llama/resolve/v2/4bit/weights/model.safetensors",
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
        let listing = ["README.md".to_owned(), "4bit/config.json".to_owned()];
        assert!(matches!(
            plan_dir_downloads(&repo, Some(&prefix), &dest, &listing),
            Err(ModelFetchError::NothingToFetch(_))
        ));
        Ok(())
    }

    #[test]
    fn an_empty_repo_listing_is_rejected() -> anyhow::Result<()> {
        let repo = hf_repo(None, None)?;
        let dest = AbsolutePath::try_new("/store/entry/blobs".to_owned())?;
        assert!(matches!(
            plan_dir_downloads(&repo, None, &dest, &[]),
            Err(ModelFetchError::NothingToFetch(_))
        ));
        Ok(())
    }

    #[test]
    fn a_traversal_filename_cannot_escape_the_model_dir() -> anyhow::Result<()> {
        // A hostile/compromised repo listing can't write outside dest_dir:
        // `AbsolutePath` rejects `..` segments, so it surfaces as InvalidDestination.
        let repo = hf_repo(None, None)?;
        let dest = AbsolutePath::try_new("/store/entry/blobs".to_owned())?;
        let listing = ["../../etc/passwd".to_owned()];
        assert!(matches!(
            plan_dir_downloads(&repo, None, &dest, &listing),
            Err(ModelFetchError::InvalidDestination(_))
        ));
        Ok(())
    }

    #[test]
    fn hashing_writer_matches_sha256_of_the_bytes() -> anyhow::Result<()> {
        // SHA-256("abc") is a NIST test vector — proves the streamed digest.
        let mut writer = HashingWriter::new(Vec::new());
        writer.write_all(b"abc")?;
        writer.flush()?;
        assert_eq!(writer.hex_digest(), SHA_A);
        Ok(())
    }

    // ---- network path: the real reqwest HTTP against a mock server ----------
    //
    // `plan_downloads` / `plan_dir_downloads` are covered purely above; these
    // drive `fetch` (download_one → stream_to_file → sha verify) and
    // `list_repo_files` (GET → JSON → siblings) over an actual socket. The
    // client is the standard TLS-configured one, which also serves plain HTTP.

    fn test_http() -> anyhow::Result<HttpClient> {
        Ok(HttpClient::new("pipette-test")?)
    }

    /// A `(declared Url source, bound Absolute dest)` pair for `fetch`. A `Url`
    /// source passes the URL through verbatim, so it needs no endpoint override.
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

        fetch_model(&test_http()?, &declared, &into, &mut Reporter::silent())?;

        hit.assert();
        assert_eq!(std::fs::read_to_string(&dest)?, "abc");
        Ok(())
    }

    #[test]
    fn fetch_rejects_a_sha256_mismatch() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/w.gguf");
            then.status(200).body("xyz"); // != the declared SHA_A
        });
        let tmp = tempfile::tempdir()?;
        let dest = tmp.path().join("w.gguf");
        let (declared, into) = url_text_model(&server.url("/w.gguf"), Some(SHA_A), &dest)?;

        match fetch_model(&test_http()?, &declared, &into, &mut Reporter::silent()) {
            Ok(()) => anyhow::bail!("expected a sha256 mismatch"),
            Err(ModelFetchError::Sha256Mismatch { .. }) => {}
            other => anyhow::bail!("unexpected error: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn fetch_surfaces_a_non_success_status() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/missing.gguf");
            then.status(404);
        });
        let tmp = tempfile::tempdir()?;
        let dest = tmp.path().join("missing.gguf");
        let (declared, into) = url_text_model(&server.url("/missing.gguf"), None, &dest)?;

        match fetch_model(&test_http()?, &declared, &into, &mut Reporter::silent()) {
            Ok(()) => anyhow::bail!("expected an HTTP error"),
            Err(ModelFetchError::Http { .. }) => {}
            other => anyhow::bail!("unexpected error: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn fetch_downloads_a_url_carrying_a_query_string() -> anyhow::Result<()> {
        // A direct `Url` source may carry a query string (not representable in the
        // model URI grammar, but a valid download); it must reach the server intact.
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

        fetch_model(&test_http()?, &declared, &into, &mut Reporter::silent())?;

        hit.assert();
        Ok(())
    }

    #[test]
    fn list_repo_files_parses_the_siblings_listing() -> anyhow::Result<()> {
        let server = MockServer::start();
        let hit = server.mock(|when, then| {
            when.method(GET)
                .path("/api/models/meta/llama/revision/main");
            then.status(200).json_body(serde_json::json!({
                "siblings": [
                    { "rfilename": "config.json" },
                    { "rfilename": "model.safetensors" },
                ]
            }));
        });

        let files = list_repo_files(&test_http()?, &server.base_url(), &hf_repo(None, None)?)?;

        hit.assert();
        assert_eq!(files, ["config.json", "model.safetensors"]);
        Ok(())
    }

    #[test]
    fn list_repo_files_sends_the_repo_bearer_token() -> anyhow::Result<()> {
        let server = MockServer::start();
        let hit = server.mock(|when, then| {
            when.method(GET)
                .path("/api/models/meta/llama/revision/main")
                .header("authorization", "Bearer hf_tok");
            then.status(200)
                .json_body(serde_json::json!({ "siblings": [{ "rfilename": "config.json" }] }));
        });

        let files = list_repo_files(
            &test_http()?,
            &server.base_url(),
            &hf_repo(None, Some("hf_tok"))?,
        )?;

        hit.assert();
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

        let size = declared_size_bytes(&test_http()?, &declared)?;

        assert!(size.is_some_and(|bytes| bytes >= 9000), "{size:?}");
        Ok(())
    }

    /// A socket that answers every request with `response`. httpmock rewrites
    /// `Content-Length` to 0 on a bodyless reply, so the size probe needs a
    /// server that says exactly what it is told to.
    fn serve(response: String) -> anyhow::Result<String> {
        use std::io::{BufRead, BufReader};
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

    /// The probe base has to be absolute by the running platform's rule, not by
    /// looking Unix-shaped. `plan_downloads` refuses a relative dest, so a base
    /// the platform reads as relative fails every remote size probe — now a
    /// reported error rather than a silent `None`, but still a quota check that
    /// never ran. Cheap to assert here, and it fails on the platform that would
    /// break.
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
            source: GgufVisionSource::Url {
                model: ResourceUrl::try_new(format!("{base}/model.gguf"))?,
                model_sha256: None,
                mmproj: ResourceUrl::try_new(format!("{base}/mmproj.gguf"))?,
                mmproj_sha256: None,
            },
        });

        assert_eq!(declared_size_bytes(&test_http()?, &declared)?, Some(1400));
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

        // A server that won't answer a HEAD is not a planning failure: the plan
        // formed, the size just isn't in it. Nothing to refuse against, so the
        // post-publish sweep becomes the only enforcement — and the fetch itself
        // will surface the 404.
        assert_eq!(declared_size_bytes(&test_http()?, &declared)?, None);
        Ok(())
    }

    /// The counterpart: when no plan can be formed, the size probe must not
    /// answer `None`. `None` skips the quota pre-flight silently and hands the
    /// job to a post-publish sweep that cannot evict the entry it just pinned,
    /// so the failure has to reach the caller. This URL names no file, which is
    /// the same wall the fetch would hit against the staging base.
    #[test]
    fn declared_size_bytes_fails_when_no_plan_can_be_formed() -> anyhow::Result<()> {
        let declared = Model::GgufText(GgufText {
            source: GgufTextSource::Url {
                url: ResourceUrl::try_new("https://example.com/models/".to_owned())?,
                sha256: None,
            },
        });

        let err = declared_size_bytes(&test_http()?, &declared)
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
                "siblings": [
                    { "rfilename": "README.md", "size": 1 },
                    { "rfilename": "4bit/config.json", "size": 40 },
                    { "rfilename": "4bit/model.safetensors", "size": 4000 },
                ]
            }));
        });
        let source = ModelSource::HuggingFace {
            repo: hf_repo(None, None)?,
            prefix: Some(RepoSubpath::try_new("4bit")?),
        };

        let size = hf_dir_size_bytes(&test_http()?, &server.base_url(), &source);

        hit.assert();
        assert_eq!(size, Some(4040), "only the prefix subtree counts");
        Ok(())
    }

    #[test]
    fn hf_dir_size_bytes_is_unknown_when_a_file_omits_its_size() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path_contains("/api/models/");
            then.status(200).json_body(serde_json::json!({
                "siblings": [
                    { "rfilename": "config.json", "size": 40 },
                    { "rfilename": "model.safetensors" },
                ]
            }));
        });
        let source = ModelSource::HuggingFace {
            repo: hf_repo(None, None)?,
            prefix: None,
        };

        assert_eq!(
            hf_dir_size_bytes(&test_http()?, &server.base_url(), &source),
            None
        );
        Ok(())
    }

    #[test]
    fn list_repo_files_surfaces_a_non_success_status() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path_contains("/api/models/");
            then.status(500);
        });

        match list_repo_files(&test_http()?, &server.base_url(), &hf_repo(None, None)?) {
            Ok(_) => anyhow::bail!("expected an HTTP error"),
            Err(e) => anyhow::ensure!(
                matches!(e, ModelFetchError::Http { .. }),
                "unexpected error: {e:?}"
            ),
        }
        Ok(())
    }
}
